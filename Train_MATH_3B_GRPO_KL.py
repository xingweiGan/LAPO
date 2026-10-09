#In this training, we donot consider epochs_per_rollout_batch not equals to 1. (i.e. all data only get used once)
#Also train_batch_size may not equal to rollout_batch_size.
#And when train_batch_size < rollout_batch_size, for each grpo_step all of generated data are finished for one rollout batch (rollout_batch_size), but after each effective batch
#(with train_batch_size), we should update the parameter and then do generation again (but actually we are using the generated output from the beginning).
#
# HF-based version: uses HuggingFace batched KV-cached generation for both training rollouts and eval
# (replaces vLLM for consistency with Main files)

from pathlib import Path
import sys
import os
import argparse
from contextlib import contextmanager
from transformers import PreTrainedModel
from typing import Literal, List, Callable, Dict, Sequence
import torch
import wandb

# Resolve bundled dependencies relative to this file, not the working directory.
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from math3b_support.adapters import run_tokenize_prompt_and_output, run_get_response_log_probs, run_sft_microbatch_train_step, run_compute_group_normalized_rewards, run_grpo_microbatch_train_step, run_compute_entropy
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.optim import AdamW
from math3b_support.drgrpo_grader import r1_zero_reward_fn, extract_boxed_answer
from math3b_support.math_baseline import r1_prompts_from_train
from datasets import load_dataset

def extract_math_gold(solution: str) -> str:
    """Extract the ground truth answer from a MATH solution (\\boxed{...})."""
    if "\\boxed" in solution:
        return extract_boxed_answer(solution)
    return solution


# ---------------------------------------------------------------------------
# Generation mode context manager
# ---------------------------------------------------------------------------

@contextmanager
def _generation_mode(model: PreTrainedModel):
    """Temporarily switch model from training mode to generation mode."""
    was_training = model.training
    had_grad_ckpt = getattr(model, "gradient_checkpointing", False)
    old_use_cache = getattr(model.config, "use_cache", False)

    model.eval()
    if had_grad_ckpt:
        model.gradient_checkpointing_disable()
    model.config.use_cache = True

    try:
        yield model
    finally:
        model.config.use_cache = old_use_cache
        if had_grad_ckpt:
            model.gradient_checkpointing_enable()
        if was_training:
            model.train()


# ---------------------------------------------------------------------------
# HF batched generation with KV cache
# ---------------------------------------------------------------------------

@torch.inference_mode()
def _hf_generate_batch(
    prompts: Sequence[str],
    model: PreTrainedModel,
    tokenizer: AutoTokenizer,
    device: torch.device,
    max_new_tokens: int = 512,
    min_new_tokens: int = 4,
    temperature: float = 1.0,
    stop_sequences: Sequence[str] = ("</answer>",),
    batch_size: int = 8,
) -> List[str]:
    """Generate using HF with KV cache, batched. Supports both sampling and greedy."""
    all_outputs: List[str] = []
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    for batch_start in range(0, len(prompts), batch_size):
        batch_prompts = prompts[batch_start : batch_start + batch_size]
        B = len(batch_prompts)

        encodings = [tokenizer(p, return_tensors="pt") for p in batch_prompts]
        prompt_lengths = [enc["input_ids"].shape[1] for enc in encodings]
        max_prompt_len = max(prompt_lengths)

        input_ids = torch.full((B, max_prompt_len), pad_id, dtype=torch.long, device=device)
        attention_mask = torch.zeros((B, max_prompt_len), dtype=torch.long, device=device)
        for i, enc in enumerate(encodings):
            plen = prompt_lengths[i]
            input_ids[i, max_prompt_len - plen :] = enc["input_ids"][0]
            attention_mask[i, max_prompt_len - plen :] = 1

        generated: List[List[int]] = [[] for _ in range(B)]
        finished = [False] * B
        past_kv = None

        for step in range(max_new_tokens):
            if step == 0:
                step_ids = input_ids
                step_mask = attention_mask
            else:
                step_ids = next_token_ids
                step_mask = torch.cat(
                    [attention_mask, torch.ones((B, 1), dtype=torch.long, device=device)], dim=1
                )
                attention_mask = step_mask

            out = model(input_ids=step_ids, attention_mask=step_mask, past_key_values=past_kv, use_cache=True)
            logits = out.logits[:, -1, :]

            if temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token_ids = torch.multinomial(probs, num_samples=1)  # (B, 1)
            else:
                next_token_ids = torch.argmax(logits, dim=-1, keepdim=True)

            past_kv = out.past_key_values

            tokens_flat = next_token_ids.squeeze(-1).tolist()
            all_done = True
            for i in range(B):
                if finished[i]:
                    continue
                generated[i].append(tokens_flat[i])
                if len(generated[i]) >= min_new_tokens:
                    if tokens_flat[i] == tokenizer.eos_token_id:
                        finished[i] = True
                        continue
                    decoded_so_far = tokenizer.decode(generated[i], skip_special_tokens=True)
                    if any(stop in decoded_so_far for stop in stop_sequences):
                        finished[i] = True
                        continue
                all_done = False
            if all_done:
                break

        for i in range(B):
            decoded = tokenizer.decode(generated[i], skip_special_tokens=True)
            for stop in stop_sequences:
                if stop in decoded:
                    decoded = decoded.split(stop)[0] + stop
                    break
            all_outputs.append(decoded)

        if (batch_start + B) % 256 == 0 or batch_start + B == len(prompts):
            print(f"  [HF Gen] {batch_start + B}/{len(prompts)}", flush=True)

    return all_outputs


def evaluate_hf(
    model: PreTrainedModel,
    tokenizer: AutoTokenizer,
    device: torch.device,
    reward_fn: Callable[[str, str], Dict[str, float]],
    prompts: List[str],
    ground_truths: List[str],
    max_new_tokens: int = 512,
    min_new_tokens: int = 4,
    batch_size: int = 8,
) -> List[float]:
    """Evaluate using HF generation (greedy). Returns [format_rate, accuracy]."""
    outputs = _hf_generate_batch(
        prompts=prompts,
        model=model,
        tokenizer=tokenizer,
        device=device,
        max_new_tokens=max_new_tokens,
        min_new_tokens=min_new_tokens,
        temperature=0.0,  # greedy
        stop_sequences=["</answer>"],
        batch_size=batch_size,
    )

    bins = {"11": 0, "10": 0, "01": 0, "00": 0}
    for text, gold in zip(outputs, ground_truths):
        scores = reward_fn(text, gold)
        fmt = 1 if scores.get("format_reward", 0.0) > 0 else 0
        ans = 1 if scores.get("answer_reward", 0.0) > 0 else 0
        bins[f"{fmt}{ans}"] += 1

    n = len(prompts)
    if n == 0:
        return [0.0, 0.0]

    format_rate = (bins["11"] + bins["10"]) / n
    accuracy = bins["11"] / n
    return [format_rate, accuracy]


def _resolve_device_layout(train_device: str = "cuda:0") -> tuple[str, str, bool]:
    eval_device = "cuda:1" if torch.cuda.is_available() and torch.cuda.device_count() > 1 else train_device
    use_dual_gpu = (
        torch.cuda.is_available()
        and train_device.startswith("cuda")
        and eval_device.startswith("cuda")
        and eval_device != train_device
    )
    return train_device, eval_device, use_dual_gpu


def main() -> None:
    hf_username = os.environ["HF_USERNAME"]
    all_logs = []
    #Parameters setup

    ################TUNNING AREA################
    beta=0.01
    learning_rate: float = 1e-5
    repo_id = f"{hf_username}/grpo-similar-small_KL_"
    repo_id= repo_id+str(beta)
    len_normalization = "mean"
    loss_type: Literal[
    "no_baseline", "reinforce_with_baseline", "grpo_clip", "grpo_clip_KL"
    ] = "grpo_clip_KL"
    use_std_normalization: bool = True
    ########################################################
    parser = argparse.ArgumentParser(description="GRPO training with KL regularization.")
    parser.add_argument("--wandb_api_key", default=os.environ.get("WANDB_API_KEY"))
    args = parser.parse_args()
    if args.wandb_api_key:
        wandb.login(key=args.wandb_api_key)

    n_grpo_steps: int = 40
    advantage_eps: float = 1e-6
    #number of (q,o) pairs per roll out including the redundant questions.
    rollout_batch_size: int = 2048
    #how many time same questions get asked
    group_size: int = 8
    sampling_temperature: float = 1.0
    sampling_min_tokens: int = 4 # As in Expiter, disallow empty string responses
    sampling_max_tokens: int = 512
    epochs_per_rollout_batch: int = 1 # On-policy
    #We restrict our choices on "no_baseline", "reinforce_with_baseline", "grpo_clip"
    n_q_per_rollout_batch = rollout_batch_size//group_size

    # Eval cadence
    eval_every_effective_batches = 8

    batch_gen_size: int = 8  # sub-batch size for HF generation

    SMOKE = os.environ.get("SMOKE", "0") == "1"
    if SMOKE:
        n_grpo_steps = 1
        rollout_batch_size = 16
        n_q_per_rollout_batch = rollout_batch_size // group_size
        eval_every_effective_batches = 1
        sampling_max_tokens = 64
        print(f"[SMOKE] Override-1: n_grpo_steps={n_grpo_steps}, rollout_batch_size={rollout_batch_size}, sampling_max_tokens={sampling_max_tokens}", flush=True)

    model_id="Qwen/Qwen2.5-3B-Instruct"
    sft_model_id: str = "xw1234gan/SFT_Qwen2.5-3B-Instruct_MATH"
    device_train, device_eval, use_dual_gpu = _resolve_device_layout("cuda:0")
    print(
        f"[Device] train={device_train}, eval={device_eval}, "
        f"dual_gpu={use_dual_gpu}",
        flush=True,
    )
    # Set-up on GPU A (trainning-only)
    model= AutoModelForCausalLM.from_pretrained(sft_model_id, torch_dtype=torch.bfloat16).to(device_train).train()
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    # ref_model on eval GPU to free VRAM on train GPU for optimizer states
    ref_model = AutoModelForCausalLM.from_pretrained(sft_model_id, torch_dtype=torch.bfloat16).to(device_eval).eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)
    optimizer = torch.optim.AdamW(
    model.parameters(), lr=learning_rate, weight_decay=0.0,
    betas=(0.9, 0.95),
    )
    optimizer.zero_grad()
    tokenizer=AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    #now let's evaluate its performance
    hf_dataset = "Maxwell-Jia/MATH"
    eval_size: int = 8 if SMOKE else 500
    ds_eval = load_dataset(hf_dataset, split="test")
    raw_eval_problems = list(ds_eval["problem"][:eval_size]) if eval_size > 0 else list(ds_eval["problem"])
    raw_eval_solutions = list(ds_eval["solution"][:eval_size]) if eval_size > 0 else list(ds_eval["solution"])
    raw_eval_gold = [extract_math_gold(sol) for sol in raw_eval_solutions]
    eval_problems, eval_gold = [], []
    for p, g in zip(raw_eval_problems, raw_eval_gold):
        if g is not None:
            eval_problems.append(p)
            eval_gold.append(g)
    n_dropped_eval = len(raw_eval_problems) - len(eval_problems)
    print(f"[Data] Eval: dropped {n_dropped_eval}/{len(raw_eval_problems)} examples with None gold answer")
    prompts_eval = r1_prompts_from_train(eval_problems)
    ground_truth_eval = eval_gold


    #Computation of size
    #train_batch_size: int = 256 # On-policy
    gradient_accumulation_steps: int = 128 # microbatch size is 2, will fit on H100
    num_egs_per_effective_batch=256 # On-policy
    microbatch_size= num_egs_per_effective_batch//gradient_accumulation_steps
    num_ebs = rollout_batch_size//num_egs_per_effective_batch
    if SMOKE:
        gradient_accumulation_steps = 2
        num_egs_per_effective_batch = 4
        microbatch_size = num_egs_per_effective_batch // gradient_accumulation_steps
        num_ebs = rollout_batch_size // num_egs_per_effective_batch
        print(f"[SMOKE] Override-2: ga={gradient_accumulation_steps}, microbatch={microbatch_size}, num_ebs={num_ebs}", flush=True)
    name = "Random"

    ###########Initialize Wandb####################
    # Initialize wandb at the beginning (before the training loop)
    model_short = model_id.split("/")[-1]
    dataset_short = "MATH"
    run_name = f"Extended_GRPO_KL_{model_short}_{dataset_short}_beta{beta}_lr{learning_rate}_mb{microbatch_size}_ga{gradient_accumulation_steps}_n{rollout_batch_size}_seed42_MATH3B_copy"
    wandb.init(
        project="MATH_3B_comparison",
        dir=str(PROJECT_ROOT),
        name=run_name,
    )
    ############################################################



    ################TRAININING####################################
    # Load MATH dataset for training
    ds_train = load_dataset(hf_dataset, split="train")
    raw_train_problems = list(ds_train["problem"])
    raw_train_solutions = list(ds_train["solution"])
    raw_train_gold = [extract_math_gold(sol) for sol in raw_train_solutions]
    train_problems, train_gold = [], []
    for p, g in zip(raw_train_problems, raw_train_gold):
        if g is not None:
            train_problems.append(p)
            train_gold.append(g)
    n_dropped_train = len(raw_train_problems) - len(train_problems)
    print(f"[Data] Train: dropped {n_dropped_train}/{len(raw_train_problems)} examples with None gold answer")

    # Double the dataset by repeating
    train_problems = train_problems * 2
    train_gold = train_gold * 2
    print(f"[Data] Doubled training data: {len(train_problems)} examples")

    train_offset = 0
    #First, sampling
    for grpo_step in range(n_grpo_steps):
        # Get next chunk of training data
        chunk_end = train_offset + n_q_per_rollout_batch
        if chunk_end <= len(train_problems):
            chunk_q = train_problems[train_offset:chunk_end]
            chunk_gold = train_gold[train_offset:chunk_end]
        else:
            chunk_q = train_problems[train_offset:] + train_problems[:chunk_end - len(train_problems)]
            chunk_gold = train_gold[train_offset:] + train_gold[:chunk_end - len(train_problems)]
        train_offset = chunk_end % len(train_problems)

    #Adjusting to r1-style prompts and answers
        prompt_train=r1_prompts_from_train(chunk_q)

        #make this duplicate times i.e make it group of prompts
        prompt_train_duplicate = [s for s in prompt_train for _ in range(group_size)]
        answer_train_duplicate_gold = [s for s in chunk_gold for _ in range(group_size)]

        #Get response using HF generation (with temperature sampling)
        print(f"[GRPO Step {grpo_step}] Generating rollouts via HF...", flush=True)
        with _generation_mode(model):
            response_train_duplicate = _hf_generate_batch(
                prompts=prompt_train_duplicate,
                model=model,
                tokenizer=tokenizer,
                device=torch.device(device_train),
                max_new_tokens=sampling_max_tokens,
                min_new_tokens=sampling_min_tokens,
                temperature=sampling_temperature,
                stop_sequences=["</answer>"],
                batch_size=batch_gen_size,
            )

        advantages_train,_,_ = run_compute_group_normalized_rewards(
        reward_fn=r1_zero_reward_fn ,
        rollout_responses= response_train_duplicate ,
        repeated_ground_truths = answer_train_duplicate_gold,
        group_size = group_size,
        advantage_eps = advantage_eps,
        normalize_by_std = True,
        )


        data_tokenized = run_tokenize_prompt_and_output(
        tokenizer=tokenizer,
        prompt_strs=prompt_train_duplicate,
        output_strs=response_train_duplicate  # Use generated responses, not ground truth!
        )

        #NEW!: Get log_prob from old policy
        whole_ids = data_tokenized["input_ids"]
        whole_lbl = data_tokenized["labels"]
        chunk_size = 2
        whole_logp_old_parts = []
        with torch.no_grad():
            for i in range(0, whole_ids.size(0), chunk_size):
                ids_chunk = whole_ids[i:i+chunk_size].to(device_train, non_blocking=True)
                lbl_chunk = whole_lbl[i:i+chunk_size].to(device_train, non_blocking=True)
                chunk_logp = run_get_response_log_probs(model, ids_chunk, lbl_chunk)["log_probs"]
                whole_logp_old_parts.append(chunk_logp.cpu())
                print(i)
        whole_logp_old = torch.cat(whole_logp_old_parts, dim=0)



    #Second, go through each sampled effective batch
        for eb in range(num_ebs):
            base = eb * num_egs_per_effective_batch
            end  = base + num_egs_per_effective_batch

        # slice one effective batch on CPU
            ids  = data_tokenized["input_ids"][base:end]
            lbls = data_tokenized["labels"][base:end]
            msk  = data_tokenized["response_mask"][base:end]
            advantages = advantages_train [base:end]
            logp_old = whole_logp_old [base:end]


            print(f"gradient_accumulation_steps: {gradient_accumulation_steps}")
            entropy_sum =0
            mask_elements_sum = 0
            loss_sum = 0

        #Third, split the effective batch to micro_batches in order to do gradient accumulation
        #eb_* means split from eb, which means it is for a single microbatch.
            for step in range(gradient_accumulation_steps):
                s = step * microbatch_size
                e = s + microbatch_size

                eb_ids = ids[s:e].to(device_train, non_blocking=True)
                eb_lbl = lbls[s:e].to(device_train, non_blocking=True)
                eb_msk = msk[s:e].to(device_train, non_blocking=True)
                eb_advantages= advantages[s:e].to(device_train, non_blocking=True)
                eb_out = run_get_response_log_probs(model, eb_ids, eb_lbl, return_token_entropy=True)
                eb_logp = eb_out["log_probs"]
                eb_logp_old= logp_old[s:e].to(device_train, non_blocking=True)
                with torch.no_grad():
                    eb_ref_logp = run_get_response_log_probs(ref_model, eb_ids.to(device_eval), eb_lbl.to(device_eval))["log_probs"]
                    eb_ref_logp = eb_ref_logp.to(device_train)
                print(f"Step {step}: advantages shape={eb_advantages.shape}, mean={eb_advantages.mean():.4f}, std={eb_advantages.std():.4f}")

                #NEW HERE!!!! prepare to edit!!!!!!!!!!!!!!
                with torch.no_grad():
                    eb_entropy = eb_out["token_entropy"]
                    entropy_sum += (eb_entropy * eb_msk).sum()
                    mask_elements_sum += eb_msk.sum()

            # scale manually for accumulation;
                eb_loss, _=run_grpo_microbatch_train_step(policy_log_probs=eb_logp,
                                               response_mask=eb_msk,
                                               gradient_accumulation_steps=gradient_accumulation_steps,
                                               loss_type=loss_type,
                                               advantages=eb_advantages,
                                               old_log_probs=eb_logp_old,
                                               ref_log_probs=eb_ref_logp,
                                               cliprange=0.2,
                                               normalization=len_normalization,
                                               beta=beta
                                               )
                loss_sum += eb_loss.detach().item()


            #!!!!!!!!!!!!!!!!Start to edit here!!!#!!!!!!!!!!!!!!!!#!!!!!!!!!!!!!!!!#!!!!!!!!!!!!!!!!
            #START TO LOG!!
            avg_entropy=entropy_sum/mask_elements_sum
            grad_norm = torch.sqrt(sum(
                (p.grad.detach().float().norm(2)**2)
                for p in model.parameters() if p.grad is not None
            ))
            torch.cuda.empty_cache()
            optimizer.step() #we update the policy here!!
            # Calculate global step
            global_step = grpo_step * num_ebs + eb

            # Only evaluate every eval_every_effective_batches using HF generation
            accuracy = None
            if (global_step + 1) % eval_every_effective_batches == 0:
                print(f"[Eval] Starting eval at step={global_step} via HF...", flush=True)
                with _generation_mode(model):
                    _, accuracy = evaluate_hf(
                        model=model,
                        tokenizer=tokenizer,
                        device=torch.device(device_train),
                        reward_fn=r1_zero_reward_fn,
                        prompts=prompts_eval,
                        ground_truths=ground_truth_eval,
                        max_new_tokens=sampling_max_tokens,
                        min_new_tokens=sampling_min_tokens,
                        batch_size=batch_gen_size,
                    )

            loss=eb_loss #This is on the trainning set

            # Log to wandb and keep a local record
            log_entry = {
                "global_step": global_step,
                "grpo_step": grpo_step,
                "effective_batch": eb,
                "train/loss": loss_sum,
                "train_loss": loss_sum,
                "train/avg_entropy": avg_entropy.item(),
                "train/grad_norm": grad_norm.item(),
            }
            if accuracy is not None:
                log_entry["eval/accuracy"] = accuracy
            print("WANDAB LOG!")
            print(grpo_step)
            print(global_step)

            wandb.log(log_entry, step=global_step)
            all_logs.append(log_entry)
            optimizer.zero_grad()

    # Push trained policy to HuggingFace
    if not SMOKE:
        repo_id = f"{hf_username}/{run_name}"
        token = os.environ["HF_TOKEN"]
        model.push_to_hub(repo_id, token=token, private=False, safe_serialization=True)
        tokenizer.push_to_hub(repo_id, token=token)
        print(f"Model pushed to HF: {repo_id}")
    else:
        print("[SMOKE] Skipping push_to_hub", flush=True)

    wandb.finish()


if __name__ == "__main__":
    main()
