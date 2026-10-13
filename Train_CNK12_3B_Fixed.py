"""Figure 4 fixed-weight LAPO training on cn_k12 (Qwen2.5-3B).

This keeps the current LAPO fixed-weight training and tracking semantics while
replacing the MATH source with the historical, disjoint
NuminaMath cn_k12 split used by the Assignment3 Figure 4 experiment.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List

import torch
import wandb
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer

from Train_MATH_3B_Fixed import (
    _generation_mode,
    _is_correct,
    _mixed_generate_batch_fast,
    _mixed_log_probs,
    _set_reproducibility,
    _vllm_policy_generate,
)
from math3b_support.SFT_policy import init_vllm, load_policy_into_vllm_instance
from math3b_support.adapters import (
    run_compute_group_normalized_rewards,
    run_grpo_microbatch_train_step,
    run_tokenize_prompt_and_output,
)
from math3b_support.cnk12_data import load_cnk12_splits
from math3b_support.drgrpo_grader import r1_zero_reward_fn
from math3b_support.math_baseline import r1_prompts_from_train


PROJECT_ROOT = Path(__file__).resolve().parent

# Keep these values aligned with the historical Figure 4 fixed experiment.
learning_rate: float = 1e-5
loss_type = "grpo_clip"
cliprange = 0.2
len_normalization = "mean"
use_std_normalization: bool = True
advantage_eps: float = 1e-6

alpha_init: float = 0.5
val_size: int = 100
eval_size: int = 500
update_interval_ebs: int = 8
eval_every_effective_batches: int = update_interval_ebs

n_grpo_steps: int = 20
rollout_temperature: float = 1.0
eval_temperature: float = 0.0
min_new_tokens: int = 4
max_new_tokens: int = 512

rollout_batch_size: int = 256 * update_interval_ebs
group_size: int = 8
num_egs_per_effective_batch: int = 256
gradient_accumulation_steps: int = 128
microbatch_size: int = num_egs_per_effective_batch // gradient_accumulation_steps
logp_chunk_size: int = 2

batch_gen_size: int = 32

model_size_label = "3B"
model_id = "Qwen/Qwen2.5-3B-Instruct"
sft_model_id = "xw1234gan/SFT_Qwen2.5-3B-Instruct_cnk12"

device_train = "cuda:1"
gpu_memory_utilization: float = 0.45

stop_sequence = "</answer>"
wandb_project = "CNK12_comparison"


def main(
    *,
    run_model_size_label: str = model_size_label,
    run_model_id: str = model_id,
    run_sft_model_id: str = sft_model_id,
    run_batch_gen_size: int = batch_gen_size,
) -> None:
    parser = argparse.ArgumentParser(
        description=(
            f"Train cn_k12 {run_model_size_label} with fixed logit mixing "
            "(alpha=0.5)."
        )
    )
    parser.add_argument("--hf_token", default=os.environ.get("HF_TOKEN"))
    parser.add_argument("--wandb_api_key", default=os.environ.get("WANDB_API_KEY"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", action="store_true")
    args = parser.parse_args()

    hf_username = os.environ["HF_USERNAME"]
    _set_reproducibility(seed=args.seed, deterministic=args.deterministic)
    assert rollout_batch_size % group_size == 0, (
        "rollout_batch_size must be divisible by group_size"
    )
    assert num_egs_per_effective_batch % gradient_accumulation_steps == 0, (
        "num_egs_per_effective_batch must be divisible by "
        "gradient_accumulation_steps"
    )

    model_short = run_model_id.split("/")[-1]
    experiment_tag = (
        f"CNK12_{run_model_size_label}_Fixed"
        f"_G{group_size}_outer{n_grpo_steps}"
    )
    n_q_per_rollout_batch = rollout_batch_size // group_size
    num_ebs = rollout_batch_size // num_egs_per_effective_batch

    device = torch.device(
        device_train if torch.cuda.device_count() > 1 else "cuda:0"
    )
    device_eval = torch.device("cuda:0") if torch.cuda.device_count() > 1 else device
    token = args.hf_token
    alpha = float(alpha_init)

    policy_model = AutoModelForCausalLM.from_pretrained(
        run_model_id,
        token=token,
        torch_dtype=torch.bfloat16,
    ).to(device).train()
    sft_model = AutoModelForCausalLM.from_pretrained(
        run_sft_model_id,
        token=token,
        torch_dtype=torch.bfloat16,
    ).to(device_eval).eval()

    policy_model.gradient_checkpointing_enable()
    policy_model.config.use_cache = False
    for parameter in sft_model.parameters():
        parameter.requires_grad_(False)

    vllm_gpu_util = gpu_memory_utilization
    if device_eval == device:
        vllm_gpu_util = min(vllm_gpu_util, 0.30)
    print(
        f"[Device] train={device}, eval/vllm={device_eval}, "
        f"vllm_gpu_memory_utilization={vllm_gpu_util}",
        flush=True,
    )
    llm = init_vllm(
        model_id=run_model_id,
        device=str(device_eval),
        seed=args.seed,
        gpu_memory_utilization=vllm_gpu_util,
    )

    tokenizer = AutoTokenizer.from_pretrained(run_model_id, token=token)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if args.wandb_api_key:
        wandb.login(key=args.wandb_api_key)

    run_name = (
        f"{experiment_tag}_{model_short}"
        f"_lr{learning_rate}_mb{microbatch_size}_ga{gradient_accumulation_steps}"
        f"_n{rollout_batch_size}_seed{args.seed}"
    )
    wandb.init(
        project=wandb_project,
        dir=str(PROJECT_ROOT),
        name=run_name,
        config={
            "seed": args.seed,
            "deterministic": args.deterministic,
            "alpha": alpha,
            "alpha_fixed": True,
        },
    )

    optimizer = AdamW(
        policy_model.parameters(),
        lr=learning_rate,
        weight_decay=0.0,
        betas=(0.9, 0.95),
    )
    optimizer.zero_grad()
    total_param_tensors = sum(1 for _ in policy_model.parameters())

    (train_split, eval_split) = load_cnk12_splits(
        train_keep=7500,
        eval_keep=eval_size,
        seed=args.seed,
    )
    train_problems, train_solutions, train_gold = train_split
    eval_problems, eval_solutions, eval_gold = eval_split
    print(
        f"[Data] cn_k12 train={len(train_problems)}, "
        f"eval={len(eval_problems)}",
        flush=True,
    )

    train_problems = train_problems * 2
    train_solutions = train_solutions * 2
    train_gold = train_gold * 2
    print(f"[Data] Doubled training data: {len(train_problems)} examples")
    eval_prompts = r1_prompts_from_train(eval_problems)
    train_offset = 0

    for grpo_step in range(n_grpo_steps):
        chunk_end = train_offset + n_q_per_rollout_batch
        if chunk_end <= len(train_problems):
            chunk_q = train_problems[train_offset:chunk_end]
            chunk_gold = train_gold[train_offset:chunk_end]
        else:
            overflow = chunk_end - len(train_problems)
            chunk_q = train_problems[train_offset:] + train_problems[:overflow]
            chunk_gold = train_gold[train_offset:] + train_gold[:overflow]
        train_offset = chunk_end % len(train_problems)

        prompt_train = r1_prompts_from_train(chunk_q)
        prompt_train_duplicate = [
            prompt for prompt in prompt_train for _ in range(group_size)
        ]
        answer_train_duplicate_gold = [
            gold for gold in chunk_gold for _ in range(group_size)
        ]

        print(f"[AdaptiveWeight Sampling] grpo_step={grpo_step}", flush=True)
        with _generation_mode(policy_model):
            rollout_responses = _mixed_generate_batch_fast(
                prompts=prompt_train_duplicate,
                policy_model=policy_model,
                sft_model=sft_model,
                tokenizer=tokenizer,
                device=device,
                alpha=alpha,
                max_new_tokens=max_new_tokens,
                min_new_tokens=min_new_tokens,
                temperature=rollout_temperature,
                stop_sequences=[stop_sequence],
                batch_size=run_batch_gen_size,
            )

        advantages, raw_rewards, reward_md = run_compute_group_normalized_rewards(
            reward_fn=r1_zero_reward_fn,
            rollout_responses=rollout_responses,
            repeated_ground_truths=answer_train_duplicate_gold,
            group_size=group_size,
            advantage_eps=advantage_eps,
            normalize_by_std=use_std_normalization,
        )

        data_tokenized = run_tokenize_prompt_and_output(
            tokenizer=tokenizer,
            prompt_strs=prompt_train_duplicate,
            output_strs=rollout_responses,
        )
        whole_ids = data_tokenized["input_ids"]
        whole_labels = data_tokenized["labels"]

        whole_logp_old_parts: List[torch.Tensor] = []
        with torch.no_grad():
            for index in range(0, whole_ids.size(0), logp_chunk_size):
                ids_chunk = whole_ids[index : index + logp_chunk_size].to(
                    device,
                    non_blocking=True,
                )
                labels_chunk = whole_labels[index : index + logp_chunk_size].to(
                    device,
                    non_blocking=True,
                )
                chunk_logp = _mixed_log_probs(
                    policy_model=policy_model,
                    sft_model=sft_model,
                    input_ids=ids_chunk,
                    labels=labels_chunk,
                    alpha=alpha,
                )["log_probs"]
                whole_logp_old_parts.append(chunk_logp.cpu())
        whole_logp_old = torch.cat(whole_logp_old_parts, dim=0)

        global_step_base = grpo_step * num_ebs
        for effective_batch in range(num_ebs):
            base = effective_batch * num_egs_per_effective_batch
            end = base + num_egs_per_effective_batch
            ids = data_tokenized["input_ids"][base:end]
            labels = data_tokenized["labels"][base:end]
            response_mask = data_tokenized["response_mask"][base:end]
            batch_advantages = advantages[base:end]
            logp_old = whole_logp_old[base:end]

            entropy_sum = torch.tensor(0.0, device=device)
            mask_elements_sum = torch.tensor(0.0, device=device)
            loss_sum = 0.0

            for accumulation_step in range(gradient_accumulation_steps):
                start = accumulation_step * microbatch_size
                stop = start + microbatch_size
                micro_ids = ids[start:stop].to(device, non_blocking=True)
                micro_labels = labels[start:stop].to(device, non_blocking=True)
                micro_mask = response_mask[start:stop].to(device, non_blocking=True)
                micro_advantages = batch_advantages[start:stop].to(
                    device,
                    non_blocking=True,
                )
                micro_logp_old = logp_old[start:stop].to(
                    device,
                    non_blocking=True,
                )

                micro_output = _mixed_log_probs(
                    policy_model=policy_model,
                    sft_model=sft_model,
                    input_ids=micro_ids,
                    labels=micro_labels,
                    alpha=alpha,
                    return_token_entropy=True,
                )
                micro_logp = micro_output["log_probs"]
                micro_entropy = micro_output["token_entropy"]
                entropy_sum += (micro_entropy * micro_mask).sum()
                mask_elements_sum += micro_mask.sum()

                micro_loss, _ = run_grpo_microbatch_train_step(
                    policy_log_probs=micro_logp,
                    response_mask=micro_mask,
                    gradient_accumulation_steps=gradient_accumulation_steps,
                    loss_type=loss_type,
                    advantages=micro_advantages,
                    old_log_probs=micro_logp_old,
                    cliprange=cliprange,
                    normalization=len_normalization,
                )
                loss_sum += micro_loss.detach().item()

            avg_entropy = entropy_sum / torch.clamp(mask_elements_sum, min=1.0)
            grad_params = [
                parameter
                for parameter in policy_model.parameters()
                if parameter.grad is not None
            ]
            sample_param = grad_params[0] if grad_params else None
            sample_before = (
                sample_param.detach().view(-1)[:8].clone().float()
                if sample_param is not None
                else None
            )
            grad_norm = torch.sqrt(
                sum(
                    parameter.grad.detach().float().norm(2) ** 2
                    for parameter in policy_model.parameters()
                    if parameter.grad is not None
                )
            )
            grad_param_befores = [
                parameter.detach().cpu().clone() for parameter in grad_params
            ]
            torch.cuda.empty_cache()
            optimizer.step()

            sample_update_max = None
            full_update_max = None
            if sample_param is not None and sample_before is not None:
                sample_after = sample_param.detach().view(-1)[:8].float()
                sample_update_max = (
                    sample_after - sample_before.to(sample_after.device)
                ).abs().max().item()
            if grad_params:
                full_update_max = max(
                    (
                        parameter.detach().cpu().float() - before.float()
                    ).abs().max().item()
                    for parameter, before in zip(grad_params, grad_param_befores)
                )
            optimizer.zero_grad()

            global_step = global_step_base + effective_batch
            print(
                f"[AdaptiveWeight Trainning] step={global_step} "
                f"loss={loss_sum:.4f} avg_entropy={avg_entropy.item():.4f} "
                f"grad_norm={grad_norm.item():.4f} alpha={alpha:.4f}"
            )
            print(
                f"Check gradients! Parameters_all={total_param_tensors} "
                f"Parameters_gradient={len(grad_params)} "
                f"sample_update_max={sample_update_max} "
                f"full_update_max={full_update_max}"
            )
            log_entry = {
                "global_step": global_step,
                "grpo_step": grpo_step,
                "effective_batch": effective_batch,
                "train/loss": loss_sum,
                "train_loss": loss_sum,
                "train/avg_entropy": avg_entropy.item(),
                "train/grad_norm": grad_norm.item(),
                "alpha": alpha,
                "train/alpha": alpha,
            }

            if (global_step + 1) % eval_every_effective_batches == 0:
                print(
                    f"[AdaptiveWeight Evaluation] starting eval at "
                    f"step={global_step} ",
                    flush=True,
                )
                with _generation_mode(policy_model):
                    eval_outputs = _mixed_generate_batch_fast(
                        prompts=eval_prompts,
                        policy_model=policy_model,
                        sft_model=sft_model,
                        tokenizer=tokenizer,
                        device=device,
                        alpha=alpha,
                        max_new_tokens=max_new_tokens,
                        min_new_tokens=min_new_tokens,
                        temperature=eval_temperature,
                        stop_sequences=[stop_sequence],
                        batch_size=run_batch_gen_size,
                    )

                load_policy_into_vllm_instance(policy_model, llm)
                eval_pi_theta_outputs = _vllm_policy_generate(
                    llm=llm,
                    prompts=eval_prompts,
                    temperature=eval_temperature,
                    max_new_tokens=max_new_tokens,
                    min_new_tokens=min_new_tokens,
                    stop_sequences=[stop_sequence],
                )

                eval_correct = sum(
                    _is_correct(output, gold)
                    for output, gold in zip(eval_outputs, eval_gold)
                )
                eval_accuracy = eval_correct / max(1, len(eval_gold))
                eval_pi_theta_correct = sum(
                    _is_correct(output, gold)
                    for output, gold in zip(eval_pi_theta_outputs, eval_gold)
                )
                eval_pi_theta = eval_pi_theta_correct / max(1, len(eval_gold))
                log_entry["eval/accuracy"] = eval_accuracy
                log_entry["eval_pi_theta"] = eval_pi_theta

            print(
                f"[AdaptiveWeight Wandb] wandb log data: {log_entry}",
                flush=True,
            )
            wandb.log(log_entry, step=global_step)

        print(
            f"[FixedAlpha] grpo_step={grpo_step} alpha={alpha:.4f} (fixed)",
            flush=True,
        )

    wandb.finish()

    # Hugging Face is the final external save: upload one final model repository.
    repo_id = f"{hf_username}/{run_name}"
    policy_model.push_to_hub(
        repo_id,
        token=token,
        private=False,
        safe_serialization=True,
    )
    tokenizer.push_to_hub(repo_id, token=token)
    print(f"Model pushed to HF: {repo_id}")


if __name__ == "__main__":
    main()
