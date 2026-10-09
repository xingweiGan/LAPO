
# Train a policy with GRPO-style RL while mixing logits with a frozen SFT model.
# The mixing weight alpha is updated after each GRPO step by comparing the
# trainable, mixed, and frozen-SFT policies on the same validation questions.
#
# EFFICIENCY version: replaces sequential token-by-token generation with:
#   - vLLM for policy-only generation (~50-60x speedup)
#   - Batched HF loop with KV cache for mixed generation (~20-30x speedup)

from __future__ import annotations

import math
import os
import random
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
import wandb
import argparse
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
from vllm import LLM, SamplingParams
from datasets import load_dataset

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from math3b_support.adapters import (  # noqa: E402
    run_compute_group_normalized_rewards,
    run_grpo_microbatch_train_step,
    run_tokenize_prompt_and_output,
)
from math3b_support.drgrpo_grader import r1_zero_reward_fn  # noqa: E402
from math3b_support.math_baseline import (  # noqa: E402
    r1_prompts_from_train,
)
# Import the bundled SFT helpers.
from math3b_support.SFT_policy import (  # noqa: E402
    init_vllm,
    load_policy_into_vllm_instance,
    extract_math_gold,
)


# ---------------------------------------------------------------------------
# Logit mixing & log-prob helpers (unchanged from original)
# ---------------------------------------------------------------------------

def _mix_logits(
    logits_theta: torch.Tensor,
    logits_sft: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    alpha_t = torch.as_tensor(alpha, dtype=logits_theta.dtype, device=logits_theta.device)
    return (1.0 - alpha_t) * logits_theta + alpha_t * logits_sft


def _mixed_log_probs(
    policy_model: PreTrainedModel,
    sft_model: PreTrainedModel,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    alpha: float,
    return_token_entropy: bool = False,
) -> Dict[str, torch.Tensor]:
    logits_theta = policy_model(input_ids=input_ids).logits
    sft_device = next(sft_model.parameters()).device
    with torch.no_grad():
        logits_sft = sft_model(input_ids=input_ids.to(sft_device)).logits

    logits_sft = logits_sft.to(dtype=logits_theta.dtype, device=logits_theta.device)
    seq_len = logits_theta.size(1)
    chunk_count = 2
    chunk_size = max(1, (seq_len + chunk_count - 1) // chunk_count)
    log_probs_parts = []
    entropy_parts = [] if return_token_entropy else None
    for start in range(0, seq_len, chunk_size):
        end = min(start + chunk_size, seq_len)

        logits_theta_chunk = logits_theta[:, start:end, :]
        logits_sft_chunk = logits_sft[:, start:end, :]
        mixed_logits = _mix_logits(logits_theta_chunk, logits_sft_chunk, alpha)
        log_probs_all = torch.log_softmax(mixed_logits, dim=-1)
        log_probs_chunk = torch.gather(
            log_probs_all, -1, labels[:, start:end].unsqueeze(-1)
        ).squeeze(-1)
        log_probs_parts.append(log_probs_chunk)
        if return_token_entropy:
            with torch.no_grad():
                log_probs_all_det = torch.log_softmax(mixed_logits.detach(), dim=-1)
                probs = torch.exp(log_probs_all_det)
                token_entropy = -(probs * log_probs_all_det).sum(dim=-1)
            entropy_parts.append(token_entropy)
    log_probs = torch.cat(log_probs_parts, dim=1)
    out = {"log_probs": log_probs}
    if return_token_entropy:
        out["token_entropy"] = torch.cat(entropy_parts, dim=1)
    return out


# ---------------------------------------------------------------------------
# Correctness checker (unchanged)
# ---------------------------------------------------------------------------

def _is_correct(text: str, gold: str) -> int:
    scores = r1_zero_reward_fn(text, gold)
    fmt = 1 if scores.get("format_reward", 0.0) > 0 else 0
    ans = 1 if scores.get("answer_reward", 0.0) > 0 else 0
    return 1 if (fmt == 1 and ans == 1) else 0


def _cyclic_indices(start: int, count: int, total: int) -> List[int]:
    """Return the next ``count`` sequential indices, wrapping at ``total``."""
    if total <= 0:
        raise ValueError("Cannot take indices from an empty dataset.")
    if count <= 0:
        raise ValueError("count must be positive.")
    return [(start + offset) % total for offset in range(count)]


def _validation_weighted_alpha_update(
    *,
    alpha: float,
    q_theta: float,
    q_mix: float,
    q_ref: float,
    temperature: float,
    update_rate: float,
    alpha_min: float,
    alpha_max: float,
) -> Dict[str, float]:
    """Compute the validation-performance-weighted controller update."""
    if temperature <= 0:
        raise ValueError("validation controller temperature must be positive.")
    if not 0 < update_rate <= 1:
        raise ValueError("alpha update rate must be in (0, 1].")
    if not 0 < alpha_min < alpha_max < 1:
        raise ValueError("alpha bounds must satisfy 0 < alpha_min < alpha_max < 1.")
    if not alpha_min <= alpha <= alpha_max:
        raise ValueError("current alpha must lie within the configured bounds.")
    if not all(0.0 <= accuracy <= 1.0 for accuracy in (q_theta, q_mix, q_ref)):
        raise ValueError("validation accuracies must be in [0, 1].")

    scaled_accuracies = [
        q_theta / temperature,
        q_mix / temperature,
        q_ref / temperature,
    ]
    max_scaled_accuracy = max(scaled_accuracies)
    exp_scores = [
        math.exp(score - max_scaled_accuracy)
        for score in scaled_accuracies
    ]
    score_sum = sum(exp_scores)
    w_theta, w_mix, w_ref = [score / score_sum for score in exp_scores]

    # Candidate reference weights are 0 for theta, current alpha for the
    # mixed policy, and 1 for the frozen reference policy.
    alpha_target = w_mix * alpha + w_ref
    alpha_smoothed = (1.0 - update_rate) * alpha + update_rate * alpha_target
    alpha_new = min(max(alpha_smoothed, alpha_min), alpha_max)

    return {
        "w_theta": float(w_theta),
        "w_mix": float(w_mix),
        "w_ref": float(w_ref),
        "alpha_target": float(alpha_target),
        "alpha_new": float(alpha_new),
    }


# ---------------------------------------------------------------------------
# Generation mode context manager (NEW)
# ---------------------------------------------------------------------------

@contextmanager
def _generation_mode(model: PreTrainedModel):
    """Temporarily switch *policy_model* from training mode to generation mode
    (eval, KV cache enabled, gradient checkpointing off) and restore afterwards."""
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
# Batched mixed generation with KV cache (NEW — replaces all _mixed_generate*)
# ---------------------------------------------------------------------------

@torch.inference_mode()
def _mixed_generate_batch_fast(
    prompts: Sequence[str],
    policy_model: PreTrainedModel,
    sft_model: PreTrainedModel,
    tokenizer: AutoTokenizer,
    device: torch.device,
    alpha: float,
    max_new_tokens: int,
    min_new_tokens: int,
    temperature: float,
    stop_sequences: Sequence[str],
    batch_size: int = 32,
) -> List[str]:
    """Generate from the mixed policy (policy + SFT) using batched KV-cached
    decoding.  Processes *batch_size* prompts at a time for ~20-30x speedup
    over the old sequential-per-prompt approach."""

    all_outputs: List[str] = []
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    for batch_start in range(0, len(prompts), batch_size):
        batch_prompts = prompts[batch_start : batch_start + batch_size]
        B = len(batch_prompts)

        # --- left-pad tokenisation for batched generation ---
        encodings = [tokenizer(p, return_tensors="pt") for p in batch_prompts]
        prompt_lengths = [enc["input_ids"].shape[1] for enc in encodings]
        max_prompt_len = max(prompt_lengths)

        # Build left-padded input_ids and attention_mask
        input_ids = torch.full((B, max_prompt_len), pad_id, dtype=torch.long, device=device)
        attention_mask = torch.zeros((B, max_prompt_len), dtype=torch.long, device=device)
        for i, enc in enumerate(encodings):
            plen = prompt_lengths[i]
            input_ids[i, max_prompt_len - plen :] = enc["input_ids"][0]
            attention_mask[i, max_prompt_len - plen :] = 1

        # Per-sequence generated tokens
        generated: List[List[int]] = [[] for _ in range(B)]
        finished = [False] * B
        past_theta = None
        past_sft = None

        for step in range(max_new_tokens):
            if step == 0:
                step_ids = input_ids
                step_mask = attention_mask
            else:
                step_ids = next_token_ids  # (B, 1) from previous iteration
                # Extend attention mask by one position
                step_mask = torch.cat(
                    [attention_mask, torch.ones((B, 1), dtype=torch.long, device=device)],
                    dim=1,
                )
                attention_mask = step_mask

            out_theta = policy_model(
                input_ids=step_ids,
                attention_mask=step_mask,
                past_key_values=past_theta,
                use_cache=True,
            )
            sft_dev = next(sft_model.parameters()).device
            out_sft = sft_model(
                input_ids=step_ids.to(sft_dev),
                attention_mask=step_mask.to(sft_dev),
                past_key_values=past_sft,
                use_cache=True,
            )

            logits_theta = out_theta.logits[:, -1, :]
            logits_sft = out_sft.logits[:, -1, :].to(dtype=logits_theta.dtype, device=logits_theta.device)
            mixed_logits = _mix_logits(logits_theta, logits_sft, alpha)
            if step < min_new_tokens and tokenizer.eos_token_id is not None:
                mixed_logits[:, tokenizer.eos_token_id] = -torch.inf

            if temperature > 0:
                probs = torch.softmax(mixed_logits / temperature, dim=-1)
                next_token_ids = torch.multinomial(probs, num_samples=1)  # (B, 1)
            else:
                next_token_ids = torch.argmax(mixed_logits, dim=-1, keepdim=True)

            past_theta = out_theta.past_key_values
            past_sft = out_sft.past_key_values

            # Record tokens & check stopping per sequence
            tokens_flat = next_token_ids.squeeze(-1).tolist()
            all_done = True
            for i in range(B):
                if finished[i]:
                    all_done = all_done and True
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

        # Decode & truncate at stop sequences
        for i in range(B):
            decoded = tokenizer.decode(generated[i], skip_special_tokens=True)
            for stop in stop_sequences:
                if stop in decoded:
                    decoded = decoded.split(stop)[0] + stop
                    break
            all_outputs.append(decoded)

        print(
            f"[AdaptiveWeight] Finished mixed-gen batch "
            f"{batch_start + B}/{len(prompts)}",
            flush=True,
        )

    return all_outputs


@torch.inference_mode()
def _frozen_generate_batch_fast(
    prompts: Sequence[str],
    model: PreTrainedModel,
    tokenizer: AutoTokenizer,
    max_new_tokens: int,
    min_new_tokens: int,
    temperature: float,
    stop_sequences: Sequence[str],
    batch_size: int = 32,
) -> List[str]:
    """Generate once from the frozen reference model with KV caching."""
    all_outputs: List[str] = []
    model_device = next(model.parameters()).device
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    for batch_start in range(0, len(prompts), batch_size):
        batch_prompts = prompts[batch_start : batch_start + batch_size]
        batch_len = len(batch_prompts)

        encodings = [tokenizer(prompt, return_tensors="pt") for prompt in batch_prompts]
        prompt_lengths = [encoding["input_ids"].shape[1] for encoding in encodings]
        max_prompt_len = max(prompt_lengths)

        input_ids = torch.full(
            (batch_len, max_prompt_len),
            pad_id,
            dtype=torch.long,
            device=model_device,
        )
        attention_mask = torch.zeros(
            (batch_len, max_prompt_len),
            dtype=torch.long,
            device=model_device,
        )
        for index, encoding in enumerate(encodings):
            prompt_len = prompt_lengths[index]
            input_ids[index, max_prompt_len - prompt_len :] = encoding["input_ids"][0].to(model_device)
            attention_mask[index, max_prompt_len - prompt_len :] = 1

        generated: List[List[int]] = [[] for _ in range(batch_len)]
        finished = [False] * batch_len
        past_key_values = None

        for step in range(max_new_tokens):
            if step == 0:
                step_ids = input_ids
                step_mask = attention_mask
            else:
                step_ids = next_token_ids
                step_mask = torch.cat(
                    [
                        attention_mask,
                        torch.ones(
                            (batch_len, 1),
                            dtype=torch.long,
                            device=model_device,
                        ),
                    ],
                    dim=1,
                )
                attention_mask = step_mask

            model_output = model(
                input_ids=step_ids,
                attention_mask=step_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )
            logits = model_output.logits[:, -1, :]
            if step < min_new_tokens and tokenizer.eos_token_id is not None:
                logits[:, tokenizer.eos_token_id] = -torch.inf
            if temperature > 0:
                probabilities = torch.softmax(logits / temperature, dim=-1)
                next_token_ids = torch.multinomial(probabilities, num_samples=1)
            else:
                next_token_ids = torch.argmax(logits, dim=-1, keepdim=True)
            past_key_values = model_output.past_key_values

            tokens_flat = next_token_ids.squeeze(-1).tolist()
            all_done = True
            for index in range(batch_len):
                if finished[index]:
                    continue
                generated[index].append(tokens_flat[index])
                if len(generated[index]) >= min_new_tokens:
                    if tokens_flat[index] == tokenizer.eos_token_id:
                        finished[index] = True
                        continue
                    decoded_so_far = tokenizer.decode(
                        generated[index],
                        skip_special_tokens=True,
                    )
                    if any(stop in decoded_so_far for stop in stop_sequences):
                        finished[index] = True
                        continue
                all_done = False

            if all_done:
                break

        for generated_ids in generated:
            decoded = tokenizer.decode(generated_ids, skip_special_tokens=True)
            for stop in stop_sequences:
                if stop in decoded:
                    decoded = decoded.split(stop)[0] + stop
                    break
            all_outputs.append(decoded)

        print(
            f"[ValWeightedAdaptive] Finished frozen-reference batch "
            f"{batch_start + batch_len}/{len(prompts)}",
            flush=True,
        )

    return all_outputs


# ---------------------------------------------------------------------------
# vLLM policy-only generation helpers
# ---------------------------------------------------------------------------

def _vllm_policy_generate(
    llm: LLM,
    prompts: Sequence[str],
    temperature: float,
    max_new_tokens: int,
    min_new_tokens: int,
    stop_sequences: Sequence[str],
) -> List[str]:
    """Generate from policy only using vLLM (batched, very fast)."""
    params = SamplingParams(
        temperature=temperature,
        top_p=1.0,
        max_tokens=max_new_tokens,
        min_tokens=min_new_tokens,
        stop=list(stop_sequences),
        include_stop_str_in_output=True,
    )
    raw_outputs = llm.generate(prompts, params)
    results: List[str] = []
    for out in raw_outputs:
        text = out.outputs[0].text if out.outputs else ""
        results.append(text)
    return results


def _load_policy_into_vllm_fresh(
    policy_model: PreTrainedModel,
    llm: LLM,
) -> None:
    """Load current policy weights and invalidate KV cache from older weights."""
    load_policy_into_vllm_instance(policy_model, llm)
    reset_prefix_cache = getattr(llm, "reset_prefix_cache", None)
    if reset_prefix_cache is None:
        raise RuntimeError(
            "This training script requires vLLM.reset_prefix_cache() after "
            "loading updated policy weights."
        )
    reset_succeeded = reset_prefix_cache()
    if reset_succeeded is False:
        raise RuntimeError("vLLM prefix cache could not be reset after loading policy weights.")


##############################
# TUNING AREA (edit as needed)
##############################
learning_rate: float = 1e-5
loss_type = "grpo_clip"
cliprange = 0.2
len_normalization = "mean"
use_std_normalization: bool = True
advantage_eps: float = 1e-6

alpha_init: float = 0.5
val_size: int = 100
val_controller_temperature: float = 0.25
alpha_update_rate: float = 0.50
alpha_min: float = 0.05
alpha_max: float = 0.95
eval_size: int = 500
update_interval_ebs: int = 8  # K effective batches per GRPO step
eval_every_effective_batches: int = update_interval_ebs

n_grpo_steps: int = 40
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

batch_gen_size: int = 8  # sub-batch size for _mixed_generate_batch_fast

model_id = "Qwen/Qwen2.5-3B-Instruct"
sft_model_id = "xw1234gan/SFT_Qwen2.5-3B-Instruct_MATH"
hf_dataset = "Maxwell-Jia/MATH"

device_train = "cuda:0"
gpu_memory_utilization: float = 0.45

stop_sequence = "</answer>"
wandb_project = "MATH_3B_comparison"


def _set_reproducibility(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True, warn_only=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train with validation-performance-weighted adaptive mixing and GRPO."
    )
    parser.add_argument("--hf_token", default=os.environ.get("HF_TOKEN"))
    parser.add_argument("--wandb_api_key", default=os.environ.get("WANDB_API_KEY"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", action="store_true")
    args = parser.parse_args()
    hf_username = os.environ["HF_USERNAME"]
    _set_reproducibility(seed=args.seed, deterministic=args.deterministic)
    assert rollout_batch_size % group_size == 0, "rollout_batch_size must be divisible by group_size"
    assert num_egs_per_effective_batch % gradient_accumulation_steps == 0, (
        "num_egs_per_effective_batch must be divisible by gradient_accumulation_steps"
    )
    assert val_size > 0, "val_size must be positive"
    assert eval_temperature == 0.0, "validation must use greedy decoding"
    assert val_controller_temperature > 0, "val_controller_temperature must be positive"
    assert 0 < alpha_update_rate <= 1, "alpha_update_rate must be in (0, 1]"
    assert 0 < alpha_min < alpha_init < alpha_max < 1, (
        "alpha bounds must satisfy 0 < alpha_min < alpha_init < alpha_max < 1"
    )

    model_short = model_id.split("/")[-1]
    dataset_short = hf_dataset.split("/")[-1]

    n_q_per_rollout_batch = rollout_batch_size // group_size
    num_ebs = rollout_batch_size // num_egs_per_effective_batch

    device = torch.device(device_train)
    device_eval = torch.device("cuda:1") if torch.cuda.device_count() > 1 else device
    token = args.hf_token
    alpha = float(alpha_init)

    # --- GPU 0: policy_model (training) ---
    policy_model = AutoModelForCausalLM.from_pretrained(
        model_id, token=token, torch_dtype=torch.bfloat16
    ).to(device).train()
    # --- SFT model on eval GPU to free VRAM on train GPU for optimizer states ---
    sft_model = AutoModelForCausalLM.from_pretrained(
        sft_model_id, token=token, torch_dtype=torch.bfloat16
    ).to(device_eval).eval()

    policy_model.gradient_checkpointing_enable()
    policy_model.config.use_cache = False

    for p in sft_model.parameters():
        p.requires_grad_(False)

    # --- GPU 1 (or 0): vLLM for policy-only generation ---
    vllm_gpu_util = gpu_memory_utilization
    if device_eval == device:
        vllm_gpu_util = min(vllm_gpu_util, 0.30)
    print(
        f"[Device] train={device}, eval/vllm={device_eval}, "
        f"vllm_gpu_memory_utilization={vllm_gpu_util}",
        flush=True,
    )
    llm = init_vllm(
        model_id=model_id,
        device=str(device_eval),
        seed=args.seed,
        gpu_memory_utilization=vllm_gpu_util,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if args.wandb_api_key:
        wandb.login(key=args.wandb_api_key)

    run_name = (
        f"ValWeightedAdaptive_Merging_{model_short}_{dataset_short}"
        f"_lr{learning_rate}_mb{microbatch_size}_ga{gradient_accumulation_steps}"
        f"_n{rollout_batch_size}_seed{args.seed}_MATH3B_copy"
    )
    wandb.init(
        project=wandb_project,
        dir=str(PROJECT_ROOT),
        name=run_name,
        config={
            "seed": args.seed,
            "deterministic": args.deterministic,
            "alpha_init": alpha_init,
            "val_size": val_size,
            "val_controller_temperature": val_controller_temperature,
            "alpha_update_rate": alpha_update_rate,
            "alpha_min": alpha_min,
            "alpha_max": alpha_max,
            "val_pool_size": n_grpo_steps * val_size,
        },
    )

    optimizer = AdamW(policy_model.parameters(), lr=learning_rate, weight_decay=0.0, betas=(0.9, 0.95))
    optimizer.zero_grad()
    total_param_tensors = sum(1 for _ in policy_model.parameters())

    # Load MATH dataset for training and eval
    ds_train = load_dataset(hf_dataset, split="train")
    ds_eval = load_dataset(hf_dataset, split="test")

    # Pre-process: filter out examples where extract_math_gold returns None
    raw_train_problems = list(ds_train["problem"])
    raw_train_solutions = list(ds_train["solution"])
    raw_train_gold = [extract_math_gold(sol) for sol in raw_train_solutions]
    train_problems, train_solutions, train_gold = [], [], []
    for p, s, g in zip(raw_train_problems, raw_train_solutions, raw_train_gold):
        if g is not None:
            train_problems.append(p)
            train_solutions.append(s)
            train_gold.append(g)
    n_dropped_train = len(raw_train_problems) - len(train_problems)
    print(f"[Data] Train: dropped {n_dropped_train}/{len(raw_train_problems)} examples with None gold answer")

    # Double the dataset by repeating
    train_problems = train_problems * 2
    train_solutions = train_solutions * 2
    train_gold = train_gold * 2
    print(f"[Data] Doubled training data: {len(train_problems)} examples")
    assert val_size <= len(train_problems), (
        "val_size cannot exceed the sequential training-data list"
    )

    raw_eval_problems = list(ds_eval["problem"][:eval_size]) if eval_size > 0 else list(ds_eval["problem"])
    raw_eval_solutions = list(ds_eval["solution"][:eval_size]) if eval_size > 0 else list(ds_eval["solution"])
    raw_eval_gold = [extract_math_gold(sol) for sol in raw_eval_solutions]
    eval_problems, eval_solutions, eval_gold = [], [], []
    for p, s, g in zip(raw_eval_problems, raw_eval_solutions, raw_eval_gold):
        if g is not None:
            eval_problems.append(p)
            eval_solutions.append(s)
            eval_gold.append(g)
    n_dropped_eval = len(raw_eval_problems) - len(eval_problems)
    print(f"[Data] Eval: dropped {n_dropped_eval}/{len(raw_eval_problems)} examples with None gold answer")

    eval_prompts = r1_prompts_from_train(eval_problems)

    # Build the exact sequential validation schedule used by the original
    # script. Each outer step first consumes its training questions and then
    # the following val_size questions. The frozen SFT policy is evaluated
    # once on all scheduled validation occurrences before training begins.
    validation_indices_by_step: List[List[int]] = []
    schedule_offset = 0
    for _ in range(n_grpo_steps):
        schedule_offset = (
            schedule_offset + n_q_per_rollout_batch
        ) % len(train_problems)
        step_indices = _cyclic_indices(
            start=schedule_offset,
            count=val_size,
            total=len(train_problems),
        )
        validation_indices_by_step.append(step_indices)
        schedule_offset = (schedule_offset + val_size) % len(train_problems)

    val_pool_size = sum(len(indices) for indices in validation_indices_by_step)
    assert val_pool_size == n_grpo_steps * val_size
    flat_validation_indices = [
        index
        for step_indices in validation_indices_by_step
        for index in step_indices
    ]
    flat_validation_problems = [
        train_problems[index] for index in flat_validation_indices
    ]
    flat_validation_gold = [train_gold[index] for index in flat_validation_indices]
    flat_validation_prompts = r1_prompts_from_train(flat_validation_problems)

    print(
        f"[ValWeightedAdaptive] Precomputing frozen-reference correctness "
        f"for N={val_pool_size} sequential validation occurrences",
        flush=True,
    )
    with _generation_mode(sft_model):
        flat_reference_outputs = _frozen_generate_batch_fast(
            prompts=flat_validation_prompts,
            model=sft_model,
            tokenizer=tokenizer,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            temperature=eval_temperature,
            stop_sequences=[stop_sequence],
            batch_size=batch_gen_size,
        )
    if len(flat_reference_outputs) != val_pool_size:
        raise RuntimeError(
            "Frozen-reference generation did not return exactly one output "
            "per scheduled validation occurrence."
        )
    flat_reference_correctness = [
        _is_correct(output, gold)
        for output, gold in zip(flat_reference_outputs, flat_validation_gold)
    ]
    reference_correctness_by_step = [
        flat_reference_correctness[
            step * val_size : (step + 1) * val_size
        ]
        for step in range(n_grpo_steps)
    ]
    del (
        flat_reference_outputs,
        flat_reference_correctness,
        flat_validation_prompts,
        flat_validation_problems,
        flat_validation_gold,
        flat_validation_indices,
    )

    # Training data pointer (cycles through dataset)
    train_offset = 0

    for grpo_step in range(n_grpo_steps):
        # Get next chunk of training data
        chunk_end = train_offset + n_q_per_rollout_batch
        if chunk_end <= len(train_problems):
            chunk_q = train_problems[train_offset:chunk_end]
            chunk_gold = train_gold[train_offset:chunk_end]
        else:
            # Wrap around
            chunk_q = train_problems[train_offset:] + train_problems[:chunk_end - len(train_problems)]
            chunk_gold = train_gold[train_offset:] + train_gold[:chunk_end - len(train_problems)]
        train_offset = chunk_end % len(train_problems)

        prompt_train = r1_prompts_from_train(chunk_q)

        prompt_train_duplicate = [s for s in prompt_train for _ in range(group_size)]
        answer_train_duplicate_gold = [s for s in chunk_gold for _ in range(group_size)]

        # ===== A: GRPO rollouts (mixed generation) =====
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
                batch_size=batch_gen_size,
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
        whole_lbl = data_tokenized["labels"]

        # old log probs under the mixed behavior policy (theta_old, alpha)
        whole_logp_old_parts: List[torch.Tensor] = []
        with torch.no_grad():
            for i in range(0, whole_ids.size(0), logp_chunk_size):
                ids_chunk = whole_ids[i : i + logp_chunk_size].to(device, non_blocking=True)
                lbl_chunk = whole_lbl[i : i + logp_chunk_size].to(device, non_blocking=True)
                chunk_logp = _mixed_log_probs(
                    policy_model=policy_model,
                    sft_model=sft_model,
                    input_ids=ids_chunk,
                    labels=lbl_chunk,
                    alpha=alpha,
                )["log_probs"]
                whole_logp_old_parts.append(chunk_logp.cpu())
        whole_logp_old = torch.cat(whole_logp_old_parts, dim=0)

        # iterate effective batches
        global_step_base = grpo_step * num_ebs
        for eb in range(num_ebs):
            base = eb * num_egs_per_effective_batch
            end = base + num_egs_per_effective_batch

            ids = data_tokenized["input_ids"][base:end]
            lbls = data_tokenized["labels"][base:end]
            msk = data_tokenized["response_mask"][base:end]
            adv = advantages[base:end]
            logp_old = whole_logp_old[base:end]

            entropy_sum = torch.tensor(0.0, device=device)
            mask_elements_sum = torch.tensor(0.0, device=device)
            loss_sum = 0.0

            for step in range(gradient_accumulation_steps):
                s = step * microbatch_size
                e = s + microbatch_size

                eb_ids = ids[s:e].to(device, non_blocking=True)
                eb_lbl = lbls[s:e].to(device, non_blocking=True)
                eb_msk = msk[s:e].to(device, non_blocking=True)
                eb_adv = adv[s:e].to(device, non_blocking=True)
                eb_logp_old = logp_old[s:e].to(device, non_blocking=True)

                eb_out = _mixed_log_probs(
                    policy_model=policy_model,
                    sft_model=sft_model,
                    input_ids=eb_ids,
                    labels=eb_lbl,
                    alpha=alpha,
                    return_token_entropy=True,
                )
                eb_logp = eb_out["log_probs"]
                eb_entropy = eb_out["token_entropy"]
                entropy_sum += (eb_entropy * eb_msk).sum()
                mask_elements_sum += eb_msk.sum()

                eb_loss, _ = run_grpo_microbatch_train_step(
                    policy_log_probs=eb_logp,
                    response_mask=eb_msk,
                    gradient_accumulation_steps=gradient_accumulation_steps,
                    loss_type=loss_type,
                    advantages=eb_adv,
                    old_log_probs=eb_logp_old,
                    cliprange=cliprange,
                    normalization=len_normalization,
                )
                loss_sum += eb_loss.detach().item()

            avg_entropy = entropy_sum / torch.clamp(mask_elements_sum, min=1.0)
            grad_params = [p for p in policy_model.parameters() if p.grad is not None]
            sample_param = grad_params[0] if grad_params else None

            sample_before = (
                sample_param.detach().view(-1)[:8].clone().float()
                if sample_param is not None
                else None
            )
            grad_norm = torch.sqrt(
                sum(
                    (p.grad.detach().float().norm(2) ** 2)
                    for p in policy_model.parameters()
                    if p.grad is not None
                )
            )
            # Store param snapshots on CPU in bf16 to avoid GPU OOM
            grad_param_befores = [p.detach().cpu().clone() for p in grad_params]
            torch.cuda.empty_cache()
            optimizer.step()
            sample_update_max = None
            full_update_max = None
            if sample_param is not None and sample_before is not None:
                sample_after = sample_param.detach().view(-1)[:8].float()
                sample_update_max = (sample_after - sample_before.to(sample_after.device)).abs().max().item()
            if grad_params:
                full_update_max = max(
                    (p.detach().cpu().float() - before.float()).abs().max().item()
                    for p, before in zip(grad_params, grad_param_befores)
                )
            optimizer.zero_grad()

            global_step = global_step_base + eb
            print(
                f"[AdaptiveWeight Trainning] step={global_step} loss={loss_sum:.4f} "
                f"avg_entropy={avg_entropy.item():.4f} grad_norm={grad_norm.item():.4f} "
                f"alpha={alpha:.4f}"
            )
            print(
                f"Check gradients! Parameters_all={total_param_tensors} "
                f"Parameters_gradient={len(grad_params)} "
                f"sample_update_max={sample_update_max} "
                f"full_update_max={full_update_max}"
            )
            log_entry = {
                "grpo_step": grpo_step,
                "effective_batch": eb,
                "train/loss": loss_sum,
                "train/avg_entropy": avg_entropy.item(),
                "train/grad_norm": grad_norm.item(),
                "train/alpha": alpha,
            }
            # ------------------------------------------------------------
            # Start to evaluate policy model!!
            if (global_step + 1) % eval_every_effective_batches == 0:
                print(
                    f"[AdaptiveWeight Evaluation] starting eval at step={global_step} ",
                    flush=True,
                )
                # ===== B: Eval mixed (500) =====
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
                        batch_size=batch_gen_size,
                    )

                # ===== C: Eval pi_theta (500) via vLLM =====
                _load_policy_into_vllm_fresh(policy_model, llm)
                eval_pi_theta_outputs = _vllm_policy_generate(
                    llm=llm,
                    prompts=eval_prompts,
                    temperature=eval_temperature,
                    max_new_tokens=max_new_tokens,
                    min_new_tokens=min_new_tokens,
                    stop_sequences=[stop_sequence],
                )

                eval_correct = sum(_is_correct(y, g) for y, g in zip(eval_outputs, eval_gold))
                eval_accuracy = eval_correct / max(1, len(eval_gold))
                eval_pi_theta_correct = sum(
                    _is_correct(y, g) for y, g in zip(eval_pi_theta_outputs, eval_gold)
                )
                eval_pi_theta = eval_pi_theta_correct / max(1, len(eval_gold))
                log_entry["eval/accuracy"] = eval_accuracy
                log_entry["eval_pi_theta"] = eval_pi_theta

            # Start to log to wandb
            print(f"[AdaptiveWeight Wandb] wandb log data: {log_entry}", flush=True)
            wandb.log(
                log_entry,
                step=global_step,
                commit=eb != num_ebs - 1,
            )

        # Update alpha using the next sequential validation batch. The current
        # theta, current mixed policy, and cached frozen SFT are compared on
        # exactly the same questions.
        print(
            f"[ValWeightedAdaptive updating alpha] at grpo_step={grpo_step} "
            f"global_step_base={global_step_base} alpha={alpha:.4f}",
            flush=True,
        )

        val_indices = _cyclic_indices(
            start=train_offset,
            count=val_size,
            total=len(train_problems),
        )
        expected_indices = validation_indices_by_step[grpo_step]
        if val_indices != expected_indices:
            raise RuntimeError(
                "Sequential validation schedule diverged from the cached "
                f"reference schedule at grpo_step={grpo_step}."
            )
        val_q = [train_problems[index] for index in val_indices]
        val_gold_list = [train_gold[index] for index in val_indices]
        c_ref = reference_correctness_by_step[grpo_step]
        train_offset = (train_offset + val_size) % len(train_problems)

        val_prompts = r1_prompts_from_train(val_q)

        # ===== D: Alpha y_theta (100) via vLLM =====
        _load_policy_into_vllm_fresh(policy_model, llm)
        y_theta = _vllm_policy_generate(
            llm=llm,
            prompts=val_prompts,
            temperature=eval_temperature,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            stop_sequences=[stop_sequence],
        )

        # ===== E: Alpha y_mix (100) via batched mixed gen =====
        with _generation_mode(policy_model):
            y_mix = _mixed_generate_batch_fast(
                prompts=val_prompts,
                policy_model=policy_model,
                sft_model=sft_model,
                tokenizer=tokenizer,
                device=device,
                alpha=alpha,
                max_new_tokens=max_new_tokens,
                min_new_tokens=min_new_tokens,
                temperature=eval_temperature,
                stop_sequences=[stop_sequence],
                batch_size=batch_gen_size,
            )

        c_theta = [_is_correct(y, g) for y, g in zip(y_theta, val_gold_list)]
        c_mix = [_is_correct(y, g) for y, g in zip(y_mix, val_gold_list)]
        if not (len(c_theta) == len(c_mix) == len(c_ref) == val_size):
            raise RuntimeError("All three policies must be scored on the same validation batch.")

        q_theta = sum(c_theta) / val_size
        q_mix = sum(c_mix) / val_size
        q_ref = sum(c_ref) / val_size
        alpha_old = alpha
        controller = _validation_weighted_alpha_update(
            alpha=alpha_old,
            q_theta=q_theta,
            q_mix=q_mix,
            q_ref=q_ref,
            temperature=val_controller_temperature,
            update_rate=alpha_update_rate,
            alpha_min=alpha_min,
            alpha_max=alpha_max,
        )
        alpha = controller["alpha_new"]
        print(
            f"[ValWeightedAdaptive updating alpha] grpo_step={grpo_step} "
            f"q_theta={q_theta:.4f} q_mix={q_mix:.4f} q_ref={q_ref:.4f} "
            f"w_theta={controller['w_theta']:.4f} "
            f"w_mix={controller['w_mix']:.4f} "
            f"w_ref={controller['w_ref']:.4f} "
            f"alpha_old={alpha_old:.4f} "
            f"alpha_target={controller['alpha_target']:.4f} "
            f"alpha_new={alpha:.4f}",
            flush=True,
        )
        wandb.log(
            {
                "alpha/target": controller["alpha_target"],
                "alpha/new": alpha,
                "alpha/q_theta": q_theta,
                "alpha/q_mix": q_mix,
                "alpha/q_ref": q_ref,
                "alpha/w_theta": controller["w_theta"],
                "alpha/w_mix": controller["w_mix"],
                "alpha/w_ref": controller["w_ref"],
            },
            step=global_step_base + num_ebs - 1,
            commit=True,
        )

    # Push only the final trained policy to Hugging Face.
    policy_model.config.lapo_alpha = float(alpha)
    policy_model.config.lapo_reference_model = sft_model_id
    policy_model.config.lapo_controller = "validation_performance_weighted"
    repo_id = f"{hf_username}/{run_name}"
    policy_model.push_to_hub(repo_id, token=token, private=False, safe_serialization=True)
    tokenizer.push_to_hub(repo_id, token=token)
    print(f"Model pushed to HF: {repo_id}")
    print(f"Final deployed mixed-policy alpha: {alpha:.6f}")

    wandb.finish()


if __name__ == "__main__":
    main()
