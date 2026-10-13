"""Figure 4 GRPO+KL baseline on cn_k12 (Qwen2.5-3B).

The optimization, W&B logging, optional phase profiling, and final Hugging
Face upload follow the current LAPO MATH GRPO+KL baseline.  Only the dataset,
model-size identifiers, and collision-free experiment names are adapted for
the historical Assignment3 Figure 4 setup.
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import sys
from typing import List, Literal

import torch
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer

from Train_MATH_3B_GRPO_KL import (
    _generation_mode,
    _hf_generate_batch,
    _resolve_device_layout,
    evaluate_hf,
)
from math3b_support.adapters import (
    run_compute_group_normalized_rewards,
    run_get_response_log_probs,
    run_grpo_microbatch_train_step,
    run_tokenize_prompt_and_output,
)
from math3b_support.cnk12_data import load_cnk12_splits
from math3b_support.drgrpo_grader import r1_zero_reward_fn
from math3b_support.math_baseline import r1_prompts_from_train
from math3b_support.profiling import StageProfiler


PROJECT_ROOT = Path(__file__).resolve().parent

beta: float = 0.01
learning_rate: float = 1e-5
len_normalization = "mean"
loss_type: Literal[
    "no_baseline",
    "reinforce_with_baseline",
    "grpo_clip",
    "grpo_clip_KL",
] = "grpo_clip_KL"
use_std_normalization: bool = True

n_grpo_steps: int = 20
advantage_eps: float = 1e-6
rollout_batch_size: int = 2048
group_size: int = 8
sampling_temperature: float = 1.0
sampling_min_tokens: int = 4
sampling_max_tokens: int = 512
epochs_per_rollout_batch: int = 1
eval_every_effective_batches: int = 8
batch_gen_size: int = 8

gradient_accumulation_steps: int = 128
num_egs_per_effective_batch: int = 256

model_size_label = "3B"
model_id = "Qwen/Qwen2.5-3B-Instruct"
sft_model_id = "xw1234gan/SFT_Qwen2.5-3B-Instruct_cnk12"
train_device_preference = "cuda:0"
wandb_project = "CNK12_comparison"


def main(
    *,
    run_model_size_label: str = model_size_label,
    run_model_id: str = model_id,
    run_sft_model_id: str = sft_model_id,
    run_batch_gen_size: int = batch_gen_size,
) -> None:
    hf_username = os.environ["HF_USERNAME"]
    all_logs = []

    parser = argparse.ArgumentParser(
        description=f"Train cn_k12 {run_model_size_label} with GRPO+KL."
    )
    parser.add_argument("--wandb_api_key", default=os.environ.get("WANDB_API_KEY"))
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Record synchronized phase timings and per-GPU memory statistics.",
    )
    parser.add_argument(
        "--profile-output",
        type=Path,
        default=None,
        help="Profiling JSON path (default: profiles/<run_name>.json).",
    )
    parser.add_argument(
        "--profile-no-nvml",
        action="store_true",
        help="Disable optional NVML physical-memory sampling.",
    )
    parser.add_argument(
        "--profile-nvml-interval-ms",
        type=float,
        default=50.0,
        help="NVML sampling interval in milliseconds (minimum: 10).",
    )
    args = parser.parse_args()
    if (
        args.profile
        and not args.profile_no_nvml
        and (
            not math.isfinite(args.profile_nvml_interval_ms)
            or args.profile_nvml_interval_ms < 10.0
        )
    ):
        parser.error("--profile-nvml-interval-ms must be finite and at least 10")
    if args.wandb_api_key:
        wandb.login(key=args.wandb_api_key)

    run_n_grpo_steps = n_grpo_steps
    run_rollout_batch_size = rollout_batch_size
    run_eval_every_effective_batches = eval_every_effective_batches
    run_sampling_max_tokens = sampling_max_tokens
    n_q_per_rollout_batch = run_rollout_batch_size // group_size

    smoke = os.environ.get("SMOKE", "0") == "1"
    if smoke:
        run_n_grpo_steps = 1
        run_rollout_batch_size = 16
        n_q_per_rollout_batch = run_rollout_batch_size // group_size
        run_eval_every_effective_batches = 1
        run_sampling_max_tokens = 64
        print(
            f"[SMOKE] Override-1: n_grpo_steps={run_n_grpo_steps}, "
            f"rollout_batch_size={run_rollout_batch_size}, "
            f"sampling_max_tokens={run_sampling_max_tokens}",
            flush=True,
        )

    device_train, device_eval, use_dual_gpu = _resolve_device_layout(
        train_device_preference
    )
    print(
        f"[Device] train={device_train}, eval={device_eval}, "
        f"dual_gpu={use_dual_gpu}",
        flush=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        run_sft_model_id,
        torch_dtype=torch.bfloat16,
    ).to(device_train).train()
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    ref_model = AutoModelForCausalLM.from_pretrained(
        run_sft_model_id,
        torch_dtype=torch.bfloat16,
    ).to(device_eval).eval()
    for parameter in ref_model.parameters():
        parameter.requires_grad_(False)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.0,
        betas=(0.9, 0.95),
    )
    optimizer.zero_grad()
    tokenizer = AutoTokenizer.from_pretrained(run_model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    eval_keep = 8 if smoke else 500
    train_split, eval_split = load_cnk12_splits(
        train_keep=7500,
        eval_keep=eval_keep,
        seed=42,
    )
    train_problems, _train_solutions, train_gold = train_split
    eval_problems, _eval_solutions, eval_gold = eval_split
    print(
        f"[Data] cn_k12 train={len(train_problems)}, "
        f"eval={len(eval_problems)}",
        flush=True,
    )
    prompts_eval = r1_prompts_from_train(eval_problems)
    ground_truth_eval = eval_gold

    run_gradient_accumulation_steps = gradient_accumulation_steps
    run_num_egs_per_effective_batch = num_egs_per_effective_batch
    microbatch_size = (
        run_num_egs_per_effective_batch // run_gradient_accumulation_steps
    )
    num_ebs = run_rollout_batch_size // run_num_egs_per_effective_batch
    if smoke:
        run_gradient_accumulation_steps = 2
        run_num_egs_per_effective_batch = 4
        microbatch_size = (
            run_num_egs_per_effective_batch // run_gradient_accumulation_steps
        )
        num_ebs = run_rollout_batch_size // run_num_egs_per_effective_batch
        print(
            f"[SMOKE] Override-2: ga={run_gradient_accumulation_steps}, "
            f"microbatch={microbatch_size}, num_ebs={num_ebs}",
            flush=True,
        )

    model_short = run_model_id.split("/")[-1]
    experiment_tag = (
        f"CNK12_{run_model_size_label}_GRPO_KL"
        f"_G{group_size}_outer{run_n_grpo_steps}"
    )
    run_name = (
        f"{experiment_tag}_{model_short}_beta{beta}_lr{learning_rate}"
        f"_mb{microbatch_size}_ga{run_gradient_accumulation_steps}"
        f"_n{run_rollout_batch_size}_seed42"
    )
    wandb.init(
        project=wandb_project,
        dir=str(PROJECT_ROOT),
        name=run_name,
    )

    profile_output = args.profile_output
    if profile_output is None:
        profile_output = PROJECT_ROOT / "profiles" / f"{run_name}.json"
    elif not profile_output.is_absolute():
        profile_output = PROJECT_ROOT / profile_output
    try:
        profile_display_path = profile_output.relative_to(PROJECT_ROOT)
    except ValueError:
        profile_display_path = Path(profile_output.name)
    profiler = StageProfiler(
        enabled=args.profile,
        devices=[device_train, device_eval],
        output_path=profile_output,
        enable_nvml=not args.profile_no_nvml,
        nvml_interval_s=args.profile_nvml_interval_ms / 1000.0,
    )
    if args.profile:
        print(
            f"[Profile] enabled; json={profile_display_path}; "
            f"cuda_devices={profiler.cuda_devices}; nvml={profiler.nvml_enabled}",
            flush=True,
        )

    train_problems = train_problems * 2
    train_gold = train_gold * 2
    print(f"[Data] Doubled training data: {len(train_problems)} examples")
    train_offset = 0

    for grpo_step in range(run_n_grpo_steps):
        profiler.begin_outer_step(grpo_step, optimizer_updates=num_ebs)
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

        print(
            f"[GRPO Step {grpo_step}] Generating rollouts via HF...",
            flush=True,
        )
        with profiler.phase("rollout"):
            with _generation_mode(model):
                response_train_duplicate = _hf_generate_batch(
                    prompts=prompt_train_duplicate,
                    model=model,
                    tokenizer=tokenizer,
                    device=torch.device(device_train),
                    max_new_tokens=run_sampling_max_tokens,
                    min_new_tokens=sampling_min_tokens,
                    temperature=sampling_temperature,
                    stop_sequences=["</answer>"],
                    batch_size=run_batch_gen_size,
                    show_progress=not args.profile,
                )

        with profiler.phase("reward"):
            advantages_train, _, _ = run_compute_group_normalized_rewards(
                reward_fn=r1_zero_reward_fn,
                rollout_responses=response_train_duplicate,
                repeated_ground_truths=answer_train_duplicate_gold,
                group_size=group_size,
                advantage_eps=advantage_eps,
                normalize_by_std=use_std_normalization,
            )

        with profiler.phase("tokenization"):
            data_tokenized = run_tokenize_prompt_and_output(
                tokenizer=tokenizer,
                prompt_strs=prompt_train_duplicate,
                output_strs=response_train_duplicate,
            )

        whole_ids = data_tokenized["input_ids"]
        whole_labels = data_tokenized["labels"]
        chunk_size = 2
        whole_logp_old_parts: List[torch.Tensor] = []
        with profiler.phase("old_logprob"):
            with torch.no_grad():
                for index in range(0, whole_ids.size(0), chunk_size):
                    ids_chunk = whole_ids[index : index + chunk_size].to(
                        device_train,
                        non_blocking=True,
                    )
                    labels_chunk = whole_labels[index : index + chunk_size].to(
                        device_train,
                        non_blocking=True,
                    )
                    chunk_logp = run_get_response_log_probs(
                        model,
                        ids_chunk,
                        labels_chunk,
                    )["log_probs"]
                    whole_logp_old_parts.append(chunk_logp.cpu())
                    if not args.profile:
                        print(index)
            whole_logp_old = torch.cat(whole_logp_old_parts, dim=0)

        for effective_batch in range(num_ebs):
            base = effective_batch * run_num_egs_per_effective_batch
            end = base + run_num_egs_per_effective_batch
            ids = data_tokenized["input_ids"][base:end]
            labels = data_tokenized["labels"][base:end]
            response_mask = data_tokenized["response_mask"][base:end]
            advantages = advantages_train[base:end]
            logp_old = whole_logp_old[base:end]

            if not args.profile:
                print(
                    "gradient_accumulation_steps: "
                    f"{run_gradient_accumulation_steps}"
                )
            entropy_sum = 0
            mask_elements_sum = 0
            loss_sum = 0.0

            for accumulation_step in range(run_gradient_accumulation_steps):
                start = accumulation_step * microbatch_size
                stop = start + microbatch_size

                with profiler.phase("forward"):
                    micro_ids = ids[start:stop].to(
                        device_train,
                        non_blocking=True,
                    )
                    micro_labels = labels[start:stop].to(
                        device_train,
                        non_blocking=True,
                    )
                    micro_mask = response_mask[start:stop].to(
                        device_train,
                        non_blocking=True,
                    )
                    micro_advantages = advantages[start:stop].to(
                        device_train,
                        non_blocking=True,
                    )
                    micro_output = run_get_response_log_probs(
                        model,
                        micro_ids,
                        micro_labels,
                        return_token_entropy=True,
                    )
                    micro_logp = micro_output["log_probs"]
                    micro_logp_old = logp_old[start:stop].to(
                        device_train,
                        non_blocking=True,
                    )
                    with torch.no_grad():
                        micro_ref_logp = run_get_response_log_probs(
                            ref_model,
                            micro_ids.to(device_eval),
                            micro_labels.to(device_eval),
                        )["log_probs"]
                        micro_ref_logp = micro_ref_logp.to(device_train)
                    with torch.no_grad():
                        micro_entropy = micro_output["token_entropy"]
                        entropy_sum += (micro_entropy * micro_mask).sum()
                        mask_elements_sum += micro_mask.sum()

                if not args.profile:
                    print(
                        f"Step {accumulation_step}: advantages "
                        f"shape={micro_advantages.shape}, "
                        f"mean={micro_advantages.mean():.4f}, "
                        f"std={micro_advantages.std():.4f}"
                    )

                micro_loss, _ = run_grpo_microbatch_train_step(
                    policy_log_probs=micro_logp,
                    response_mask=micro_mask,
                    gradient_accumulation_steps=run_gradient_accumulation_steps,
                    loss_type=loss_type,
                    advantages=micro_advantages,
                    old_log_probs=micro_logp_old,
                    ref_log_probs=micro_ref_logp,
                    cliprange=0.2,
                    normalization=len_normalization,
                    beta=beta,
                    profile_phase=profiler.phase if args.profile else None,
                )
                loss_sum += micro_loss.detach().item()

            with profiler.phase("grad_metrics"):
                avg_entropy = entropy_sum / mask_elements_sum
                grad_norm = torch.sqrt(
                    sum(
                        parameter.grad.detach().float().norm(2) ** 2
                        for parameter in model.parameters()
                        if parameter.grad is not None
                    )
                )
            with profiler.phase("allocator_cleanup"):
                torch.cuda.empty_cache()
            with profiler.phase("optimizer"):
                optimizer.step()

            global_step = grpo_step * num_ebs + effective_batch
            accuracy = None
            if (
                (global_step + 1) % run_eval_every_effective_batches == 0
            ):
                print(
                    f"[Eval] Starting eval at step={global_step} via HF...",
                    flush=True,
                )
                with profiler.phase("evaluation"):
                    with _generation_mode(model):
                        _, accuracy = evaluate_hf(
                            model=model,
                            tokenizer=tokenizer,
                            device=torch.device(device_train),
                            reward_fn=r1_zero_reward_fn,
                            prompts=prompts_eval,
                            ground_truths=ground_truth_eval,
                            max_new_tokens=run_sampling_max_tokens,
                            min_new_tokens=sampling_min_tokens,
                            batch_size=run_batch_gen_size,
                            show_progress=not args.profile,
                        )

            loss = micro_loss
            log_entry = {
                "global_step": global_step,
                "grpo_step": grpo_step,
                "effective_batch": effective_batch,
                "train/loss": loss_sum,
                "train_loss": loss_sum,
                "train/avg_entropy": avg_entropy.item(),
                "train/grad_norm": grad_norm.item(),
            }
            if accuracy is not None:
                log_entry["eval/accuracy"] = accuracy
            with profiler.phase("logging"):
                print("WANDB LOG!")
                print(grpo_step)
                print(global_step)
                wandb.log(
                    log_entry,
                    step=global_step,
                    commit=not (
                        args.profile and effective_batch == num_ebs - 1
                    ),
                )
                all_logs.append(log_entry)
            with profiler.phase("zero_grad"):
                optimizer.zero_grad()

        profiler.end_outer_step()
        if args.profile:
            wandb.log(
                profiler.last_outer_scalars(),
                step=grpo_step * num_ebs + num_ebs - 1,
                commit=True,
            )

    if args.profile and wandb.run is not None:
        try:
            wandb.run.summary.update(profiler.summary_scalars())
        except Exception as exc:
            print(
                f"[Profile] W&B summary update failed "
                f"({type(exc).__name__}); continuing to final save.",
                file=sys.stderr,
                flush=True,
            )
    try:
        profile_path = profiler.close()
    except Exception as exc:
        print(
            f"[Profile] JSON finalization failed ({type(exc).__name__}); "
            "continuing to final save.",
            file=sys.stderr,
            flush=True,
        )
    else:
        if profile_path is not None:
            print(f"[Profile] wrote {profile_display_path}", flush=True)

    try:
        wandb.finish()
    except Exception as exc:
        print(
            f"[W&B] finish failed ({type(exc).__name__}); "
            "continuing to final Hugging Face save.",
            file=sys.stderr,
            flush=True,
        )

    if not smoke:
        repo_id = f"{hf_username}/{run_name}"
        token = os.environ["HF_TOKEN"]
        model.push_to_hub(
            repo_id,
            token=token,
            private=False,
            safe_serialization=True,
        )
        tokenizer.push_to_hub(repo_id, token=token)
        print(f"Model pushed to HF: {repo_id}")
    else:
        print("[SMOKE] Skipping push_to_hub", flush=True)


if __name__ == "__main__":
    main()
