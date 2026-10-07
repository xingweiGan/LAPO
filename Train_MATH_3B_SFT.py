# MATH 3B supervised fine-tuning entrypoint.
import sys
from unittest.mock import patch
from transformers import PreTrainedModel, AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams
import torch
from vllm.model_executor import set_random_seed as vllm_set_random_seed
import pandas as pd
from torch.optim import AdamW
from typing import Optional, Callable, Dict, List
from pathlib import Path
import wandb
from huggingface_hub import create_repo
import os
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from math3b_support.adapters import (
    run_tokenize_prompt_and_output,
    run_get_response_log_probs,
    run_sft_microbatch_train_step,
)
from math3b_support.drgrpo_grader import r1_zero_reward_fn, extract_boxed_answer
from math3b_support.math_baseline import r1_prompts_from_train, extract_gsm8k_gold
from datasets import load_dataset


# ---------------- Core utilities (safe to import) ----------------

def init_vllm(model_id: str, device: str, seed: int, gpu_memory_utilization: float = 0.45) -> LLM:
    """
    Start the inference process; use vLLM on a specific device.
    Safe to import; no side effects beyond constructing the LLM when called.
    """
    vllm_set_random_seed(seed)

    #Patch means changing a value to a assigned one
    #Skip the distributed setting
    world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
    #Skip the profling test
    profiling_patch = patch(
        "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
        return_value=None
    )
    with world_size_patch, profiling_patch:
        #Notes: dtypes is the data type used for storing weights and do acomputation
        #enable_prefix_caching=True, so attention key values for the prompt will be reused
        #gpu_memory_utilization=gpu_memory_utilization, how much memory is allowed for KV cache and model weights
        #KV cache (sth that need to be used again) is saved attention memory from previous tokens
        return LLM(
            model=model_id,
            device=device,
            dtype=torch.bfloat16,
            enable_prefix_caching=True,
            gpu_memory_utilization=gpu_memory_utilization,
            enforce_eager=True,
        )


def load_policy_into_vllm_instance(policy: PreTrainedModel, llm: LLM) -> None:
    """Load HF policy weights into the live vLLM model runner."""
    state_dict = policy.state_dict()
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())


def r1_answer_from_train(ds: list[str]) -> list[str]:
    """
    Convert GSM8K answers to R1 format:
      replace final line prefix '#### ' with '</think> <answer>'
      and append '</answer>'
    """
    return [a.replace("\n#### ", "\n</think> <answer>") + "</answer>" for a in ds]


def r1_answer_from_math(solutions: list[str]) -> list[str]:
    """
    Convert MATH solutions to R1 format:
      use the solution as the thinking, extract \\boxed{} as the answer.
    """
    results = []
    for sol in solutions:
        boxed = extract_boxed_answer(sol) if "\\boxed" in sol else sol
        results.append(f"{sol}\n</think> <answer>{boxed}</answer>")
    return results


def extract_math_gold(solution: str) -> str:
    """Extract the ground truth answer from a MATH solution (\\boxed{...})."""
    if "\\boxed" in solution:
        return extract_boxed_answer(solution)
    return solution


def evaluate_vllm_new(
    vllm_model: LLM,
    reward_fn: Callable[[str, str], Dict[str, float]],
    prompts: List[str],
    ground_truths: List[str],
    eval_sampling_params: SamplingParams,
) -> List[float]:
    """
    Generate model outputs for `prompts`, score them with `reward_fn`, and
    return [format_rate, accuracy] without any file I/O.

    - format_rate = P(format_reward > 0)
    - accuracy    = P(format_reward > 0 AND answer_reward > 0)
    """
    # 1) Generate
    outputs = vllm_model.generate(prompts, eval_sampling_params)

    # 2) Score and tally
    bins = {"11": 0, "10": 0, "01": 0, "00": 0}
    for out, gold in zip(outputs, ground_truths):
        # vLLM returns a RequestOutput with .outputs list (take the first)
        text = out.outputs[0].text if out.outputs else ""
        scores = reward_fn(text, gold)

        fmt = 1 if scores.get("format_reward", 0.0) > 0 else 0
        ans = 1 if scores.get("answer_reward", 0.0) > 0 else 0
        bins[f"{fmt}{ans}"] += 1

    # 3) Aggregate metrics
    n = len(prompts)
    if n == 0:
        return [0.0, 0.0]

    format_rate = (bins["11"] + bins["10"]) / n
    accuracy = bins["11"] / n

    return [format_rate, accuracy]



# --------------Main code---------------

def main(
    model_id: str = "Qwen/Qwen2.5-3B-Instruct",
    device_train: str = "cuda:0",
    device_eval: str = "cuda:1",
    num_unique: int = 768,
    gradient_accumulation_steps: int = 16,
    microbatch_size: int = 1,
    lr: float = 2e-5,
    eval_temperature: float = 0.0,
    eval_max_tokens: int = 192,
    hf_dataset: str = "Maxwell-Jia/MATH",
    seed: int = 42,
    do_eval_each_eb: bool = True,
    wandb_project: str = "MATH_3B_comparison",
    wandb_run_name: Optional[str] = None,
) -> None:
    hf_username = os.environ["HF_USERNAME"]
    model_short = model_id.split("/")[-1]
    dataset_short = hf_dataset.split("/")[-1]
    repo_id = f"{hf_username}/SFT_{model_short}_{dataset_short}_MATH3B_copy"
    token = os.environ["HF_TOKEN"]  # set before running
    create_repo(repo_id, private=False, exist_ok=True, token=token)

    wandb.login(key=os.environ["WANDB_API_KEY"])

    run_name = wandb_run_name or (
        f"SFT_{model_short}_{dataset_short}"
        f"_lr{lr}_mb{microbatch_size}_ga{gradient_accumulation_steps}"
        f"_n{num_unique}_seed{seed}_MATH3B_copy"
    )
    wandb_run = wandb.init(
        project=wandb_project,
        name=run_name,
        dir=str(PROJECT_ROOT),
    )

    # Load training data from HuggingFace MATH dataset
    ds_train = load_dataset(hf_dataset, split="train")
    # Take the first num_unique examples
    if len(ds_train) > num_unique:
        ds_train = ds_train.select(range(num_unique))
    sft_q = ds_train["problem"]
    sft_sol = ds_train["solution"]
    num_unique_actual = len(ds_train)
    if num_unique_actual < num_unique:
        print(f"[SFT] Requested {num_unique} examples but only found {num_unique_actual}; continuing with available data.")

    prompts_train = r1_prompts_from_train(sft_q)
    answers_train = r1_answer_from_math(sft_sol)

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    #"input_ids": the tokenized prompt and output strings, with the final token sliced off.
    #"labels" shifted input_ids (i.e., the input_ids without the first token).
    #"response_mask" a mask on the response tokens in `labels`.
    data_tokenized = run_tokenize_prompt_and_output(prompts_train, answers_train, tokenizer)

    # vLLM for eval
    llm = init_vllm(model_id=model_id, device=device_eval, seed=seed)

    # Eval set from HuggingFace MATH dataset
    ds_eval = load_dataset(hf_dataset, split="test")
    prompts_eval = r1_prompts_from_train(ds_eval["problem"])
    ground_truth_eval = [extract_math_gold(sol) for sol in ds_eval["solution"]]
    #Here, the temperature decides the distribution of output tokens, when T<1, distribution is sharper->pick
    #the token with highest logits and the distribution become more determinstic. And when T>1, it makes it become
    #more random in the sens the difference between logits of each tokens become smaller
    sampling = SamplingParams(
        temperature=eval_temperature,
        top_p=1.0,
        max_tokens=eval_max_tokens,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )

    # Policy on train device (bf16 to save VRAM for 3B)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16).to(device_train).train()
    model.gradient_checkpointing_enable()
    optimizer = AdamW(model.parameters(), lr=lr)
    optimizer.zero_grad()

    # Effective batch math
    num_egs_per_effective_batch = microbatch_size * gradient_accumulation_steps
    num_ebs = num_unique_actual // num_egs_per_effective_batch
    if num_unique_actual % num_egs_per_effective_batch != 0:
        print(f"[SFT] Dropping last {num_unique_actual % num_egs_per_effective_batch} samples to fit whole effective batches.")

    for eb in range(num_ebs):
        print(eb)
        base = eb * num_egs_per_effective_batch
        end = base + num_egs_per_effective_batch

        ids = data_tokenized["input_ids"][base:end]
        lbls = data_tokenized["labels"][base:end]
        msk = data_tokenized["response_mask"][base:end]

        loss_sum = 0.0
        for step in range(gradient_accumulation_steps):
            s = step * microbatch_size
            e = s + microbatch_size

            eb_ids = ids[s:e].to(device_train, non_blocking=True)
            eb_lbl = lbls[s:e].to(device_train, non_blocking=True)
            eb_msk = msk[s:e].to(device_train, non_blocking=True)

            eb_logp = run_get_response_log_probs(model, eb_ids, eb_lbl)["log_probs"]
            eb_loss, _ = run_sft_microbatch_train_step(
                eb_logp, eb_msk, gradient_accumulation_steps=gradient_accumulation_steps
            )
            print(f"loss: {float(eb_loss):.4f}")
            loss_sum += float(eb_loss)

        optimizer.step()
        optimizer.zero_grad()
        #These updates happens on trainning cuda, we need to move the updated policy to evaluation cuda and later
        #can be used for evaluating the evaluation dataset.
        load_policy_into_vllm_instance(model, llm)

        #In general, in trainning cuda, we use the HF model and in the evaluation cuda, we use vLLM for evaluation
        #Normally, we use HF model for trainning and vLLM for inference/evaluation.
        #We have turned on the trainning mode for HF model at the beginning and there is no need to switch between
        #eval and training bc we never use the eval mode for HF model.

        eval_metrics = {}
        if do_eval_each_eb:
            format_rate, accuracy = evaluate_vllm_new(
                llm, r1_zero_reward_fn, prompts_eval, ground_truth_eval, sampling
            )
            eval_metrics = {
                "eval/format_rate": format_rate,
                "eval/accuracy": accuracy,
            }
            print(
                f"[Eval] eb={eb} format_rate={format_rate:.3f} accuracy={accuracy:.3f}"
            )

        log_payload = {
            "effective_batch": eb,
            "train/loss": loss_sum,
            **eval_metrics,
        }
        wandb.log(log_payload, step=eb)

    #Save it to HF
    model.push_to_hub(repo_id, token=token, private=False, safe_serialization=True)
    tokenizer.push_to_hub(repo_id, token=token)
    print("Model pushed to HF!")

    if wandb_run is not None:
        wandb_run.finish()


# Only run the demo if this file is executed directly, NOT on import.
if __name__ == "__main__":
    main()
