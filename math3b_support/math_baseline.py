from datasets import load_dataset
from math3b_support.drgrpo_grader import r1_zero_reward_fn
from typing import List, Callable, Dict
from vllm import LLM, SamplingParams
import json
import re

#Transform the ds to r1-format prompts (list)
def r1_prompts_from_train(ds):
    header = (
        "A conversation between User and Assistant. "
        "The Assistant first thinks step-by-step inside <think>...</think>, "
        "then gives ONLY the final result inside <answer>...</answer>.\n"
    )
    return [f"{header}User: {q}\nAssistant: <think>" for q in ds]

# Extract text after "####" and normalize a bit.
def extract_gsm8k_gold(s: str) -> str | None:
    m = re.search(r"####\s*(.+)$", s, flags=re.M)
    if not m:
        return None
    gold = m.group(1)

    gold = re.sub(r"[.\s]+$", "", gold)   # trim trailing punctuation/space
    gold = gold.lstrip()                  # <-- remove any leading space
    gold = gold.replace(",", "").replace("$", "").replace("%", "")

    m2 = re.search(r"(-?\d+(?:\.\d+)?|\d+/\d+)", gold)
    return m2.group(1) if m2 else gold

def evaluate_vllm(
    vllm_model: LLM,
    reward_fn: Callable[[str, str], Dict[str, float]],
    prompts: List[str],
    ground_truths: List[str],                  # <-- moved into args
    eval_sampling_params: SamplingParams,
    index: int = 1,
) -> None:
    """
    Evaluate a language model on a list of prompts,
    compute evaluation metrics, and serialize results to disk.
    """

    # 1) Generate
    outputs = vllm_model.generate(prompts, eval_sampling_params)

    # 2) Score + write per-example records
    bins = {"11": 0, "10": 0, "01": 0, "00": 0}
    with open(f"eval.records_{index}.jsonl", "w", encoding="utf-8") as recf:
        for i, (out, gold) in enumerate(zip(outputs, ground_truths)):
            text = out.outputs[0].text
            scores = reward_fn(text, gold)  # e.g., r1_zero_reward_fn

            fmt = 1 if scores.get("format_reward", 0.0) > 0 else 0
            ans = 1 if scores.get("answer_reward", 0.0) > 0 else 0
            bins[f"{fmt}{ans}"] += 1

            rec = {
                "id": i,
                "prompt": prompts[i],
                "output": text,
                "gold": gold,
                "scores": scores,
            }
            recf.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # 3) Write summary metrics
    n = len(prompts)
    summary = {
        "n": n,
        "counts": {
            "format=1,answer=1": bins["11"],
            "format=1,answer=0": bins["10"],
            "format=0,answer=1": bins["01"],
            "format=0,answer=0": bins["00"],
        },
        "format_rate": (bins["11"] + bins["10"]) / n if n else 0.0,
        "accuracy": bins["11"] / n if n else 0.0,
    }
    with open(f"eval.summary_{index}.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)



if __name__ == "__main__":  # minimal fix: prevent side effects on import
    # Step1: Preprocess the format for answers and questions.
    ds = load_dataset("openai/gsm8k", "main")

    # Lists of gold answers aligned with each split
    train_gold = [extract_gsm8k_gold(a) for a in ds["train"]["answer"]]
    prompts = r1_prompts_from_train(ds["train"]["question"])
    MODEL = "Qwen/Qwen2.5-Math-1.5B"
    llm = LLM(
        model=MODEL,
        gpu_memory_utilization=0.90,
        max_model_len=2048,
        trust_remote_code=True,  # safe for Qwen-family models
    )
    sampling = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=192,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )
    evaluate_vllm(llm, r1_zero_reward_fn, prompts, train_gold, sampling)
