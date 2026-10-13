"""Deterministic train/eval splits for the NuminaMath cn_k12 subset.

The filtering and split semantics mirror the Figure 4 experiments from the
historical Assignment3 implementation: keep cn_k12 math-word-problems with a
short scalar boxed answer, deduplicate by normalized problem text, shuffle
with a fixed seed, then take disjoint evaluation and training slices.
"""

from __future__ import annotations

import random
import re
from typing import List, Tuple

from datasets import load_dataset

from math3b_support.drgrpo_grader import extract_boxed_answer


HF_ID = "AI-MO/NuminaMath-1.5"
KEEP_SOURCE = "cn_k12"
KEEP_QTYPE = "math-word-problem"
MAX_SOLUTION_CHARS = 2000


def _is_clean_scalar_gold(gold: str) -> bool:
    if not gold:
        return False
    gold = gold.strip()
    if not 1 <= len(gold) <= 30:
        return False

    if re.fullmatch(r"-?\d+(\.\d+)?", gold):
        return True

    bad_markers = (
        r"\\text\{",
        r"\\textbf\{",
        r"\\mathrm\{",
        r"\\parallel",
        r"\\perp",
        r"\\cong",
        r"\\sim",
        r"Q\.E\.D",
        r"\\square",
        r"<",
        r">",
        r"\\leq",
        r"\\geq",
        r"\\neq",
    )
    if any(re.search(pattern, gold) for pattern in bad_markers):
        return False
    if re.search(r"[,;]|\\;", gold):
        return False
    if re.fullmatch(r"\(?[A-E]\)?", gold):
        return False
    if re.search(r"[a-zA-Z]\s*\(|=", gold):
        return False

    if re.fullmatch(r"-?\\frac\{-?\d+\}\{-?\d+\}", gold):
        return True
    if re.fullmatch(r"-?\\sqrt\{-?\d+\}", gold):
        return True
    if re.fullmatch(r"-?\d+/-?\d+", gold):
        return True
    return len(gold) <= 10 and re.search(r"[A-Za-z\\]", gold) is None


def _extract_gold(solution: str) -> str | None:
    if "\\boxed" not in solution:
        return None
    gold = extract_boxed_answer(solution)
    if gold is None:
        return None
    gold = gold.strip()
    return gold if _is_clean_scalar_gold(gold) else None


def _filtered_pool(seed: int = 42) -> tuple[List[str], List[str], List[str]]:
    dataset = load_dataset(HF_ID, split="train")
    seen: set[str] = set()
    problems: List[str] = []
    solutions: List[str] = []
    golds: List[str] = []

    for row in dataset:
        if row.get("source") != KEEP_SOURCE:
            continue
        if row.get("question_type") != KEEP_QTYPE:
            continue
        solution = row.get("solution") or ""
        if not solution or len(solution) > MAX_SOLUTION_CHARS:
            continue
        gold = _extract_gold(solution)
        if gold is None:
            continue
        problem = row.get("problem") or ""
        if not problem:
            continue
        key = " ".join(problem.split())
        if key in seen:
            continue
        seen.add(key)
        problems.append(problem)
        solutions.append(solution)
        golds.append(gold)

    order = list(range(len(problems)))
    random.Random(seed).shuffle(order)
    problems = [problems[index] for index in order]
    solutions = [solutions[index] for index in order]
    golds = [golds[index] for index in order]
    print(f"[cnk12_data] Pool after filter+dedup: {len(problems)} rows")
    return problems, solutions, golds


def load_cnk12_splits(
    train_keep: int = 768,
    eval_keep: int = 500,
    seed: int = 42,
) -> Tuple[
    Tuple[List[str], List[str], List[str]],
    Tuple[List[str], List[str], List[str]],
]:
    """Return disjoint ``(train, eval)`` problem/solution/gold tuples."""
    problems, solutions, golds = _filtered_pool(seed=seed)
    required = eval_keep + train_keep
    if len(problems) < required:
        raise RuntimeError(
            f"Pool has {len(problems)} rows but need {eval_keep}+{train_keep}="
            f"{required}."
        )

    eval_problems = problems[:eval_keep]
    eval_solutions = solutions[:eval_keep]
    eval_golds = golds[:eval_keep]
    train_end = eval_keep + train_keep
    train_problems = problems[eval_keep:train_end]
    train_solutions = solutions[eval_keep:train_end]
    train_golds = golds[eval_keep:train_end]

    eval_set = {" ".join(problem.split()) for problem in eval_problems}
    train_set = {" ".join(problem.split()) for problem in train_problems}
    overlap = eval_set & train_set
    assert not overlap, (
        f"[cnk12_data] FATAL: train/eval overlap by {len(overlap)} problems"
    )

    print(
        f"[cnk12_data] SPLIT OK  train={len(train_problems)}  "
        f"eval={len(eval_problems)}  overlap=0"
    )
    return (
        (train_problems, train_solutions, train_golds),
        (eval_problems, eval_solutions, eval_golds),
    )
