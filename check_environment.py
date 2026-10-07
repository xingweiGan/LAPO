"""Check the bundle without downloading models or starting training."""
from __future__ import annotations

import argparse
import ast
import importlib
import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
ENTRYPOINTS = (
    "Train_MATH_3B_SFT",
    "Train_MATH_3B_GRPO_KL",
    "Train_MATH_3B_Fixed",
    "Train_MATH_3B_Adaptive",
)
DEPENDENCIES = (
    "torch", "transformers", "vllm", "datasets", "wandb", "numpy",
    "pandas", "huggingface_hub", "accelerate", "sympy", "math_verify",
    "latex2sympy2_extended", "pylatexenc",
)


def check_sources() -> bool:
    errors = []
    paths = [ROOT / f"{name}.py" for name in ENTRYPOINTS]
    paths += sorted((ROOT / "math3b_support").glob("*.py"))
    for path in paths:
        try:
            tree = ast.parse(path.read_text(), filename=str(path))
            compile(tree, str(path), "exec")
        except (OSError, SyntaxError) as exc:
            errors.append(f"{path.name}: {exc}")
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom) or not node.module:
                continue
            if node.module.startswith(("tests.", "cs336_alignment.")) or node.module == "SFT_policy":
                errors.append(f"{path.name}: dependency still points outside bundle: {node.module}")
            if node.module.startswith("math3b_support."):
                dependency = ROOT.joinpath(*node.module.split(".")).with_suffix(".py")
                if not dependency.is_file():
                    errors.append(f"{path.name}: missing {dependency}")
    for error in errors:
        print(f"FAIL: {error}")
    if not errors:
        print(f"PASS: syntax and bundled dependency paths ({len(paths)} Python files)")
    return not errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--static-only", action="store_true",
                        help="Check source and local paths without importing third-party libraries.")
    args = parser.parse_args()
    if not check_sources():
        return 1
    if args.static_only:
        return 0

    missing = [name for name in DEPENDENCIES if importlib.util.find_spec(name) is None]
    if missing:
        print("Missing Python dependencies: " + ", ".join(missing))
        print(f"In {ROOT}, run uv sync --locked --no-install-package flash-attn, then uv sync --locked.")
        print(f"Then activate {ROOT / '.venv/bin/activate'} and rerun this check.")
        return 1

    sys.path.insert(0, str(ROOT))
    modules = ["math3b_support.adapters", "math3b_support.drgrpo_grader",
               "math3b_support.math_baseline", "math3b_support.SFT_policy", *ENTRYPOINTS]
    errors = []
    for name in modules:
        try:
            module = importlib.import_module(name)
            if not Path(module.__file__).resolve().is_relative_to(ROOT):
                raise ImportError(f"Resolved outside bundle: {module.__file__}")
            print(f"PASS: import {name}")
        except Exception as exc:
            errors.append(f"{name}: {type(exc).__name__}: {exc}")
    for error in errors:
        print(f"FAIL: {error}")
    if errors:
        return 1

    import torch
    if not torch.cuda.is_available():
        print("FAIL: CUDA is unavailable; these training scripts require an NVIDIA GPU.")
        return 1
    print(f"PASS: CUDA available ({torch.cuda.device_count()} visible GPU(s))")
    print("No training or model downloads performed. Model access and available VRAM remain to be checked at training time.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
