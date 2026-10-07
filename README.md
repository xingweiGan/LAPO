# How to Run

Use a machine with an NVIDIA GPU and a working CUDA environment.

## 1. First-time setup

Replace the path below with the actual location of this folder:

```bash
cd /path/to/ProjectII_MATH_3B
pip install uv
uv sync --no-install-package flash-attn
uv sync
#uv run pytest
source .venv/bin/activate
export HF_TOKEN="YOUR_HUGGING_FACE_TOKEN"
export WANDB_API_KEY="YOUR_WANDB_API_KEY"
export HF_USERNAME="YOUR_HF_USERNAME_OR_ORGANIZATION"
export SFT_MODEL_ID="YOUR_HF_USERNAME/YOUR_SFT_MODEL_REPOSITORY"
pip install vllm
```

`HF_USERNAME` selects the account or organization for model uploads. `SFT_MODEL_ID` is needed only for GRPO, Fixed, and Adaptive. Replace the placeholders in your terminal; do not put your actual credentials or personal repository paths in this README.

Your Hugging Face token must allow access to the required models and uploading training results. The first run requires internet access to download models and the dataset.

## 2. Run one training script

All four scripts use the Qwen2.5-3B model family and the `Maxwell-Jia/MATH` dataset. The GRPO, Fixed, and Adaptive scripts load the SFT checkpoint specified by `SFT_MODEL_ID`. Each script runs independently.

SFT (`Train_MATH_3B_SFT.py`): fine-tunes `Qwen/Qwen2.5-3B-Instruct` on the first 768 MATH training examples using the provided solutions as supervised targets. It evaluates with vLLM and uploads the trained model to `${HF_USERNAME}/SFT_Qwen2.5-3B-Instruct_MATH_MATH3B_copy`.

```bash
python Train_MATH_3B_SFT.py
```

This SFT script uses two visible GPUs: `cuda:0` for training and `cuda:1` for evaluation. To use its new checkpoint in the other three scripts, set `SFT_MODEL_ID` to the output repository above before running them. You can also set `SFT_MODEL_ID` to an existing checkpoint and skip SFT training.

GRPO + KL (`Train_MATH_3B_GRPO_KL.py`): fine-tunes the MATH SFT model using clipped GRPO with a KL penalty (`beta=0.01`) against a frozen copy of the same SFT model.

```bash
python Train_MATH_3B_GRPO_KL.py
```

Fixed (`Train_MATH_3B_Fixed.py`): trains a Qwen2.5-3B-Instruct policy using GRPO with its logits mixed with a frozen MATH SFT model. The mixing weight stays fixed at `alpha=0.5`.

```bash
python Train_MATH_3B_Fixed.py
```

Adaptive (`Train_MATH_3B_Adaptive.py`): uses the same policy initialization and logit mixing as Fixed, but starts at `alpha=0.5` and adjusts the mixing weight based on evaluation results.

```bash
python Train_MATH_3B_Adaptive.py
```

Choose one command to start the corresponding training run.
