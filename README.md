# How to Run

Use a machine with an NVIDIA GPU and a working CUDA environment.

## 1. First-time setup

Replace the path below with the actual location of this folder:

```bash
cd /path/to/LAPO
pip install uv
uv sync --no-install-package flash-attn
uv sync
#uv run pytest
source .venv/bin/activate
export HF_TOKEN="YOUR_HUGGING_FACE_TOKEN"
export WANDB_API_KEY="YOUR_WANDB_API_KEY"
export HF_USERNAME="YOUR_HF_USERNAME_OR_ORGANIZATION"
pip install vllm
```

On Runpod, provide these same three names through the template environment variables/secrets. `HF_USERNAME` selects the account or organization for model uploads. Replace the placeholders only when running locally; do not put actual credentials in this README.

Your Hugging Face token must allow access to the required models and uploading training results. The first run requires internet access to download models and the dataset.

## Prebuilt RunPod image

The dependency-only image is published by GitHub Actions to:

```text
ghcr.io/xingweigan/lapo:cu124-py312
```

Use that value as the container image in the RunPod template. Keep
`HF_TOKEN`, `WANDB_API_KEY`, and `HF_USERNAME` in the RunPod template
environment variables/secrets; they are not stored in the image.

The first GHCR publication may create a private package. Before using the
image in RunPod, either change the package visibility to Public in GitHub or
configure the RunPod container registry credentials with read-only package
access.

The image already contains Python 3.12, PyTorch/CUDA, vLLM, flash-attn, and all
dependencies locked by `uv.lock`. On a new Pod, only fetch the current source:

```bash
cd /workspace
git clone https://github.com/xingweiGan/LAPO.git
cd LAPO
python check_environment.py
```

Then run the selected training script directly. Do not run `uv sync`,
`pip install vllm`, or activate a project `.venv` when using this image. The
image places its environment on `PATH`, and Hugging Face downloads are cached
under `/workspace/.cache/huggingface` so a persistent RunPod volume can reuse
them.

## 2. Run one training script

The original five entrypoints use the Qwen2.5-3B model family and the `Maxwell-Jia/MATH` dataset. The four post-SFT scripts use the fixed SFT checkpoint `xw1234gan/SFT_Qwen2.5-3B-Instruct_MATH`; no additional SFT checkpoint environment variable is required. Each script runs independently.

SFT (`Train_MATH_3B_SFT.py`): fine-tunes `Qwen/Qwen2.5-3B-Instruct` on the first 768 MATH training examples using the provided solutions as supervised targets. It evaluates with vLLM and uploads the trained model to `${HF_USERNAME}/SFT_Qwen2.5-3B-Instruct_MATH_MATH3B_copy`.

```bash
python Train_MATH_3B_SFT.py
```

This SFT script uses two visible GPUs: `cuda:0` for training and `cuda:1` for evaluation. Its uploaded checkpoint is a separate experiment output and does not automatically replace the fixed SFT checkpoint used by the four post-SFT scripts.

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

Adaptive validation-weighted (`Train_MATH_3B_Adaptive_ValWeighted.py`): updates the mixing weight from the validation accuracies of the trainable policy, mixed policy, and frozen SFT policy on the same validation questions.

```bash
python Train_MATH_3B_Adaptive_ValWeighted.py
```

Choose one command to start the corresponding training run.
