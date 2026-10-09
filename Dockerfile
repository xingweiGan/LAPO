# syntax=docker/dockerfile:1.7

FROM ghcr.io/astral-sh/uv:0.9.0 AS uv

# RunPod's CUDA base keeps the standard /start.sh entrypoint used for SSH,
# Jupyter, and template environment variables. The digest pins the exact
# multi-platform image manifest; the workflow builds only linux/amd64.
FROM runpod/base:0.7.0-cuda1241-ubuntu2204@sha256:2635bcc00cdd1f51b6748328890522ee0d6dafc0e0c947766a9302a1b93ce9c1

ARG DEBIAN_FRONTEND=noninteractive
ARG PYTHON_VERSION=3.12.3

LABEL org.opencontainers.image.source="https://github.com/xingweiGan/LAPO" \
      org.opencontainers.image.description="Reproducible CUDA 12.4 environment for LAPO training on RunPod"

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        git \
        libglib2.0-0 \
        libgl1 \
        libnuma1 \
        openssh-client \
        tmux \
    && rm -rf /var/lib/apt/lists/*

COPY --from=uv /uv /uvx /usr/local/bin/

ENV CUDA_HOME=/usr/local/cuda \
    UV_PYTHON_INSTALL_DIR=/opt/uv-python \
    UV_PROJECT_ENVIRONMENT=/opt/lapo-venv \
    UV_LINK_MODE=copy \
    UV_NO_PROGRESS=1 \
    VIRTUAL_ENV=/opt/lapo-venv \
    PATH=/opt/lapo-venv/bin:/usr/local/cuda/bin:${PATH} \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/workspace/.cache/huggingface \
    HF_HUB_CACHE=/workspace/.cache/huggingface/hub \
    WANDB_DIR=/workspace/wandb

WORKDIR /opt/lapo-env

# Only dependency metadata is copied. LAPO source remains in /workspace so a
# new Pod can clone or pull the latest code without rebuilding this image.
COPY pyproject.toml uv.lock .python-version README.md ./

# flash-attn needs torch present before its wheel can be selected or compiled.
RUN --mount=type=cache,target=/workspace/.cache/uv \
    uv python install "${PYTHON_VERSION}" \
    && uv sync \
        --python "${PYTHON_VERSION}" \
        --frozen \
        --no-install-project \
        --no-install-package flash-attn

RUN --mount=type=cache,target=/workspace/.cache/uv \
    MAX_JOBS=2 \
    NVCC_THREADS=1 \
    uv sync \
        --python "${PYTHON_VERSION}" \
        --frozen \
        --no-install-project

# Fail the image build if the core GPU-training stack is not importable.
RUN python -c "import flash_attn, torch, transformers, vllm, wandb; print('python environment OK:', 'torch=' + torch.__version__, 'cuda=' + str(torch.version.cuda), 'vllm=' + vllm.__version__, 'flash_attn=' + flash_attn.__version__, 'transformers=' + transformers.__version__, 'wandb=' + wandb.__version__)"

WORKDIR /workspace

# CMD and ENTRYPOINT are inherited from runpod/base. /start.sh keeps the Pod
# alive and exposes the usual RunPod SSH/Jupyter services.
