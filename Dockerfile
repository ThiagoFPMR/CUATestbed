# CUDA 12.8 is the floor for NVIDIA Blackwell (B100/B200, RTX 50xx; sm_100/sm_120).
# The cu128 toolchain + torch wheels below also cover Ampere/Hopper, so one image
# runs on both old and new GPUs.
FROM nvidia/cuda:12.8.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/workspace/hf-cache

# Python 3.12 (Ubuntu 22.04 ships 3.10), build tooling, git, ssh client, and
# p7zip (download.py uses `7z` to extract the AgentNet split-zip image archives).
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common ca-certificates curl git openssh-client \
        build-essential p7zip-full && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && apt-get install -y --no-install-recommends \
        python3.12 python3.12-dev && \
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1 && \
    rm -rf /var/lib/apt/lists/*

# Clone the training code from the private repo. BuildKit forwards the host SSH
# agent via `--mount=type=ssh`; no key is written into any image layer. Bump
# OMNIBROWSE_REF (or build with --no-cache) to pull newer code.
ARG OMNIBROWSE_REF=train/reference-model
RUN --mount=type=ssh \
    mkdir -p -m 0700 ~/.ssh && \
    ssh-keyscan github.com >> ~/.ssh/known_hosts 2>/dev/null && \
    git clone git@github.com:ThiagoFPMR/OmniBrowse.git /workspace/OmniBrowse && \
    git -C /workspace/OmniBrowse checkout "${OMNIBROWSE_REF}"

WORKDIR /workspace/OmniBrowse

# unsloth pulls a CUDA build of torch plus transformers, trl, peft, datasets,
# bitsandbytes and accelerate. torch wheels bundle their own CUDA/cuDNN runtime.
# The pip cache mount keeps rebuilds fast even when the clone layer is busted.
#
# Install torch first from the cu128 wheel index so it carries Blackwell
# (sm_100/sm_120) kernels; the default PyPI wheel may not. Pinning it up front
# means the unsloth install below sees torch already satisfied and won't pull a
# non-Blackwell build over it. Override TORCH_CUDA_CHANNEL to retarget the index.
ARG TORCH_CUDA_CHANNEL=cu128
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --upgrade pip setuptools wheel && \
    python -m pip install torch torchvision \
        --index-url "https://download.pytorch.org/whl/${TORCH_CUDA_CHANNEL}" && \
    python -m pip install -r requirements.txt

# vLLM powers fast generation during GRPO/GSPO (Unsloth `fast_inference` +
# trainer `use_vllm`). It must be present or `fast_inference=True` fails at model
# load. vLLM pins torch tightly, so install it AFTER the cu128 torch above and
# point its --extra-index-url at the same channel: that lets any torch dep
# resolve to the Blackwell (sm_120) build instead of pulling a default-PyPI
# torch over it. If pip still tries to downgrade torch, pin a vLLM release whose
# torch matches the cu128 wheel via --build-arg VLLM_SPEC="vllm==<ver>".
# NOTE: vLLM drags in flashinfer/xformers — on Blackwell these need cu128 wheels
# too; verify the built image actually imports vllm on the target GPU.
ARG VLLM_SPEC=vllm
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install "${VLLM_SPEC}" \
        --extra-index-url "https://download.pytorch.org/whl/${TORCH_CUDA_CHANNEL}" && \
    python -c "import torch, importlib.metadata as m; print('torch', torch.__version__); print('vllm', m.version('vllm'))"

# Pre-download the base model weights into the image's HF cache so the first
# training run doesn't burn time pulling from the Hub. Defaults to just the 4B
# variant to keep the image lean; override with
# --build-arg PREFETCH_MODELS="<repo> [<repo> ...]" to pick a different one (or
# multiple), or pass an empty string to skip prefetching entirely.
ARG PREFETCH_MODELS="unsloth/Qwen3-VL-4B-Instruct-unsloth-bnb-4bit"
RUN if [ -n "${PREFETCH_MODELS}" ]; then \
        for repo in ${PREFETCH_MODELS}; do \
            echo "Prefetching ${repo} into ${HF_HOME}" && \
            python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='${repo}', repo_type='model')" || exit 1; \
        done; \
    fi

# The AgentNet dataset is NOT baked in. Either mount a host copy to
# dataset/agentnet/data, or let `python download.py` fetch + extract it at
# runtime (uses hf_transfer + pipelined extraction; safe to re-run).
CMD ["bash"]
