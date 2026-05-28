FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

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
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --upgrade pip setuptools wheel && \
    python -m pip install -r requirements.txt

# Pre-download the base model weights into the image's HF cache so the first
# training run doesn't burn time pulling from the Hub. Defaults to just the 4B
# variant to keep the image lean; override with
# --build-arg PREFETCH_MODELS="<repo> [<repo> ...]" to pick a different one (or
# multiple), or pass an empty string to skip prefetching entirely.
ARG PREFETCH_MODELS="unsloth/Qwen3-VL-8B-Instruct-unsloth-bnb-4bit"
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
