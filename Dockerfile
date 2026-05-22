# syntax=docker/dockerfile:1
#
# Training image for OmniBrowse (Unsloth + TRL GRPO/GSPO VLM RL training).
# The training code is baked into the image — cloned from the private GitHub
# repo at build time — so at runtime only the dataset needs to be provided.
#
# Build (BuildKit forwards your SSH agent so the private clone authenticates
# without the key ever entering the image):
#   eval "$(ssh-agent -s)" && ssh-add ~/.ssh/id_ed25519   # if not already loaded
#   docker build --ssh default -t omnibrowse:latest .
#   # rebuild against newer code — bump the ref (or add --no-cache):
#   docker build --ssh default --build-arg OMNIBROWSE_REF=<branch|tag|sha> \
#       -t omnibrowse:latest .
#
# Run (host needs the NVIDIA driver + nvidia-container-toolkit):
#   docker run --gpus all -it --rm \
#       -v /path/to/agentnet:/workspace/OmniBrowse/dataset/agentnet/data \
#       -v $HOME/hf-cache:/workspace/hf-cache \
#       -e WANDB_API_KEY=... \
#       omnibrowse:latest \
#       python train.py --experiment_name run1 \
#           --model_config qwen3-vl-8b-4bit.yml --trainer_config gspo.yml \
#           --rewards gui_r1 --s3_uri s3://your-bucket/omnibrowse-runs

FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/workspace/hf-cache

# Python 3.12 (Ubuntu 22.04 ships 3.10), build tooling, git and the ssh client.
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common ca-certificates curl git openssh-client \
        build-essential && \
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

# The ~305 GB AgentNet dataset is NOT baked in — at runtime mount it to
# dataset/agentnet/data, or download it inside the container:
#   huggingface-cli download xlangai/AgentNet --repo-type dataset \
#       --local-dir dataset/agentnet/data
CMD ["bash"]
