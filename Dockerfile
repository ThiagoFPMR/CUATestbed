# syntax=docker/dockerfile:1
# CUDA 12.8 is the floor for NVIDIA Blackwell (B100/B200, RTX 50xx; sm_100/sm_120).
# The cu128 toolchain + torch wheels below also cover Ampere/Hopper, so one image
# runs on both old and new GPUs.
#
# This image is vLLM-free by design: GRPO/GSPO rollouts use TRL's native HF
# generation (model/trainer configs set use_vllm=false). vLLM 0.10.2 has no native
# Qwen3-VL impl (its Transformers-backend bnb loader can't read unsloth's pre-packed
# nf4 checkpoint), and newer vLLM drags transformers>5.5 which unsloth rejects — so
# it was pure cost here. Re-enabling means: pip install vllm==0.10.2, patch its
# transformers_utils/configs to be dataclass-safe, and flip use_vllm back on.
FROM nvidia/cuda:12.8.1-devel-ubuntu22.04

# Everything the image bakes lives under /opt, NOT /workspace: RunPod (and most
# pod hosts) mount the persistent data volume on /workspace at runtime, which
# would shadow any baked code/cache placed there. Keeping code + HF cache on /opt
# makes the image immune to wherever the data volume lands.
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/opt/hf-cache

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

# Clone the training code from the private repo into /opt (see note above: NOT
# /workspace, so a runtime data volume can't shadow it). BuildKit forwards the
# host SSH agent via `--mount=type=ssh`; no key is written into any image layer.
# Bump OMNIBROWSE_REF (or build with --no-cache) to pull newer code.
ARG OMNIBROWSE_REF=train/reference-model
RUN --mount=type=ssh \
    mkdir -p -m 0700 ~/.ssh && \
    ssh-keyscan github.com >> ~/.ssh/known_hosts 2>/dev/null && \
    git clone git@github.com:ThiagoFPMR/OmniBrowse.git /opt/OmniBrowse && \
    git -C /opt/OmniBrowse checkout "${OMNIBROWSE_REF}"

WORKDIR /opt/OmniBrowse

# Dependency install. Order matters: install torch FIRST, pinned, from the cu128
# wheel index so it carries Blackwell (sm_120) kernels; the unsloth install then
# sees torch already satisfied and won't pull a non-Blackwell build over it.
# Override TORCH_CUDA_CHANNEL to retarget the index. The pip cache mount keeps
# rebuilds fast even when the clone layer is busted.
#
# Explicit pins (requirements.txt only lists `unsloth` unpinned; without these the
# resolver drifts up to transformers 5.x and breaks Qwen3-VL loading):
#   torch==2.8.0          - Blackwell-capable cu128 wheel, kept deterministic.
#   transformers==4.57.6  - >=4.57.0 for Qwen3-VL support; <5.0 to avoid the 5.x
#                           dataclass config refactor (and its Qwen3-VL load bug).
#   unsloth==2026.6.9     - the verified-working release.
ARG TORCH_CUDA_CHANNEL=cu128
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --upgrade pip setuptools wheel && \
    python -m pip install torch==2.8.0 torchvision==0.23.0 \
        --index-url "https://download.pytorch.org/whl/${TORCH_CUDA_CHANNEL}" && \
    python -m pip install -r requirements.txt \
        "transformers==4.57.6" "unsloth==2026.6.9" && \
    python -c "import torch; print('torch', torch.__version__)"

# Fix a trl 0.24.0 x transformers incompatibility. transformers'
# _is_package_available returns a (bool, version) tuple even when
# return_version=False; trl assigns that raw tuple to its `_<pkg>_available`
# flags (vllm_ascend, llm_blender, mergekit, ...), so every `if is_X_available():`
# guard is always-truthy and trl tries to import absent optional deps, crashing
# the import of GRPOTrainer. With vLLM removed this shim is load-bearing: it's what
# makes trl's is_vllm_available() return a clean False instead of sending trl into
# `from vllm import LLM`. Shim the function so the default yields a plain bool; the
# assert makes the build fail loudly if trl's file layout ever changes.
RUN python <<'PY'
p = "/usr/local/lib/python3.12/dist-packages/trl/import_utils.py"
src = open(p).read()
anchor = "from transformers.utils.import_utils import _is_package_available\n"
shim = anchor + (
    "__trl_raw_ipa = _is_package_available\n"
    "def _is_package_available(pkg_name, return_version=False):\n"
    "    _r = __trl_raw_ipa(pkg_name, return_version=return_version)\n"
    "    return _r if return_version else (_r[0] if isinstance(_r, tuple) else _r)\n"
)
assert anchor in src, "trl import_utils anchor not found — trl layout changed"
open(p, "w").write(src.replace(anchor, shim, 1))
print("trl guard shim applied")
PY

# Build-time gate (GPU-FREE): docker build has no accelerator and unsloth's import
# hard-raises without one, so we can't import the full stack here. We can still
# cheaply confirm the trl guard shim landed. The full unsloth + trl-GRPO import is
# exercised at runtime by the probe on the GPU pod (or `docker run --gpus all`).
RUN grep -q "__trl_raw_ipa" /usr/local/lib/python3.12/dist-packages/trl/import_utils.py \
 && echo "trl guard shim present"

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
