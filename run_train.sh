#!/usr/bin/env bash
# In-container entrypoint for the RunPod training run.
#
#   ./run_train.sh <experiment_name>
#
# Detects how many GPUs the box has, generates a trainer config scaled to that
# count (scale_trainer_config.py), then launches multi-GPU training over all of
# them via `accelerate launch`. The AgentNet dataset is expected to already be on
# the persistent network volume at /workspace/agentnet (fill it once with
# prep_volume_s3.py); we symlink it into place rather than re-downloading.
#
# Required env vars (forward with `docker run -e VAR ...` / RunPod secrets):
#   WANDB_API_KEY, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY
# Optional:
#   AWS_DEFAULT_REGION   (defaults to us-east-1)
#   RUNPOD_API_KEY       enables the train.py crash guard rail that terminates
#                        the pod to stop billing (RUNPOD_POD_ID is injected by
#                        RunPod). Without it, a crash leaves the box billing.

set -euo pipefail

: "${WANDB_API_KEY:?set WANDB_API_KEY (forward with docker run -e WANDB_API_KEY)}"
: "${AWS_ACCESS_KEY_ID:?set AWS_ACCESS_KEY_ID (forward with docker run -e AWS_ACCESS_KEY_ID)}"
: "${AWS_SECRET_ACCESS_KEY:?set AWS_SECRET_ACCESS_KEY (forward with docker run -e AWS_SECRET_ACCESS_KEY)}"
: "${AWS_DEFAULT_REGION:=us-east-1}"
export AWS_DEFAULT_REGION

EXPERIMENT_NAME="${1:?usage: ./run_train.sh <experiment_name>}"

# --- Tunables (edit here; not CLI args) -------------------------------------
MODEL_CONFIG="qwen3-vl-4b-4bit.yml"
NUM_GENERATIONS=6        # GRPO group size. Was 4 on the limited-hw run; 6 is the
                         # chosen value for this run (still limited-hw-defensible).
PER_DEVICE_BS=2          # THROUGHPUT KNOB: raise until OOM, lower if generation
                         # OOMs. Per-GPU micro-batch; peak memory scales with it.
LR_SCALING="sqrt"        # none | sqrt | linear. sqrt recommended for AdamW.
S3_URI="s3://omnibrowse-s3/runs"
S3_UPLOAD_EVERY=3        # mirror to S3 every Nth checkpoint (volume is canonical).
DATA_VOLUME="/workspace/agentnet"
TRAINER_SRC="policy/trainer/configs/gspo.yml"
TRAINER_OUT_NAME="gspo_runpod.yml"
TRAINER_OUT="policy/trainer/configs/${TRAINER_OUT_NAME}"
LOCAL_DATA="dataset/agentnet/data"
# ----------------------------------------------------------------------------

# 1. Point the dataset loader (hardcoded to dataset/agentnet/data) at the volume.
if [ ! -e "${DATA_VOLUME}/agentnet_ubuntu_5k.jsonl" ]; then
  echo "ERROR: AgentNet not found on the volume at ${DATA_VOLUME} (expected" \
       "agentnet_ubuntu_5k.jsonl). Fill it first with prep_volume_s3.py." >&2
  exit 1
fi
if [ -L "${LOCAL_DATA}" ] || [ ! -e "${LOCAL_DATA}" ]; then
  ln -sfn "${DATA_VOLUME}" "${LOCAL_DATA}"
  echo "[data] symlinked ${LOCAL_DATA} -> ${DATA_VOLUME}"
else
  echo "[data] ${LOCAL_DATA} already exists as a real path; using it as-is."
fi

# 2. Count GPUs (respect CUDA_VISIBLE_DEVICES if the operator pinned a subset).
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
  NUM_GPUS=$(echo "${CUDA_VISIBLE_DEVICES}" | tr ',' '\n' | grep -c .)
else
  NUM_GPUS=$(nvidia-smi -L | wc -l)
fi
if [ "${NUM_GPUS}" -lt 1 ]; then
  echo "ERROR: no GPUs detected (nvidia-smi -L returned nothing)." >&2
  exit 1
fi
echo "[gpus] detected ${NUM_GPUS} GPU(s)"

# 3. Generate the GPU-count-scaled trainer config. --auto-grad-accum guarantees a
#    TRL-legal global batch for whatever GPU count we just found.
python scale_trainer_config.py \
  --src "${TRAINER_SRC}" \
  --out "${TRAINER_OUT}" \
  --per-device-bs "${PER_DEVICE_BS}" \
  --num-gpus "${NUM_GPUS}" \
  --num-generations "${NUM_GENERATIONS}" \
  --auto-grad-accum \
  --lr-scaling "${LR_SCALING}"

# 4. Launch multi-GPU training. accelerate sets LOCAL_RANK/WORLD_SIZE; one process
#    per GPU, each with its own colocated vLLM engine (vllm_mode: colocate). bf16
#    is driven by the trainer config (bf16: true), not an accelerate flag, to
#    avoid a double mixed-precision config.
accelerate launch \
  --num_processes "${NUM_GPUS}" \
  --num_machines 1 \
  train.py \
  --experiment_name "${EXPERIMENT_NAME}" \
  --model_config "${MODEL_CONFIG}" \
  --trainer_config "${TRAINER_OUT_NAME}" \
  --rewards gui_r1 \
  --s3_uri "${S3_URI}" \
  --s3_upload_every "${S3_UPLOAD_EVERY}"
