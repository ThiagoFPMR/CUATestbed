#!/usr/bin/env bash
# In-container entrypoint: fetch + extract AgentNet (skipped if already on disk),
# then launch training. Intended to be passed as the command to `docker run`.
#
# Required env vars (forward them with `docker run -e VAR ...`):
#   WANDB_API_KEY, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY
# Optional: AWS_DEFAULT_REGION (defaults to us-east-1).
#
# Example invocation from the host:
#   WANDB_API_KEY=... AWS_ACCESS_KEY_ID=... AWS_SECRET_ACCESS_KEY=... \
#   docker run --gpus all -it --rm \
#     -v omnibrowse-agentnet:/workspace/OmniBrowse/dataset/agentnet/data \
#     -e WANDB_API_KEY -e AWS_ACCESS_KEY_ID -e AWS_SECRET_ACCESS_KEY \
#     -e AWS_DEFAULT_REGION \
#     omnibrowse:latest ./run_train.sh <experiment_name> [data_source]
#
# data_source (optional second arg) selects where the dataset comes from:
#   hf  (default) — full dataset from HuggingFace, downloaded + 7z-extracted
#   s3            — full uncompressed dataset pulled straight from S3 (loose
#                   files, no extract step), via `python download.py --full-s3`
#
# The dataset volume is optional but recommended on expensive boxes so the
# ~140GB pull only happens once. For quick tests, `python download.py --reduced`
# pulls the small S3 slice; build that slice on a box with the full dataset via
# `python download.py --make-reduced --n-trajs 20`.

set -euo pipefail

: "${WANDB_API_KEY:?set WANDB_API_KEY (forward with docker run -e WANDB_API_KEY)}"
: "${AWS_ACCESS_KEY_ID:?set AWS_ACCESS_KEY_ID (forward with docker run -e AWS_ACCESS_KEY_ID)}"
: "${AWS_SECRET_ACCESS_KEY:?set AWS_SECRET_ACCESS_KEY (forward with docker run -e AWS_SECRET_ACCESS_KEY)}"
: "${AWS_DEFAULT_REGION:=us-east-1}"
export AWS_DEFAULT_REGION

EXPERIMENT_NAME="${1:?set experiment name as the first argument: ./run_train.sh <experiment_name> [data_source]}"
DATA_SOURCE="${2:-hf}"

case "$DATA_SOURCE" in
  hf)
    python download.py
    ;;
  s3)
    python download.py --full-s3
    ;;
  *)
    echo "Unknown data_source '$DATA_SOURCE' (expected 'hf' or 's3')" >&2
    exit 1
    ;;
esac


python train.py \
  --experiment_name "$EXPERIMENT_NAME" \
  --model_config qwen3-vl-8b-4bit.yml \
  --trainer_config gspo.yml \
  --rewards gui_r1 \
  --s3_uri s3://omnibrowse-s3/runs
