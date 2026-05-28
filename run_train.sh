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
#     omnibrowse:latest ./run_train.sh
#
# The dataset volume is optional but recommended on expensive boxes so the
# ~140GB pull only happens once. For quick tests, `python download.py --reduced`
# below already pulls the small S3 slice; build that slice on a box with the
# full dataset via `python download.py --make-reduced --n-trajs 20`.

set -euo pipefail

: "${WANDB_API_KEY:?set WANDB_API_KEY (forward with docker run -e WANDB_API_KEY)}"
: "${AWS_ACCESS_KEY_ID:?set AWS_ACCESS_KEY_ID (forward with docker run -e AWS_ACCESS_KEY_ID)}"
: "${AWS_SECRET_ACCESS_KEY:?set AWS_SECRET_ACCESS_KEY (forward with docker run -e AWS_SECRET_ACCESS_KEY)}"
: "${AWS_DEFAULT_REGION:=us-east-1}"
export AWS_DEFAULT_REGION

python download.py --reduced
python train.py \
  --experiment_name io_test \
  --model_config qwen3-vl-2b-4bit.yml \
  --trainer_config gspo.yml \
  --rewards gui_r1 \
  --s3_uri s3://omnibrowse-s3/runs
