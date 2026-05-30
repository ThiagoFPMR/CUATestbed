#!/usr/bin/env bash
# In-container entrypoint for the CPU-only data-prep image: download + extract
# AgentNet from HF, then upload the full uncompressed dataset to S3 as loose
# files (producer side of `python download.py --full-s3`).
#
# Required env vars (forward them with `docker run -e VAR ...`):
#   AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY
# Optional:
#   AWS_DEFAULT_REGION  (default: us-east-1)
#   S3_URI              (default: s3://omnibrowse-s3/datasets/agentnet-full)
#   UPLOAD_WORKERS      (default: 64) — concurrent S3 uploads
#
# Example:
#   AWS_ACCESS_KEY_ID=... AWS_SECRET_ACCESS_KEY=... \
#   docker run --rm \
#     -e AWS_ACCESS_KEY_ID -e AWS_SECRET_ACCESS_KEY -e AWS_DEFAULT_REGION \
#     -v /mnt/scratch:/workspace/OmniBrowse/dataset/agentnet/data \
#     omnibrowse-dataprep:latest
#
# Mounting a big scratch disk at the data dir is recommended: the extracted
# dataset is ~305 GB and peaks higher mid-extraction.

set -euo pipefail

: "${AWS_ACCESS_KEY_ID:?set AWS_ACCESS_KEY_ID (forward with docker run -e AWS_ACCESS_KEY_ID)}"
: "${AWS_SECRET_ACCESS_KEY:?set AWS_SECRET_ACCESS_KEY (forward with docker run -e AWS_SECRET_ACCESS_KEY)}"
: "${AWS_DEFAULT_REGION:=us-east-1}"
export AWS_DEFAULT_REGION

S3_URI="${S3_URI:-s3://omnibrowse-s3/datasets/agentnet-full}"
UPLOAD_WORKERS="${UPLOAD_WORKERS:-64}"

python dataprep.py --s3-uri "$S3_URI" --workers "$UPLOAD_WORKERS"
