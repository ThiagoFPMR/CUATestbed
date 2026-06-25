#!/usr/bin/env bash
# Find the largest per_device_train_batch_size that fits on ONE GPU, calling
# probe_per_device_bs.py in a FRESH process per candidate (an OOM can leave the
# CUDA allocator / colocate vLLM engine in a poisoned state, so reusing a process
# would corrupt later trials).
#
# Two modes (fitness is monotone: if N fits, N-1 fits):
#
#   ASCENDING SWEEP (default) -- walk up until the first OOM, report the max.
#     ./probe_sweep.sh [START] [MAX]
#
#   FIRST-FIT (time-pressed) -- probe an explicit DESCENDING candidate list and
#   stop at the first one that FITs. One or two runs instead of a full climb.
#     CANDIDATES="16 8" ./probe_sweep.sh
#
# Env overrides: MODEL, TRAINER, NUM_GEN (forwarded to the probe), CANDIDATES.
#
# Examples:
#   ./probe_sweep.sh 1 12
#   NUM_GEN=6 MODEL=qwen3-vl-4b-4bit.yml CANDIDATES="16 8" ./probe_sweep.sh

set -uo pipefail

START="${1:-1}"
MAX="${2:-16}"
MODEL="${MODEL:-qwen3-vl-8b-4bit.yml}"
TRAINER="${TRAINER:-gspo.yml}"
NUM_GEN="${NUM_GEN:-6}"
CANDIDATES="${CANDIDATES:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Probe one per_device_bs in a fresh process. Echoes the candidate, returns the
# probe's exit code (0 = fit, 2 = OOM, other = error).
probe_one() {
  local bs="$1"
  echo ""
  echo "######## per_device_bs=${bs} ########"
  python "${SCRIPT_DIR}/probe_per_device_bs.py" \
    --per-device-bs "${bs}" \
    --model-config "${MODEL}" \
    --trainer-config "${TRAINER}" \
    --num-generations "${NUM_GEN}"
}

last_ok=0
if [ -n "${CANDIDATES}" ]; then
  # First-fit: try each candidate (intended descending) and stop at the first FIT.
  for bs in ${CANDIDATES}; do
    probe_one "${bs}"
    rc=$?
    if [ "${rc}" -eq 0 ]; then
      last_ok="${bs}"
      echo "[sweep] FIT at per_device_bs=${bs}; stopping (first-fit)."
      break
    elif [ "${rc}" -eq 2 ]; then
      echo "[sweep] OOM at per_device_bs=${bs}; trying next smaller candidate."
    else
      echo "[sweep] non-OOM error (rc=${rc}) at per_device_bs=${bs}; stopping." >&2
      break
    fi
  done
else
  # Ascending sweep: climb until the first OOM, report the last value that fit.
  for ((bs = START; bs <= MAX; bs++)); do
    probe_one "${bs}"
    rc=$?
    if [ "${rc}" -eq 0 ]; then
      last_ok="${bs}"
    elif [ "${rc}" -eq 2 ]; then
      echo "[sweep] OOM at per_device_bs=${bs}; stopping."
      break
    else
      echo "[sweep] non-OOM error (rc=${rc}) at per_device_bs=${bs}; stopping." >&2
      break
    fi
  done
fi

echo ""
echo "============================================================"
if [ "${last_ok}" -eq 0 ]; then
  if [ -n "${CANDIDATES}" ]; then
    echo "[sweep] nothing fit (candidates: ${CANDIDATES}). Add a smaller candidate,"
  else
    echo "[sweep] nothing fit (even per_device_bs=${START} failed). Lower START,"
  fi
  echo "        drop num_generations, or raise gpu_memory_utilization headroom."
else
  rec=$(( last_ok > 1 ? last_ok - 1 : last_ok ))
  echo "[sweep] largest per_device_bs that FIT on this GPU: ${last_ok}"
  echo "[sweep] recommended for the multi-GPU run:           ${rec}"
  echo "        (backed off by 1 for NCCL/DDP buffers; set PER_DEVICE_BS=${rec}"
  echo "         in run_train.sh. Valid only for this exact GPU SKU.)"
fi
echo "============================================================"
