#!/usr/bin/env python
"""
Probe the largest per_device_train_batch_size that fits on ONE GPU.

Runs a few *real* GRPO training steps (HF in-process generation + forward/backward
+ optimizer step) on the real model + AgentNet data + system prompt, at a single
candidate per_device_bs, and reports FIT / OOM. Because the stack is plain DDP
(no FSDP/ZeRO sharding), per-GPU memory is the same on 1 GPU as on N of the SAME
card -- so the answer found here transfers to a bigger pod of the same SKU. See
probe_sweep.sh for the driver that sweeps candidates in fresh processes.

NOTE: vLLM was dropped, so generation runs via HF `.generate()` *inside* the torch
allocator, not a separate colocate engine. Generation KV/activations are therefore
part of the measured torch peak and scale with
num_generations * per_device_bs * max_completion_length -- so num_generations and
the trainer's max_completion_length now move the OOM line directly (previously they
were bounded by vLLM's fixed reservation).

Caveats (read these or the number lies to you):
  * Run on the EXACT GPU SKU you'll rent. The activation + in-torch generation
    budget is absolute VRAM, so a 48GB probe says nothing about an 80GB card.
  * Multi-GPU adds a few hundred MB of NCCL/DDP buffers not present here, so the
    single-GPU max is slightly optimistic -- use (max - 1) on the real run.
  * Only valid while you stay on DDP. FSDP/ZeRO would shard optimizer/grads and
    let multi-GPU fit MORE than this probe suggests.

Exit codes (consumed by probe_sweep.sh): 0 = fit, 2 = CUDA OOM, 1 = other error.

Examples
--------
# Does per_device_bs=4 fit with the production 8B config + num_generations=6?
python probe_per_device_bs.py --per-device-bs 4 --num-generations 6

# Sweep is driven externally; this script tests ONE value per process.
"""
import unsloth  # noqa: F401  (must be imported before torch / transformers)

import argparse
import sys
from math import gcd

import torch

from model import load_model_from_file
from policy.rewards import parse_rewards
from policy.trainer import load_trainer_from_file
from dataset.agentnet.agentnet import load_agentnet
from prompts import base_prompt

FIT, OTHER_ERROR, OOM = 0, 1, 2


def _gib(n_bytes: int) -> float:
    return n_bytes / (1024 ** 3)


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--per-device-bs", type=int, required=True,
                   help="Candidate per_device_train_batch_size to test.")
    p.add_argument("--model-config", default="qwen3-vl-4b-4bit.yml",
                   help="Model YAML (default: the production 4B config).")
    p.add_argument("--trainer-config", default="gspo.yml",
                   help="Trainer YAML to base settings on (default: gspo.yml).")
    p.add_argument("--num-generations", type=int, default=6,
                   help="GRPO group size to test under (default: 6, the run value). "
                        "Without vLLM, generation runs in-torch, so this directly "
                        "drives peak memory -- probe at the real run value.")
    p.add_argument("--history-length", type=int, default=12,
                   help="load_agentnet history_length (default: 12, matches train.py).")
    p.add_argument("--max-steps", type=int, default=3,
                   help="Training steps to run; enough to hit the real peak (default 3).")
    args = p.parse_args()

    if not torch.cuda.is_available():
        print("[probe] no CUDA device visible; aborting.", file=sys.stderr)
        return OTHER_ERROR

    # At num_gpus=1, TRL requires generation_batch = per_device_bs * grad_accum to
    # be divisible by num_generations. Pick the smallest grad_accum that satisfies
    # it -- grad_accum runs micro-batches sequentially, so it does NOT change peak
    # memory; it only makes the combo TRL-legal so trainer.train() will start.
    num_gen = args.num_generations
    grad_accum = num_gen // gcd(args.per_device_bs, num_gen)

    dev = torch.cuda.current_device()
    total = torch.cuda.get_device_properties(dev).total_memory
    name = torch.cuda.get_device_name(dev)
    print(f"[probe] GPU: {name} ({_gib(total):.1f} GiB total)")
    print(f"[probe] testing per_device_bs={args.per_device_bs}, "
          f"num_generations={num_gen}, grad_accum={grad_accum} (auto, for TRL "
          f"legality at 1 GPU), max_steps={args.max_steps}")

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(dev)

    model, tokenizer = load_model_from_file(args.model_config)
    train_dataset = load_agentnet(base_prompt, history_length=args.history_length)
    reward_funcs = parse_rewards(["gui_r1"])

    trainer = load_trainer_from_file(
        args.trainer_config,
        run_name="probe",
        output_dir="/tmp/probe_out",
        per_device_train_batch_size=args.per_device_bs,
        gradient_accumulation_steps=grad_accum,
        num_generations=num_gen,
        max_steps=args.max_steps,
        num_train_epochs=1,
        save_strategy="no",
        report_to="none",
        logging_steps=1,
        log_completions=False,
    )(
        model=model,
        reward_funcs=reward_funcs,
        processing_class=tokenizer,
        train_dataset=train_dataset,
    )

    try:
        trainer.train()
    except torch.cuda.OutOfMemoryError as e:
        peak = torch.cuda.max_memory_reserved(dev)
        print(f"\n[probe] RESULT: OOM at per_device_bs={args.per_device_bs}")
        print(f"[probe]   torch peak reserved before OOM: {_gib(peak):.1f} GiB "
              f"(includes in-torch generation now). {str(e)[:120]}")
        return OOM
    except Exception as e:  # noqa: BLE001 -- surface anything else as a non-OOM failure
        print(f"\n[probe] RESULT: ERROR (not OOM) at per_device_bs="
              f"{args.per_device_bs}: {type(e).__name__}: {e}", file=sys.stderr)
        return OTHER_ERROR

    peak = torch.cuda.max_memory_reserved(dev)
    # True headroom: the CUDA driver's free count includes EVERYTHING on the card
    # (the CUDA context, cuDNN/NCCL workspaces, etc.), not just torch's allocator.
    # Whatever the driver sees used that torch doesn't account for is the "non-torch"
    # hold -- now mostly just the CUDA context (vLLM's separate reservation is gone),
    # so it's small. Add it to torch's peak to estimate true usage; the rest is free.
    free_now, _ = torch.cuda.mem_get_info(dev)
    non_torch = max(0, total - free_now - torch.cuda.memory_reserved(dev))
    est_peak = peak + non_torch
    margin = max(0, total - est_peak)
    # The real run is multi-GPU: NCCL/DDP add buffers not present in this 1-GPU
    # probe. Keep this much free or the production run can OOM where the probe fit.
    DDP_RESERVE_GIB = 1.5
    tight = _gib(margin) < DDP_RESERVE_GIB + 1.0  # +1 GiB of slop on top of NCCL.

    print(f"\n[probe] RESULT: FIT at per_device_bs={args.per_device_bs}")
    print(f"[probe]   torch peak reserved: {_gib(peak):.1f} GiB | "
          f"non-torch (CUDA ctx): {_gib(non_torch):.1f} GiB | "
          f"est. peak usage: {_gib(est_peak):.1f} / {_gib(total):.1f} GiB")
    print(f"[probe]   estimated free margin: {_gib(margin):.1f} GiB "
          f"({100 * margin / total:.0f}% of card)")
    if tight:
        print(f"[probe]   WARNING: cutting it close (< ~{DDP_RESERVE_GIB + 1.0:.1f} "
              f"GiB free). Multi-GPU adds ~{DDP_RESERVE_GIB} GiB of NCCL/DDP buffers, "
              f"so this per_device_bs may OOM on the real run -- back off one step.")
    else:
        print(f"[probe]   margin looks safe for the multi-GPU run "
              f"(comfortably above the ~{DDP_RESERVE_GIB} GiB NCCL/DDP reserve).")
    return FIT


if __name__ == "__main__":
    sys.exit(main())
