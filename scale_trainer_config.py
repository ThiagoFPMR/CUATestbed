#!/usr/bin/env python
"""
Generate a scaled copy of a trainer YAML config (e.g. gspo.yml) for
multi-GPU hardware.

Rewrites `per_device_train_batch_size` and `gradient_accumulation_steps`;
optionally overrides `num_generations` and rescales `learning_rate`.
`num_gpus` is a *launch-time* quantity (set by accelerate/torchrun, not the
YAML); it's used here purely to compute and validate the resulting global batch.

This is pure batch/lr arithmetic and is generation-backend-agnostic: every other
key in the source YAML (e.g. `use_vllm`, `max_completion_length`) is copied
through verbatim, so the scaled config inherits whatever the source sets. Pick a
`per_device_bs` that the probe found to FIT -- with vLLM dropped the probe's peak
now includes in-torch generation, so that number already accounts for it.

TRL constraint enforced:
    global_batch = per_device_bs * num_gpus * grad_accum
    must be divisible by num_generations.
(With 8 GPUs the *8 already supplies a factor of 2, so num_generations=6 needs
per_device_bs * grad_accum to be a multiple of 3.)

Learning-rate scaling (opt-in via --lr-scaling) is relative to the limited-
hardware baseline effective batch (--base-global-batch, default 4) and uses the
*unique prompt groups* (global_batch / num_generations) as the batch dimension,
since num_generations lowers within-group variance rather than adding diversity.
Default is "none" (lr untouched) because GRPO is lr-sensitive; "sqrt" is the
recommended starting point for AdamW.

Examples
--------
# 8 GPUs, fit 4 samples/GPU, no accumulation -> global 32 = 8 prompt groups
python scale_trainer_config.py --per-device-bs 4 --grad-accum 1 --num-gpus 8

# num_generations=6 (needs a factor of 3) with sqrt lr scaling
python scale_trainer_config.py --per-device-bs 3 --num-generations 6 --lr-scaling sqrt

# Preview without writing
python scale_trainer_config.py --per-device-bs 8 --dry-run
"""

import argparse
import math
import os
from math import gcd as _gcd

import yaml

DEFAULT_SRC = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "policy", "trainer", "configs", "gspo.yml",
)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", default=DEFAULT_SRC,
                   help="Source YAML to copy from (default: configs/gspo.yml)")
    p.add_argument("--out", default=None,
                   help="Output path. Default: <src-stem>_runpod.yml next to src.")
    p.add_argument("--per-device-bs", type=int, required=True,
                   help="per_device_train_batch_size (tune up until OOM).")
    p.add_argument("--grad-accum", type=int, default=1,
                   help="gradient_accumulation_steps (default: 1). With "
                        "--auto-grad-accum this is treated as a floor.")
    p.add_argument("--auto-grad-accum", action="store_true",
                   help="Pick the smallest grad_accum (>= --grad-accum) that makes "
                        "global_batch divisible by num_generations. Use when the GPU "
                        "count isn't known ahead of time (e.g. launched from a script "
                        "that detects GPUs), so the combo is always TRL-legal.")
    p.add_argument("--num-gpus", type=int, default=8,
                   help="GPUs the launcher will use; for math/validation only.")
    p.add_argument("--num-generations", type=int, default=None,
                   help="Override num_generations (default: keep source value).")
    p.add_argument("--lr-scaling", choices=("none", "sqrt", "linear"), default="none",
                   help="Rescale learning_rate vs the baseline batch. 'sqrt' "
                        "recommended for AdamW; 'none' (default) leaves lr as-is.")
    p.add_argument("--base-global-batch", type=int, default=4,
                   help="Limited-hardware baseline effective batch for lr scaling "
                        "(default: 4 = the 1-GPU run).")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the resulting config and stats without writing.")
    args = p.parse_args()

    with open(args.src) as f:
        cfg = yaml.safe_load(f)

    num_gen = args.num_generations if args.num_generations is not None else cfg.get("num_generations")
    if num_gen is None:
        raise SystemExit("No num_generations in source config and none passed; aborting.")

    grad_accum = args.grad_accum
    if args.auto_grad_accum:
        # Smallest grad_accum step that divides num_gen given the per-step rollouts
        # already supplied by per_device_bs * num_gpus, then round the floor up to a
        # multiple of it. Guarantees global_batch % num_gen == 0 for any GPU count.
        per_step = args.per_device_bs * args.num_gpus
        unit = num_gen // _gcd(per_step, num_gen)
        floor = max(1, args.grad_accum)
        grad_accum = ((floor + unit - 1) // unit) * unit

    global_batch = args.per_device_bs * args.num_gpus * grad_accum

    if global_batch % num_gen != 0:
        hint = ""
        if args.num_gpus % num_gen != 0:
            need = num_gen // _gcd(args.num_gpus, num_gen)
            hint = (f" (per_device_bs * grad_accum must be a multiple of {need} "
                    f"to cover num_generations={num_gen} on {args.num_gpus} GPUs; "
                    f"or pass --auto-grad-accum)")
        raise SystemExit(
            f"Invalid combo: global_batch = {args.per_device_bs} * {args.num_gpus} "
            f"* {grad_accum} = {global_batch} is NOT divisible by "
            f"num_generations={num_gen}. TRL will reject this.{hint}"
        )

    unique_prompts = global_batch // num_gen

    # Rewrite the throughput knobs.
    cfg["per_device_train_batch_size"] = args.per_device_bs
    cfg["gradient_accumulation_steps"] = grad_accum
    cfg["num_generations"] = num_gen

    # Optional lr rescaling against the baseline's *unique prompt groups*.
    base_lr = float(cfg["learning_rate"])
    base_groups = args.base_global_batch / num_gen
    ratio = unique_prompts / base_groups
    if args.lr_scaling == "sqrt":
        new_lr = base_lr * math.sqrt(ratio)
    elif args.lr_scaling == "linear":
        new_lr = base_lr * ratio
    else:
        new_lr = base_lr
    new_lr = float(f"{new_lr:.3e}")  # keep the YAML tidy
    cfg["learning_rate"] = new_lr

    print("Resulting configuration")
    print("-----------------------")
    print(f"  num_generations:                {num_gen}"
          f"{' (overridden)' if args.num_generations is not None else ''}")
    print(f"  per_device_train_batch_size:    {args.per_device_bs}")
    print(f"  gradient_accumulation_steps:    {grad_accum}"
          f"{' (auto)' if args.auto_grad_accum else ''}")
    print(f"  num_gpus (launch-time):         {args.num_gpus}")
    print(f"  -> global / effective batch:    {global_batch}")
    print(f"  -> unique prompt groups / step: {unique_prompts}")
    print(f"  (old single-GPU run was: global {args.base_global_batch} "
          f"= {args.base_global_batch // num_gen if args.base_global_batch % num_gen == 0 else round(base_groups, 2)} prompt group(s)/step)")
    if args.lr_scaling == "none":
        print(f"  learning_rate:                  {base_lr:.2e} (unchanged; "
              f"consider --lr-scaling sqrt -> {base_lr * math.sqrt(ratio):.2e})")
    else:
        print(f"  learning_rate:                  {base_lr:.2e} -> {new_lr:.2e} "
              f"({args.lr_scaling} x{ratio:.1f} in groups)")

    if args.dry_run:
        print("\n--dry-run: not writing. YAML would be:\n")
        print(yaml.safe_dump(cfg, sort_keys=False))
        return

    out = args.out
    if out is None:
        stem, ext = os.path.splitext(args.src)
        out = f"{stem}_runpod{ext}"

    with open(out, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
