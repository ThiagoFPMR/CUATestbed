"""
Backfill ``retention_state.json`` into checkpoints of a run that started before
CheckpointRetentionCallback persisted its state.

Why this works: the retention cascade is a pure function of the save count — it
reads nothing but the save index and the step numbers. So we can replay it from
zero (mirroring CheckpointRetentionCallback.on_save) up to any save and recover
the exact (counter, buckets, latest) that the live run held. The only piece tied
to runtime metrics — the best-reward checkpoint — is recoverable too, because HF
stores every logged metric in each checkpoint's trainer_state.json log_history.

We write the state as-of *each* surviving checkpoint, so resuming from any of
them restores correctly (not just the latest).

Assumption: ``save_steps`` was constant for the whole run, so save i happened at
step ``i * save_steps``. If it changed mid-run, the replayed bucket step numbers
won't line up with what's on disk — the script flags that per checkpoint.

Usage:
    python backfill_retention_state.py <output_dir> --save_steps 60 [--n_ckpts 10] \
        [--metric_key reward] [--dry_run]
"""
import argparse
import json
import os
import re

CHECKPOINT_RE = re.compile(r"^checkpoint-(\d+)$")
STATE_FILENAME = "retention_state.json"


def replay(n_ckpts: int, save_steps: int, n_saves: int):
    """
    Replay the retention cascade for ``n_saves`` saves and return
    ``(counter, buckets, latest)`` as it stood right after the last one.

    Mirrors CheckpointRetentionCallback.on_save exactly — keep them in sync.
    """
    counter = 0
    buckets: list[int | None] = [None] * n_ckpts
    latest: int | None = None
    for i in range(1, n_saves + 1):
        step = i * save_steps
        counter += 1
        for k in range(n_ckpts - 1, -1, -1):
            if counter % (2 ** (k + 1)) == 0:
                buckets[k] = latest if k == 0 else buckets[k - 1]
        if counter == 2 ** n_ckpts:
            counter = 0
        latest = step
    return counter, buckets, latest


def best_from_trainer_state(ckpt_dir: str, metric_key: str, allowed_steps: set):
    """
    Recover (best_step, best_value) from a checkpoint's trainer_state.json,
    considering only ``allowed_steps`` — the checkpoints that still exist on
    disk. "Best" must be a checkpoint we can actually keep: not the best of
    every logged step (a phantom like step 19, never saved), and not the best
    saved step if it was already pruned (the prune loop can't retain a dir
    that's gone).
    """
    path = os.path.join(ckpt_dir, "trainer_state.json")
    if not os.path.exists(path):
        return None, None
    with open(path) as f:
        ts = json.load(f)
    best_step, best_value = None, float("-inf")
    # log_history is ascending by step; >= makes best the latest checkpoint tied
    # at the max (reward is bounded at 1), matching CheckpointRetentionCallback.
    for entry in ts.get("log_history", []):
        step = entry.get("step")
        if metric_key in entry and step in allowed_steps:
            value = entry[metric_key]
            if value >= best_value:
                best_value, best_step = value, int(step)
    if best_step is None:
        return None, None
    return best_step, best_value


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("output_dir", help="Experiment dir containing checkpoint-N/ subdirs.")
    parser.add_argument("--save_steps", type=int, required=True, help="save_steps used during the run (must have been constant).")
    parser.add_argument("--n_ckpts", type=int, default=10, help="n_ckpts the run used (default 10, matching train.py).")
    parser.add_argument("--metric_key", default="reward", help="Metric used to pick the best checkpoint (default: reward).")
    parser.add_argument("--dry_run", action="store_true", help="Print what would be written without writing.")
    args = parser.parse_args()

    checkpoints = {}
    for name in os.listdir(args.output_dir):
        m = CHECKPOINT_RE.match(name)
        if m:
            checkpoints[int(m.group(1))] = os.path.join(args.output_dir, name)
    if not checkpoints:
        raise SystemExit(f"No checkpoint-N dirs found under {args.output_dir}")

    on_disk = set(checkpoints)
    for step in sorted(checkpoints):
        ckpt_dir = checkpoints[step]
        if step % args.save_steps != 0:
            print(f"⏭️  checkpoint-{step}: step not a multiple of save_steps={args.save_steps} "
                  f"(save_steps changed mid-run?) — skipping, cannot replay reliably.")
            continue

        n_saves = step // args.save_steps
        counter, buckets, latest = replay(args.n_ckpts, args.save_steps, n_saves)
        # Best is the best *surviving* checkpoint at or before this one.
        allowed_steps = {s for s in on_disk if s <= step}
        best_step, best_value = best_from_trainer_state(ckpt_dir, args.metric_key, allowed_steps)

        keep = {latest} | {b for b in buckets if b is not None}
        if best_step is not None:
            keep.add(best_step)
        # For the newest checkpoint, everything it wants to keep should still be
        # on disk; older checkpoints legitimately reference siblings the live run
        # already pruned, so only flag the latest.
        if step == max(checkpoints):
            missing = sorted(keep - on_disk)
            if missing:
                print(f"⚠️  checkpoint-{step}: reconstructed keep-set references missing "
                      f"checkpoints {missing} (manual deletion or save_steps change?).")

        state = {
            "n_ckpts": args.n_ckpts,
            "counter": counter,
            "buckets": buckets,
            "latest": latest,
            "best_step": best_step,
            "best_value": best_value if best_step is not None else None,
        }
        dest = os.path.join(ckpt_dir, STATE_FILENAME)
        print(f"{'(dry-run) ' if args.dry_run else ''}checkpoint-{step}: counter={counter} "
              f"best_step={best_step} keep={sorted(keep)}")
        if not args.dry_run:
            with open(dest, "w") as f:
                json.dump(state, f)

    if not args.dry_run:
        print(f"✅ Backfilled {STATE_FILENAME} into {len(checkpoints)} checkpoint(s).")


if __name__ == "__main__":
    main()
