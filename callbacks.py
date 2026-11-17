"""
TrainerCallbacks for the VLM RL training loop.
"""
import json
import os
import re
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse

import torch
from transformers import TrainerCallback

from dataset.agentnet import agentnet as agentnet_module
from policy.rewards import logging_rewards


class StepTimerCallback(TrainerCallback):
    """
    Callback to track and log the time taken for each training step, as well as GPU memory usage.
     - Logs "step_time_s" for the duration of each step.
     - Logs "gpu_mem_free_mib", "gpu_mem_alloc_mib", "gpu_mem_reserved_mib", and "gpu_mem_frag_mib" for GPU memory stats.
     - Logs "example_n_images_mean" and "example_image_pixels_mean" for dataset image stats if available.
     - Optionally empties the GPU cache every N steps to manage memory fragmentation.
     (Note: frequent cache clearing can impact performance, so use with caution and adjust the frequency
    """
    EMPTY_CACHE_EVERY_N_STEPS = 2

    def __init__(self):
        self._t0 = None
        self._last = None

    def on_step_begin(self, args, state, control, **kwargs):
        self._t0 = time.perf_counter()

    def on_step_end(self, args, state, control, **kwargs):
        if self._t0 is not None:
            self._last = time.perf_counter() - self._t0
        if (
            torch.cuda.is_available()
            and self.EMPTY_CACHE_EVERY_N_STEPS > 0
            and state.global_step > 0
            and state.global_step % self.EMPTY_CACHE_EVERY_N_STEPS == 0
        ):
            torch.cuda.empty_cache()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        if self._last is not None:
            logs["step_time_s"] = round(self._last, 3)
        if torch.cuda.is_available():
            mib = 1024 * 1024
            free, _ = torch.cuda.mem_get_info()
            logs["gpu_mem_free_mib"] = round(free / mib, 1)
            logs["gpu_mem_alloc_mib"] = round(torch.cuda.memory_allocated() / mib, 1)
            logs["gpu_mem_reserved_mib"] = round(torch.cuda.memory_reserved() / mib, 1)
            # gap between reserved and allocated == cached-but-unused == fragmentation proxy
            logs["gpu_mem_frag_mib"] = round(
                (torch.cuda.memory_reserved() - torch.cuda.memory_allocated()) / mib, 1
            )
        counts = agentnet_module.LAST_EXAMPLE_IMAGE_COUNTS
        if counts:
            logs["example_n_images_mean"] = round(sum(counts) / len(counts), 3)
            agentnet_module.LAST_EXAMPLE_IMAGE_COUNTS = []
        hws = agentnet_module.LAST_EXAMPLE_IMAGE_HW
        if hws:
            pixels = [h * w for h, w in hws]
            logs["example_image_pixels_mean"] = round(sum(pixels) / len(pixels), 0)
            agentnet_module.LAST_EXAMPLE_IMAGE_HW = []
        history_lens = agentnet_module.LAST_EXAMPLE_HISTORY_LENGTHS
        if history_lens:
            logs["example_history_len_mean"] = round(sum(history_lens) / len(history_lens), 3)
            agentnet_module.LAST_EXAMPLE_HISTORY_LENGTHS = []
        # Total wall time spent loading/decoding images this step. If this tracks
        # step_time_s on a spike, the latency is data-loading (I/O), not compute.
        data_load_times = agentnet_module.LAST_DATA_LOAD_TIMES
        if data_load_times:
            logs["data_load_s"] = round(sum(data_load_times), 3)
            agentnet_module.LAST_DATA_LOAD_TIMES = []


class GenerationLoggingCallback(TrainerCallback):
    """
    Persist a sample of the model's generations during training, picking only
    the best route per step (highest combined GUI-R1 reward of the GRPO group).

    Two cadences, written under ``{output_dir}/generation_logs/``:
      - ``predicted_actions.log``: best route's predicted action vs ground truth,
        every ``pred_action_every`` steps. Cheap, dense — for eyeballing
        grounding/accuracy drift.
      - ``full_outputs.log``: the best route's entire raw completion (observation
        + reason + answer) plus the ground-truth label, every
        ``full_output_every`` steps. Verbose — for inspecting full behavior.

    Generations are captured by ``generation_logging_reward_func`` (a neutral
    reward func) into ``logging_rewards.PENDING_BEST``; this callback drains it.
    """

    def __init__(self, pred_action_every: int = 5, full_output_every: int = 60):
        self.pred_action_every = pred_action_every
        self.full_output_every = full_output_every
        self._log_dir = None

    def _ensure_dir(self, args):
        if self._log_dir is None:
            self._log_dir = os.path.join(args.output_dir, "generation_logs")
            os.makedirs(self._log_dir, exist_ok=True)
        return self._log_dir

    def on_step_end(self, args, state, control, **kwargs):
        record = logging_rewards.PENDING_BEST
        logging_rewards.PENDING_BEST = None  # drain regardless, so it can't go stale
        if record is None:
            return

        step = state.global_step
        want_pred = self.pred_action_every > 0 and step % self.pred_action_every == 0
        want_full = self.full_output_every > 0 and step % self.full_output_every == 0
        if not (want_pred or want_full):
            return

        full_entry, pred_entry = logging_rewards.format_best_record(record, step)
        log_dir = self._ensure_dir(args)
        if want_pred:
            with open(os.path.join(log_dir, "predicted_actions.log"), "a") as f:
                f.write(pred_entry)
        if want_full:
            with open(os.path.join(log_dir, "full_outputs.log"), "a") as f:
                f.write(full_entry)


class CheckpointRetentionCallback(TrainerCallback):
    """
    Retain checkpoints at exponentially-spaced offsets from the latest via a
    cascading bucket update. Buckets[k] holds the checkpoint at offset
    ~2^(k+1) saves back. On each save, buckets are updated in reverse order so
    each one captures its predecessor's pre-update value; bucket 0 pulls from
    the latest. Counter resets after 2^n_ckpts saves.

    Total retained: up to n_ckpts + 2 (buckets + latest + best).

    State (counter/buckets/best) is the running memory of the cascade. The HF
    Trainer does not persist callback attributes across a resume, so we write
    them into each checkpoint dir as ``retention_state.json`` on save and reload
    them in ``on_train_begin``. Runs that predate this persistence can be
    backfilled offline by replaying the cascade — see
    ``backfill_retention_state.py``.
    """

    CHECKPOINT_RE = re.compile(r"^checkpoint-(\d+)$")
    STATE_FILENAME = "retention_state.json"

    def __init__(self, n_ckpts: int = 10, metric_key: str = "reward",
                 keep_all: bool = False):
        self.n_ckpts = n_ckpts
        self.counter = 0
        self.buckets: list[int | None] = [None] * n_ckpts
        self.latest: int | None = None
        self.metric_key = metric_key
        self.keep_all = keep_all
        self._best_step: int | None = None
        self._best_value = float("-inf")
        self._last_metric: float | None = None

    def on_train_begin(self, args, state, control, **kwargs):
        # Fresh run (global_step==0) → nothing to restore. On resume, reload the
        # state stashed inside the checkpoint we're resuming from.
        if not state.global_step:
            return
        path = os.path.join(
            args.output_dir, f"checkpoint-{state.global_step}", self.STATE_FILENAME
        )
        if not os.path.exists(path):
            print(
                f"⚠️  CheckpointRetentionCallback: no {self.STATE_FILENAME} in "
                f"checkpoint-{state.global_step}; retention history starts blank, so "
                f"the next save may prune pre-existing checkpoints. Backfill with "
                f"backfill_retention_state.py to avoid this."
            )
            return
        with open(path) as f:
            data = json.load(f)
        if data.get("n_ckpts") != self.n_ckpts:
            print(
                f"⚠️  CheckpointRetentionCallback: saved n_ckpts={data.get('n_ckpts')} "
                f"!= configured n_ckpts={self.n_ckpts}; ignoring stored state."
            )
            return
        self.counter = data["counter"]
        self.buckets = data["buckets"]
        self.latest = data["latest"]
        self._best_step = data.get("best_step")
        best_value = data.get("best_value")
        self._best_value = best_value if best_value is not None else float("-inf")

    def _save_state(self, args) -> None:
        ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{self.latest}")
        if not os.path.isdir(ckpt_dir):
            return
        state = {
            "n_ckpts": self.n_ckpts,
            "counter": self.counter,
            "buckets": self.buckets,
            "latest": self.latest,
            "best_step": self._best_step,
            "best_value": self._best_value if self._best_step is not None else None,
        }
        with open(os.path.join(ckpt_dir, self.STATE_FILENAME), "w") as f:
            json.dump(state, f)

    def on_log(self, args, state, control, logs=None, **kwargs):
        # Cache the most recent metric value only. "Best" is decided at save time
        # against actual checkpoints — tracking it over every logged step would
        # pin best_step to a non-save step that can never be retained.
        if logs and self.metric_key in logs:
            self._last_metric = logs[self.metric_key]

    def on_save(self, args, state, control, **kwargs):
        self.counter += 1

        # Reverse cascade: largest offset first so each bucket reads its
        # predecessor's pre-update value. Bucket 0's predecessor is the
        # previous latest (still held in self.latest at this point).
        for k in range(self.n_ckpts - 1, -1, -1):
            if self.counter % (2 ** (k + 1)) == 0:
                if k == 0:
                    self.buckets[0] = self.latest
                else:
                    self.buckets[k] = self.buckets[k - 1]

        if self.counter == 2 ** self.n_ckpts:
            self.counter = 0

        self.latest = state.global_step

        # Update "best" only here, so best_step is always a real checkpoint. HF
        # logs (on_log) before it saves within a step, so _last_metric holds this
        # step's metric by now (given logging aligns with this save step).
        # Use >= so that among checkpoints tied at the max (reward is bounded at
        # 1 and plateaus there), best advances to the latest, more-trained one.
        if self._last_metric is not None and self._last_metric >= self._best_value:
            self._best_value = self._last_metric
            self._best_step = self.latest

        if not self.keep_all:
            keep_steps = {self.latest}
            keep_steps.update(s for s in self.buckets if s is not None)
            if self._best_step is not None:
                keep_steps.add(self._best_step)

            for name in os.listdir(args.output_dir):
                m = self.CHECKPOINT_RE.match(name)
                if not m:
                    continue
                step = int(m.group(1))
                if step not in keep_steps:
                    shutil.rmtree(os.path.join(args.output_dir, name), ignore_errors=True)

        # Persist the post-save state into the surviving latest checkpoint so a
        # resume can restore it (HF doesn't serialize callback attributes).
        self._save_state(args)


def _parse_s3_prefix(s3_uri: str, experiment_name: str) -> tuple[str, str]:
    """Resolve (bucket, prefix) for an experiment, matching S3UploadCallback's layout."""
    parsed = urlparse(s3_uri)
    if parsed.scheme != "s3" or not parsed.netloc:
        raise ValueError(f"--s3_uri must look like s3://bucket/prefix, got: {s3_uri}")
    base = parsed.path.strip("/")
    prefix = f"{base}/{experiment_name}".strip("/")
    return parsed.netloc, prefix


def download_checkpoint_from_s3(
    s3_uri: str,
    experiment_name: str,
    checkpoint: str | None = None,
    max_workers: int = 8,
) -> str | None:
    """
    Pull a checkpoint dir from S3 into the local experiment dir so training can
    resume from it (for cloud boxes whose local checkpoints are ephemeral).

    `checkpoint` is a name like "checkpoint-500"; pass None to pick the
    highest-step checkpoint present under the experiment's S3 prefix. Files land
    at `<experiment_name>/<checkpoint>/...`, mirroring the upload layout.

    Returns the local checkpoint path, or None if no matching checkpoint exists
    on S3.
    """
    import boto3  # lazy: only required when --s3_uri is passed

    bucket, prefix = _parse_s3_prefix(s3_uri, experiment_name)
    client = boto3.client("s3")
    paginator = client.get_paginator("list_objects_v2")

    if checkpoint is None:
        steps: dict[int, str] = {}
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix + "/", Delimiter="/"):
            for cp in page.get("CommonPrefixes", []):
                name = cp["Prefix"].rstrip("/").rsplit("/", 1)[-1]
                m = re.match(r"checkpoint-(\d+)$", name)
                if m:
                    steps[int(m.group(1))] = name
        if not steps:
            return None
        checkpoint = steps[max(steps)]

    ckpt_prefix = f"{prefix}/{checkpoint}"
    keys = [
        obj["Key"]
        for page in paginator.paginate(Bucket=bucket, Prefix=ckpt_prefix + "/")
        for obj in page.get("Contents", [])
    ]
    if not keys:
        return None

    local_ckpt = os.path.join(experiment_name, checkpoint)

    def _fetch(key: str) -> None:
        rel = key[len(prefix) + 1:]  # "checkpoint-N/<file>"
        dest = os.path.join(experiment_name, rel)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        client.download_file(bucket, key, dest)

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for fut in [ex.submit(_fetch, k) for k in keys]:
            fut.result()

    return local_ckpt


class S3UploadCallback(TrainerCallback):
    """
    Mirror checkpoints to S3 via a background thread pool so training never
    blocks on network I/O. After training, call `finish()` to push the final
    top-level artifacts (adapter, tokenizer, config copies, generation logs) and
    block until every upload has completed.

    `upload_every` throttles the S3 mirror: only every Nth checkpoint save is
    pushed (default 1 = every save). Local checkpoints are written every save
    regardless — this just lowers the S3 backup cadence, which is all you need
    when the canonical copy lives on a persistent network volume. On each S3 sync
    the `generation_logs/` dir is mirrored too (append-only files re-uploaded
    whole, so S3 holds the latest). `finish()` always pushes a final complete
    copy regardless of the cadence, so the end state on S3 is never partial.

    Uploads are additive — nothing is deleted from S3 — so the local
    CheckpointRetentionCallback's pruning does not propagate to the bucket.
    """

    def __init__(self, s3_uri: str, experiment_name: str, max_workers: int = 4,
                 upload_every: int = 1):
        import boto3  # lazy: only required when --s3_uri is passed

        parsed = urlparse(s3_uri)
        if parsed.scheme != "s3" or not parsed.netloc:
            raise ValueError(f"--s3_uri must look like s3://bucket/prefix, got: {s3_uri}")
        self.bucket = parsed.netloc
        base = parsed.path.strip("/")
        self.prefix = f"{base}/{experiment_name}".strip("/")
        self.upload_every = max(1, upload_every)
        self._save_count = 0
        self._client = boto3.client("s3")
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._futures: list = []

    def _upload_file(self, local_path: str, key: str) -> None:
        try:
            self._client.upload_file(local_path, self.bucket, key)
        except Exception as e:
            print(f"S3 upload failed for {local_path} -> s3://{self.bucket}/{key}: {e}")

    def _submit_tree(self, local_root: str, key_prefix: str, skip_checkpoints: bool = False) -> None:
        for dirpath, dirnames, filenames in os.walk(local_root):
            if skip_checkpoints:
                dirnames[:] = [d for d in dirnames if not d.startswith("checkpoint-")]
            for name in filenames:
                local_path = os.path.join(dirpath, name)
                rel = os.path.relpath(local_path, local_root).replace(os.sep, "/")
                key = f"{key_prefix}/{rel}"
                self._futures.append(self._executor.submit(self._upload_file, local_path, key))

    def on_save(self, args, state, control, **kwargs):
        # Throttle the S3 mirror: the checkpoint is already on the (persistent)
        # local volume every save; only back a subset up to S3.
        self._save_count += 1
        if self._save_count % self.upload_every != 0:
            return

        ckpt = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        if os.path.isdir(ckpt):
            self._submit_tree(ckpt, f"{self.prefix}/checkpoint-{state.global_step}")
        # Mirror the generation logs alongside this checkpoint so the S3 backup
        # carries them too (they otherwise only reach S3 at finish()).
        gen_logs = os.path.join(args.output_dir, "generation_logs")
        if os.path.isdir(gen_logs):
            self._submit_tree(gen_logs, f"{self.prefix}/generation_logs")

    def upload_now(self, local_root: str, key_subprefix: str = "") -> None:
        """Submit a tree and block until those specific uploads complete."""
        key_prefix = self.prefix if not key_subprefix else f"{self.prefix}/{key_subprefix}"
        before = len(self._futures)
        self._submit_tree(local_root, key_prefix)
        for f in self._futures[before:]:
            f.result()

    def finish(self, local_dir: str) -> None:
        """Upload final top-level artifacts and block until every upload is done."""
        self._submit_tree(local_dir, self.prefix, skip_checkpoints=True)
        for f in self._futures:
            f.result()
        self._executor.shutdown(wait=True)
