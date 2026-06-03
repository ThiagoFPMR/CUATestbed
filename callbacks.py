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


class CheckpointRetentionCallback(TrainerCallback):
    """
    Retain checkpoints at exponentially-spaced offsets from the latest via a
    cascading bucket update. Buckets[k] holds the checkpoint at offset
    ~2^(k+1) saves back. On each save, buckets are updated in reverse order so
    each one captures its predecessor's pre-update value; bucket 0 pulls from
    the latest. Counter resets after 2^n_ckpts saves.

    Total retained: up to n_ckpts + 2 (buckets + latest + best).
    """

    CHECKPOINT_RE = re.compile(r"^checkpoint-(\d+)$")
    STATE_FILE = "retention_state.json"

    def __init__(self, n_ckpts: int = 10, metric_key: str = "reward"):
        self.n_ckpts = n_ckpts
        self.counter = 0
        self.buckets: list[int | None] = [None] * n_ckpts
        self.latest: int | None = None
        self.metric_key = metric_key
        self._best_step: int | None = None
        self._best_value = float("-inf")

    def on_train_begin(self, args, state, control, **kwargs):
        # On resume, the bucket cascade / best-tracking state lives in the
        # resumed checkpoint dir (written by on_save below, and pulled back from
        # S3 with the rest of the checkpoint). Restore it so retention continues
        # exactly as if the run had never stopped instead of restarting from zero.
        if not state.global_step:
            return
        path = os.path.join(args.output_dir, f"checkpoint-{state.global_step}", self.STATE_FILE)
        if not os.path.exists(path):
            return
        with open(path) as f:
            data = json.load(f)
        self.counter = data["counter"]
        self.buckets = data["buckets"]
        self.latest = data["latest"]
        self._best_step = data["best_step"]
        self._best_value = data["best_value"]

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs or self.metric_key not in logs:
            return
        value = logs[self.metric_key]
        if value > self._best_value:
            self._best_value = value
            self._best_step = state.global_step

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

        # Persist retention state inside the just-saved checkpoint so a later
        # resume (local or from S3) can pick the cascade back up. Runs before
        # S3UploadCallback.on_save, so the file is included in the upload.
        ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        if os.path.isdir(ckpt_dir):
            with open(os.path.join(ckpt_dir, self.STATE_FILE), "w") as f:
                json.dump(
                    {
                        "counter": self.counter,
                        "buckets": self.buckets,
                        "latest": self.latest,
                        "best_step": self._best_step,
                        "best_value": self._best_value,
                    },
                    f,
                )


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
    Mirror each checkpoint to S3 as it is written, via a background thread pool
    so training never blocks on network I/O. Intended for cloud boxes with
    ephemeral local storage. After training, call `finish()` to push the final
    top-level artifacts (adapter, tokenizer, config copies) and block until
    every upload has completed.

    Uploads are additive — nothing is deleted from S3 — so the local
    CheckpointRetentionCallback's pruning does not propagate to the bucket.
    """

    def __init__(self, s3_uri: str, experiment_name: str, max_workers: int = 4):
        import boto3  # lazy: only required when --s3_uri is passed

        parsed = urlparse(s3_uri)
        if parsed.scheme != "s3" or not parsed.netloc:
            raise ValueError(f"--s3_uri must look like s3://bucket/prefix, got: {s3_uri}")
        self.bucket = parsed.netloc
        base = parsed.path.strip("/")
        self.prefix = f"{base}/{experiment_name}".strip("/")
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
        ckpt = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        if os.path.isdir(ckpt):
            self._submit_tree(ckpt, f"{self.prefix}/checkpoint-{state.global_step}")

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
