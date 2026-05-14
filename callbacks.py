"""
TrainerCallbacks for the VLM RL training loop.
"""
import os
import re
import shutil
import time
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
    EMPTY_CACHE_EVERY_N_STEPS = 10

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

    def __init__(self, n_ckpts: int = 10, metric_key: str = "reward"):
        self.n_ckpts = n_ckpts
        self.counter = 0
        self.buckets: list[int | None] = [None] * n_ckpts
        self.latest: int | None = None
        self.metric_key = metric_key
        self._best_step: int | None = None
        self._best_value = float("-inf")

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
