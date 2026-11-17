"""
General training script for vision-language models using reinforcement learning.
"""
import unsloth
import os
import torch
import argparse
from safetensors import safe_open

from utils import save_training_params
from model import load_model_from_file
from policy.rewards import parse_rewards
from policy.rewards.logging_rewards import generation_logging_reward_func
from policy.trainer import load_trainer_from_file
from callbacks import (
    StepTimerCallback,
    GenerationLoggingCallback,
    CheckpointRetentionCallback,
    S3UploadCallback,
    download_checkpoint_from_s3,
)
from transformers.trainer_utils import get_last_checkpoint

from dataset.agentnet.agentnet import load_agentnet
from prompts import base_prompt

os.environ.setdefault("WANDB_PROJECT", "cuatestbed")

import peft.utils.save_and_load as _peft_save_and_load

_peft_save_and_load._maybe_shard_state_dict_for_tp = lambda *args, **kwargs: None

# --- Multi-GPU (DDP) compatibility shim for Unsloth's compiled GRPO trainer ---
# Under `accelerate launch` with >1 GPU the model is wrapped in
# DistributedDataParallel. Unsloth's compiled UnslothGRPOTrainer.compute_loss
# reads `model.config` (for the logit-softcapping check) directly on that wrapper,
# but DDP doesn't forward attribute access to the module it wraps, raising:
#   AttributeError: 'DistributedDataParallel' object has no attribute 'config'
# at the first loss step (generation runs fine; the crash is in compute_loss).
# Delegate unknown attribute lookups on a DDP instance to the wrapped .module so
# `ddp.config` transparently resolves to `ddp.module.config`. No-op on 1 GPU
# (there's no DDP wrapper). `.module` itself resolves via nn.Module.__getattr__'s
# _modules lookup, so this does not recurse.
from torch.nn.parallel import DistributedDataParallel as _DDP

_ddp_base_getattr = _DDP.__getattr__


def _ddp_attr_proxy(self, name):
    try:
        return _ddp_base_getattr(self, name)
    except AttributeError:
        return getattr(self.module, name)


_DDP.__getattr__ = _ddp_attr_proxy


def _resolve_resume_checkpoint(
        resume: bool | str,
        experiment_name: str,
        s3_uri: str | None,
) -> bool | str:
    """
    Turn `resume_from_checkpoint` into a concrete local checkpoint path, pulling
    it from S3 when the requested checkpoint isn't on the (ephemeral) local disk.

    - resume is a local dir that exists        -> use it as-is.
    - resume is a "checkpoint-N" / step value  -> use local copy if present, else
                                                  fetch that checkpoint from S3.
    - resume is True (auto)                     -> prefer the latest local
                                                  checkpoint, else the latest one
                                                  on S3.
    """
    if isinstance(resume, str):
        if os.path.isdir(resume):
            return resume
        name = resume if resume.startswith("checkpoint-") else f"checkpoint-{resume}"
        local = os.path.join(experiment_name, name)
        if os.path.isdir(local):
            return local
        if s3_uri:
            got = download_checkpoint_from_s3(s3_uri, experiment_name, name)
            if got:
                return got
        raise FileNotFoundError(
            f"Checkpoint {resume!r} not found locally ({local}) or on S3 "
            f"({s3_uri or 'no --s3_uri set'})."
        )

    # resume is True -> auto-detect.
    local = get_last_checkpoint(experiment_name) if os.path.isdir(experiment_name) else None
    if local:
        return local
    if s3_uri:
        got = download_checkpoint_from_s3(s3_uri, experiment_name)
        if got:
            return got
    return resume  # nothing to pull; let the trainer surface "no checkpoint" itself.


def _terminate_pod(reason: str) -> None:
    """Terminate this RunPod pod to stop compute billing. Opt-in via env.

    Guard rail for ephemeral cloud boxes: a crashed/finished container does NOT
    stop billing on its own -- only terminating the *pod* does. Gated on
    RUNPOD_API_KEY (pass as a pod secret); RUNPOD_POD_ID is injected by RunPod.
    Checkpoints live on S3 (S3UploadCallback), so killing the pod is safe -- a
    fresh pod resumes from the latest S3 checkpoint.

    Uses the REST API (DELETE /v1/pods/{id}, Bearer auth). The legacy GraphQL
    endpoint with `?api_key=` is now rejected with 403, which is what bit the
    prep scripts. Best-effort: on any failure we log loudly and leave the pod up
    rather than risk a wrongly-killed box (a stuck pod is recoverable by hand).
    """
    api_key = os.environ.get("RUNPOD_API_KEY")
    pod_id = os.environ.get("RUNPOD_POD_ID")
    if not api_key:
        print("[shutdown] RUNPOD_API_KEY not set; leaving pod up. Stop it manually "
              "to halt billing.", flush=True)
        return
    if not pod_id:
        print("[shutdown] RUNPOD_POD_ID not set (not on a RunPod pod?); cannot "
              "self-terminate. Stop the pod manually!", flush=True)
        return

    import urllib.request
    import urllib.error

    req = urllib.request.Request(
        f"https://rest.runpod.io/v1/pods/{pod_id}",
        headers={"Authorization": f"Bearer {api_key}"},
        method="DELETE",
    )
    print(f"[shutdown] {reason}: terminating pod {pod_id} to stop billing...", flush=True)
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            print(f"[shutdown] runpod api -> {r.status} {r.read().decode()[:200]}", flush=True)
    except urllib.error.HTTPError as e:
        body = e.read().decode()[:200] if hasattr(e, "read") else ""
        print(f"[shutdown] terminate request failed: HTTP {e.code} {body!r}. "
              "Stop the pod manually!", flush=True)
    except Exception as e:
        print(f"[shutdown] terminate request failed: {e!r}. Stop the pod manually!",
              flush=True)


def train_vlm_rl(
        base_prompt: str,
        experiment_name: str,
        model_config: str,
        trainer_config: str,
        rewards: list[str],
        resume_from_checkpoint: bool | str = False,
        s3_uri: str | None = None,
        s3_upload_every: int = 10,
        keep_all_checkpoints: bool = False,
) -> None:
    """
    Train a vision-language model using reinforcement learning.

    Args:
        experiment_name (str): Name of the training experiment.
        model_config (str): Path to the model configuration file. If does not contain /, it is assumed to be in the model configs directory.
        trainer_config (str): Path to the trainer configuration file. If does not contain /, it is assumed to be in the trainer configs directory.
        output_dir (str): Directory to save the trained model.
        s3_uri (str | None): If set (e.g. s3://bucket/prefix), mirror checkpoints and final artifacts to S3.
        s3_upload_every (int): Push to S3 only every Nth checkpoint save (default 1). Local saves are unaffected; use >1 to lower the S3 backup cadence when checkpoints already persist on a network volume.
        keep_all_checkpoints (bool): If True, retain every checkpoint saved at the configured save_steps cadence instead of pruning down to the exponentially-spaced retention set.
    """
    torch.cuda.empty_cache()
    model, tokenizer = load_model_from_file(model_config)

    train_dataset = load_agentnet(base_prompt, history_length=12)
    
    reward_funcs = parse_rewards(rewards)
    reward_funcs.append(generation_logging_reward_func)

    trainer = load_trainer_from_file(trainer_config, run_name=experiment_name, output_dir=experiment_name)(
        model = model,
        reward_funcs = reward_funcs,
        processing_class = tokenizer,
        train_dataset = train_dataset,
    )
    # Insert at the front so our additions to the log dict run BEFORE PrinterCallback
    # and WandbCallback consume it.
    trainer.callback_handler.callbacks.insert(0, StepTimerCallback())
    trainer.callback_handler.callbacks.append(
        GenerationLoggingCallback(pred_action_every=10, full_output_every=60)
    )
    trainer.callback_handler.callbacks.append(
        CheckpointRetentionCallback(n_ckpts=10, keep_all=keep_all_checkpoints)
    )

    s3_callback = None
    if s3_uri:
        s3_callback = S3UploadCallback(s3_uri, experiment_name, upload_every=s3_upload_every)
        trainer.callback_handler.callbacks.append(s3_callback)

    # Save config copies before training so a crashed/interrupted run still
    # leaves them behind (locally and on S3 if configured).
    save_training_params(
        base_prompt,
        experiment_name,
        model_config,
        trainer_config,
        rewards
    )
    print("💾 Saved copies of config files.")
    if s3_callback is not None:
        s3_callback.upload_now(f"{experiment_name}/configs", key_subprefix="configs")
        print("☁️  Configs uploaded to S3.")

    if resume_from_checkpoint:
        resume_from_checkpoint = _resolve_resume_checkpoint(
            resume_from_checkpoint, experiment_name, s3_uri
        )
        print(f"↩️  Resuming from checkpoint: {resume_from_checkpoint}")

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    print("🏁 Training completed successfully!")

    # Local saving
    model.save_pretrained(experiment_name)
    tokenizer.save_pretrained(experiment_name)
    print(f"📁 Model and tokenizer saved to {experiment_name}!")

    # Verify that the LoRA model was saved correctly    
    with safe_open(f"{experiment_name}/adapter_model.safetensors", framework = "pt") as f:
        tensors = {}
        for key in f.keys():
            tensor = f.get_tensor(key)
            n_zeros = (tensor == 0).sum() / tensor.numel()
            assert(n_zeros.item() != tensor.numel())
    print("✅ LoRA model saved and verified successfully!")

    # Push final artifacts and block until all background uploads finish.
    if s3_callback is not None:
        s3_callback.finish(experiment_name)
        print(f"☁️  Experiment data uploaded to {s3_uri}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a vision-language model using reinforcement learning. Output directory is specified in the trainer config.")
    parser.add_argument("--experiment_name", type=str, required=False, default="vlm_rl_experiment", help="Name of the training experiment.")
    parser.add_argument("--model_config", type=str, required=False, default="qwen3-vl-2b-4bit.yml", help="Path to the model configuration file.")
    parser.add_argument("--trainer_config", type=str, required=False, default="gspo.yml", help="Path to the trainer configuration file.")
    parser.add_argument("--rewards", type=str, nargs="+", required=False, default=["gui_r1"], help="List of reward function names.")
    parser.add_argument("--resume_from_checkpoint", nargs="?", const=True, default=False, help="Resume training. Pass with no value to auto-detect the latest checkpoint (preferring a local one, else the latest on S3 if --s3_uri is set). Pass a local path, or a 'checkpoint-N' name / step to fetch that specific checkpoint from S3 when it isn't on local disk.")
    parser.add_argument("--s3_uri", type=str, required=False, default=None, help="If set (e.g. s3://bucket/prefix), mirror checkpoints and the final artifacts to S3. For cloud boxes with ephemeral local storage.")
    parser.add_argument("--s3_upload_every", type=int, required=False, default=1, help="Mirror to S3 only every Nth checkpoint save (default 1 = every save). Local checkpoints are always written; raise this to lower the S3 backup cadence when checkpoints persist on a network volume. Each S3 sync also includes the generation_logs.")
    parser.add_argument("--keep_all_checkpoints", action="store_true", help="Keep every checkpoint saved at the configured save_steps cadence, disabling the exponentially-spaced retention pruning (CheckpointRetentionCallback).")
    args = parser.parse_args()

    try:
        train_vlm_rl(base_prompt, args.experiment_name, args.model_config, args.trainer_config, args.rewards, args.resume_from_checkpoint, args.s3_uri, args.s3_upload_every, args.keep_all_checkpoints)
    except BaseException as e:
        # Guard rail: on any crash (including KeyboardInterrupt/OOM kill) stop the
        # billing on the cloud box. Latest checkpoint is already on S3, so a fresh
        # pod can resume. Re-raise so the traceback + non-zero exit still surface.
        print(f"[error] training aborted: {e!r}", flush=True)
        _terminate_pod(f"training crashed ({type(e).__name__})")
        raise