"""
Merge a trained LoRA adapter into its base model to produce a standalone,
vLLM-servable checkpoint.

Point this at a run (experiment) directory and, optionally, a specific
checkpoint inside it. The base model is read from the adapter's
`adapter_config.json` (`base_model_name_or_path`) — no need to remember which
config the run used. Because the base trained as QLoRA (bnb 4-bit), the default
`merged_16bit` save method dequantizes the base to 16-bit and folds the adapter
in, yielding a plain HF Qwen3-VL checkpoint that vLLM can serve directly.

Examples:
    # Merge the final adapter saved at the run root.
    python merge.py --run my_run

    # Merge a specific checkpoint (any of: checkpoint-500, 500, "latest").
    python merge.py --run my_run --checkpoint 500
    python merge.py --run my_run --checkpoint latest

    # Choose an explicit output dir and/or 4-bit re-quantized merge.
    python merge.py --run my_run --output my_run-merged --save_method merged_4bit
"""
import unsloth
from unsloth import FastVisionModel

import argparse
import json
import os
import re


def _resolve_adapter_dir(run: str, checkpoint: str | None) -> str:
    """
    Resolve the directory that holds the adapter to merge.

    - checkpoint is None        -> the run root (the final adapter from train.py),
                                   falling back to the latest checkpoint if the
                                   root has no adapter.
    - checkpoint == "latest"    -> the highest-step checkpoint-N dir.
    - checkpoint is "checkpoint-N" / N -> that specific checkpoint dir.
    """
    if not os.path.isdir(run):
        raise FileNotFoundError(f"Run directory not found: {run}")

    def _has_adapter(d: str) -> bool:
        return os.path.exists(os.path.join(d, "adapter_config.json"))

    def _latest_checkpoint() -> str:
        steps: dict[int, str] = {}
        for name in os.listdir(run):
            m = re.match(r"^checkpoint-(\d+)$", name)
            if m and os.path.isdir(os.path.join(run, name)):
                steps[int(m.group(1))] = name
        if not steps:
            raise FileNotFoundError(f"No checkpoint-N directories found under {run}")
        return os.path.join(run, steps[max(steps)])

    if checkpoint is None:
        if _has_adapter(run):
            return run
        return _latest_checkpoint()

    if checkpoint == "latest":
        return _latest_checkpoint()

    name = checkpoint if checkpoint.startswith("checkpoint-") else f"checkpoint-{checkpoint}"
    path = os.path.join(run, name)
    if not _has_adapter(path):
        raise FileNotFoundError(f"No adapter found at {path}")
    return path


def _base_model_of(adapter_dir: str) -> str:
    """Read the base model the adapter was trained on from adapter_config.json."""
    cfg_path = os.path.join(adapter_dir, "adapter_config.json")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"adapter_config.json not found in {adapter_dir}")
    with open(cfg_path) as f:
        base = json.load(f).get("base_model_name_or_path")
    if not base:
        raise ValueError(f"'base_model_name_or_path' missing from {cfg_path}")
    return base


def _max_seq_length_of(run: str, default: int = 16384) -> int:
    """Best-effort read of max_seq_length from the run's copied model config."""
    cfg_dir = os.path.join(run, "configs")
    if not os.path.isdir(cfg_dir):
        return default
    import yaml
    for name in os.listdir(cfg_dir):
        if not name.endswith((".yml", ".yaml")):
            continue
        with open(os.path.join(cfg_dir, name)) as f:
            data = yaml.safe_load(f) or {}
        msl = (data.get("quantization") or {}).get("max_seq_length")
        if msl:
            return int(msl)
    return default


def merge(
        run: str,
        checkpoint: str | None,
        output: str | None,
        save_method: str,
) -> None:
    adapter_dir = _resolve_adapter_dir(run, checkpoint)
    base_model = _base_model_of(adapter_dir)
    max_seq_length = _max_seq_length_of(run)

    if output is None:
        tag = os.path.basename(adapter_dir.rstrip("/"))
        suffix = "merged" if tag == os.path.basename(run.rstrip("/")) else f"merged-{tag}"
        output = f"{run.rstrip('/')}-{suffix}"

    print(f"📂 Adapter:      {adapter_dir}")
    print(f"🧩 Base model:   {base_model}")
    print(f"📐 max_seq_len:  {max_seq_length}")
    print(f"💾 Save method:  {save_method}")
    print(f"📤 Output:       {output}")

    # Loading the adapter dir makes Unsloth pull the base (from adapter_config),
    # apply the LoRA, and leave the model ready to merge. 4-bit load matches the
    # bnb base it was trained on; merged_16bit dequantizes on save.
    model, tokenizer = FastVisionModel.from_pretrained(
        model_name=adapter_dir,
        max_seq_length=max_seq_length,
        load_in_4bit=True,
    )

    model.save_pretrained_merged(output, tokenizer, save_method=save_method)
    print(f"✅ Merged model written to {output}")
    print(f"   Serve with: vllm serve {output} --trust-remote-code "
          f"--max-model-len {max_seq_length} --limit-mm-per-prompt image=13")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Merge a trained LoRA adapter into its base model for vLLM serving."
    )
    parser.add_argument("--run", required=True, help="Run/experiment directory containing the adapter and/or checkpoints.")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint to merge: 'checkpoint-N', N, or 'latest'. Omit to use the run's final adapter.")
    parser.add_argument("--output", default=None, help="Output directory for the merged model. Defaults to '<run>-merged[-checkpoint-N]'.")
    parser.add_argument("--save_method", default="merged_16bit", choices=["merged_16bit", "merged_4bit"], help="merged_16bit dequantizes to fp16 (recommended for vLLM); merged_4bit re-quantizes.")
    args = parser.parse_args()

    merge(args.run, args.checkpoint, args.output, args.save_method)
