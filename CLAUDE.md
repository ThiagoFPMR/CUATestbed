# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

OmniBrowse trains vision-language models (VLMs) as autonomous desktop/browser GUI agents using reinforcement learning. The core loop: load a VLM via Unsloth → build a dataset of (screenshot, instruction) → (action) pairs → train with GRPO/GSPO using custom reward functions that score the format and accuracy of model-generated action codes.

## Environment

The project uses the `cua` virtualenv located at `./cua/`. Activate with:
```bash
source cua/bin/activate
```

Key dependencies: `unsloth`, `trl`, `transformers`, `peft`, `datasets`, `safetensors`, `torch`.

## Commands

**Run the forward pass test** (loads model, dataset, scores completions):
```bash
python test_forward.py
```

**Launch training**:
```bash
python train.py --experiment_name my_run --model_config qwen3-vl-2b-4bit.yml --trainer_config grpo.yml --rewards gui_r1
```

**Download the AgentNet dataset** (required before using `load_agentnet`):
```bash
huggingface-cli download xlangai/AgentNet --repo-type dataset --local-dir dataset/agentnet/data
```

## Architecture

### `train.py` — entry point
Wires together model loading, dataset loading, reward parsing, and trainer instantiation. Contains a `base_prompt` that defines the agent's action vocabulary and output format. Currently has `pdb.set_trace()` breakpoints.

### `model/` — model loading
`load_model.py` reads a YAML config from `model/configs/` and calls `FastVisionModel.from_pretrained` + `get_peft_model` (LoRA). Config files: `qwen3-vl-{2b,4b,8b}-4bit.yml`. Short names (no `/`) are resolved relative to `model/configs/`.

### `policy/rewards/` — reward functions
The reward system is the core of the RL training signal.

- `parse_rewards.py`: `parse_rewards(["gui_r1"])` → list of callables. All reward functions follow the signature `(completions, labels, img_sizes) -> list[float]`. The GRPO trainer passes `completions` as `list[list[dict]]` (message lists); reward functions extract the final assistant message's content internally.
- `format_rewards.py`: `gui_r1_formatting_reward_func` — checks `<reason>...</reason> <answer>...</answer>` structure and validates each action code against `ACTION_PATTERNS`. Returns 1.0 or 0.0.
- `action_rewards.py`: `accuracy_reward_func` — compares parsed predicted actions against ground-truth label dicts. Scores action type, point coordinates (within tolerance radius `r`), text messages (F1 ≥ threshold), and other kwargs as exact matches.
- `parse_rewards.py`: `gui_r1_reward_func` — combined reward: `0.8 * format + 0.2 * accuracy`.
- `utils.py`: `actions_from_completion` extracts `[{'description': ..., 'action': ...}]` from the `<answer>` tag via `ast.literal_eval`.

### `dataset/` — dataset loaders

**AgentNet** (`dataset/agentnet/`): The primary training dataset.
- `agentnet.py`: Loads JSONL trajectories from `dataset/agentnet/data/`, filters trajectories (`filter_trajectory`) and steps (`filter_step`), builds multi-turn conversations (`conversation_from_step`) with history of up to N prior screenshots and actions, then returns a HuggingFace `Dataset` with a `set_transform` that loads PIL images lazily.
- `action_utils.py`: Defines `ACTION_PATTERNS` (regex dict for all supported action types), `parse_action_arguments` (parses action call strings into label dicts), `labels_from_code`, `verify_action_code`.

**GUI-R1** (`dataset/gui_r1/`): Simpler single-turn dataset loaded from disk via `load_from_disk`. Requires running `prepare.py` first.

**MathVista** (`dataset/math_vista/`): Separate dataset loader for math reasoning tasks.

### `policy/trainer/` — trainer loading
`load_trainer_from_file` reads a YAML from `policy/trainer/configs/` (e.g., `grpo.yml`, `gspo.yml`), looks up the `config_type` and `trainer_type` on the `trl` module, instantiates `TrainingArguments`, and returns a factory function that accepts model/reward/dataset kwargs.

### Action format
Model outputs must follow:
```
<reason>...</reason> <answer>[{'description': '...', 'action': 'click(x=0.5, y=0.3)'}]</answer>
```
Coordinates are normalized 0–1. The `dataset/agentnet/data/` raw format uses `pyautogui`-prefixed action codes (e.g., `pyautogui.click(x=0.5, y=0.5)`); the dot-prefix is stripped during dataset processing.

## Key design decisions

- **Labels are JSON strings**, not dicts. `dataset` returns `labels` as `json.dumps(list[dict])`, and reward functions call `json.loads` internally.
- **`dataset/__init__.py` is intentionally empty** to avoid eager import of heavy dependencies (unsloth/torch). Import loaders directly: `from dataset.agentnet.agentnet import load_agentnet`.
- **Config names without `/`** are resolved to their respective `configs/` subdirectories automatically by both `load_model_from_file` and `load_trainer_from_file`.
- **`actions_from_completion` requires both `description` and `action` keys** in each answer item. History-only entries in `conversation_from_step` omit `description` and only get scored against the current step's label.
