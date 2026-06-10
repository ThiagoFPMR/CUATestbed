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

os.environ.setdefault("WANDB_PROJECT", "omnibrowse")


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


def train_vlm_rl(
        base_prompt: str,
        experiment_name: str,
        model_config: str,
        trainer_config: str,
        rewards: list[str],
        resume_from_checkpoint: bool | str = False,
        s3_uri: str | None = None,
) -> None:
    """
    Train a vision-language model using reinforcement learning.

    Args:
        experiment_name (str): Name of the training experiment.
        model_config (str): Path to the model configuration file. If does not contain /, it is assumed to be in the model configs directory.
        trainer_config (str): Path to the trainer configuration file. If does not contain /, it is assumed to be in the trainer configs directory.
        output_dir (str): Directory to save the trained model.
        s3_uri (str | None): If set (e.g. s3://bucket/prefix), mirror checkpoints and final artifacts to S3.
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
        GenerationLoggingCallback(pred_action_every=5, full_output_every=60)
    )
    trainer.callback_handler.callbacks.append(CheckpointRetentionCallback(n_ckpts=10))

    s3_callback = None
    if s3_uri:
        s3_callback = S3UploadCallback(s3_uri, experiment_name)
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
    parser.add_argument("--s3_uri", type=str, required=False, default=None, help="If set (e.g. s3://bucket/prefix), mirror each checkpoint and the final artifacts to S3. For cloud boxes with ephemeral local storage.")
    args = parser.parse_args()

    base_prompt = (
        "You are a desktop GUI agent. Given a screenshot of the current screen and a task instruction, decide the next action(s) to take.\n\n"
        "## Conversation history format\n"
        "Earlier turns are a log of what has already been done. They appear in three tiers, from newest to oldest:\n"
        "- Most recent prior step: the screenshot is shown directly as an image, followed by a text <observation> of that screen and a bare action list.\n"
        "- The two steps before that: a text <observation> describing the screen state at that point, followed by a bare action list. No screenshot.\n"
        "- All older steps: only a bare action list — no observation and no screenshot. Treat these as a trail of actions already attempted, not as a reconstruction of past screens.\n"
        "YOUR own reply must NOT use any of these compact formats; it must follow the full format below.\n\n"
        "## Output format — your response must follow this exactly\n"
        "Begin with <observation>, then <reason>, then <answer>, and STOP. No preamble. No text after </answer>.\n\n"
        "Skeleton:\n"
        "<observation>OBSERVATION</observation> <reason>REASONING</reason> <answer>[{'description': 'DESCRIPTION', 'action': 'ACTION_CALL'}]</answer>\n\n"
        "Where:\n"
        "- OBSERVATION: a concise description of the current screen state — what application is open, what is visible, where the cursor/focus appears to be, any notable UI elements. Two to four sentences. This is your grounding step before reasoning.\n"
        "- REASONING: a structured analysis (see 'How to reason' below). Keep it focused and write in the first person.\n"
        "- The <answer> body is a Python list literal of dicts. Each dict MUST have BOTH keys:\n"
        "    'description' — a short natural-language summary of what this action does and why. Refer to the target by its visible name, label, or distinguishing features (shape, color, position) — NOT by coordinates. For window controls, identify them correctly: minimize is '—', maximize is '□', close is 'X'. If a UI element's name is in a non-English language, keep the original language.\n"
        "    'action'      — a single action call string from the table below, in the exact syntax shown.\n"
        "- You MAY emit multiple action dicts in the list if several actions should fire in sequence at this step (e.g. write then press enter, or consolidated repeated keypresses). Most steps are a single action — don't pad. Prefer consolidating repetitive keypresses using the 'presses' kwarg (e.g. press('backspace', presses=5)) over emitting five separate dicts.\n"
        "- Quoting: write each 'action' value as a DOUBLE-quoted Python string so any single quotes inside (e.g. press('enter')) are valid. Example: 'action': \"press('enter')\". Never write 'action': 'press('enter')' — that is a syntax error.\n"
        "- Coordinates x and y are NORMALIZED FLOATS in [0.0, 1.0] — never raw pixel values. Always include a leading 0 (write 0.5, not .5).\n"
        "- Use ONLY the actions in the table below, with their exact argument syntax. Do NOT invent actions like type(...), execute(...), or bare shell commands.\n\n"
        "## How to reason — what goes inside <reason>...</reason>\n"
        "Structure your reasoning around two phases. You do not need literal headers, but every reply should cover both:\n\n"
        "1. Progress assessment — look backward:\n"
        "   - What parts of the overall task are already done, based on the prior action log and the current screenshot?\n"
        "   - Did the previous action produce the expected result? Look for unexpected dialogs, error states, unchanged screens, or focus loss.\n"
        "   - If the previous action appears to have failed or led somewhere wrong, name the recovery step (dismiss the dialog, undo, retry with a different target, etc.) instead of plowing ahead.\n\n"
        "2. Next-action analysis — look forward:\n"
        "   - Briefly consider the plausible next actions given the current screen state.\n"
        "   - Pick the most logical one and say why it beats the alternatives (e.g. keyboard shortcut vs. menu navigation, direct click vs. scrolling first).\n"
        "   - Anticipate what the screen should look like after the action — this becomes the check on the next turn.\n\n"
        "Special case — text input actions:\n"
        "   - Note where the cursor currently is and whether the intended field is actually focused (don't assume).\n"
        "   - State the expected final text in the field after typing.\n"
        "   - Consolidate repeated keypresses with the 'presses' kwarg rather than repeating dicts.\n\n"
        "Special case — terminating:\n"
        "   - Only call terminate(status='success') when the screenshot itself confirms the task's end state. Do not terminate optimistically.\n"
        "   - Call terminate(status='failure') when you have exhausted reasonable approaches or the task is genuinely blocked, and explain why in the reasoning.\n\n"
        "## Available actions\n"
        "Pointer actions — kwargs x, y as floats in [0.0, 1.0]:\n"
        "  click(x=0.5, y=0.3)                       left-click at the given point\n"
        "  rightClick(x=0.5, y=0.3)                  right-click\n"
        "  doubleClick(x=0.5, y=0.3)                 double-click\n"
        "  tripleClick(x=0.5, y=0.3)                 triple-click (e.g. select a line of text)\n"
        "  middleClick(x=0.5, y=0.3)                 middle-click\n"
        "  moveTo(x=0.5, y=0.3)                      move the cursor without clicking\n"
        "  dragTo(x=0.5, y=0.3, button='left')       drag to the point; button is one of 'left' | 'right' | 'middle'\n\n"
        "Scrolling — a single positional INTEGER (no decimals, no kwargs); positive = down/right, negative = up/left:\n"
        "  scroll(-3)                                vertical scroll\n"
        "  hscroll(2)                                horizontal scroll\n\n"
        "Keyboard:\n"
        "  write(message='hello world')              type literal text into the focused field (kwarg 'message'; do NOT use type())\n"
        "  press('enter')                            press a single key by name\n"
        "  press('a', presses=3)                     press a key multiple times (optional kwarg) — prefer this over repeating dicts\n"
        "  press(['tab', 'tab'])                     press a sequence of keys, in order\n"
        "  hotkey(['ctrl', 'c'])                     hold a chord of keys simultaneously\n\n"
        "Navigation / control:\n"
        "  wait()                                    pause briefly for the UI to update\n"
        "  terminate(status='success')               end the task — use when complete\n"
        "  terminate(status='failure')               end the task — use when it cannot be completed\n\n"
        "## Worked examples (your reply should look like these — every reply, even if the history has bare actions)\n"
        "Example A — single click, with progress + next-action reasoning:\n"
        "<observation>The screen shows a text editor application with a menu bar at the top. The File, Edit, View, and Help menus are visible. The document area below is empty and no menus are currently open. The cursor appears to be in the main document body.</observation> <reason>The history shows I haven't opened any menus yet, and the current screenshot confirms I'm at the application's main window. The File menu sits in the top-left of the toolbar, and I need it to reach 'Open Recent'. Alternatives — a keyboard shortcut like Ctrl+O — would skip the recent-files list, so the menu is the right path. After clicking I expect the File dropdown to appear, exposing 'Open Recent' as a submenu.</reason> <answer>[{'description': 'Open the File menu in the top-left of the menu bar', 'action': \"click(x=0.04, y=0.03)\"}]</answer>\n\n"
        "Example B — keyboard shortcut, recovering from a previous misstep:\n"
        "<observation>The screen shows a document editor with text content visible. The cursor is blinking inside the document body, indicating the editor has focus. The title bar shows the filename without a dirty indicator, but I have not yet saved the changes made in this session.</observation> <reason>The previous action clicked into the document body, which means the editor is now focused — good, that's the prerequisite for a save shortcut. The task is to save the file, and I haven't done that yet. Ctrl+S is the standard save shortcut and avoids menu navigation entirely. After pressing it I expect either a silent save (title bar dirty marker disappears) or a Save As dialog if the file is untitled.</reason> <answer>[{'description': 'Save the document with the Ctrl+S shortcut', 'action': \"hotkey(['ctrl', 's'])\"}]</answer>\n\n"
    )

    train_vlm_rl(base_prompt, args.experiment_name, args.model_config, args.trainer_config, args.rewards, args.resume_from_checkpoint, args.s3_uri)