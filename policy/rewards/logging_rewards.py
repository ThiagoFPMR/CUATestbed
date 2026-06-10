"""
Generation logging for the RL training loop.

TRL's GRPO trainer only hands the generated completions to *reward functions*,
never to TrainerCallbacks — but only a callback can see ``state.global_step``.
So we bridge the two the same way the dataset stats do (see
``StepTimerCallback`` reading ``agentnet_module.LAST_EXAMPLE_*``): a neutral
reward func stashes the single best generation of the step into a module global,
and ``GenerationLoggingCallback`` (in callbacks.py) drains it on ``on_step_end``,
where it knows the step number and can gate on the logging intervals.

"Best route" = the generation with the highest combined GUI-R1 reward in the
group; with GRPO that is the best sampled rollout for the prompt.
"""
import json
import logging

from .format_rewards import gui_r1_formatting_reward_func
from .action_rewards import accuracy_reward_func
from .utils import actions_from_completion

logger = logging.getLogger(__name__)

# Ranking weights mirror parse_rewards.GUI_R1_*_WEIGHT so "best route" matches
# the reward the model is actually optimized against.
_FORMAT_WEIGHT = 0.6
_ACCURACY_WEIGHT = 0.4

# Stash drained by GenerationLoggingCallback.on_step_end. Holds the best
# generation seen so far in the current optimizer step (across grad-accum
# micro-batches), or None once drained.
PENDING_BEST: dict | None = None


def generation_logging_reward_func(completions, labels, image_sizes, **kwargs):
    """
    Neutral reward func (always returns 0.0) whose real job is to record the
    best generation of the step for logging. Returning a constant keeps it from
    affecting GRPO advantages while still being called with the completions.
    """
    global PENDING_BEST

    if isinstance(completions[0], list):
        contents = [completion[-1]["content"] for completion in completions]
    else:
        contents = completions

    try:
        fmt = gui_r1_formatting_reward_func(completions)
        acc = accuracy_reward_func(contents, labels, image_sizes, point_tol=8, scroll_tol=16)
        combined = [_FORMAT_WEIGHT * f + _ACCURACY_WEIGHT * a for f, a in zip(fmt, acc)]

        best_i = max(range(len(combined)), key=combined.__getitem__)
        candidate = {
            "content": contents[best_i],
            "label": labels[best_i],
            "image_size": image_sizes[best_i],
            "format": fmt[best_i],
            "accuracy": acc[best_i],
            "combined": combined[best_i],
            "n_candidates": len(contents),
        }
        # Across grad-accum micro-batches in one step, keep the overall best.
        if PENDING_BEST is None or candidate["combined"] > PENDING_BEST["combined"]:
            PENDING_BEST = candidate
    except Exception as e:  # logging must never break training
        logger.warning("generation_logging_reward_func skipped a batch: %s", e)

    return [0.0] * len(contents)


def format_best_record(record: dict, step: int) -> tuple[str, str]:
    """
    Render a stashed best-generation record into (full_output_entry,
    predicted_action_entry) text blocks for the two log files.
    """
    label = record["label"]
    try:
        gt_actions = [a.get("action") for a in json.loads(label)]
    except (ValueError, TypeError):
        gt_actions = label
    pred_actions = [a.get("action", "") for a in actions_from_completion(record["content"])]

    header = (
        f"step {step} | best of {record['n_candidates']} | "
        f"combined={record['combined']:.3f} "
        f"format={record['format']:.2f} accuracy={record['accuracy']:.2f}"
    )

    full_entry = (
        f"{'=' * 80}\n"
        f"{header}\n"
        f"{'-' * 80}\n"
        f"MODEL OUTPUT:\n{record['content']}\n"
        f"{'-' * 80}\n"
        f"GROUND TRUTH: {label}\n\n"
    )

    pred_entry = f"[step {step}] pred={pred_actions} | gt={gt_actions} | acc={record['accuracy']:.2f}\n"

    return full_entry, pred_entry
