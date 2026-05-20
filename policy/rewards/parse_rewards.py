"""
Module for parsing reward function names into their corresponding RewardFunction instances.
"""

from .action_rewards import *
from .format_rewards import *
from typing import Callable


# GUI-R1 paper combines format (weight 0.6) and accuracy (weight 0.4). Split into two
# pre-weighted reward funcs so the trainer sums them (preserving the original total)
# while wandb logs each component independently.
GUI_R1_FORMAT_WEIGHT = 0.6
GUI_R1_ACCURACY_WEIGHT = 0.4


def gui_r1_format_reward_func(completions, labels, image_sizes, **kwargs):
    """Pre-weighted format component of the GUI-R1 reward."""
    format_rewards = gui_r1_formatting_reward_func(completions)
    return [GUI_R1_FORMAT_WEIGHT * f for f in format_rewards]


def gui_r1_accuracy_reward_func(completions, labels, image_sizes, **kwargs):
    """Pre-weighted accuracy component of the GUI-R1 reward."""
    if isinstance(completions[0], list):
        contents = [completion[-1]["content"] for completion in completions]
    else:
        contents = completions
    acc_rewards = accuracy_reward_func(contents, labels, image_sizes, point_tol=8, scroll_tol=16)
    return [GUI_R1_ACCURACY_WEIGHT * a for a in acc_rewards]


# Reward-name aliases that expand into multiple underlying funcs.
_REWARD_GROUPS = {
    "gui_r1": ["gui_r1_format", "gui_r1_accuracy"],
}


def parse_rewards(reward_names: list[str]) -> list[Callable]:
    """
    Parse a list of reward function names into their corresponding RewardFunction instances.

    Args:
        reward_names (list[str]): List of reward function names as strings.

    Returns:
        list[Callable]: List of instantiated reward functions.
    """
    reward_funcs = []
    for reward_name in reward_names:
        names = _REWARD_GROUPS.get(reward_name, [reward_name])
        for name in names:
            try:
                reward_funcs.append(globals()[f"{name}_reward_func"])
            except KeyError:
                raise ValueError(f"🚫 Unknown reward function: {name}")

    return reward_funcs

if __name__ == "__main__":
    reward_names = ["gui_r1"]
    import pdb; pdb.set_trace()
    reward_funcs = parse_rewards(reward_names)
    print(reward_funcs)