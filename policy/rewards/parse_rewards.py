"""
Module for parsing reward function names into their corresponding RewardFunction instances.
"""

from .action_rewards import *
from .format_rewards import *
from typing import Callable


def gui_r1_reward_func(completions, labels, image_sizes, **kwargs):
    """
    Combined reward from GUI-R1 paper that incorporates both formatting and accuracy rewards.
    """
    # GRPO trainer passes a list of message lists (e.g. [[{"role": "assistant", "content": "..."}]])
    # Extract the strings to feed to our unified reward functions
    if isinstance(completions[0], list):
        contents = [completion[-1]["content"] for completion in completions]
    else:
        contents = completions

    # Pass the extracted string contents directly 
    # (Update format_rewards.py or action_rewards.py if they try to re-extract!)
    format_rewards = gui_r1_formatting_reward_func(completions)
    acc_rewards = accuracy_reward_func(contents, labels, image_sizes)
    total_rewards = [0.8 * f + 0.2 * a for f, a in zip(format_rewards, acc_rewards)]
    return total_rewards

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
        try: 
            reward_func = globals()[f"{reward_name}_reward_func"]
            reward_funcs.append(reward_func)
        except KeyError:
            raise ValueError(f"🚫 Unknown reward function: {reward_name}")
        
    return reward_funcs

if __name__ == "__main__":
    reward_names = ["gui_r1"]
    import pdb; pdb.set_trace()
    reward_funcs = parse_rewards(reward_names)
    print(reward_funcs)