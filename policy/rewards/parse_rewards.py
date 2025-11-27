"""
Module for parsing reward function names into their corresponding RewardFunction instances.
"""

import rewards


def parse_rewards(reward_names: list[str]) -> list[function]:
    """
    Parse a list of reward function names into their corresponding RewardFunction instances.

    Args:
        reward_names (list[str]): List of reward function names as strings.

    Returns:
        list[function]: List of instantiated reward functions.
    """
    reward_funcs = []
    for reward_name in reward_names:
        try: 
            reward_func = getattr(rewards, f"{reward_name}_reward_func")
            reward_funcs.append(reward_func)
        except AttributeError:
            raise ValueError(f"🚫 Unknown reward function: {reward_name}")
        
    return reward_funcs