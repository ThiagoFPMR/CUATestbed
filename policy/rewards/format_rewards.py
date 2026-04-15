"""
Reward functions for evaluating model completions according to the GUI-R1 paper.
"""
import re
import json

from dataset.agentnet.agentnet import ACTION_PATTERNS
from policy.rewards.utils import get_actions_from_answer


def gui_r1_formatting_reward_func(completions, **kwargs):
    """
    Reward function to evaluate the formatting of model completions
    based on the presence of reasoning and solution tags.

    Process should be as follows:
    <think>...</think>
    <answer>
        [{
            'description': ....,
            'action': action(parameters),
        }],...
    </answer>

    Args:
        completions (list of str): List of model-generated completions.
    Returns:
        list of float: Reward scores for each completion.
    """
    contents = [completion[0]["content"] for completion in completions]

    format_rewards = []
    for content in contents:
        format_rewards.append(__gui_r1_formatting_reward_func(content))

def __gui_r1_formatting_reward_func(completion: str):
    """
    Helper function to evaluate the formatting of a single completion.

    Args:
        content (str): The model-generated completion string.

    Returns:
        float: Reward score for the completion.
    """
    target_format = re.compile(r"<think>.*?</think>\s*<answer>.*?</answer>", re.DOTALL)
    if not re.fullmatch(target_format, completion):
        return 0.0

    answer_match = re.search(r"<answer>(.*?)</answer>", completion, re.DOTALL)
    if not answer_match:
        return 0.0

    actions = actions_from_completion(completion)
    if not actions:
        return 0.0

    for action in actions:
        action_code = action.get("action", "")
        if not re.match(r"^\w+\(.*\)$", action_code):
            return 0.0
        action_type = action_code.split("(")[0]
        pattern = ACTION_PATTERNS.get(action_type)

        if not pattern:
            return 0.0
        if not pattern.match(action_code):
            return 0.0

    return 1.0