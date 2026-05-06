"""
Reward functions for evaluating model actions from our ingestion of the AgentNet dataset.
"""
import re
import json

from policy.rewards.utils import f1_score, actions_from_completion
from dataset.agentnet.action_utils import parse_action_arguments


def accuracy_reward_func(
    completions: list[str], 
    labels: list[str], 
    image_sizes: list[tuple[int, int]],
    f1_thr: float = 0.5,
    r: int = 0
) -> list[float]:
    """
    Compute accuracy reward for a batch based on whether the predicted actions match the ground truth actions.

    Args:
        completions (list[str]): List of model-generated completion strings.
        labels (list[str]): List of JSON strings representing ground truth actions and parameters.
        image_sizes (list[tuple[int, int]]): List of image sizes.
        f1_thr (float): The threshold for F1 score when comparing messages. Default is 0.5.
        r (int): The tolerance radius in pixels for point-based actions. Default is 0 for exact match.
    Returns:
        list[float]: List of accuracy rewards for the batch, between 0 and 1.
    """
    rewards = []
    for completion, label, img_size in zip(completions, labels, image_sizes):
        rewards.append(__accuracy_reward_func(completion, label, img_size, f1_thr, r))
    return rewards

def __accuracy_reward_func(
    completion: str, 
    label: str, 
    img_size: tuple[int, int], 
    f1_thr: float = 0.5, 
    r: int = 0
) -> float:
    """
    Compute accuracy reward for a single completion and label pair.

    Args:
        completion (str): The model-generated completion string.
        label (str): The ground truth actions and parameters as a JSON string.
        img_size (tuple[int, int]): The size of the image (height, width).
        f1_thr (float): The threshold for F1 score when comparing messages. Default is 0.5.
        r (int): The tolerance radius in pixels for point-based actions. Default is 0 for exact match.
    Returns:
        float: Accuracy reward, 1.0 if the predicted action matches the ground truth, else 0.0.
    """
    label: list[dict] = json.loads(label)
    pred_actions: list[dict] = actions_from_completion(completion)
    pred_actions: list[str] = [action.get("action", "") for action in pred_actions]

    effective_len = min(len(pred_actions), len(label))
    if effective_len == 0:
        return 0.0

    reward_sum = 0.0
    for i, action_code in enumerate(pred_actions[:effective_len]):
        action_type = action_code.split("(")[0]
        if action_type != label[i]["action"]:
            return reward_sum / effective_len
        if "x" in label[i] and "y" in label[i]:
            if img_size is None:
                return reward_sum / effective_len
            x_range = (label[i]["x"] - (0.5 + r) / img_size[1], label[i]["x"] + (0.5 + r) / img_size[1])
            y_range = (label[i]["y"] - (0.5 + r) / img_size[0], label[i]["y"] + (0.5 + r) / img_size[0])

            x_pred, y_pred = re.search(r"x=(\d+\.?\d*),\s*y=(\d+\.?\d*)", action_code).groups()
            x_pred, y_pred = float(x_pred), float(y_pred)

            if not (x_range[0] <= x_pred <= x_range[1] and y_range[0] <= y_pred <= y_range[1]):
                return reward_sum / effective_len
        if "message" in label[i]:
            message_pred = re.search(r"message='(.*?)'", action_code).group(1)
            if f1_score(message_pred, label[i]["message"]) < f1_thr:
                return reward_sum / effective_len

        for key in label[i]:
            if key in ["action", "x", "y", "message"]:
                continue
            if key.startswith("arg_"):
                parsed_pred = parse_action_arguments(action_code)
                if parsed_pred.get(key) != label[i][key]:
                    
                    return reward_sum / effective_len
            else:
                if isinstance(label[i][key], str):
                    if (f"{key}='{label[i][key]}'" not in action_code and
                            f'{key}="{label[i][key]}"' not in action_code):
                        return reward_sum / effective_len
                else:
                    if f"{key}={label[i][key]}" not in action_code:
                        return reward_sum / effective_len
        
        reward_sum += 1.0
   
    return reward_sum / effective_len