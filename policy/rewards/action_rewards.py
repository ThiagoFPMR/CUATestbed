"""
Reward functions for evaluating model actions from our ingestion of the AgentNet dataset.
"""
import re
import json

from policy.rewards.utils import f1_score, actions_from_completion


def accuracy_reward_func(
    completions: list[str], 
    labels: list[str], 
    img_sizes: list[tuple[int, int]],
    f1_thr: float = 0.5,
    r: int = 0
) -> list[float]:
    """
    Compute accuracy reward for a batch based on whether the predicted actions match the ground truth actions.

    Args:
        completions (list[str]): List of model-generated completion strings.
        labels (list[str]): List of JSON strings representing ground truth actions and parameters.
        img_sizes (list[tuple[int, int]]): List of image sizes.
        f1_thr (float): The threshold for F1 score when comparing messages. Default is 0.5.
        r (int): The tolerance radius in pixels for point-based actions. Default is 0 for exact match.
    Returns:
        list[float]: List of accuracy rewards for the batch, between 0 and 1.
    """
    rewards = []
    for completion, label, img_size in zip(completions, labels, img_sizes):
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

    # check that the number of predicted actions matches the number of ground truth actions
    if len(pred_actions) != len(label):
        print(f"⛔️ Mismatch in number of predicted actions and ground truth actions. Predicted: {len(pred_actions)}, Ground Truth: {len(label)}")
        return 0.0

    reward_sum = 0.0
    for i, action_code in enumerate(pred_actions):
        action_type = action_code.split("(")[0]
        # Match the action type before checking parameters
        if action_type != label[i]["action"]:
            return reward_sum / len(pred_actions)  
        # If the action has point parameters, check that they match as well
        if "x" in label[i] and "y" in label[i]: 
            x_range = (label[i]["x"] - (0.5 + r) / img_size[1], label[i]["x"] + (0.5 + r) / img_size[1])
            y_range = (label[i]["y"] - (0.5 - r) / img_size[0], label[i]["y"] + (0.5 + r) / img_size[0])
            
            x_pred, y_pred = re.search(r"x=(\d+\.?\d*),\s*y=(\d+\.?\d*)", action_code).groups()
            x_pred, y_pred = float(x_pred), float(y_pred)

            if not (x_range[0] <= x_pred <= x_range[1] and y_range[0] <= y_pred <= y_range[1]):
                return reward_sum / len(pred_actions)
        # If the action has a message parameter, check that it matches as well
        if "message" in label[i]:
            message_pred = re.search(r"message='(.*?)'", action_code).group(1)
            if f1_score(message_pred, label[i]["message"]) < f1_thr:
                return reward_sum / len(pred_actions)

        # every other parameters must be exact match
        for key in label[i]:
            if key in ["action", "x", "y", "message"]:
                continue
            # if key is arg_n, check that the nth argument in the action code matches the label
            if key.startswith("arg_"):
                arg_idx = int(key.split("_")[1])
                arg_pred = re.search(r"\((.*?)\)", action_code).group(1).split(",")[arg_idx].strip()
                if arg_pred != str(label[i][key]):
                    return reward_sum / len(pred_actions)
            else:
                # otherwise, check that the key=value pair is in the action code
                if f"{key}={label[i][key]}" not in action_code:
                    return reward_sum / len(pred_actions)
        
        reward_sum += 1.0

    return reward_sum / len(pred_actions)