"""
Reward functions for evaluating model completions according to the GUI-R1 paper.
"""
import re


def get_action_from_answer(answer):
    """
    Extract the list of actions from the model completion answer.

    Args:
        completion (str): The answer extracted from the model-generated completion string.

    Returns:
        list of dict or None: List of action dictionaries if parsing is successful, else None.
    """
    action_match = re.search("'action':\s*'(\w+)'", answer)
    if not action_match:
        return None
    return action_match.group(1)

def get_point_from_answer(answer):
    """
    Extract the point coordinates from the model completion answer.

    Args:
        completion (str): The answer extracted from the model-generated completion string.

    Returns:
        list of int or None: List of point coordinates [x, y] if parsing is successful, else None.
    """
    point_match = re.search("'point':\s*\[(\d+),\s*(\-?\d+)\]", answer)
    if not point_match:
        return None
    return [int(point_match.group(1)), int(point_match.group(2))]

def get_input_text_from_answer(answer):
    """
    Extract the input text from the model completion answer.

    Args:
        completion (str): The answer extracted from the model-generated completion string.

    Returns:
        str or None: The input text if parsing is successful, else None.
    """
    input_text_match = re.search("'input_text':\s*'(.*?)'", answer)
    if not input_text_match:
        return None
    return input_text_match.group(1)

def f1_score(pred, gt):
    """
    Compute F1 score between predicted and ground truth actions.
    """
    pred = pred.replace("[", "").replace("]", "")
    gt = gt.replace("[", "").replace("]", "")
    
    pred_set = set(pred)
    gt_set = set(gt)

    tp = len(pred_set.intersection(gt_set)) # true positives
    fp = len(pred_set.difference(gt_set)) # false positives
    fn = len(gt_set.difference(pred_set)) # false negatives

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return f1

def gui_r1_formatting_reward_func(completions, **kwargs):
    """
    Reward function to evaluate the formatting of model completions
    based on the presence of reasoning and solution tags.

    Process should be as follows:
    <think>...</think>
    <answer>
        [{
            'action': enum[...], 
            'point': [x,y],
            'input_text': 'no input text [default]'
        }],...
    </answer>

    Args:
        completions (list of str): List of model-generated completions.
    
    Returns:
        list of float: Reward scores for each completion.
    """
    target_format = re.compile(r"<think>.*?</think>\s*<answer>.*?</answer>", re.DOTALL)
    contents = [completion[0]["content"] for completion in completions]

    format_rewards = []
    for content in contents:
        # Check if the content matches the target format
        if not re.fullmatch(target_format, content):
            format_rewards.append(0.0)
            continue
        
        # Check for answer presence
        answer_match = re.search(r"<answer>(.*?)</answer>", content, re.DOTALL)
        if not answer_match:
            format_rewards.append(0.0)
            continue

        # Turn answer string into dict
        try:
            actions = eval(answer_match.group(1).strip())
        except Exception:
            format_rewards.append(0.0)
            continue

        if not isinstance(actions, list) or len(actions) == 0:
            format_rewards.append(0.0)
            continue
        
        # Validate each action dict
        valid_actions = True
        for action in actions:
            if not all(key in action for key in ["action", "point", "input_text"]):
                valid_actions = False
                break
            if not isinstance(action["action"], str) or not isinstance(action["input_text"], str):
                valid_actions = False
                break
            if not isinstance(action["point"], list) or len(action["point"]) != 2:
                valid_actions = False
                break
            if not isinstance(action["point"][0], int) or not isinstance(action["point"][1], int):
                valid_actions = False
                break
            if action["action"] in ["type", "select", "open_app"] and action["input_text"] == "no input text":
                valid_actions = False
                break
            if action["action"] in ["scroll"] and action["input_text"] not in ["left", "right", "up", "down"]:
                valid_actions = False
                break

        if valid_actions:
            format_rewards.append(1.0)
        else:
            format_rewards.append(0.0)

    return format_rewards

def gui_r1_accuracy_reward_func(completions, gt_actions, gt_bboxs, gt_input_texts, **kwargs):
    """
    Reward function to evaluate the accuracy of model completions based on the correctness of predicted actions, points, and input text.

    Args:
        completions (list of str): List of model-generated completions.
        gt_actions (list of str): List of ground truth actions for each completion.
        gt_bboxs (list of list): List of ground truth bounding boxes or points for each completion.
        gt_input_texts (list of str): List of ground truth input texts for each completion
    
    Returns:
        list of float: Reward scores for each completion.
    """
    answer_pattern = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
    answers = [answer_pattern.search(completion).group(1) for completion in completions]

    acc_rewards = []
    for idx, answer in enumerate(answers):
        action = get_action_from_answer(answer)

        if action != gt_actions[idx]:
            acc_rewards.append(0.0)
            continue

        point = get_point_from_answer(answer)
        if action in ['click', 'long_press', 'press_back']:
            gt_bbox = gt_bboxs[idx]
            if point is None:
                acc_rewards.append(0.0)
                continue
            x, y = point

            # If gt_bbox is a single point, allow some error radius.
            if len(gt_bbox) == 2:
                if (x - gt_bbox[0])**2 + (y - gt_bbox[1])**2 <= 140**2: # allow 140 pixel error radius
                    acc_rewards.append(1.0)
                else:
                    acc_rewards.append(0.0)
                continue
            
            # If gt_bbox is a bounding box, check if the point is within the box.
            x1, y1, x2, y2 = gt_bbox
            if x1 <= x <= x2 and y1 <= y <= y2:
                acc_rewards.append(1.0)
            else:
                acc_rewards.append(0.0)
            continue

        input_text = get_input_text_from_answer(answer)
        if action in ['type', 'select', 'scroll']:
            gt_input_text = gt_input_texts[idx]
            if input_text is None:
                acc_rewards.append(0.0)
                continue

            f1 = f1_score(input_text, gt_input_text)
            acc_rewards.append(f1 >= 0.5) # reward if F1 score is above threshold
            continue

        acc_rewards.append(1.0)

    return acc_rewards  
        
def gui_r1_reward_func(completions, **kwargs):
    format_rewards = gui_r1_formatting_reward_func(completions, **kwargs)
    acc_rewards = gui_r1_accuracy_reward_func(completions, **kwargs)
    # Combine rewards (simple sum for now)
    total_rewards = [0.8 * f + 0.2 * a for f, a in zip(format_rewards, acc_rewards)]
    return total_rewards