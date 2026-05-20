"""
Utility functions for reward computation in the OmniBrowse policy module.
"""
import re
import ast


def f1_score(pred: str, gt: str) -> float:
    """Compute F1 score between two strings based on word overlap."""
    pred_tokens = set(pred.split())
    gt_tokens = set(gt.split())
    if not pred_tokens and not gt_tokens:
        return 1.0
    if not pred_tokens or not gt_tokens:
        return 0.0
    precision = len(pred_tokens & gt_tokens) / len(pred_tokens)
    recall = len(pred_tokens & gt_tokens) / len(gt_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)

def actions_from_completion(completion: str) -> list[dict[str, str]]:
    """
    Extracts a list of dictionaries from the <answer> tag in the completion.
    Expected format inside the tag: [{'description': '...', 'action': '...'}]

    Args:
        completion (str): The text containing the <answer> tag.
    
    Returns:
        list[dict[str, str]]: A list of dictionaries with 'description' and 'action' keys.
                              Returns an empty list if parsing or validation fails.
    """
    answer_match = re.search(r"<answer>(.*?)</answer>", completion, re.DOTALL)
    if not answer_match:
        return []
    
    try:
        # Extract the string and strip whitespace
        actions_str = answer_match.group(1).strip()
        
        # Safely parse the stringified list of dicts
        actions = ast.literal_eval(actions_str)
        
        # Validate that the result is a list
        if not isinstance(actions, list):
            return []
            
        # 'description' is optional (interpretability only); 'action' is required for scoring.
        is_valid = all(
            isinstance(item, dict) and
            'action' in item and
            isinstance(item.get('action'), str) and
            (item.get('description') is None or isinstance(item.get('description'), str))
            for item in actions
        )
        
        if is_valid:
            return actions
        else:
            return []
            
    except (ValueError, SyntaxError):
        # Catches malformed string representations
        return []