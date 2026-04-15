import re
import ast

def get_action_patterns() -> dict:
    """Returns a dictionary mapping action names to their compiled regex patterns."""
    pattern_groups = {
        r"^(click|rightClick|doubleClick|moveTo|tripleClick|middleClick)\(\s*x=(0(\.\d+)?|1(\.0+)?)\s*,\s*y=(0(\.\d+)?|1(\.0+)?)\s*\)$": 
            ("click", "rightClick", "doubleClick", "moveTo", "tripleClick", "middleClick"),
        r"^(scroll|scrollUp|scrollDown|hscroll)\(\s*-?\d+\s*\)$": 
            ("scroll", "scrollUp", "scrollDown", "hscroll"),
        r"^wait\(\s*\)$": ("wait",),
        r"^write\(\s*message=(['\"]).*?\1\s*\)$": ("write",),
        r"^hotkey\(\s*\[\s*(['\"]).*?\1\s*(,\s*(['\"]).*?\3\s*)*\]\s*\)$": ("hotkey",),
        r"^dragTo\(\s*x=(0(\.\d+)?|1(\.0+)?)\s*,\s*y=(0(\.\d+)?|1(\.0+)?)\s*,\s*button=(['\"])(left|right|middle)\7\s*\)$": ("dragTo",),
        r"^press\(\s*(?:\[\s*(['\"]).*?\1\s*(,\s*(['\"]).*?\3\s*)*\]|(['\"]).*?\4(\s*,\s*presses=\d+)?)\s*\)$": ("press",),
        r"^terminate\(\s*status=(['\"])(success|failure|fail)\2\s*\)$": ("terminate",),
        r"^navigate\(\s*(['\"]).*?\1\s*\)$": ("navigate",),
        r"^screenshot\(\s*\)$": ("screenshot",)
    }

    action_patterns = {}
    for pattern_str, keys in pattern_groups.items():
        compiled_pattern = re.compile(pattern_str)
        for key in keys:
            action_patterns[key] = compiled_pattern
    return action_patterns

# Keep a pre-compiled singleton instance for easy imports if desired
ACTION_PATTERNS = get_action_patterns()

def verify_action_code(code: str) -> bool:
    """
    Runs the action code through a series of regex checks to ensure it conforms
    to the expected format for each action type. 
    
    Args:
        action_code (str): The action code to verify.
    Returns:
        bool: True if the action code is valid, False otherwise.
    """
    for action_code in code.split("\n"):
        action_code = action_code.split(".")[1]
        action_type = action_code.split("(")[0]
        
        pattern = ACTION_PATTERNS.get(action_type)

        if not pattern:
            return False
        if not pattern.match(action_code):
            return False

    return True

def get_actions_from_step(code: str) -> str:
    """Extract the action from a step's code."""
    actions = "["
    for action_code in code.split("\n"):
        action_code = action_code.split(".")[1]
        if action_code == "terminate(status='fail')":
            import pdb; pdb.set_trace()
            action_code ="terminate(status='failure')"
        actions += f"{action_code}, "
    actions = actions.rstrip(", ") + "]"
    return actions

def labels_from_code(code: str) -> dict:
    """Converts a block of action code into a list of label dictionaries for each action."""
    labels = []
    for action_code in code.split("\n"):
        label = parse_action_arguments(action_code)
        if label:
            labels.append(label)
    return labels

def parse_action_arguments(action_code: str) -> dict:
    """
    Extracts the arguments from an action string turning it into a dictionary 
    where the action name is a key and each argument is a separate key.
    Example: 'click(x=0.5, y=0.5)' -> {'action': 'click', 'x': 0.5, 'y': 0.5}

    Args:
        action_code (str): The raw action code string to parse.
    Returns:
        dict: A dictionary containing the action name and its arguments.
    """
    try:
        # Remove any module prefix if present (e.g., 'pyautogui.click' -> 'click')
        if "." in action_code:
            action_code = action_code.split(".", 1)[1]
            
        tree = ast.parse(action_code.strip(), mode='eval').body
        if not isinstance(tree, ast.Call):
            return {}
            
        # Get the action name
        action_name = tree.func.id if isinstance(tree.func, ast.Name) else getattr(tree.func, 'attr', str(tree.func))
        
        result = {"action": action_name}
        
        # Extract positional arguments
        for i, arg in enumerate(tree.args):
            try:
                result[f"arg_{i}"] = ast.literal_eval(arg)
            except Exception:
                pass # Skip if unable to parse the literal
                
        # Extract keyword arguments (x=0.5, message="hello")
        for kw in tree.keywords:
            try:
                result[kw.arg] = ast.literal_eval(kw.value)
            except Exception:
                pass
                
        return result
    except Exception as e:
        print(f"Failed to parse action '{action_code}': {e}")
        return {}

