import json
import os
import cv2

def show_trajectory(example, imgs_dir):
    """
    Utility function to visualize the trajectory of an example.
    Shows each step's screenshot and code in sequence.
    Also says whether the action was redundant or incorrect by looking ahead to the next step.
    The code and redundant/incorrect info is shown as text displayed in the image window, so you can see it while looking at the screenshot.
    """
    trajectories = example.get("traj", [])
    n_steps = len(trajectories)
    if n_steps == 0:
        return
        
    instruction = example.get("instruction", example.get("goal", "Instruction not found under 'instruction' or 'goal' keys"))
    print(f"\n========================================")
    print(f"Viewing Trajectory - Instruction:")
    print(f"{instruction}")
    print(f"========================================")

    i = 0
    while i < n_steps:
        step = trajectories[i]
        image_file = step.get("image", step.get("screenshot", ""))
        image_path = os.path.join(imgs_dir, image_file)
        
        try:
            img = cv2.imread(image_path)
            if img is None:
                print(f"[Step {i+1}] Could not read image at {image_path}")
                i += 1
                continue
        except Exception as e:
            print(f"[Step {i+1}] Error loading image at {image_path}: {e}")
            i += 1
            continue

        code = step.get("value", {}).get("code", "No code found")
        is_redundant = step.get("value", {}).get("last_step_redundant", False)
        is_incorrect = not step.get("value", {}).get("last_step_correct", True)

        status_text = f"Step {i+1}/{n_steps}\nCode: {code}\nRedundant: {is_redundant}\nIncorrect: {is_incorrect}"
        
        # Display the status text on the image
        y0, dy = 30, 30
        for j, line in enumerate(status_text.split('\n')):
            y = y0 + j*dy
            cv2.putText(img, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if not is_incorrect else (0, 0, 255), 2)

        # Scale down the image to fit the screen better
        scale_percent = 50 # percent of original size
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        dim = (width, height)
        resized_img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

        cv2.imshow('Trajectory Step', resized_img)
        key = cv2.waitKeyEx(0)  # waitKeyEx required for Windows arrow keys occasionally
        
        # Press 'q' or 'ESC' to exit the trajectory viewing entirely
        if key == ord('q') or key == 27:
            cv2.destroyAllWindows()
            return
        # Left arrow (Linux=65361, Windows=2424832, Mac=2) or 'a' moves back
        elif key in [81, 2, 2424832, 65361, ord('a'), ord('A')]:
            i = max(0, i - 1)
        # Right arrow or any other key progresses forward
        else:
            i += 1
        
    cv2.destroyAllWindows()
        

def count_has_redundant_actions(example, stats):
    """
    Counts how many examples have redundant actions (does not count the number of redundant actions).
    """
    if "redundant_examples" not in stats:
        stats["redundant_examples"] = 0

    trajectories = example.get("traj", [])
    for traj in trajectories:
        is_redundant = traj.get("value", {}).get("last_step_redundant", False)
        if is_redundant:
            # import pdb; pdb.set_trace()
            stats["redundant_examples"] = stats.get("redundant_examples", 0) + 1
            break  # Count this example only once, even if it has multiple redundant actions

def count_has_incorrect_actions(example, stats):
    """
    Counts how many examples have incorrect actions (does not count the number of incorrect actions).
    """
    if "incorrect_examples" not in stats:
        stats["incorrect_examples"] = 0

    trajectories = example.get("traj", [])
    for traj in trajectories:
        is_incorrect = not traj.get("value", {}).get("last_step_correct", True)
        if is_incorrect:
            # import pdb; pdb.set_trace()
            stats["incorrect_examples"] = stats.get("incorrect_examples", 0) + 1
            break  # Count this example only once, even if it has multiple incorrect actions

def count_completed_tasks(example, stats):
    """
    Counts how many examples have completed tasks.
    """
    if "completed_feasible_examples" not in stats:
        stats["completed_feasible_examples"] = 0
    if "total_completed_examples" not in stats:
        stats["total_completed_examples"] = 0
    if "incomplete_w_alignment_gt_7" not in stats:
        stats["incomplete_w_alignment_gt_7"] = 0
        stats["incomplete_w_alignment_lt_7"] = 0

    is_completed = example.get("task_completed", False)
    if is_completed:
        stats["completed_feasible_examples"] += 1
        stats["total_completed_examples"] += 1
    elif example.get("domain", "") != "infeasible" and example.get("alignment_score", 0) >= 7:
        stats["incomplete_w_alignment_gt_7"] += 1
        # import pdb; pdb.set_trace()
    elif example.get("domain", "") != "infeasible" and example.get("alignment_score", 10) < 7:
        stats["incomplete_w_alignment_lt_7"] += 1

    # How many examples successfully identified that the task was infeasible
    if example.get("domain", "") == "infeasible":
        trajectories = example.get("traj", [])
        for traj in trajectories[::-1]:
            code = traj.get("value", {}).get("code", "")
            if code.find("terminate") == -1:
                continue
            if code.find("failure") != -1:
                stats["total_completed_examples"] = stats["total_completed_examples"] + 1
                break

def track_domains(example, stats):
    """
    Tracks the distribution of domains in the dataset.
    """
    if "domain_counts" not in stats:
        stats["domain_counts"] = {}
    if "domains" not in stats:
        stats["domains"] = set()

    domain = example.get("domain", "not found")
    stats["domain_counts"][domain] = stats["domain_counts"].get(domain, 0) + 1
    stats["domains"].add(domain)

def track_action_types(example, stats):
    """
    Tracks the distribution of action types in the dataset.
    """
    if "action_type_counts" not in stats:
        stats["action_type_counts"] = {}
        stats["action_types"] = set()
        stats["action_arguments"] = {}

    trajectories = example.get("traj", [])
    for traj in trajectories:
        code = traj.get("value", {}).get("code", "not found") 
        # code should follow a format like "pyautogui.action(args)", we can extract the action name
        for action_code in code.split("\n"):
            try:
                action_name = action_code.split(".")[1].split("(")[0] if "." in action_code else "not found"
            except:
                action_name = "does not follow expected format"
            stats["action_type_counts"][action_name] = stats["action_type_counts"].get(action_name, 0) + 1
            stats["action_types"].add(action_name)

            if action_name not in stats["action_arguments"]:
                stats["action_arguments"][action_name] = set()

            args = action_code.split("(")[1].replace(")", "") if "(" in action_code else ""
            stats["action_arguments"][action_name].add(args)

def verify_action_formats(example, stats):
    if "non_conforming_actions" not in stats:
        stats["non_conforming_actions"] = {}
    trajectories = example.get("traj", [])

    def add_non_conforming(name, reason, code_str):
        if name not in stats["non_conforming_actions"]:
            stats["non_conforming_actions"][name] = []
        stats["non_conforming_actions"][name].append(code_str)
        print(f"Non-conforming {name} action code: '{code_str}' - reason: {reason}")

    for traj in trajectories:
        code = traj.get("value", {}).get("code", "not found") 
        for action_code in code.split("\n"):
            action_name = action_code.split(".")[1].split("(")[0] if "." in action_code else "not found"
            action_args = action_code.split("(")[1].replace(")", "") if "(" in action_code else ""
            if action_name == "not found":
                print(f"Non-conforming action code: '{action_code}'")
                continue
            
            # start by checking the 'click' actions
            if action_name in ["click", "rightClick", "doubleClick",  "moveTo", "tripleClick", "middleClick"]:
                # ensure that action args follows the format "x=number, y=number" where both numbers are in the range [0, 1]
                if not all(arg.strip().startswith(("x=", "y=")) for arg in action_args.split(",")):
                    add_non_conforming(action_name, "click action does not have x= and y= arguments", action_code)
                    continue
                try:
                    x_arg = next(arg for arg in action_args.split(",") if arg.strip().startswith("x="))
                    y_arg = next(arg for arg in action_args.split(",") if arg.strip().startswith("y="))
                    x_value = float(x_arg.split("=")[1])
                    y_value = float(y_arg.split("=")[1])
                    if not (0 <= x_value <= 1 and 0 <= y_value <= 1):
                        add_non_conforming(action_name, "click action has x or y argument out of range [0, 1]", action_code)
                except Exception as e:
                    add_non_conforming(action_name, f"error parsing click action arguments: {e}", action_code)
            # wait action should have no arguments
            elif action_name == "wait":
                if action_args.strip() != "":
                    add_non_conforming(action_name, "wait action should have no arguments", action_code)
            # write should have (message="some text") as its argument
            elif action_name == "write":
                if not action_args.strip().startswith("message="):
                    add_non_conforming(action_name, "write action should have message= argument", action_code)
            # scroll and hscroll should have a single integer argument (e.g. scroll(5) or hscroll(-3))
            elif action_name in ["scroll", "hscroll"]:
                try:
                    scroll_value = int(action_args.strip())
                except Exception as e:
                    add_non_conforming(action_name, f"scroll action should have a single integer argument: {e}", action_code)
            # dragTo should have arguments "x=number, y=number" where both numbers are in the range [0, 1] and a button argument like "button='left'"
            elif action_name == "dragTo":
                if not all(arg.strip().startswith(("x=", "y=", "button=")) for arg in action_args.split(",")):
                    add_non_conforming(action_name, "dragTo action does not have x=, y=, and button= arguments", action_code)
                    continue
                try:
                    x_arg = next(arg for arg in action_args.split(",") if arg.strip().startswith("x="))
                    y_arg = next(arg for arg in action_args.split(",") if arg.strip().startswith("y="))
                    button_arg = next(arg for arg in action_args.split(",") if arg.strip().startswith("button="))
                    x_value = float(x_arg.split("=")[1])
                    y_value = float(y_arg.split("=")[1])
                    button_value = button_arg.split("=")[1].strip("'\"")
                    if not (0 <= x_value <= 1 and 0 <= y_value <= 1):
                        add_non_conforming(action_name, "dragTo action has x or y argument out of range [0, 1]", action_code)
                    if button_value not in ["left", "right", "middle"]:
                        add_non_conforming(action_name, "dragTo action has invalid button argument (should be 'left', 'right', or 'middle')", action_code)
                except Exception as e:
                    add_non_conforming(action_name, f"error parsing dragTo action arguments: {e}", action_code)
            # press follows ("key", presses=number) format, where the first is a string representing the key to press and the second is an integer representing how many times to press it
            elif action_name == "press":
                if not (action_args.strip().startswith('"') or action_args.strip().startswith("'")):
                    add_non_conforming(action_name, "press action should have format (\"key\", [presses=number])", action_code)
                else:
                    try:
                        args_list = action_args.split(",")
                        key_part = args_list[0].strip()
                        key_value = key_part.strip('"').strip("'")
                        
                        presses_part = next((arg for arg in args_list if arg.strip().startswith("presses=")), None)
                        if presses_part is not None:
                            presses_value = int(presses_part.split("=")[1])
                    except Exception as e:
                        add_non_conforming(action_name, f"error parsing press action arguments: {e}", action_code)
            # terminate follows (status="success") where status can be "success" or "failure" or "fail"
            elif action_name == "terminate":
                if not action_args.strip().startswith("status="):
                    add_non_conforming(action_name, "terminate action should have status= argument", action_code)
                else:
                    try:
                        status_part = action_args.strip()
                        status_value = status_part.split("=")[1].strip('"').strip("'")
                        if status_value not in ["success", "failure", "fail"]:
                            add_non_conforming(action_name, "terminate action has invalid status argument (should be 'success', 'failure', or 'fail')", action_code)
                    except Exception as e:
                        add_non_conforming(action_name, f"error parsing terminate action arguments: {e}", action_code)
            #hotkey should give a list of keys to press following (["key1", "key2", ...]) format
            elif action_name == "hotkey":
                if not action_args.strip().startswith("[") or not action_args.strip().endswith("]"):
                    add_non_conforming(action_name, "hotkey action should have format ([\"key1\", \"key2\", ...])", action_code)
                else:
                    try:
                        keys_part = action_args.strip()[1:-1].strip()  # remove the surrounding brackets
                        keys = [key.strip().strip('"').strip("'") for key in keys_part.split(",")]
                    except Exception as e:
                        add_non_conforming(action_name, f"error parsing hotkey action arguments: {e}", action_code)
            

def track_example_props(example, stats):
    """
    Tracks the presence of the many keys in the examples, counting how often they appear.
    (Some keys are optional or missing in some examples, so this helps us understand the data better.)
    """
    keys_to_track = [
        'task_id', 
        'instruction', 
        'task_completed', 
        'alignment_score', 
        'efficiency_score', 
        'task_difficulty', 
        'reason', 
        'natural_language_task', 
        'actual_task', 
        'traj', 
        'domain'
    ]
    for key in keys_to_track:
        if key not in stats:
            stats[key] = 0
        if key in example:
            stats[key] += 1

def track_instruction_length(example, stats):
    """
    Tracks the mean and standard deviation of instruction lengths in the dataset.
    Uses Welford's online algorithm for numeric stability.
    """
    if "instruction_count" not in stats:
        stats["instruction_count"] = 0
        stats["mean_instruction_length"] = 0.0
        stats["M2_instruction_length"] = 0.0
        stats["std_instruction_length"] = 0.0

    instruction = example.get("instruction", "")
    length = len(instruction.split())  # Count words in the instruction
    
    stats["instruction_count"] += 1
    count = stats["instruction_count"]
    
    # Welford's online algorithm
    delta = length - stats["mean_instruction_length"]
    stats["mean_instruction_length"] += delta / count
    delta2 = length - stats["mean_instruction_length"]
    stats["M2_instruction_length"] += delta * delta2
    
    if count > 1:
        # Sample standard deviation
        import math
        stats["std_instruction_length"] = math.sqrt(stats["M2_instruction_length"] / (count - 1))
    else:
        stats["std_instruction_length"] = 0.0

def profile_dataset(jsonl_files: list[str], analysis_funcs: list[callable]) -> dict:
    """
    Profiles a dataset by applying a list of analysis functions to each example.

    Args:
        jsonl_files: List of paths to JSONL files containing the dataset.
        analysis_funcs: List of functions that analyze each example and populate stats.

    Returns:
        A dictionary where keys are the names of the analysis functions and values are lists of collected stats
    """
    stats = {func.__name__: {} for func in analysis_funcs}
    
    for jsonl_file in jsonl_files:
        print(f"Profiling data from: {jsonl_file}...")

        imgs_dir = DATASET_DIR + "/ubuntu_images"

        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                example = json.loads(line)
                for func in analysis_funcs:
                    func(example, stats[func.__name__])
    
    return stats

if __name__ == "__main__":
    DATASET_DIR = "data"
    # Dynamically build the paths using the DATASET_DIR variable
    sources = [
        os.path.join(DATASET_DIR, "agentnet_ubuntu_5k.jsonl"), 
        os.path.join(DATASET_DIR, "agentnet_win_mac_18k.jsonl"),
    ]

    funcs = [
        count_has_redundant_actions,
        count_has_incorrect_actions,
        count_completed_tasks,
        track_domains,
        track_action_types,
        track_example_props,
        track_instruction_length,
        verify_action_formats,
    ]

    # 1. Load all records into a single Python list
    stats = profile_dataset(sources, funcs)

    import pdb; pdb.set_trace()

