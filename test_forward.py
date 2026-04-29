import json
import torch
from model import load_model_from_file
from dataset.agentnet.agentnet import load_agentnet
from policy.rewards.parse_rewards import parse_rewards

NUM_SAMPLES = 5


def label_to_action_code(label: dict) -> str:
    """Reconstruct a valid action code string from a parsed label dict."""
    action = label["action"]
    parts = []
    i = 0
    while f"arg_{i}" in label:
        val = label[f"arg_{i}"]
        if isinstance(val, list):
            items = ", ".join(f"'{v}'" if isinstance(v, str) else str(v) for v in val)
            parts.append(f"[{items}]")
        elif isinstance(val, str):
            parts.append(f"'{val}'")
        else:
            parts.append(str(val))
        i += 1
    for key, val in label.items():
        if key == "action" or key.startswith("arg_"):
            continue
        if isinstance(val, str):
            parts.append(f"{key}='{val}'")
        else:
            parts.append(f"{key}={val}")
    return f"{action}({', '.join(parts)})"


def make_correct_completion(label_str: str) -> str:
    """Build a perfectly formatted, correct completion from a label JSON string."""
    labels = json.loads(label_str or "[]")
    action_codes = [label_to_action_code(lbl) for lbl in labels]
    actions = [{"description": "performing the correct action", "action": code} for code in action_codes]
    return f"<think>Analyzing the UI to determine the correct action.</think> <answer>{repr(actions)}</answer>"


def score_completion(gui_reward, completion_text, labels, img_sizes):
    completions = [[{"role": "assistant", "content": completion_text}]]
    return gui_reward(completions, labels, img_sizes)[0]


def test_forward_pass():
    print("Loading model and tokenizer...")
    model_config = "qwen3-vl-2b-4bit.yml"
    model, tokenizer = load_model_from_file(model_config)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading dataset...")
    base_prompt = (
        "You are GUI-R1, a reasoning GUI Agent Assistant. In this UI screenshot <image>, I want you to continue executing the command '[INSTRUCT]', with the action history being '[HISTORY]'.\n"
        "Please provide the action to perform (enumerate from ['wait', 'long_press', 'click', 'press_back', 'type', 'open_app', 'scroll']), the point where the cursor is moved to (integer) if a click is performed, and any input text required to complete the action.\n"
        "Output the thinking process in <think> </think> tags, and the final answer in <answer> </answer> tags as follows:\n"
        "<think> ... </think> <answer>[{'action': enum['wait', 'long_press', 'click', 'press_back', 'type', 'open_app', 'scroll'], 'point': [x, y], 'input_text': 'no input text [default]'}]</answer>\n"
        "Note:\n specific input text (no default) is necessary for actions enum['type', 'open_app', 'scroll'] \n Example:\n"
        "[{'action': enum['wait', 'press_back'], 'point': [-100, -100], 'input_text': 'no input text'}]\n"
        "[{'action': enum['click', 'long_press'], 'point': [123, 300], 'input_text': 'no input text'}]\n"
        "[{'action': enum['type', 'open_app'], 'point': [-100, -100], 'input_text': 'shanghai shopping mall'}]\n"
        "[{'action': enum['scroll'], 'point': [-100, -100], 'input_text': enum['up', 'left', 'right', 'down']}]"
    )

    dataset = load_agentnet(base_prompt)
    reward_funcs = parse_rewards(["gui_r1"])
    gui_reward = reward_funcs[0]

    print(f"\nRunning tests on {NUM_SAMPLES} samples...\n")

    for sample_idx in range(NUM_SAMPLES):
        sample = dataset[sample_idx]
        label_str = sample.get("labels", "[]")
        img_sizes = sample.get("image_sizes", [(100, 100)])
        if not isinstance(img_sizes, list):
            img_sizes = [img_sizes]
        labels = [label_str]

        print(f"\n{'='*60}")
        print(f"Sample {sample_idx + 1}/{NUM_SAMPLES}")
        print(f"Label:      {label_str}")
        print(f"Image size: {img_sizes}")

        # Artificial correct completion — should always yield reward ~1.0
        correct_completion = make_correct_completion(label_str)
        print(f"\n[Correct Completion]\n{correct_completion}")
        correct_reward = score_completion(gui_reward, correct_completion, labels, img_sizes)
        print(f"[Correct Reward]: {correct_reward:.4f}")

        # Model generation
        print("\n[Model Generation]")
        images = []
        for msg in sample["messages"]:
            for content in msg.get("content", []):
                if isinstance(content, dict) and content.get("type") == "image":
                    images.append(content["image"])

        text = tokenizer.apply_chat_template(sample["messages"], tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text=[text], images=images, return_tensors="pt", padding=True).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                pad_token_id=tokenizer.eos_token_id,
            )

        completion_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        completion_text = tokenizer.decode(completion_tokens, skip_special_tokens=True)
        print(f"[Model Completion]\n{completion_text}")
        model_reward = score_completion(gui_reward, completion_text, labels, img_sizes)
        print(f"[Model Reward]: {model_reward:.4f}")

    print(f"\n{'='*60}")
    print("Done.")


if __name__ == "__main__":
    test_forward_pass()
