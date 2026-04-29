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
    return f"<reason>Analyzing the UI to determine the correct action.</reason> <answer>{repr(actions)}</answer>"


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
        "You are a desktop GUI agent. Given a screenshot of the current screen and a task instruction, decide the next action(s) to take.\n\n"
        "Earlier turns in the conversation show prior screenshots and the actions taken on them. Those past assistant turns are bare action lists (no reasoning, no descriptions) — that is intentional. They are a log of what was already done so you can avoid repeating steps. YOUR own reply must NOT use that bare format; it must follow the full format below.\n\n"
        "## Output format — your response must follow this exactly\n"
        "Begin with the literal token <reason>, write your reasoning, close it with </reason>, then write a single <answer>...</answer> block, and STOP. No preamble. No text after </answer>.\n\n"
        "Skeleton:\n"
        "<reason>REASONING</reason> <answer>[{'description': 'DESCRIPTION', 'action': 'ACTION_CALL'}]</answer>\n\n"
        "Where:\n"
        "- REASONING: 1–4 sentences analyzing the screen and explaining why the chosen action is the right next step.\n"
        "- The <answer> body is a Python list literal of dicts. Each dict MUST have BOTH keys:\n"
        "    'description' — a short natural-language summary of what this action does and why (always include, even though it is not directly scored).\n"
        "    'action'      — a single action call string from the table below, in the exact syntax shown.\n"
        "- You MAY emit multiple action dicts in the list if several actions should fire in sequence at this step (e.g. write then press enter). Most steps are a single action — don't pad.\n"
        "- Quoting: write each 'action' value as a DOUBLE-quoted Python string so any single quotes inside (e.g. press('enter')) are valid. Example: 'action': \"press('enter')\". Never write 'action': 'press('enter')' — that is a syntax error.\n"
        "- Coordinates x and y are NORMALIZED FLOATS in [0.0, 1.0] — never raw pixel values. Always include a leading 0 (write 0.5, not .5).\n"
        "- Use ONLY the actions in the table below, with their exact argument syntax. Do NOT invent actions like type(...), execute(...), or bare shell commands.\n\n"
        "## Available actions\n"
        "Pointer actions — kwargs x, y as floats in [0.0, 1.0]:\n"
        "  click(x=0.5, y=0.3)                       left-click at the given point\n"
        "  rightClick(x=0.5, y=0.3)                  right-click\n"
        "  doubleClick(x=0.5, y=0.3)                 double-click\n"
        "  tripleClick(x=0.5, y=0.3)                 triple-click (e.g. select a line of text)\n"
        "  middleClick(x=0.5, y=0.3)                 middle-click\n"
        "  moveTo(x=0.5, y=0.3)                      move the cursor without clicking\n"
        "  dragTo(x=0.5, y=0.3, button='left')       drag to the point; button is one of 'left' | 'right' | 'middle'\n\n"
        "Scrolling — a single positional INTEGER (no decimals, no kwargs); positive = down/right, negative = up/left:\n"
        "  scroll(-3)                                vertical scroll\n"
        "  scrollUp(5) / scrollDown(5)               directional vertical scroll, positive integer\n"
        "  hscroll(2)                                horizontal scroll\n\n"
        "Keyboard:\n"
        "  write(message='hello world')              type literal text into the focused field (kwarg 'message'; do NOT use type())\n"
        "  press('enter')                            press a single key by name\n"
        "  press('a', presses=3)                     press a key multiple times (optional kwarg)\n"
        "  press(['tab', 'tab'])                     press a sequence of keys, in order\n"
        "  hotkey(['ctrl', 'c'])                     hold a chord of keys simultaneously\n\n"
        "Navigation / control:\n"
        "  navigate('https://example.com')           open a URL in the browser (positional string)\n"
        "  screenshot()                              capture the current screen\n"
        "  wait()                                    pause briefly for the UI to update\n"
        "  terminate(status='success')               end the task — use when complete\n"
        "  terminate(status='failure')               end the task — use when it cannot be completed\n\n"
        "## Worked examples (your reply should look like these — every reply, even if the history has bare actions)\n"
        "Example A — single click:\n"
        "<reason>The File menu is in the top-left of the toolbar. Clicking it opens the menu so I can pick Open Recent next.</reason> <answer>[{'description': 'Open the File menu in the top-left toolbar', 'action': \"click(x=0.04, y=0.03)\"}]</answer>\n\n"
        "Example B — keyboard shortcut:\n"
        "<reason>The document is focused and I need to save it. Ctrl+S is the standard save shortcut and avoids navigating menus.</reason> <answer>[{'description': 'Save the document with the Ctrl+S shortcut', 'action': \"hotkey(['ctrl', 's'])\"}]</answer>\n\n"
        "Example C — typing then submitting (two actions in one step):\n"
        "<reason>The address bar is already focused. I should type the URL, then press Enter to navigate. These two actions belong to a single step.</reason> <answer>[{'description': 'Type the target URL into the address bar', 'action': \"write(message='https://news.ycombinator.com')\"}, {'description': 'Submit the address bar to navigate to the page', 'action': \"press('enter')\"}]</answer>"
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
