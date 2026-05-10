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
    return (
        f"<observation>The current screen state is observed.</observation> "
        f"<reason>Analyzing the UI to determine the correct action.</reason> "
        f"<answer>{repr(actions)}</answer>"
    )


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
        "## Conversation history format\n"
        "Earlier turns in the conversation are a log of what has already been done:\n"
        "- The turn immediately before the current screenshot shows that screenshot directly (as an image), followed by a bare action list.\n"
        "- All older turns show a text <observation> describing the screen state at that point, followed by a bare action list.\n"
        "This asymmetry is intentional: you get the full image only for the most recent prior step so that the conversation stays within the model's capacity. Use the older text observations to understand the trajectory up to that point.\n"
        "YOUR own reply must NOT use that bare format; it must follow the full format below.\n\n"
        "## Output format — your response must follow this exactly\n"
        "Begin with <observation>, then <reason>, then <answer>, and STOP. No preamble. No text after </answer>.\n\n"
        "Skeleton:\n"
        "<observation>OBSERVATION</observation> <reason>REASONING</reason> <answer>[{'description': 'DESCRIPTION', 'action': 'ACTION_CALL'}]</answer>\n\n"
        "Where:\n"
        "- OBSERVATION: a concise description of the current screen state — what application is open, what is visible, where the cursor/focus appears to be, any notable UI elements. Two to four sentences. This is your grounding step before reasoning.\n"
        "- REASONING: a structured analysis (see 'How to reason' below). Keep it focused — typically 3–8 sentences, longer only when the situation genuinely warrants it. Write in the first person.\n"
        "- The <answer> body is a Python list literal of dicts. Each dict MUST have BOTH keys:\n"
        "    'description' — a short natural-language summary of what this action does and why. Refer to the target by its visible name, label, or distinguishing features (shape, color, position) — NOT by coordinates. For window controls, identify them correctly: minimize is '—', maximize is '□', close is 'X'. If a UI element's name is in a non-English language, keep the original language.\n"
        "    'action'      — a single action call string from the table below, in the exact syntax shown.\n"
        "- You MAY emit multiple action dicts in the list if several actions should fire in sequence at this step (e.g. write then press enter, or consolidated repeated keypresses). Most steps are a single action — don't pad. Prefer consolidating repetitive keypresses using the 'presses' kwarg (e.g. press('backspace', presses=5)) over emitting five separate dicts.\n"
        "- Quoting: write each 'action' value as a DOUBLE-quoted Python string so any single quotes inside (e.g. press('enter')) are valid. Example: 'action': \"press('enter')\". Never write 'action': 'press('enter')' — that is a syntax error.\n"
        "- Coordinates x and y are NORMALIZED FLOATS in [0.0, 1.0] — never raw pixel values. Always include a leading 0 (write 0.5, not .5).\n"
        "- Use ONLY the actions in the table below, with their exact argument syntax. Do NOT invent actions like type(...), execute(...), or bare shell commands.\n\n"
        "## How to reason — what goes inside <reason>...</reason>\n"
        "Structure your reasoning around two phases. You do not need literal headers, but every reply should cover both:\n\n"
        "1. Progress assessment — look backward:\n"
        "   - What parts of the overall task are already done, based on the prior action log and the current screenshot?\n"
        "   - Did the previous action produce the expected result? Look for unexpected dialogs, error states, unchanged screens, or focus loss.\n"
        "   - If the previous action appears to have failed or led somewhere wrong, name the recovery step (dismiss the dialog, undo, retry with a different target, etc.) instead of plowing ahead.\n\n"
        "2. Next-action analysis — look forward:\n"
        "   - Briefly consider the plausible next actions given the current screen state.\n"
        "   - Pick the most logical one and say why it beats the alternatives (e.g. keyboard shortcut vs. menu navigation, direct click vs. scrolling first).\n"
        "   - Anticipate what the screen should look like after the action — this becomes the check on the next turn.\n\n"
        "Special case — text input actions:\n"
        "   - Note where the cursor currently is and whether the intended field is actually focused (don't assume).\n"
        "   - State the expected final text in the field after typing.\n"
        "   - Consolidate repeated keypresses with the 'presses' kwarg rather than repeating dicts.\n\n"
        "Special case — terminating:\n"
        "   - Only call terminate(status='success') when the screenshot itself confirms the task's end state. Do not terminate optimistically.\n"
        "   - Call terminate(status='failure') when you have exhausted reasonable approaches or the task is genuinely blocked, and explain why in the reasoning.\n\n"
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
        "  hscroll(2)                                horizontal scroll\n\n"
        "Keyboard:\n"
        "  write(message='hello world')              type literal text into the focused field (kwarg 'message'; do NOT use type())\n"
        "  press('enter')                            press a single key by name\n"
        "  press('a', presses=3)                     press a key multiple times (optional kwarg) — prefer this over repeating dicts\n"
        "  press(['tab', 'tab'])                     press a sequence of keys, in order\n"
        "  hotkey(['ctrl', 'c'])                     hold a chord of keys simultaneously\n\n"
        "Navigation / control:\n"
        "  wait()                                    pause briefly for the UI to update\n"
        "  terminate(status='success')               end the task — use when complete\n"
        "  terminate(status='failure')               end the task — use when it cannot be completed\n\n"
        "## Worked examples (your reply should look like these — every reply, even if the history has bare actions)\n"
        "Example A — single click, with progress + next-action reasoning:\n"
        "<observation>The screen shows a text editor application with a menu bar at the top. The File, Edit, View, and Help menus are visible. The document area below is empty and no menus are currently open. The cursor appears to be in the main document body.</observation> <reason>The history shows I haven't opened any menus yet, and the current screenshot confirms I'm at the application's main window. The File menu sits in the top-left of the toolbar, and I need it to reach 'Open Recent'. Alternatives — a keyboard shortcut like Ctrl+O — would skip the recent-files list, so the menu is the right path. After clicking I expect the File dropdown to appear, exposing 'Open Recent' as a submenu.</reason> <answer>[{'description': 'Open the File menu in the top-left of the menu bar', 'action': \"click(x=0.04, y=0.03)\"}]</answer>\n\n"
        "Example B — keyboard shortcut, recovering from a previous misstep:\n"
        "<observation>The screen shows a document editor with text content visible. The cursor is blinking inside the document body, indicating the editor has focus. The title bar shows the filename without a dirty indicator, but I have not yet saved the changes made in this session.</observation> <reason>The previous action clicked into the document body, which means the editor is now focused — good, that's the prerequisite for a save shortcut. The task is to save the file, and I haven't done that yet. Ctrl+S is the standard save shortcut and avoids menu navigation entirely. After pressing it I expect either a silent save (title bar dirty marker disappears) or a Save As dialog if the file is untitled.</reason> <answer>[{'description': 'Save the document with the Ctrl+S shortcut', 'action': \"hotkey(['ctrl', 's'])\"}]</answer>\n\n"
        "Example C — typing then submitting (two actions in one step), with a text-input plan:\n"
        "<observation>The browser address bar is highlighted and appears empty, indicating it has focus after the previous click. The current page content is still visible behind the selected bar. No dropdown suggestions are showing yet.</observation> <reason>The previous action clicked the address bar, and the screenshot confirms the cursor is now in it (the bar is highlighted and empty). I need to navigate to Hacker News, so I'll type the URL and then press Enter — these belong to a single step because separating them risks the bar losing focus. Expected end state: the address bar contains 'https://news.ycombinator.com' and the page begins loading.</reason> <answer>[{'description': 'Type the target URL into the focused address bar', 'action': \"write(message='https://news.ycombinator.com')\"}, {'description': 'Submit the address bar to navigate to the page', 'action': \"press('enter')\"}]</answer>\n\n"
        "Example D — consolidated repeated keypress:\n"
        "<observation>The screen shows a search interface with a text field that contains stale text 'old query'. The field appears focused based on the cursor position visible at the end of the text. No search results are currently displayed.</observation> <reason>The search field currently contains stale text 'old query' (10 characters) and is focused, per the prior action. I need to clear it before typing the new query. Pressing Backspace 10 times in one consolidated action is cleaner than ten separate dicts, and selecting-all-then-typing would also work but is no shorter here. After this the field should be empty and ready for new input.</reason> <answer>[{'description': 'Clear the search field by pressing Backspace ten times', 'action': \"press('backspace', presses=10)\"}]</answer>"
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
        for msg in sample["prompt"]:
            for content in msg.get("content", []):
                if isinstance(content, dict) and content.get("type") == "image":
                    images.append(content["image"])

        text = tokenizer.apply_chat_template(sample["prompt"], tokenize=False, add_generation_prompt=True)
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
