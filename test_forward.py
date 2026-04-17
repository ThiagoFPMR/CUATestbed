import torch
from model import load_model_from_file
from dataset.agentnet.agentnet import load_agentnet
from policy.rewards.parse_rewards import parse_rewards

def test_forward_pass():
    print("Loading model and tokenizer...")
    model_config = "qwen3-vl-2b-4bit.yml"  # Adjust as needed
    model, tokenizer = load_model_from_file(model_config)
    
    # Use GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("Loading dataset...")
    # Using the same base_prompt from train.py
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
    sample = dataset[0]  # Take the first example
    
    # The new GRPO trainer uses dict completions and list of batch objects
    # This varies depending on exact Unsloth/HuggingFace implementation, but we'll try a manual inference:
    print("Formatting inputs...")
    # Extract images from messages to pass to the tokenizer/processor
    images = []
    for msg in sample["messages"]:
        for content in msg.get("content", []):
            if isinstance(content, dict) and content.get("type") == "image":
                images.append(content["image"])

    # AgentNet already returns a "messages" field natively formatted for VLM
    # and populated with images (via load_batch_images)
    text = tokenizer.apply_chat_template(sample["messages"], tokenize=False, add_generation_prompt=True)
    
    # Note: image processing might be needed if Qwen requires it via processor
    # But assuming unsloth's FastVisionModel handles it if we pass properly:
    # (Just testing text/dummy generation in case images are handled by processor instead of tokenizer)
    print("Tokenizing...")
    # Pass text explicitly as a kwarg and provide the images!
    inputs = tokenizer(text=[text], images=images, return_tensors="pt", padding=True).to(device)
    
    print("Running forward pass (Generation)...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    completion_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    completion_text = tokenizer.decode(completion_tokens, skip_special_tokens=True)
    print("\n--- GENERATED COMPLETION ---")
    print(completion_text)
    print("----------------------------\n")
    
    print("Testing Reward Calculation...")
    reward_funcs = parse_rewards(["gui_r1"])
    gui_reward = reward_funcs[0]
    
    # Provide the completion, mock label and image sizes depending on what your reward func expects
    # Action/format rewards usually expect a batch:
    try:
        # For AgentNet, it provides `labels` and `image_sizes` directly.
        completions = [[{"role": "assistant", "content": completion_text}]]
        labels = [sample.get("labels", "")] 
        img_sizes = sample.get("image_sizes", [(100, 100)]) # expects tuple or list
        if not isinstance(img_sizes, list): img_sizes = [img_sizes]
        
        rewards = gui_reward(completions, labels, img_sizes)
        print(f"Calculated Reward: {rewards}")
    except Exception as e:
        raise e
        print(f"Reward calculation error: {e}")

if __name__ == "__main__":
    test_forward_pass()