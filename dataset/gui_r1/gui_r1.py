import os
from datasets import load_from_disk

def format_for_vlm(example, base_prompt):
    """
    Formats the baked row into a chat-template compatible structure.
    Adjusted to match your new schema (instruction, history, etc.)
    """
    # Build the text content using the instruction and history if available
    # You can customize this logic based on your specific planning strategy
    text_content = base_prompt.replace("[INSTRUCT]", example["instruction"])
    
    if example["history"] and example["history"] != "None":
        text_content = f"History: {example['history']}\n\n{text_content}"

    # Construct the standard multi-modal message format
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"}, # The VLM will swap this for the image tensor
                {"type": "text", "text": text_content},
            ],
        },
    ]

    return {
        "messages": messages,
        "image": example["image"], # This is loaded lazily from disk
        "label": example["gt_action"],
        "bbox": example["gt_bbox"]
    }

def get_baked_gui_dataset(path="./data/gui_r1_raw_test", base_prompt="[INSTRUCT]"):
    """
    Loads the memory-mapped dataset and prepares it for use.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Baked data not found at {path}. Run prepare.py first!")

    # 1. Load from disk (Instant, ~0 RAM overhead)
    dataset = load_from_disk(path)
    
    print(f"Dataset loaded. Total examples: {len(dataset)}")

    # 2. Apply formatting
    # We use .map() but since the dataset is local, it's very fast.
    # For inference testing, you might not even need .map(), you could just 
    # format on-the-fly to save even more memory.
    formatted_dataset = dataset.map(
        lambda x: format_for_vlm(x, base_prompt),
        remove_columns=dataset.column_names # Clean up raw columns
    )

    return formatted_dataset

if __name__ == "__main__":
    # Test it out
    PROMPT_TEMPLATE = (
        "You are a GUI agent. Goal: [INSTRUCT]\n"
        "Provide your reasoning in <REASONING> tags and action in <SOLUTION> tags."
    )
    
    DATA_PATH = "./data/gui_r1_raw_test"
    
    print("Initializing baked dataset...")
    ds = get_baked_gui_dataset(DATA_PATH, PROMPT_TEMPLATE)

    # Test random access (impossible with streaming, instant with baking!)
    print("\nAccessing example #42...")
    sample = ds[42]
    
    print(f"Action: {sample['label']}")
    print(f"BBox: {sample['bbox']}")
    print(f"Instruction: {sample['messages'][0]['content'][1]['text'][:100]}...")
    
    # Check image
    img = sample['image']
    print(f"Image mode: {img.mode}, Size: {img.size}")
    # img.show() # Uncomment to see the screenshot