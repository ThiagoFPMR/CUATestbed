"""
GUI-R1 Dataset Loader and Preprocessor
This module provides functions to load and preprocess the GUI-R1 dataset for multi-modal tasks.
"""

from datasets import load_dataset, Image


def make_conversation(
        example: dict, 
        base_prompt: str, 
        key_pairs: dict = {"[INSTRUCT]": "instruction"}
    ) -> dict:
    """
    Construct a multi-modal conversation prompt for the GUI-R1 dataset.
    """
    # The user's text prompt
    for key, value in key_pairs.items():
        text_content = base_prompt.replace(key, example[value])

    # Construct the prompt in the desired multi-modal format
    prompt = [
        {
            "role": "user",
            "content": [
                {"type": "image"},  # Placeholder for the image
                {"type": "text", "text": text_content},  # The text part of the prompt
            ],
        },
    ]

    return {
        "prompt": prompt, 
        "image": example["image"], 
        "gt_action": example["gt_action"], 
        "gt_bbox": example["gt_bbox"], 
        "gt_input_text": example["gt_input_text"]
    }

def load_gui_r1(
        base_prompt: str,
        split: str = "train", 
        resize_to: tuple = None,  
        key_pairs: dict = {"[INSTRUCT]": "instruction"},
        streaming: bool = True
    ) -> load_dataset:
    """
    Load and preprocess the GUI-R1 dataset for multi-modal tasks.

    Args:
        base_prompt (str): Base prompt template with placeholders for dynamic content.
        split (str, optional): Dataset split to load. Defaults to "train".
        resize_to (tuple, optional): Desired size to resize images to. Defaults to None.
        key_pairs (dict, optional): Mapping of placeholders to example keys. Defaults to {"[INSTRUCT]": "instruction"}.
        streaming (bool, optional): Whether to load the dataset in streaming mode. Defaults to True.
        
    Returns:
        load_dataset: The preprocessed GUI-R1 dataset.
    """
    dataset = load_dataset("ritzzai/GUI-R1", streaming=streaming, split=split)
    
    dataset = dataset.cast_column("image", Image())

    def preprocess_example(example):
        if resize_to is None:
            return make_conversation(example, base_prompt, key_pairs)
        
        example["image"] = example["image"].resize(resize_to)
        return make_conversation(example, base_prompt, key_pairs)

    dataset = dataset.map(preprocess_example)
    return dataset


if __name__ == "__main__":
    BASE_PROMPT = (
        "[INSTRUCT]\n"
        "Please provide detailed reasoning between <REASONING> and </REASONING>, "
        "followed by your final answer between <SOLUTION> and </SOLUTION>."
    )

    dataset = load_gui_r1(base_prompt=BASE_PROMPT, split="train")

    print("Dataset ready. Fetching first example...")

    first_example = next(iter(dataset))
    import pdb; pdb.set_trace()