"""
Load and preprocess the MathVista dataset.
"""

from unsloth import FastVisionModel
from transformers import PreTrainedTokenizerBase
from datasets import Dataset, load_dataset
from .utils import resize_images, convert_to_rgb, make_conversation, is_numeric_answer


def load_math_vista(tokenizer: PreTrainedTokenizerBase) -> Dataset:
    """
    Load and preprocess the MathVista dataset for training.
    """

    dataset = load_dataset("AI4Math/MathVista", split = "testmini")
    dataset = dataset.filter(is_numeric_answer)
    dataset = dataset.map(resize_images)
    dataset = dataset.map(convert_to_rgb)
    train_dataset = dataset.map(make_conversation)
    train_dataset = train_dataset.remove_columns("image")
    train_dataset = train_dataset.rename_column("decoded_image", "image")

    train_dataset = train_dataset.map(
        lambda example: {
            "prompt": tokenizer.apply_chat_template(
                example["prompt"],
                tokenize = False,
                add_generation_prompt = True, # Must add assistant
            )
        }
    )

    return train_dataset