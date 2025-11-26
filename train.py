"""
General training script for vision-language models using reinforcement learning.
"""

import argparse
from safetensors import safe_open

from policy.rewards import *
from model import load_model_from_file
from policy.trainer import load_trainer_from_file

from dataset.math_vista import load_math_vista


def train_vlm_rl(model_config: str, trainer_config: str) -> None:
    """
    Train a vision-language model using reinforcement learning.

    Args:
        model_config (str): Path to the model configuration file. If does not contain /, it is assumed to be in the model configs directory.
        trainer_config (str): Path to the trainer configuration file. If does not contain /, it is assumed to be in the trainer configs directory.
        output_dir (str): Directory to save the trained model.
    """

    model, tokenizer = load_model_from_file(model_config)

    train_dataset = load_math_vista(tokenizer)

    trainer = load_trainer_from_file(trainer_config)(
        model = model,
        reward_funcs = [
            formatting_reward_func,
            correctness_reward_func,
        ],
        processing_class = tokenizer,
        train_dataset = train_dataset,
    )
    trainer.train()

    model.save_pretrained("grpo_lora")  # Local saving

    tensors = {}
    with safe_open("grpo_lora/adapter_model.safetensors", framework = "pt") as f:
        # Verify both A and B are non zero
        for key in f.keys():
            tensor = f.get_tensor(key)
            n_zeros = (tensor == 0).sum() / tensor.numel()
            assert(n_zeros.item() != tensor.numel())


if __name__ == "__main__":
    # Use argparse to get model and trainer config paths
    parser = argparse.ArgumentParser(description="Train a vision-language model using reinforcement learning.")
    parser.add_argument("--model_config", type=str, required=False, default="qwen3-vl-8b-4bit.yml", help="Path to the model configuration file.")
    parser.add_argument("--trainer_config", type=str, required=False, default="grpo.yml", help="Path to the trainer configuration file.")
    args = parser.parse_args()

    train_vlm_rl(args.model_config, args.trainer_config)