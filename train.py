"""
General training script for vision-language models using reinforcement learning.
"""

import argparse
from safetensors import safe_open

from utils import save_training_params
from model import load_model_from_file
from policy.rewards import parse_rewards
from policy.trainer import load_trainer_from_file

from dataset.math_vista import load_math_vista


def train_vlm_rl(
        experiment_name: str, 
        model_config: str, 
        trainer_config: str,
        rewards: list[str]
) -> None:
    """
    Train a vision-language model using reinforcement learning.

    Args:
        experiment_name (str): Name of the training experiment.
        model_config (str): Path to the model configuration file. If does not contain /, it is assumed to be in the model configs directory.
        trainer_config (str): Path to the trainer configuration file. If does not contain /, it is assumed to be in the trainer configs directory.
        output_dir (str): Directory to save the trained model.
    """

    model, tokenizer = load_model_from_file(model_config)

    train_dataset = load_math_vista(tokenizer)

    trainer = load_trainer_from_file(trainer_config)(
        model = model,
        reward_funcs = parse_rewards(rewards),
        processing_class = tokenizer,
        train_dataset = train_dataset,
    )
    trainer.train()
    print("🏁 Training completed successfully!")

    # Local saving
    model.save_pretrained(experiment_name) 
    tokenizer.save_pretrained(experiment_name)
    print(f"📁 Model and tokenizer saved to {experiment_name}!")

    # Save copies of the config files used to output directory
    save_training_params(experiment_name, model_config, trainer_config, rewards)
    print("💾 Saved copies of config files.")

    # Verify that the LoRA model was saved correctly    
    with safe_open(f"{experiment_name}/adapter_model.safetensors", framework = "pt") as f:
        tensors = {}
        for key in f.keys():
            tensor = f.get_tensor(key)
            n_zeros = (tensor == 0).sum() / tensor.numel()
            assert(n_zeros.item() != tensor.numel())
    print("✅ LoRA model saved and verified successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a vision-language model using reinforcement learning. Output directory is specified in the trainer config.")
    parser.add_argument("--experiment_name", type=str, required=False, default="vlm_rl_experiment", help="Name of the training experiment.")
    parser.add_argument("--model_config", type=str, required=False, default="qwen3-vl-2b-4bit.yml", help="Path to the model configuration file.")
    parser.add_argument("--trainer_config", type=str, required=False, default="gspo.yml", help="Path to the trainer configuration file.")
    parser.add_argument("--rewards", type=str, nargs="+", required=False, default=["formatting", "correctness"], help="List of reward function names.")
    args = parser.parse_args()

    train_vlm_rl(args.experiment_name, args.model_config, args.trainer_config, args.rewards)