"""
Utility functions.
"""

import os
import yaml
import shutil


def save_training_params(
        base_prompt: str,
        experiment_name: str, 
        model_config: str, 
        trainer_config: str,
        rewards: list[str]
) -> None:
    """
    Copy the model and trainer configuration files to the experiment's config directory.

    Args:
        experiment_name (str): Name of the training experiment.
        model_config (str): Path to the model configuration file.
        trainer_config (str): Path to the trainer configuration file.
        rewards (list[str]): List of reward function names.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))

    args_path, model_path = "", ""
    if not "/" in trainer_config:
        args_path = os.path.join(current_dir, "policy", "trainer", "configs")
    if not "/" in model_config:
        model_path = os.path.join(current_dir, "model", "configs")

    trainer_config_path = os.path.join(args_path, trainer_config)
    model_config_path = os.path.join(model_path, model_config)

    with open(trainer_config_path, "r") as f:
        config = yaml.safe_load(f)
    config["reward_functions"] = rewards

    os.makedirs(f"{experiment_name}/configs", exist_ok=True)
    shutil.copy(model_config_path, f"{experiment_name}/configs/{model_config_path.split(os.sep)[-1]}")

    with open(f"{experiment_name}/configs/{trainer_config_path.split(os.sep)[-1]}", "w") as f:
        yaml.safe_dump(config, f)

    with open(f"{experiment_name}/configs/prompt.txt", "w") as f:
        f.write(base_prompt)