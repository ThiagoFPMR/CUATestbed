"""
Module to load training configuration for optimizer and training arguments.
"""

import os
import trl
import yaml
from typing import Callable
from transformers import Trainer, TrainingArguments


def load_args(config_obj: TrainingArguments, train_args: dict) -> TrainingArguments:
    """
    Load training arguments based on the provided configuration.
    
    Args:
        config_obj (TrainingArguments): The TrainingArguments class to instantiate.
        train_config (dict): A dictionary containing training configuration parameters.

    Returns:
        TrainingArguments: An instance of TrainingArguments populated with the provided configuration.
    """
    train_args['learning_rate'] = float(train_args['learning_rate'])
    training_args = config_obj(
        **train_args
    )

    return training_args

def load_trainer_from_file(args_path: str, **config_overrides) -> Callable:
    """
    Load trainer configuration from a YAML file and return a trainer loader function.

    Args:
        args_path (str): Path to the YAML file containing trainer configuration.

    Returns:
        Callable: A function that loads and returns the trainer instance when called.
    """
    
    # Use the config file path relative to this script if no path is provided
    if not "/" in args_path:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        args_path = os.path.join(current_dir, 'configs', args_path)

    with open(args_path, "r") as f:
        trainer_config = yaml.safe_load(f)

    trainer_config.update(config_overrides)

    def load_trainer(**kwargs) -> Trainer:
        """
        Load and return the trainer instance based on the configuration.

        Args:
            **kwargs: Additional keyword arguments to pass to the trainer constructor.
        """
        config_class = getattr(trl, trainer_config["config_type"])
        trainer_class = getattr(trl, trainer_config["trainer_type"])

        del trainer_config["config_type"]
        del trainer_config["trainer_type"]

        trainer_args = load_args(config_class, trainer_config)
        trainer = trainer_class(args=trainer_args, **kwargs)

        return trainer

    return load_trainer