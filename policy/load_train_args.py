"""
Module to load training configuration for optimizer and training arguments.
"""

import os
import yaml
from transformers import TrainingArguments


def load_args(config_obj: TrainingArguments, train_args: dict) -> TrainingArguments:
    """
    Load training arguments based on the provided configuration.
    
    Args:
        config_obj (TrainingArguments): The TrainingArguments class to instantiate.
        train_config (dict): A dictionary containing training configuration parameters.

    Returns:
        TrainingArguments: An instance of TrainingArguments populated with the provided configuration.
    """

    train_args["optim"] = train_args["optimizer"]
    train_args["lr_scheduler_type"] = train_args["lr_scheduler"]
    del train_args["lr_scheduler"]
    del train_args["optimizer"]

    training_args = config_obj(
        **train_args
    )

    return training_args

def load_args_from_file(config_obj: TrainingArguments, args_path: str) -> TrainingArguments:
    """
    Load training arguments from a configuration file.

    Args:
        config_obj (TrainingArguments): The TrainingArguments class to instantiate.
        config_path (str): Path to the YAML configuration file.

    Returns:
        TrainingArguments: An instance of TrainingArguments populated with the provided configuration.
    """
    
    # Use the config file path relative to this script if no path is provided
    if not "/" in args_path:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        args_path = os.path.join(current_dir, 'configs', args_path)

    with open(args_path, "r") as f:
        model_config = yaml.safe_load(f)

    return load_args(config_obj, model_config)