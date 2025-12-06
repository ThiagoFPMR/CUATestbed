"""
Module to load a FastVisionModel and its tokenizer based on a configuration file.
"""
import unsloth
from unsloth import FastVisionModel
from transformers import PreTrainedTokenizerBase
from peft import TaskType
import yaml
import os


def load_model_from_config(model_config: dict) -> tuple[FastVisionModel, PreTrainedTokenizerBase]:
    """Load the model and tokenizer based on the provided configuration."""

    quantization_config = model_config["quantization"]
    finetuning_config = model_config["finetuning"]

    qat_scheme = None
    if quantization_config["qat_scheme"] != "none":
        qat_scheme = quantization_config["qat_scheme"]

    model, tokenizer = FastVisionModel.from_pretrained(
        model_name = model_config["base_model"],
        max_seq_length = quantization_config["max_seq_length"],
        load_in_4bit = quantization_config["fp_precision"] == 4,
        load_in_8bit = quantization_config["fp_precision"] == 8,
        load_in_16bit = quantization_config["fp_precision"] == 16,
        full_finetuning = quantization_config["full_finetuning"],
        qat_scheme = qat_scheme,
        fast_inference = quantization_config["use_vllm"],
        gpu_memory_utilization = quantization_config["gpu_memory_utilization"],
    )

    task_type = finetuning_config["task_type"]
    if task_type == "causal_lm":
        task_type = TaskType.CAUSAL_LM
    elif task_type == "seq2seq_lm":
        task_type = TaskType.SEQ_2_SEQ_LM
    elif task_type == "token_cls":
        task_type = TaskType.TOKEN_CLS
    elif task_type == "seq_cls":
        task_type = TaskType.SEQ_CLS
    elif task_type == "question_ans":
        task_type = TaskType.QUESTION_ANS
    elif task_type == "feature_extraction":
        task_type = TaskType.FEATURE_EXTRACTION
    finetuning_config["task_type"] = task_type

    if finetuning_config["loftq_config"]["loftq_iter"] == 0:
        del finetuning_config["loftq_config"]

    model = FastVisionModel.get_peft_model(
        model,
        **finetuning_config
    )

    return model, tokenizer

def load_model_from_file(config_path: str) -> tuple[FastVisionModel, object]:
    """Load the model and tokenizer based on the provided configuration file path."""
    
    # Use the config file path relative to this script if no path is provided
    if not "/" in config_path:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, 'configs', config_path)

    with open(config_path, "r") as f:
        model_config = yaml.safe_load(f)

    return load_model_from_config(model_config)