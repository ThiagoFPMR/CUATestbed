"""
General training script for vision-language models using reinforcement learning.
"""

import argparse
from safetensors import safe_open

from utils import save_training_params
from model import load_model_from_file
from policy.rewards import parse_rewards
from policy.trainer import load_trainer_from_file

from dataset import load_gui_r1


def train_vlm_rl(
        base_prompt: str,
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

    train_dataset = load_gui_r1(
        base_prompt,
        tokenizer
    )

    import pdb; pdb.set_trace()

    trainer = load_trainer_from_file(trainer_config)(
        model = model,
        reward_funcs = parse_rewards(rewards),
        processing_class = tokenizer,
        train_dataset = train_dataset,
    )

    import pdb; pdb.set_trace()
    trainer.train()
    print("🏁 Training completed successfully!")

    # Local saving
    model.save_pretrained(experiment_name) 
    tokenizer.save_pretrained(experiment_name)
    print(f"📁 Model and tokenizer saved to {experiment_name}!")

    # Save copies of the config files used to output directory
    save_training_params(
        base_prompt,
        experiment_name, 
        model_config, 
        trainer_config, 
        rewards
    )
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
    parser.add_argument("--rewards", type=str, nargs="+", required=False, default=["gui_r1"], help="List of reward function names.")
    args = parser.parse_args()

    # GUI-R1 base prompt
    base_prompt = (
        f"You are GUI-R1, a reasoning GUI Agent Assistant. In this UI screenshot <image>, I want you to continue executing the command '[INSTRUCT]', with the action history being '[HISTORY]'.\n"
            "Please provide the action to perform (enumerate from ['wait', 'long_press', 'click', 'press_back', 'type', 'open_app', 'scroll']), the point where the cursor is moved to (integer) if a click is performed, and any input text required to complete the action.\n"
            "Output the thinking process in <think> </think> tags, and the final answer in <answer> </answer> tags as follows:\n"
            "<think> ... </think> <answer>[{'action': enum['wait', 'long_press', 'click', 'press_back', 'type', 'open_app', 'scroll'], 'point': [x, y], 'input_text': 'no input text [default]'}]</answer>\n"
            "Note:\n specific input text (no default) is necessary for actions enum['type', 'open_app', 'scroll'] \n Example:\n"
            "[{'action': enum['wait', 'press_back'], 'point': [-100, -100], 'input_text': 'no input text'}]\n"
            "[{'action': enum['click', 'long_press'], 'point': [123, 300], 'input_text': 'no input text'}]\n"
            "[{'action': enum['type', 'open_app'], 'point': [-100, -100], 'input_text': 'shanghai shopping mall'}]\n"
            "[{'action': enum['scroll'], 'point': [-100, -100], 'input_text': enum['up', 'left', 'right', 'down']}]"
    )

    train_vlm_rl(base_prompt, args.experiment_name, args.model_config, args.trainer_config, args.rewards)