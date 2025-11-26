"""
General training script for vision-language models using reinforcement learning.
"""

from policy.rewards import *
from model.load_model import load_model_from_file
from policy.load_train_args import load_args_from_file

from dataset.math_vista import load_math_vista

from safetensors import safe_open
from trl import GRPOConfig, GRPOTrainer

model, tokenizer = load_model_from_file("qwen3-vl-8b-4bit.yml")

train_dataset = load_math_vista()
train_dataset = train_dataset.map(
    lambda example: {
        "prompt": tokenizer.apply_chat_template(
            example["prompt"],
            tokenize = False,
            add_generation_prompt = True, # Must add assistant
        )
    }
)

trainer = GRPOTrainer(
    model = model,
    reward_funcs = [
        formatting_reward_func,
        correctness_reward_func,
    ],
    processing_class = tokenizer,
    args = load_args_from_file(GRPOConfig, "grpo.yml"),
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
