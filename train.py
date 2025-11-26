"""
General training script for vision-language models using reinforcement learning.
"""

from safetensors import safe_open

from policy.rewards import *
from model import load_model_from_file
from policy.trainer import load_trainer_from_file

from dataset.math_vista import load_math_vista


model, tokenizer = load_model_from_file("qwen3-vl-8b-4bit.yml")

train_dataset = load_math_vista(tokenizer)
trainer = load_trainer_from_file("grpo.yml")(
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
