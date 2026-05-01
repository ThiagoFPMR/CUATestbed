# OmniBrowse
Autonomous browsing agent.

```
source cua/bin/activate && python train.py \
  --experiment_name qwen3-8b-agentnet-v1 \
  --model_config qwen3-vl-8b-4bit.yml \
  --trainer_config gspo.yml \
  --rewards gui_r1
```