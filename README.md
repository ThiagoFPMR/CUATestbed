# OmniBrowse
Autonomous browsing agent.

```
source cua/bin/activate && python train.py \
  --experiment_name qwen3-8b-agentnet-v1 \
  --model_config qwen3-vl-8b-4bit.yml \
  --trainer_config gspo.yml \
  --rewards gui_r1
```

```
eval "$(ssh-agent -s)" && ssh-add ~/.ssh/id_ed25519
docker build --ssh default -t omnibrowse:latest .
docker run --gpus all -it --rm \
  -v /fast/local/agentnet:/workspace/OmniBrowse/dataset/agentnet/data \
  -e WANDB_API_KEY=... omnibrowse:latest \
  python train.py --experiment_name io_test --model_config qwen3-vl-8b-4bit.yml \
    --trainer_config gspo.yml --rewards gui_r1 --s3_uri s3://your-bucket/runs
```
