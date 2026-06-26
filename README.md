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
WANDB_API_KEY=... AWS_ACCESS_KEY_ID=... AWS_SECRET_ACCESS_KEY=... \
docker run --gpus all -it --rm \
  -v omnibrowse-agentnet:/opt/OmniBrowse/dataset/agentnet/data \
  -e WANDB_API_KEY -e AWS_ACCESS_KEY_ID -e AWS_SECRET_ACCESS_KEY \
  -e AWS_DEFAULT_REGION \
  omnibrowse:latest ./run_train.sh
```

`run_train.sh` pulls the reduced AgentNet slice from S3 (via `python download.py
--reduced`) and then launches `train.py`. The dataset volume is optional but
recommended on expensive boxes so the pull only happens once. To rebuild the
reduced slice from a box that already has the full dataset:

```
python download.py --make-reduced --n-trajs 20
```
