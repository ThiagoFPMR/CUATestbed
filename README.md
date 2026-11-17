# CUATestbed

An experimentation repo for **training small vision-language models as computer-use agents (CUAs) with off-policy RL**.

It's a sandbox for trying the recipes from recent RL-for-GUI-agents work on a compact model and a modest compute budget:

- **[GUI-R1](https://arxiv.org/abs/2504.10458) / [UI-R1](https://arxiv.org/abs/2503.21620)**: rule-based, verifiable rewards (format + action accuracy) with group-relative policy optimization, applied to single-step GUI action prediction.
- [**OpenCUA**](https://arxiv.org/abs/2508.09123): the AgentNet dataset of human-demonstrated desktop trajectories (Windows / Ubuntu / macOS), its `pyautogui` action space, and its layered history format (recent steps carry the screenshot and reasoning, older steps only the action).

The model is **Qwen3-VL (2B / 4B / 8B)**, loaded 4-bit through [Unsloth](https://github.com/unslothai/unsloth) and fine-tuned with LoRA. Training uses TRL's `GRPOTrainer` in [**GSPO**](https://arxiv.org/abs/2507.18071) mode (sequence-level importance sampling). Each AgentNet step becomes an independent one-step episode: history from the logged trajectory, the current screenshot and the goal go in, and the model's predicted actions are scored against the rest of the logged trajectory.

> This is research code, not a library. Expect hard-coded paths, cloud-specific launch scripts and ongoing changes.

---

## How it works

```
AgentNet trajectory ──► filter (alignment / redundant / incorrect steps)
                     ──► per-step chat sample: system prompt + history + screenshot + goal
                     ──► label = remaining ground-truth actions of the episode

Qwen3-VL (QLoRA) ──► G sampled completions per sample ──► rewards ──► GSPO update
```

**Output format.** The model must emit:

```
<observation>what's on screen</observation>
<reason>why the next action moves toward the goal</reason>
<answer>[{'description': 'Click the search box', 'action': 'click(x=0.31, y=0.12)'}, ...]</answer>
```

Pointer coordinates are normalized to `[0, 1]`, so the policy doesn't depend on screenshot resolution. The action vocabulary follows `pyautogui`: `click`, `doubleClick`, `tripleClick`, `rightClick`, `middleClick`, `moveTo`, `dragTo`, `scroll`, `hscroll`, `write`, `press`, `hotkey`, `wait`, `terminate`.

**Reward** (`--rewards gui_r1`), loosely following GUI-R1:

```
R = 0.6 · R_format + 0.4 · R_accuracy
```

- `R_format` is 1 only when the three tags are present and in order and every action parses against its regex signature.
- `R_accuracy` matches the N predicted actions against the next N ground-truth actions **in order**. Everything before the first mismatch scores, and everything after it scores zero. At the failure point, getting the action type right earns partial credit. The result is averaged over N, so long speculative action chains don't pay unless the whole chain is right.
- Tolerances: 8 px radius for pointer targets, 16 px for scroll amounts, and token-F1 ≥ 0.5 for typed text.

---

## A few findings worth noting

These are observations from running the recipe above on **Qwen3-VL-4B**. Experiments were run on small portions of AgentNet, never completing a full training run over the data, so they should be read simply as practical notes found empirically.

### 1. Instruct models learn the format almost immediately, so the format reward stops helping

Qwen3-VL is already heavily instruction-tuned. In one run, the format reward hit its 0.6 ceiling by about **step 80**. In another, it was already saturated at the first step. After that the total reward sat at around **0.74**, and almost all of that came from a component that no longer varies.

Since advantages are computed *relative to the group*, an identical reward term  across all samples drops out of the gradient. The GUI-R1-style 0.6 format weight therefore doesn't keep reinforcing anything. It shrinks the reward range left for accuracy, which over ~550 steps of that first run moved only from about 0.13 to 0.15. With an instruct base, consider:

- annealing the format weight toward accuracy once it saturates, or
- treating format as a **gate** (zero reward if malformed) rather than a large additive bonus.

### 2. You get grounding gains without a grounding objective

No training stage targets localization. The only pressure on coordinates comes from the 8 px tolerance in the accuracy reward. Even so, after seeing only **about 10% of AgentNet**, the model improved on [ScreenSpot-Pro](https://arxiv.org/abs/2504.07981):


| Qwen3-VL-4B      | Text     | Icon     | Overall  |
| ---------------- | -------- | -------- | -------- |
| Base             | 46.7     | 12.7     | 33.7     |
| + RL on AgentNet | **54.2** | **13.7** | **38.7** |


Most of the gain is on **text** targets. That fits Qwen's OCR-heavy pretraining: the policy gets better at matching "click *Save*" to a label it can already read.

### 3. Shorter completions mean less learning signal

With G = 4 and rewards that behave close to binary, many groups receive identical rewards for every sample and so produce no gradient. Over the course of one run:

- mean completion length fell from about **250 to 223 tokens**,
- completion length and within-group reward std were correlated at about **+0.65**,
- the fraction of zero-std groups climbed **above 90%**.

As the policy grows more confident and concise, its samples get more alike, and GSPO has less to learn from. Possible fixes include larger groups and denser rewards (for example, coordinate reward proportional to distance instead of a hard radius).

### 4. You don't need many completion tokens

With a 1024-token budget, completions settled at about **200 tokens**, and with a 512-token budget at about **220–250**. Neither came close to the cap, so **512** is plenty. A tighter cap saves a lot of time and VRAM, because HF `generate()` runs until the longest sequence in the batch finishes.

### 5. Single-step framing makes `terminate` a rare class, so the model doesn't learn to stop

When every step of a trajectory is its own sample, an episode of about 20 steps yields one `terminate` target. In the full action space, `terminate` is the correct answer in only **~5%** of training samples, which pushes the policy toward always continuing.

In a run with pointer/grounding actions removed from the action space, `terminate` rose to **~30%** of samples. There the model **did** learn a calibrated stopping policy. Late in training, it predicted `terminate` at roughly the true base rate, with **~71% recall and ~73% precision**, and held that for about 13k steps.

So the model is perfectly capable of learning when to stop, as long as it sees enough examples of it. When training on single steps, it's worth oversampling terminal steps or adding a trajectory-level signal that rewards stopping at the right time.

### 6. Other things to keep in mind

- **AgentNet's action distribution is very long-tailed.** After filtering there are about 248k `click`, 36k `moveTo`, 34k `write`, 19k `terminate`, 9k `hotkey` and 40 `middleClick`. Rare actions get very little gradient unless you resample.
- **Matching against a single demonstration underestimates the model.** GUIs often allow several valid routes (menu vs. shortcut), and the reward penalizes all but the logged one. A flat accuracy curve can mean "stopped learning" or "learned valid routes the reward can't see."
- **Data curation matters.** AgentNet includes redundant or incorrect steps and abandoned tasks. The loader keeps successful trajectories whole. For incomplete ones it applies an alignment-score cutoff, drops the final `terminate`, and drops steps flagged as redundant or incorrect.
- **It runs on consumer hardware.** Training fits on a single **RTX 4080 (16 GB)**: GSPO has no critic, and 4-bit QLoRA keeps memory down. Wall-clock time is the real constraint: on that card, one run took about 16 days to get through roughly half of a reduced-action subset of AgentNet.

---

## Repo layout

```
train.py                    entry point: model + dataset + rewards + trainer
prompts.py                  system prompt (action space, output format, examples)
callbacks.py                S3 mirroring, checkpoint retention, generation logging
utils.py                    run bookkeeping (saves configs/prompt alongside the run)

model/
  load_model.py             Unsloth FastVisionModel + LoRA from YAML
  configs/                  qwen3-vl-{2b,4b,8b}-4bit.yml

dataset/agentnet/
  agentnet.py               trajectory/step filtering, history construction, lazy image loading
  action_utils.py           ACTION_PATTERNS, pyautogui → label dict parsing
  stats.py                  dataset inspection / plotting helpers

policy/
  rewards/                  format, accuracy (sequential match), and logging reward funcs
  trainer/                  TRL trainer factory + configs (grpo.yml, gspo.yml)

download.py                 fetch + extract AgentNet from HF (or a reduced slice from S3)
merge.py                    merge a LoRA adapter into a standalone, vLLM-servable checkpoint

run_train.sh                in-container multi-GPU launcher (scales config to GPU count)
probe_per_device_bs.py      find the largest per-GPU batch that fits (driven by probe_sweep.sh)
scale_trainer_config.py     rescale batch / grad-accum / LR for N GPUs
prep_volume_s3.py           fill a network volume with AgentNet from an S3 mirror
backfill_retention_state.py recover checkpoint-retention state for older runs
Dockerfile                  training image
```

---

## Running it

```bash
python -m venv cua && source cua/bin/activate
pip install -r requirements.txt

# AgentNet (large: hundreds of GB extracted)
python download.py                     # → dataset/agentnet/data/

python train.py \
  --experiment_name qwen3-4b-agentnet \
  --model_config qwen3-vl-4b-4bit.yml \
  --trainer_config gspo.yml \
  --rewards gui_r1
```

Config names without a `/` resolve to `model/configs/` and `policy/trainer/configs/`. Add `--resume_from_checkpoint` to continue a run, and `--s3_uri s3://bucket/prefix` to mirror checkpoints off an ephemeral box.

### Exporting

```bash
python merge.py --run qwen3-4b-agentnet --checkpoint latest   # LoRA → full HF checkpoint
```

---

## Acknowledgements

Part of the compute used in testing was provided by the Model2AI lab (DCA/UFRN). 

The reward design follows GUI-R1 and UI-R1, the data and action space come from OpenCUA/AgentNet, and the base models are Qwen3-VL.
