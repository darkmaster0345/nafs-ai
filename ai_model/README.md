# Nafs AI — Standalone Brain Folder

This folder contains **fully self-contained** Adam and Eve brain checkpoints.
You can copy this folder anywhere and run it **without the rest of the project**.

## File Format

All model files are **PyTorch `.pt` checkpoints** (a Python pickled dict).

```
adam.pt   /  eve.pt
├── model_state_dict   ← BabyBrain (PPO+GRU) weights — 348,169 params
├── thinker_state      ← ThoughtTransformer weights — 539,935 params
├── agent              ← "Adam" / "Eve"
├── project            ← "Nafs AI"
├── version            ← "v0.3"
├── arch               ← "PPO+GRU + ThoughtTransformer"
├── input_dim          ← 21
├── num_actions        ← 8
├── action_names       ← ["EXPLORE","EAT","DRINK","SLEEP","HIDE","MOVE","FLEE","IDLE"]
├── starting_vocab     ← 23 primitive sensation words (hot, cold, pain, ...)
├── tick               ← Tick at which checkpoint was saved (if trained)
├── source_checkpoint  ← Original filename inside the parent project
└── note               ← Free-text note
```

To inspect a checkpoint without running it:

```python
import torch
ckpt = torch.load("adam.pt", weights_only=False)
print(ckpt.keys())
print(ckpt["agent"], ckpt.get("tick"), ckpt.get("source_checkpoint"))
```

## Files in this folder

| File | Purpose |
|---|---|
| `nafs_brain.py` | Standalone model definitions + tokenizer + loader (no project deps) |
| `adam.pt`       | Adam's brain checkpoint (PyTorch state_dict) |
| `eve.pt`        | Eve's brain checkpoint  (PyTorch state_dict) |
| `run_adam.py`   | Standalone runner — loads Adam, runs inference on a 21-dim sensory vector |
| `run_eve.py`    | Standalone runner for Eve |
| `README.md`     | This file |

## How to run WITHOUT the project

```bash
# 1. Copy this folder anywhere
cp -r ai_model /tmp/my_nafs

# 2. Install only torch (no other deps)
pip install torch

# 3. Run Adam on a random sensory vector
cd /tmp/my_nafs
python run_adam.py

# 4. Run Eve on a specific 21-dim sensory vector
python run_eve.py 0.5 0.2 0.1 0.0 0.3 0.8 0.1 0.0 0.0 0.5 0.5 0.5 0.5 0.2 0.1 0.9 0.0 0.1 0.5 0.5 0.5

# 5. Interactive REPL (type 21 comma-separated numbers, get thoughts back)
python run_adam.py --interactive
```

## Architecture (per agent, ~888K params total)

| Component | Purpose | Params |
|---|---|---|
| `BabyBrain` (PPO + GRU)            | Action selection + value estimation | 348,169 |
| `ThoughtTransformer` (char-level)  | Generates inner-voice thought string | 539,935 |

**Total per agent: ~888K params.**

Both Adam and Eve use the **identical architecture** — they are perfectly symmetric.
The only difference is which weights are inside the `.pt` file (different lived
experience → different learned brain).

## What is a "sensory vector"?

A 21-dimensional float vector describing the agent's immediate experience:

```
[0]   health          [8]  near_water    [16] wet
[1]   hunger          [9]  near_danger   [17] dry
[2]   energy         [10]  biome_temp    [18] loud
[3]   temperature    [11]  biome_food    [19] quiet
[4]   pain            [12] biome_water   [20] time_of_day
[5]   pleasure        [13] biome_danger
[6]   fear            [14] weather_id
[7]   near_food       [15] light_level
```

All values are normalized to roughly [0, 1] or [-1, 1].

## Notes

- If the checkpoint was generated from a fresh (untrained) brain, the model
  will produce **random outputs**. That is intentional — the agent has zero
  experience. Train via `train_multi_agent.py` in the parent project to
  produce a brain with learned weights.
- The `ThoughtTransformer` starts untrained in fresh checkpoints. Until the
  parent sim has run enough ticks for `LearnedThinker.is_ready()` to return
  True, the generated thought will be near-random character noise.
- Memory.json, lineage.db, and other agent-specific state are NOT included
  here. This folder contains ONLY the model weights.
