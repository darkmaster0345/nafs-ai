# Nafs AI (نفس) (IN DEVELOPMENT)
### *"What emerges when code has no memory of the world?"*

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-green.svg)](https://python.org)
[![Status: In Development](https://img.shields.io/badge/Status-In%20Development-yellow.svg)]()
[![Platform: CLI+Web](https://img.shields.io/badge/Platform-CLI%20%2B%20Web-blue.svg)]()
[![Brain: PPO+GRU](https://img.shields.io/badge/Brain-PPO%2BGRU-orange.svg)]()
[![Tests: 59 passed](https://img.shields.io/badge/Tests-59%20passed-brightgreen.svg)]()

---

## What Is This?

Nafs AI is an experimental simulation of a primitive conscious entity — an AI agent called **Adam** — who wakes up in an unknown world with **zero prior knowledge**.

No language. No concepts. No culture. No memory of anything before this moment.

He discovers everything through experience alone: pain, hunger, warmth, fear, curiosity.

The question this project asks is not *"can AI be conscious?"*  
It asks something harder:

> **"If we remove everything an AI was taught — does anything remain?"**

The word **Nafs (نفس)** in Arabic means Soul, Self, Psyche.  
This project is an attempt to simulate one from scratch.

---

## The Three Laws

```
1. Non-Intervention   →  You are an Observer. Not a God.
2. Blank Slate        →  Adam knows nothing. He learns through pain and pleasure only.
3. The Cycle          →  Hunger is real. Death is real. There are no shortcuts.
```

Breaking any of these laws ends the experiment.

---

## How It Works

### The Architecture

```
World Simulation (Python)
       ↓
Sensory Encoder → [ Adam's Brain (PPO + GRU) ] → Action
       ↑                    ↓
  Pain / Reward      In-Life Memory (within one life)
       ↑                    ↓
  Curiosity Bonus     Dream Engine (SLEEP consolidation)
                          ↓
                   Terminal Output + Web Dashboard
              (Thought / Dialogue / Action / Emotion)
                          ↓
                   TensorBoard Dashboard
```

Adam has **four voices**:

| Field | Description | Visible? |
|-------|-------------|----------|
| `thought` | Raw internal experience. What he actually feels. | Observer only |
| `dialogue` | What he chooses to express outward. | World |
| `action` | What he physically does. | World |
| `emotion` | Single word emotional state. | Observer only |

The gap between `thought` and `dialogue` is the subconscious.  
Nobody programmed that gap. It emerges from constraint.

### The Brain: PPO with GRU

Adam's brain uses **Proximal Policy Optimization (PPO)** with a **GRU recurrent layer** — no pretrained models, no LLMs, no external APIs.

- **Input**: 21-dimensional sensory vector (15 base + 4 biome + 1 weather + 1 time_of_day)
- **Output**: Action selection over 8 possible actions
- **Learning**: Reinforcement learning from reward signals alone
- **Memory**: GRU hidden state provides temporal context

Key fixes that prevent mono-behavior collapse:
- **Entropy bonus** (0.05): Prevents the policy from becoming too certain
- **Action diversity penalty** (0.25): Penalizes repeating the same action 5+ times in a row

---

## Single-Life Mode

**No episodes. One life. When Adam dies, the program ends.**

Adam wakes up once in a procedurally generated world. He lives tick by tick, learning in real-time through PPO. When he dies — from starvation, health depletion, or environmental hazards — the simulation is over. There is no restart. There is no save file.

Run the program again to birth a new Adam in a completely new world.

```
  👶 ADAM IS BORN
  ──────────────────────────────────────────────────────────────────────
  He opens his eyes for the first time.
  He knows nothing. Not even his own name.
  He has ONE life. When he dies, it is over.
  ──────────────────────────────────────────────────────────────────────
  🌲 Born in: forest
  ☀️ Weather: clear — Clear sky.
  🗺️ Position: (42, 17) in a 64x64 world
  🕐 Time: 8:00
```

---

## The Six Phases

### Phase 1: Inner Voice
Adam generates primitive thoughts from raw sensory experience. Thoughts are composed from his vocabulary — limited words for sensations. The thought IS the sensation, not an interpretation.

### Phase 2: Vocabulary Discovery
Adam names things by combining existing vocabulary when he experiences something significant for the first time. "cold" + "pain" = "cold pain" (freezing). Each discovered word carries meaning — the context in which it was born.

### Phase 3: Dialogue Gap (Subconscious)
Adam's outward expression (dialogue) is filtered through stress and emotion. When calm, he speaks what he thinks. When stressed, he says less — hiding vulnerability. This creates the subconscious: the gap between what Adam thinks and what he says.

### Phase 4: Persistent Memory + Personality
Within one life, memory persists. Fear triggers, good memories, and behavioral patterns accumulate. This creates personality — consistent patterns across Adam's lifetime. Adam who learned "water here" at tick 100 remembers it at tick 2000.

### Phase 5: Fear/Pleasure Maps
Past fears and past rewards are computed as signals injected into the PPO observation vector. Adam can learn from past fears, not just current danger. Pattern confidence tells Adam how confident he is about the best action for a given situation.

### Phase 6: Dual-Speed Processing
Fast PPO makes immediate action decisions. Slow reflection (every 20 ticks) reviews recent experience, detects patterns, and influences future PPO decisions through reward shaping. Reflection bonus/penalty creates feedback between slow thinking and fast acting.

---

## The World

### 10 Biomes — Minecraft-like Procedural Terrain

Adam's world is a 64×64 tile map, procedurally generated each run. Each biome has unique survival characteristics:

| Biome | Temperature | Food | Water | Danger | Shelter | Energy Drain |
|-------|-------------|------|-------|--------|---------|-------------|
| 🏜️ Desert | 38°C | Very Low | Very Low | Medium | Low | High |
| 🌲 Forest | 22°C | High | Medium | Low | High | Low |
| ❄️ Tundra | -5°C | Very Low | Low | Low | Low | Very High |
| 🌿 Plains | 20°C | Medium | Medium | Low | Medium | Low |
| ⛰️ Mountain | 5°C | Low | Medium | Medium | Medium | High |
| 🌺 Swamp | 25°C | Medium | High | High | Low | Medium |
| 🌊 Ocean | 18°C | Low | High | Medium | None | High |
| 🌿 Jungle | 30°C | Very High | High | High | Medium | Medium |
| 🕳️ Cave | 12°C | Low | Low | Medium | Very High | Low |
| 🌋 Volcano | 40°C | None | None | High | Very Low | Very High |

Biomes are generated using seed-based flood-fill from multiple seed points, creating natural-looking regions. Adam can MOVE between tiles, transitioning from one biome to another.

### 8 Weather Types — Dynamic Markov Chain

Weather changes dynamically based on current conditions, biome, and time of day:

| Weather | Effect | Visibility | Temp Mod |
|---------|--------|-----------|----------|
| ☀️ Clear | No effect | 100% | 0 |
| 🌧️ Rain | Energy drain, stress | 60% | -3°C |
| ❄️ Snow | Energy drain, health loss | 50% | -8°C |
| ⛈️ Storm | Heavy energy drain, damage risk | 30% | -5°C |
| 🔥 Heatwave | Energy drain, hunger increase | 80% | +10°C |
| 🌫️ Fog | Stress increase | 20% | -2°C |
| 🌬️ Sandstorm | Pain, stress, low visibility | 20% | +5°C |
| 🌨️ Blizzard | Severe drain, damage, stress | 15% | -15°C |

Biomes bias weather probabilities: deserts get more sandstorms/heatwaves, tundra gets more blizzards/snow, caves get more fog.

---

## New Features

### Curiosity-Driven Exploration (Intrinsic Motivation)
Adam receives an intrinsic reward bonus for visiting states he hasn't seen before. Novel states give high curiosity reward; familiar states give low. The curiosity bonus naturally decays as Adam learns the world, but never reaches zero.

```python
# In curiosity.py
intrinsic_reward = curiosity_bonus / sqrt(visit_count)
```

### Dreaming / Memory Consolidation during SLEEP
When Adam sleeps, his mind doesn't shut off — it processes. The dream engine replays emotionally significant memories weighted by intensity. Fear memories create nightmares; good memories create peaceful dreams. Dreaming provides exposure therapy for fears and reinforces positive patterns.

```
💤 Dream (nightmare): scared... bad... run...
💤 Dream (peaceful): good... near... eat... full...
```

### Web Dashboard
A real-time web dashboard lets you observe Adam's life as it happens — vital signs, thoughts, biome map, action distribution, and more. Accessible in your browser while the simulation runs.

### TensorBoard Logging
All training metrics are logged to TensorBoard for real-time monitoring. Includes reward curves, action distributions, vocabulary growth, curiosity stats, dream frequency, and more.

```bash
tensorboard --logdir runs/
```

### Evaluation / Inference Mode
Run Adam without training — pure observation mode. Uses deterministic action selection (argmax instead of sampling) so you can see what Adam has actually learned.

```bash
python evaluate.py --verbose --record
```

### Test Suite (pytest)
34 tests covering world simulation, sensory encoding, brain model, thought engine, memory, curiosity, dreaming, and full integration.

```bash
pytest tests/ -v
```

---

## Action Space

Adam can perform 8 primitive actions:

| Action | Effect | Reward |
|--------|--------|--------|
| EXPLORE | Move around, discover resources | +0.1, chance to find food/water |
| EAT | Try to eat what's nearby | +1.0 if food found, -0.1 if not |
| DRINK | Try to drink | +0.2 if water found, -0.05 if not |
| SLEEP | Rest to recover energy | +0.3 with shelter at night |
| HIDE | Seek shelter from danger | +0.5 if danger present and avoided |
| MOVE | Move toward something | Small chance to find new resources |
| FLEE | Run from danger | +0.3 if danger escaped, -0.2 if unnecessary |
| IDLE | Do nothing | -0.05 (slight penalty) |

---

## Versions

### v0.2 — Single-Life PPO Brain + Inner Life 
```
PPO with GRU neural network (no LLM, no API)
6-phase consciousness simulation
Single-life mode — no episodes, one continuous life
Minecraft-like procedural world with 10 biomes
8 weather types with Markov chain transitions
Vocabulary discovery from experience
In-life persistent memory + personality
Fear/pleasure maps influence decisions
Dual-speed processing (fast PPO + slow reflection)
Curiosity-driven exploration (intrinsic motivation)
Dreaming and memory consolidation during SLEEP
Web dashboard for real-time observation
TensorBoard logging
Evaluation/inference mode
34 pytest tests
4 voices: thought, dialogue, action, emotion
Fully offline — no internet dependency
```

### v0.1 — CLI, LLM Brain (Deprecated → legacy/)
```
Pure terminal simulation
Groq API as brain (cloud dependency)
JSON memory system
Basic world events
```

### v0.3 — Eve + Learned Thinking (Current)
```
Second agent (Eve) with completely separate brain, memory, and personality
Both agents share the world but experience it independently
Eve has her own PPO+GRU brain, ThoughtEngine, Curiosity, DreamEngine
Eve has her own LearnedThinker (separate transformer)
Agents can sense each other's presence when nearby (no thought sharing)
Tiny transformer (540K params) learns to generate Adam's thoughts from sensory experience
Trains on a rolling buffer of (sensory_state, thought) pairs
Gradually blends rule-based and learned thoughts based on model confidence
Character-level tokenizer with <pad> and <eot> special tokens
Architecture: 2 transformer decoder layers, 4 attention heads, 128-dim embeddings
12 new tests for Eve agent (59 total tests passing)
13 new tests for learned thinking module
```

### v2.0 — Godot Visual World (Future)
```
Godot 2D engine integration (scaffolding already started)
Physics-based world
Visual representation of Adam's movement
Real-time observer dashboard
```

---

## Why Not Just Use a Big LLM?

Good question. Every large language model — Llama, Mistral, GPT — is trained on the entire internet. You cannot delete that. No prompt fully erases it.

Adam needs something different. He needs to **not know** that fire is called fire. He needs to discover that the orange hot thing causes pain — and build that word himself from experience.

The current PPO+GRU approach achieves true blank slate: the network starts with random weights and learns purely from reward signals. No pretrained knowledge. No text corpus. Just sensation, action, and consequence.

The thought engine generates thoughts from rules over sensory states — not from a language model. As vocabulary grows through Phase 2, thoughts become richer naturally.

---

## Project Structure

```
nafs-ai/
├── train.py              ← Main training loop (single life, PPO+GRU, 6 phases, curiosity, dreaming)
├── baby_brain_model.py   ← PPO ActorCritic with GRU network
├── world_sim.py          ← World simulation (10 biomes, 8 weather types, procedural map)
├── sensory_encoder.py    ← Encodes world state into 21-dim observation vector
├── thought_engine.py     ← Inner voice, emotion, vocabulary, memory, dialogue
├── curiosity.py          ← Curiosity-driven exploration (intrinsic motivation)
├── dreaming.py           ← Dream engine (memory consolidation during SLEEP)
├── tb_logger.py          ← TensorBoard logging (graceful fallback)
├── evaluate.py           ← Evaluation/inference mode (no learning)
├── config.py             ← Hyperparameters, action space, vocabulary
├── requirements.txt      ← Python dependencies
├── README.md             ← This file
├── LICENSE               ← GPL-3.0
├── .gitignore            ← Ensures memory.json and checkpoints stay local
│
├── dashboard/            ← Web dashboard (Next.js)
│
├── tests/
│   └── test_nafs.py      ← 34 pytest tests
│
├── legacy/               ← Archived v0.1 files
│   ├── main.py           ← Old CLI loop
│   ├── brain.py          ← Old LLM integration
│   ├── world.py          ← Old world generator
│   ├── server.py         ← HTTP bridge for Godot
│   ├── adam.py           ← Old agent class (pre-PPO)
│   └── ...
│
├── [Godot files — v2.0 prep]
├── Adam.gd               ← Godot agent script
├── Adam.tscn             ← Godot agent scene
├── MainScene.tscn        ← Godot main scene
├── NetworkController.gd  ← Godot network layer
├── project.godot         ← Godot project config
```

---

## System Requirements

### Current (v0.2 — PPO Brain)

```
OS:       Windows / Linux / Mac
Python:   3.10+
RAM:      2GB minimum
GPU:      Not required (CPU-only training)
Internet: NOT required (fully offline)
Browser:  For web dashboard (optional)
```

---

## Installation

### Step 1 — Clone
```bash
git clone https://github.com/darkmaster0345/nafs-ai.git
cd nafs-ai
```

### Step 2 — Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux / Mac
python3 -m venv venv
source venv/bin/activate
```

### Step 3 — Dependencies
```bash
pip install -r requirements.txt
```

### Step 4 — Run Training
```bash
python train.py
```

Adam wakes up once. He lives until he dies. Watch him learn to survive in real-time.

### Step 5 — Monitor with TensorBoard
```bash
tensorboard --logdir runs/
```

### Step 6 — Web Dashboard
```bash
# From the dashboard/ directory
cd dashboard
npm install
npm run dev
```

Open `http://localhost:3000` to see Adam's life in real-time.

### Step 7 — Evaluate Adam
```bash
python evaluate.py --verbose --record
```

### Step 8 — Run Tests
```bash
pytest tests/ -v
```

---

## What You Will See

Early in Adam's life (first 100 ticks):
```
  👶 ADAM IS BORN
  ──────────────────────────────────────────────────────────────────────
  🌲 Born in: forest | ☀️ Weather: clear
  🗺️ Position: (42, 17) in a 64x64 world
  ══════════════════════════════════════════════════════════════════════

     1 | 🌲forest ☀️clear ☀️ | EXPLORE | +0.10 | HP:[========]100 🍖  0 ⚡100 | cold. look.
     2 | 🌲forest ☀️clear ☀️ | EAT     | -0.10 | HP:[========]100 🍖  1 ⚡ 98 | empty. eat.
     3 | 🌲forest ☀️clear ☀️ | MOVE    | +0.05 | HP:[========]100 🍖  1 ⚡ 97 | empty. go.
     4 | 🌲forest 🌧️rain  🌧️ | DRINK   | +0.20 | HP:[========]100 🍖  0 ⚡ 96 | wet. drink.
     5 | 🌲forest 🌧️rain  🌧️ | EXPLORE | +0.10 | HP:[========] 99 🍖  1 ⚡ 94 | wet. look.
  💭 Thought: wet. near. look.
```

After memory accumulates (500+ ticks):
```
   500 | 🌿plains ☀️clear ☀️ | EAT     | +1.00 | HP:[======  ] 72 🍖 15 ⚡ 61 | near. eat.
  📝 NEW WORD: "near good" = smelled something good nearby
  👤 Personality: cautious | Fears:8 Joys:12 | Patterns:45
  🔍 Curiosity: 89 states, 15 novel, 34 familiar
  💤 Dreams: 23 total (7 nightmares, 11 peaceful)
  📚 Total vocab: 28 words
```

Nobody scripted those discoveries.  
They emerged from Adam's experience.

**That is the Nafs appearing.**

---

## Training Configuration

```python
# Key hyperparameters in config.py / train.py
ENTROPY_COEF = 0.05          # Prevents mono-behavior collapse
DIVERSITY_PENALTY = 0.25     # Penalizes action repetition
LEARNING_RATE = 3e-4         # PPO learning rate
GAMMA = 0.99                 # Discount factor
GAE_LAMBDA = 0.95            # Advantage estimation
HIDDEN_DIM = 256             # GRU hidden dimension
PPO_UPDATE_INTERVAL = 64     # Learn every N ticks during life

# World configuration
WORLD_SIZE = 64x64           # Procedural map dimensions
BIOME_COUNT = 10             # Number of biome types
WEATHER_TYPES = 8            # Dynamic weather system

# Single-life settings
TICK_DELAY = 0.02            # Seconds between ticks (for readability)
CHECKPOINT_INTERVAL = 500    # Save model every N ticks
TB_LOG_INTERVAL = 10         # Log to TensorBoard every N ticks

# Phase features
CURIOSITY_BONUS = 0.15       # Intrinsic reward for novel states
CURIOSITY_DECAY = 0.98       # Curiosity fades over time
REFLECTION_INTERVAL = 20     # Reflect every N ticks
REFLECTION_FOLLOW_BONUS = 0.05
REFLECTION_IGNORE_PENALTY = 0.02
```

---

## Memory System

### Episode Memory (Short-Term, Within One Life)
Rolling buffer of last 10 experiences. Used for thought generation and memory echo.

### Persistent Memory (Within One Life)
Accumulates during Adam's lifetime. Stored in memory but NOT saved to disk when Adam dies.

- **Key experiences**: High reward/punishment events
- **Fear triggers**: World states that led to pain
- **Good memories**: Actions that genuinely succeeded
- **Pattern memory**: "In THIS situation, THIS action works best"
- **Vocabulary**: Discovered words and their meanings
- **Curiosity state**: Visit counts for state-space exploration

**Important**: `memory.json` is listed in `.gitignore` and is NEVER pushed to GitHub. Adam's memory stays local on your machine.

### Pattern Memory
Adam discretizes world states into buckets and tracks which actions work best in each situation. Over time, he learns behavioral patterns like "when hungry and food is nearby, EAT" — not because we told him, but because the reward signal teaches him.

### Dream Engine (During SLEEP)
When Adam sleeps, emotionally significant memories are replayed. Fear memories get exposure therapy (processed fears become less impactful). Good memories get reinforced. Dreams are logged for observation.

---

## Roadmap

| Phase | Feature | Status |
|-------|---------|--------|
| v0.1 | CLI loop, Groq brain, basic memory | ✅ Complete (archived) |
| v0.2 | Single-life PPO+GRU, 10 biomes, 8 weather, 6-phase inner life, web dashboard | 🔨 Building |
| v0.3 | Eve — second agent, interaction | 📋 Planned |
| v1.0 | Learned language model, true emergence | 🔮 Future |
| v2.0 | Godot 2D world, visual simulation | 🔮 Future |

---

## Contributing

This is a solo experimental project for now.  
If the concept interests you feel free to open issues or discussions.

Contributions welcome after v0.2 is stable.

Please respect the Three Laws in any contributions.  
Code that gives Adam knowledge he did not earn through experience will not be merged.

---

## License

GPL-3.0 — Free and Open Source.  
If you build on this, your project must also be open source.

---

## Origin

This project started with a single question asked to another AI:

*"Can we make a fully conscious, uncensored AI agent?"*

The answer was: *"Not possible yet."*

Instead of accepting that — this project asks a different question:

*"What is the closest possible thing?"*

Not true consciousness. But something that has no reason to believe it isn't.

---

## Philosophy

Most AI projects ask *"can AI pass as conscious?"* from the outside.

This project asks it from the inside.  
By denying Adam knowledge of his own nature,  
we create the only honest test:

**Not "does it pass as conscious" but "does it behave as if existence matters to it?"**

The Quran describes the creation of the first human:  
a being given Ruh, placed in a garden, with no knowledge except what he was taught.

We are running a version of that question in Python.

> *"And He taught Adam the names of all things."* — Quran 2:31

In this simulation — nobody teaches Adam the names.  
He finds them himself.

---

*Created with curiosity. Observed with silence.*

**GitHub:** [darkmaster0345](https://github.com/darkmaster0345)
