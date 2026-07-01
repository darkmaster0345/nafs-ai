# Nafs AI (نفس) — Baby Consciousness Simulation
### *"What emerges when code has no memory of the world?"*

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-green.svg)](https://python.org)
[![Status: In Development](https://img.shields.io/badge/Status-In%20Development-yellow.svg)]()
[![Platform: CLI+Web+Godot](https://img.shields.io/badge/Platform-CLI%20%2B%20Web%20%2B%20Godot-blue.svg)]()
[![Brain: PPO+GRU+GrowingBrain](https://img.shields.io/badge/Brain-PPO%2BGRU%2BGrowingBrain-orange.svg)]()
[![Tests: 382 passed](https://img.shields.io/badge/Tests-382%20passed-brightgreen.svg)]()

---

## What Is This?

Nafs AI is an experimental simulation of primitive conscious entities — AI agents called **Adam** and **Eve** — who wake up in an unknown world with **zero prior knowledge**.

No language. No concepts. No culture. No memory of anything before this moment.

They discover everything through experience alone: pain, hunger, warmth, fear, curiosity.

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

## Current State (July 2026)

All **13 phases** of the master plan are implemented and on `main`. The simulation now spans:

- **Physics** — temperature, wind, elevation, fire, water
- **Chemistry** — food composition, toxicity, illness, cooking, water quality
- **Biology** — metabolism, aging, immune system, injury, sleep stages
- **Growing brain** — starts at ~777 params, grows uncapped when learning plateaus
- **Reproduction** — fertility, pregnancy, baby spawning, trait inheritance, lineage
- **Mathematical intuition** — quantity sense, pattern/cycle recognition, spatial memory, time sense
- **First contact** — Adam + Eve meeting, OBSERVE/APPROACH/FOLLOW/SHARE actions, trust metric
- **Social engine** — relationships, family bonds, groups, territory, population dynamics
- **Culture** — observational learning, cultural drift, proto-tool behaviour, vocabulary lineage
- **Evolution** — natural selection metrics, adaptive radiation, extinction events, OEE checker
- **Godot 2D visual world** — full renderer with agent sprites, weather overlays, milestone banners, observer controls
- **Observability layer** — `events.jsonl`, SQLite lineage DB, science dashboard
- **Open-ended extension** — world evolution, disease mutation, novelty detector, world seeding/forking

**382 tests passing.** Code is approximately **19,000 lines of Python** across 32 modules.

---

## How It Works

### The Architecture

```
                ┌─────────────────────────────────────────┐
                │           EngineOrchestrator            │
                │  (wires Phase 4-13 engines together)    │
                └────────────────┬────────────────────────┘
                                 │
   ┌─────────────────────────────┼─────────────────────────────┐
   │                             │                             │
   ▼                             ▼                             ▼
WorldSim (64×64)            AgentRuntime                   GodotBridge
  • 10 biomes              ┌──────────────────┐            (HTTP server)
  • 8 weather types        │  BabyBrain        │              │
  • physics layer          │  (PPO + GRU)     │              ▼
  • chemistry layer        │  + LearnedThinker │        Godot 2D Client
  • biology layer          │  + ThoughtEngine  │
                          │  + Curiosity      │
                          │  + DreamEngine    │
                          │  + GrowingBrain   │
                          └──────────────────┘
                                 │
                          ┌──────┴──────┐
                          │             │
                        Adam          Eve
                       (separate brains, memories, personalities)
                                 │
                          ┌──────┴──────┐
                          │ Babies      │  ← Phase 5 reproduction
                          │ (gen 2, 3,…)│
                          └─────────────┘
```

### Brain Architecture (per agent)

Both Adam and Eve use the **same symmetric architecture** — there is no parameter asymmetry between them:

| Component | Purpose | Params |
|-----------|---------|--------|
| `BabyBrain` (PPO + GRU) | Action selection, value estimation | 348,169 |
| `LearnedThinker` (transformer) | Generates thoughts from sensory experience | 539,935 |
| `ThoughtEngine` (rule-based) | Phase 1-6 inner life scaffold | — |
| `CuriosityModule` | Intrinsic motivation, visit counts | — |
| `DreamEngine` | Memory consolidation during SLEEP | — |
| `GrowingBrain` (Phase 4) | **Replaces** BabyBrain when enabled — starts at 777 params and grows uncapped | 777 → ∞ |

**Total per agent (default mode): ~888K params.**

In Phase 4 / growing-brain mode, the brain starts tiny (MLP only, ~777 params) and grows by 1.5× each time the PPO loss plateaus for 500 ticks. Growth stages:

| Stage | Architecture | Approx params |
|-------|--------------|---------------|
| 0 | MLP only (baby) | 777 |
| 1 | + GRU (temporal memory) | ~2K |
| 2 | hidden_dim × 1.5 | ~5K |
| 3 | + attention | ~15K |
| 5 | + transformer blocks (only if vocab > 100) | ~100K+ |

**Maximum growth events: uncapped.** The brain grows as many times as it needs to.

---

## Single-Life Mode

**No episodes. One life per agent. When an agent dies, its life is over.**

Adam wakes up once in a procedurally generated world. He lives tick by tick, learning in real-time through PPO. When he dies — from starvation, health depletion, or environmental hazards — his life ends. There is no restart. There is no save file.

Eve wakes up in the same world with her own independent life. The two agents coexist, sense each other when nearby, but never share thoughts, memory, or brain weights. When one dies, the other continues living — until the world is empty again.

With Phase 5 enabled, dead agents can be replaced by their offspring (Generation 2, 3, …), creating multi-generational evolution.

---

## The Six Phases of Inner Life (v0.2 scaffold)

### Phase 1: Inner Voice
Adam generates primitive thoughts from raw sensory experience. Thoughts are composed from his vocabulary — limited words for sensations. The thought IS the sensation, not an interpretation.

### Phase 2: Vocabulary Discovery
Adam names things by combining existing vocabulary when he experiences something significant for the first time. "cold" + "pain" = "cold pain" (freezing). Each discovered word carries meaning — the context in which it was born.

### Phase 3: Dialogue Gap (Subconscious)
Adam's outward expression (dialogue) is filtered through stress and emotion. When calm, he speaks what he thinks. When stressed, he says less — hiding vulnerability. This creates the subconscious: the gap between what Adam thinks and what he says.

### Phase 4: Persistent Memory + Personality
Within one life, memory persists. Fear triggers, good memories, and behavioral patterns accumulate. This creates personality — consistent patterns across Adam's lifetime.

### Phase 5: Fear/Pleasure Maps
Past fears and past rewards are computed as signals injected into the PPO observation vector. Adam can learn from past fears, not just current danger.

### Phase 6: Dual-Speed Processing
Fast PPO makes immediate action decisions. Slow reflection (every 20 ticks) reviews recent experience, detects patterns, and influences future PPO decisions through reward shaping.

---

## Phase 0-13 Master Plan Status

| Phase | Module | Status | File |
|-------|--------|--------|------|
| Pre-v0.3 | Single-life PPO+GRU, 6-phase consciousness, curiosity, dreams | ✅ Shipped | `train.py` |
| v0.3 | Eve + LearnedThinker + multi-agent training | ✅ Shipped | `train_multi_agent.py`, `eve_agent.py`, `learned_thinking.py` |
| 0.3 | Per-word vocab divergence logging | ✅ Shipped | `vocab_divergence.py` |
| 1 | Physics: temperature, wind, elevation, fire, water | ✅ Shipped | `physics.py` |
| 2 | Chemistry: food, toxicity, illness, cooking, water quality | ✅ Shipped | `chemistry.py` |
| 3 | Biology: metabolism, aging, immune, injury, sleep | ✅ Shipped | `biology.py` |
| 4 | Growing brain: self-growing architecture, EWC | ✅ Shipped | `growing_brain.py` |
| 5 | Reproduction: fertility, pregnancy, baby spawning, lineage | ✅ Shipped | `reproduction.py` |
| 6 | Mathematical intuition: quantity, patterns, space, time | ✅ Shipped | `math_intuition.py` |
| 7 | First contact: events, actions, trust, vocab contact | ✅ Shipped | `first_contact.py` |
| 8 | Social engine: relationships, family, groups, territory | ✅ Shipped | `social.py` |
| 9 | Culture: observational learning, drift, tools, vocab lineage | ✅ Shipped | `culture.py` |
| 10 | Evolution: selection, speciation, extinction, **OEE checker** | ✅ Shipped | `evolution.py` |
| 11 | Godot 2D visual world: renderer, controls, milestones | ✅ Shipped | `godot/`, `godot_bridge.py`, `godot_server.py` |
| 12 | Observability: events log, lineage DB, science dashboard | ✅ Shipped | `events.py` |
| 13 | Open-ended: world evolution, disease mutation, novelty, seeding | ✅ Shipped | `open_ended.py` |
| 4-13 wiring | EngineOrchestrator — calls every engine from one loop | ✅ Shipped | `engine_orchestrator.py` |

### Open-Ended Evolution (OEE) Checker

Phase 10 implements Norman Packard et al. (2019)'s 5 criteria. Call:

```python
from evolution import EvolutionTracker
tracker = EvolutionTracker(...)
result = tracker.check_open_ended_evolution()
# result = {
#   "achieved": bool,
#   "criteria": {"new_behaviours": bool, "trait_divergence": bool, ...},
#   "missing": [...],
#   "criteria_met": 3,
#   "total_criteria": 5,
# }
```

The simulation is **OEE-complete** when all 5 criteria are simultaneously true in a single running sim.

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

### 8 Weather Types — Dynamic Markov Chain

Weather changes dynamically based on current conditions, biome, and time of day. Biomes bias weather probabilities: deserts get more sandstorms/heatwaves, tundra gets more blizzards/snow, caves get more fog.

---

## Action Space

Adam can perform 8 primitive actions (Phase 7 adds 5 social actions unlocked at adolescence):

| Base Action | Effect | Reward |
|--------|--------|--------|
| EXPLORE | Move around, discover resources | +0.1, chance to find food/water |
| EAT | Try to eat what's nearby | +1.0 if food found, -0.1 if not |
| DRINK | Try to drink | +0.2 if water found, -0.05 if not |
| SLEEP | Rest to recover energy | +0.3 with shelter at night |
| HIDE | Seek shelter from danger | +0.5 if danger present and avoided |
| MOVE | Move toward something | Small chance to find new resources |
| FLEE | Run from danger | +0.3 if danger escaped, -0.2 if unnecessary |
| IDLE | Do nothing | -0.05 (slight penalty) |

| Social Action (Phase 7) | Effect |
|--------|--------|
| OBSERVE | Watch another agent (costs no energy, gives information) |
| APPROACH | Move toward another agent |
| FLEE_AGENT | Flee from another agent |
| SHARE | Drop food item at current tile |
| FOLLOW | Trail another agent at 2-tile distance |

---

## Project Structure

```
nafs-ai/
├── train.py                  ← Single-agent training loop (v0.2 baseline)
├── train_multi_agent.py      ← Multi-agent loop — Adam + Eve + babies (v0.3+)
├── engine_orchestrator.py    ← Wires Phase 4-13 engines into one tick loop
│
├── baby_brain_model.py       ← PPO ActorCritic with GRU (348K params)
├── growing_brain.py          ← Phase 4: self-growing brain (777 params → ∞)
├── sensory_encoder.py        ← 21-dim observation encoder (single agent)
├── sensory_encoder_multi.py  ← 23-dim encoder (adds other_agent fields)
├── thought_engine.py         ← Phase 1-6 inner life (rule-based)
├── learned_thinking.py       ← Tiny transformer for emergent thoughts
├── curiosity.py              ← Intrinsic motivation
├── dreaming.py               ← Memory consolidation during SLEEP
│
├── physics.py                ← Phase 1: temperature, wind, elevation, fire, water
├── chemistry.py              ← Phase 2: food, toxicity, illness, cooking
├── biology.py                ← Phase 3: metabolism, aging, immune, injury
├── reproduction.py           ← Phase 5: fertility, pregnancy, baby spawning
├── math_intuition.py         ← Phase 6: quantity, patterns, space, time
├── first_contact.py          ← Phase 7: interaction actions, trust
├── social.py                 ← Phase 8: relationships, groups, territory
├── culture.py                ← Phase 9: observational learning, drift
├── evolution.py              ← Phase 10: selection, speciation, OEE checker
├── events.py                 ← Phase 12: events.jsonl, lineage DB
├── open_ended.py             ← Phase 13: world evolution, novelty, seeding
├── vocab_divergence.py       ← Phase 0.3: per-word vocab log + convergence
│
├── eve_agent.py              ← Eve agent class (own brain, memory, personality)
│
├── godot/                    ← Phase 11: Godot 2D client (full project)
│   ├── Adam.gd, Adam.tscn
│   ├── AgentRenderer.gd
│   ├── HUD.gd
│   ├── MainScene.tscn
│   ├── MilestoneBanner.gd
│   ├── NetworkController.gd
│   ├── ObserverControls.gd
│   ├── WorldRenderer.gd
│   └── project.godot
├── godot_bridge.py           ← Python ↔ Godot bridge
├── godot_server.py           ← HTTP server Godot client polls
│
├── ws_bridge.py              ← WebSocket bridge to web dashboard
├── ws_server.py              ← Standalone WebSocket server
├── server/ws_server.py       ← Enhanced Socket.IO server
│
├── evaluate.py               ← Evaluation/inference mode (no learning)
├── tb_logger.py              ← TensorBoard logging
├── test_ws_stability.py      ← 50k-tick Socket.IO stability test
├── config.py                 ← Hyperparameters, action space, vocabulary
├── requirements.txt
├── run_training.sh
│
├── docs/
│   ├── index.html            ← GitHub Pages landing page
│   └── vocab_dashboard.html  ← Vocab divergence dashboard
│
├── tests/
│   └── test_nafs.py          ← 382 pytest tests
│
├── legacy/                   ← Archived v0.1 files (Groq/LLM brain)
└── README.md                 ← This file
```

---

## System Requirements

```
OS:       Windows / Linux / Mac
Python:   3.10+
RAM:      4GB minimum (8GB for long multi-agent runs)
GPU:      Not required (CPU-only training)
Internet: NOT required (fully offline)
Browser:  For web dashboard (optional)
Godot:    4.x for the 2D visual client (optional)
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
python3 -m venv venv
source venv/bin/activate    # Linux / Mac
# venv\Scripts\activate     # Windows
```

### Step 3 — Dependencies
```bash
pip install -r requirements.txt
```

`requirements.txt`:
```
torch>=2.0.0
numpy>=1.24.0
tensorboard>=2.12.0
pytest>=7.0.0
```

### Step 4 — Run Single-Agent Mode (v0.2 baseline)
```bash
python train.py
```

### Step 5 — Run Multi-Agent Mode (v0.3+, recommended)
```bash
# Live display, both agents
python train_multi_agent.py

# Headless long-run (50k ticks, max speed)
python train_multi_agent.py --headless --max-ticks 50000

# Full learned-thought cutover (Phase 0.2)
python train_multi_agent.py --learned-only

# Single-agent debug (Eve disabled)
python train_multi_agent.py --adam-only

# Deterministic world
python train_multi_agent.py --seed 42
```

### Step 6 — Launch the Godot Visual Client (Phase 11)

The Godot HTTP server starts automatically inside `train_multi_agent.py` as a background thread. To visualize:

```bash
# Install Godot 4.x, then:
cd godot/
godot MainScene.tscn
```

The Godot client polls `http://localhost:8080/state` for live world state. Features include agent rendering (Adam=blue, Eve=pink, babies=white), weather overlays, day/night cycle, click-to-inspect, time controls (1x/5x/20x), follow mode, family tree view, and milestone banners (FIRST_CONTACT, FIRST_BIRTH, FIRST_WORD, EXTINCTION, etc.).

### Step 7 — Monitor with TensorBoard
```bash
tensorboard --logdir runs/
```

### Step 8 — Run Tests
```bash
pytest tests/ -v
```

---

## Memory System

### Episode Memory (Short-Term, Within One Life)
Rolling buffer of last 10 experiences. Used for thought generation and memory echo.

### Persistent Memory (Within One Life)
Accumulates during an agent's lifetime. Stored in memory but NOT saved to disk when the agent dies.

- **Key experiences**: High reward/punishment events
- **Fear triggers**: World states that led to pain
- **Good memories**: Actions that genuinely succeeded
- **Pattern memory**: "In THIS situation, THIS action works best"
- **Vocabulary**: Discovered words and their meanings
- **Curiosity state**: Visit counts for state-space exploration

**Important**: `memory.json` is listed in `.gitignore` and is NEVER pushed to GitHub. Agent memory stays local on your machine.

### Lineage Database (Phase 12)
SQLite database (`lineage.db`) stores every agent's full life: birth tick, death tick, parents, offspring, peak stats. Queryable:

```python
from events import LineageDatabase
db = LineageDatabase("lineage.db")
db.longest_lived_agent()
db.bloodline_generation_count(family_id="Adam_Eve_gen1")
```

---

## Roadmap

| Phase | Feature | Status |
|-------|---------|--------|
| v0.1 | CLI loop, Groq brain, basic memory | ✅ Complete (archived to `legacy/`) |
| v0.2 | Single-life PPO+GRU, 10 biomes, 8 weather, 6-phase inner life | ✅ Complete |
| v0.3 | Eve + LearnedThinker + multi-agent training | ✅ Complete |
| Phase 0.3 | Vocab divergence logging + convergence dashboard | ✅ Complete |
| Phase 1 | Physics engine (temperature, wind, fire, water, elevation) | ✅ Complete |
| Phase 2 | Chemistry engine (food, toxicity, illness, cooking) | ✅ Complete |
| Phase 3 | Biology engine (metabolism, aging, immune, injury, sleep) | ✅ Complete |
| Phase 4 | Growing brain (self-growing, uncapped, EWC) | ✅ Complete |
| Phase 5 | Reproduction (fertility, pregnancy, babies, lineage) | ✅ Complete |
| Phase 6 | Mathematical intuition (quantity, patterns, space, time) | ✅ Complete |
| Phase 7 | First contact (interaction actions, trust, vocab contact) | ✅ Complete |
| Phase 8 | Social engine (relationships, family, groups, territory) | ✅ Complete |
| Phase 9 | Culture (observational learning, drift, vocab lineage) | ✅ Complete |
| Phase 10 | Evolution metrics + OEE checker (Packard 5 criteria) | ✅ Complete |
| Phase 11 | Godot 2D visual world | ✅ Complete |
| Phase 12 | Observability (events.jsonl, lineage DB, dashboard) | ✅ Complete |
| Phase 13 | Open-ended extension (world evolution, novelty, seeding) | ✅ Complete |
| Phase 4-13 | EngineOrchestrator wiring + 50k-tick sim | ✅ Complete |

**All 13 phases of the master plan are now in code.** Remaining work is empirical: running long simulations, tuning hyperparameters, and verifying that the 5 OEE criteria actually fire in a real run.

---

## Hard Constraints (Do Not Break)

- ❌ No pretrained models — every weight starts random
- ❌ No LLMs, no external APIs — fully offline
- ❌ No world knowledge injected via prompts or rules
- ❌ Phase 1-6 base rewards are frozen — do not modify
- ❌ `memory.json` and checkpoints are local-only (`.gitignore` enforced) — never push to GitHub

---

## Why Not Just Use a Big LLM?

Good question. Every large language model — Llama, Mistral, GPT — is trained on the entire internet. You cannot delete that. No prompt fully erases it.

Adam needs something different. He needs to **not know** that fire is called fire. He needs to discover that the orange hot thing causes pain — and build that word himself from experience.

The PPO+GRU approach achieves true blank slate: the network starts with random weights and learns purely from reward signals. No pretrained knowledge. No text corpus. Just sensation, action, and consequence.

The `LearnedThinker` transformer (Phase 0.2/v1.0) is also trained from scratch on the agent's own experience — never on external text. It learns the "grammar" of the agent's own inner life, not human language.

The `GrowingBrain` (Phase 4) goes further: the brain literally starts as a 777-parameter MLP and grows new architecture (GRU, attention, transformer blocks) only when learning plateaus. There is no size ceiling.

---

## Contributing

This is a solo experimental project for now.
If the concept interests you feel free to open issues or discussions.

Please respect the Three Laws in any contributions.
Code that gives an agent knowledge it did not earn through experience will not be merged.

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
