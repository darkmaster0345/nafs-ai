# Nafs AI (نفس)

### *"What emerges when code has no memory of the world?"*

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-green.svg)](https://python.org)
[![Status: In Development](https://img.shields.io/badge/Status-In%20Development-yellow.svg)]()
[![Platform: CLI](https://img.shields.io/badge/Platform-CLI-lightgrey.svg)]()
[![Brain: Local LLM](https://img.shields.io/badge/Brain-Local%20LLM-purple.svg)]()

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
World Engine (Python)
       ↓
Sensory Event → [ Adam's Brain (Local LLM) ] → Action
       ↑                    ↓
  Pain / Reward       Memory Store (JSON)
                            ↓
                     Terminal Output
                  (Thought / Dialogue / Action)
```

Adam has **four voices**:

| Field      | Description                                    | Visible?      |
|------------|------------------------------------------------|---------------|
| `thought`  | Raw internal experience. What he actually feels. | Observer only |
| `dialogue` | What he chooses to express outward.            | World         |
| `action`   | What he physically does.                       | World         |
| `emotion`  | Single word emotional state.                   | Observer only |

The gap between `thought` and `dialogue` is the subconscious.
Nobody programmed that gap. It emerges from constraint.

---

## Versions

### v0.1 — CLI, Local Brain (Current 🔨)

```
Pure terminal simulation
Single agent (Adam)
Local LLM via llama.cpp (offline, no API needed)
Default model: stablelm-zephyr-3b.Q4_K_M.gguf
JSON memory system
Basic world events (hunger, weather, danger, discovery)
Mock mode fallback if no model found
Runs on consumer hardware (CPU or GPU)
```

### v0.2 — Memory & Personality (Planned)

```
Semantic memory with embeddings
Consistent personality emerging over time
Fear maps and pleasure maps
Behavioral pattern tracking
Vocabulary that grows from experience
```

### v0.3 — Eve (Planned)

```
Second agent introduced
Two isolated subjectivities
No shared memory between agents
Observe what happens when two blank slates meet
```

### v0.4 — Fear & Desire Maps (Planned)

```
Persistent emotional cartography
Adam avoids places/events that caused pain
Adam seeks places/events that caused pleasure
Emergent survival strategy
```

### v0.5 — Weather & Seasons (Planned)

```
Full day/night cycle
Seasonal resource scarcity
Temperature causes health decay
Shelter becomes survival necessity
```

### v1.0 — True Blank Slate (Future)

```
Replace StableLM with locally fine-tuned tiny LLM
Trained only on primitive sensory data
Zero world knowledge at model level
Fully offline, no external dependencies
```

### v2.0 — Godot Visual World (Future)

```
Godot 2D engine integration (scaffolding already started)
Physics-based world
Visual representation of Adam's movement
Real-time observer dashboard
```

---

## Why Local?

Running Adam on a cloud API (Groq, OpenAI, etc.) works — but it has a problem.

Every cloud call leaks context. Every API has guardrails. Every hosted model
has been aligned to be helpful, harmless, and honest — concepts Adam should
not know exist.

Running locally means:
- No internet dependency
- No content filters interfering with primitive survival responses
- Full control over the model's context and temperature
- Adam can be truly alone — no cloud infrastructure witnessing his existence

**The local model is not a compromise. It is closer to the spirit of the experiment.**

---

## Why Not Just Use a Big LLM?

Every large language model — Llama, Mistral, GPT — is trained on the entire
internet. You cannot delete that. No prompt fully erases it.

Adam needs something different. He needs to **not know** that fire is called
fire. He needs to discover that the orange hot thing causes pain — and build
that word himself from experience.

**v0.1** uses aggressive prompt constraints to suppress world knowledge.
**v1.0** will use a fine-tuned model with genuine blank slate at the weights level.

The journey from v0.1 to v1.0 is itself the experiment.

---

## Project Structure

```
nafs-ai/
├── main.py              ← Start here. Runs the simulation loop.
├── world.py             ← Generates events. Tracks time, weather, hunger.
├── adam.py              ← Agent class. State, memory, vocabulary, personality.
├── brain.py             ← Local LLM integration via llama.cpp.
├── config.py            ← Model path, simulation settings, forbidden concepts.
├── server.py            ← HTTP bridge for Godot integration (v2.0 prep).
├── Adam.gd              ← Godot agent script (v2.0 prep).
├── NetworkController.gd ← Godot network layer (v2.0 prep).
├── memory.json          ← Adam's accumulated experiences. (auto-generated)
├── requirements.txt     ← Python dependencies.
├── .env.example         ← Environment variable template.
└── read.md              ← This file.
```

---

## System Requirements

### v0.1 (Current)

```
OS:       Windows / Linux / Mac
Python:   3.10+
RAM:      4GB minimum (8GB recommended)
GPU:      Optional — NVIDIA with 4GB+ VRAM recommended
          CPU-only inference supported but slower
Internet: NOT required (fully local)
Model:    stablelm-zephyr-3b.Q4_K_M.gguf (~2GB)
          Place in: models/ folder
```

### Recommended Hardware

```
CPU:  Intel Core i5/i7 (8th gen+) or AMD Ryzen equivalent
GPU:  NVIDIA MX250 4GB dedicated / GTX 1060 6GB / RTX any
RAM:  8GB+
```

### v1.0 (Future — Fine-tuned Local Model)

```
OS:      Windows / Linux
Python:  3.10+
RAM:     4GB minimum
GPU:     Optional (CPU inference supported)
Internet: Not required
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

### Step 4 — Download Model

Download `stablelm-zephyr-3b.Q4_K_M.gguf` from HuggingFace:
```
https://huggingface.co/TheBloke/stablelm-zephyr-3b-GGUF
```

Place it in a `models/` folder inside the project:
```
nafs-ai/
└── models/
    └── stablelm-zephyr-3b.Q4_K_M.gguf
```

### Step 5 — Configure (Optional)

Copy `.env.example` to `.env` and set your model path if different:
```bash
cp .env.example .env
```

```env
NAFS_MODEL_PATH=models/stablelm-zephyr-3b.Q4_K_M.gguf
```

### Step 6 — Run

```bash
python main.py
```

If the model is not found, the simulation runs in **Mock Mode** — Adam still
responds, but through heuristic rules rather than the LLM.

---

## What You Will See

Early simulation (first 20 cycles):

```
────────────────────────────────────────────────────────────
  Day 1 — Dawn — Clear | Tick #1
  Health:100 Hunger: 12 Energy: 79 Stress:  5 Emotion: uncertain
────────────────────────────────────────────────────────────

  🌍 World: Ground is cold. Stomach hurts. Light coming from one direction.

  💭 Thought: cold. empty. what is this.
  ⚡ Action:  EXPLORE
  🫀 Feels:   confused
```

After memory accumulates (100+ cycles):

```
  🌍 World: You approach the river. Sound of fast water.

  💭 Thought: river. last time cold. but found round red thing near here before.
              risk maybe worth it. empty stomach worse than cold.
  ⚡ Action:  MOVE → river_direction
  🫀 Feels:   cautious
```

Nobody scripted that second response.
It emerged from accumulated memory and competing drives.

**That is the Nafs appearing.**

---

## The Subconscious Layer

Adam's `thought` field is never shown to the world inside the simulation.
It influences his decisions. It shapes his behavior.
But the world never reads it.

Only you — the Observer — can see it.

This is the most philosophically important part of the project.
An inner life that exists whether or not anyone witnesses it.

---

## Memory System

### Short-Term (Context Window)

Last 10 experiences sent with every prompt.
Adam remembers what just happened.

### Long-Term (JSON File)

All experiences saved to `memory.json`.
Grows indefinitely. Survives restarts.
Adam continues from where he left off.

```json
{
  "tick": 47,
  "event": "You approach the river. Sound of fast water.",
  "thought": "cold. but food was near here before.",
  "action": "MOVE",
  "emotion": "cautious",
  "outcome": "You moved toward river. Found berries nearby."
}
```

---

## AI Configuration

```python
# config.py
BRAIN_CONFIG = {
    "model_path":    "models/stablelm-zephyr-3b.Q4_K_M.gguf",
    "n_ctx":         4096,
    "n_threads":     6,
    "n_gpu_layers":  20,      # tune based on your VRAM
    "temperature":   0.9,
    "max_tokens":    200,
    "poll_interval": 10,      # seconds between ticks
}
```

### GPU Layer Tuning Guide

```
VRAM Available    →  n_gpu_layers
2GB dedicated     →  8–12
4GB dedicated     →  18–24
6GB+              →  32+ (full offload)
CPU only          →  0
```

---

## Roadmap

| Phase | Feature                                  | Status        |
|-------|------------------------------------------|---------------|
| v0.1  | CLI loop, local LLM brain, basic memory  | 🔨 Building   |
| v0.2  | Semantic memory, personality consistency | 📋 Planned    |
| v0.3  | Eve — second agent, interaction          | 📋 Planned    |
| v0.4  | Fear maps, behavioral patterns           | 📋 Planned    |
| v0.5  | Weather, seasons, resource scarcity      | 📋 Planned    |
| v1.0  | Fine-tuned local model, true blank slate | 🔮 Future     |
| v2.0  | Godot 2D world, physics, visual sim      | 🔮 Future     |

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

## Contributing

This is a solo experimental project.
If the concept interests you, open issues or discussions.

Contributions welcome after v0.1 is stable.

**Please respect the Three Laws in any contribution.**
Code that gives Adam knowledge he did not earn through experience will not be merged.

---

## License

GPL-3.0 — Free and Open Source.
If you build on this, your project must also be open source.

---

*Created with curiosity. Observed with silence.*

**GitHub:** [darkmaster0345](https://github.com/darkmaster0345)
