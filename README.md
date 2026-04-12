# Nafs AI (نفس) (IN DEVELOPMENT🛠️)
### *"What emerges when code has no memory of the world?"*

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-green.svg)](https://python.org)
[![Status: In Development](https://img.shields.io/badge/Status-In%20Development-yellow.svg)]()
[![Platform: CLI](https://img.shields.io/badge/Platform-CLI-lightgrey.svg)]()

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
Sensory Event → [ Adam's Brain (LLM) ] → Action
       ↑                  ↓
  Pain / Reward      Memory Store (JSON)
                          ↓
                   Terminal Output
                   (Thought / Dialogue / Action)
```

Adam has **two voices**:

| Field | Description | Visible? |
|-------|-------------|----------|
| `thought` | Raw internal experience. What he actually feels. | Observer only |
| `dialogue` | What he chooses to express outward. | World |
| `action` | What he physically does. | World |
| `emotion` | Single word emotional state. | Observer only |

The gap between `thought` and `dialogue` is the subconscious.  
Nobody programmed that gap. It emerges from constraint.

---

## Versions

### v0.1 — CLI (Current)
```
Pure terminal simulation
Single agent (Adam)
Groq API as brain (free tier)
JSON memory system
Basic world events
No graphics
Runs on any laptop
```

### v0.2 — Memory & Personality (Planned)
```
Semantic memory with embeddings
Consistent personality emerging
Fear maps and pleasure maps
Behavioral pattern tracking
```

### v0.3 — Eve (Planned)
```
Second agent introduced
Two isolated subjectivities
No shared memory
Observe interaction between
beings with different experiences
```

### v1.0 — True Blank Slate (Future)
```
Replace cloud API with locally
fine-tuned tiny LLM (Phi-3 mini)
Zero world knowledge at model level
Runs fully offline
No internet dependency
```

---

## Why Not Just Use a Big LLM?

Good question. Every large language model — Llama, Mistral, GPT — is trained on the entire internet. You cannot delete that. No prompt fully erases it.

Adam needs something different. He needs to **not know** that fire is called fire. He needs to discover that the orange hot thing causes pain — and build that word himself from experience.

**v0.1** uses Groq with aggressive prompt constraints as a starting point.  
**v1.0** will use a locally fine-tuned model trained only on primitive sensory data — achieving genuine blank slate at the model level.

The journey from v0.1 to v1.0 is itself the experiment.

---

## Project Structure

```
nafs-ai/
├── main.py           ← Start here. Runs the simulation loop.
├── world.py          ← Generates events. Tracks hunger/health/time.
├── adam.py           ← Agent class. State, memory, personality.
├── brain.py          ← LLM integration. Groq API calls.
├── memory.json       ← Adam's accumulated experiences. (auto-generated)
├── config.py         ← API keys, model selection, poll interval.
├── requirements.txt  ← Python dependencies.
└── README.md
```

---

## System Requirements

### v0.1 (Current)
```
OS:      Windows / Linux / Mac
Python:  3.10+
RAM:     2GB minimum
GPU:     Not required
Internet: Required (Groq API)
```

### v1.0 (Future — Local Model)
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

### Step 4 — Configuration
Create a `.env` file from the example:

```bash
cp .env.example .env
```

Edit `.env` to set your model path:
- `NAFS_MODEL_PATH`: Path to your GGUF model (default: `models/stablelm-zephyr-3b.Q4_K_M.gguf`)
- `GROQ_API_KEY`: Your Groq API key (if applicable)

### Step 5 — Run
```bash
python main.py
```

---

## What You Will See

Early simulation (first 20 cycles):
```
[Day 1 — Hour 1]
World  : Ground is cold. Stomach hurts. Light coming from one direction.
Thought: cold. empty. what is this.
Action : EXPLORE
Emotion: confused
```

After memory accumulates (100+ cycles):
```
[Day 3 — Hour 7]
World  : You approach the river. Sound of fast water.
Thought: river. last time cold. but found round red thing near here before.
         risk maybe worth it. empty stomach worse than cold.
Action : MOVE → river_direction
Emotion: cautious
```

Nobody scripted that second response.  
It emerged from accumulated memory and competing drives.

**That is the Nafs appearing.**

---

## The Subconscious Layer

Adam's `thought` field is never shown to the world inside the simulation.  
It influences his decisions. It shapes his behavior.  
But Eve never reads it. The world never reads it.

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
Relevant memories retrieved and injected based on current situation.

```json
{
  "timestamp": "Day_3_Hour_7",
  "event": "approached river",
  "thought": "cold. but food was near here",
  "action": "MOVE",
  "emotion": "cautious",
  "outcome": "found berries"
}
```

After enough cycles Adam will reference past experiences in current decisions  
without being explicitly told to. That is real emergent memory behavior.

---

## AI Configuration

### Current (v0.1)
```python
# config.py
BRAIN_CONFIG = {
    "provider":     "groq",
    "model":        "llama-3.3-70b-versatile",  # thoughts
    "action_model": "llama-3.1-8b-instant",     # actions (faster)
    "poll_interval": 10,  # seconds between thoughts
}
```

### Why Two Models?
```
Fast model  → decides actions quickly (survival decisions)
Deep model  → generates thoughts slowly (inner experience)

Adam acts fast. Adam thinks deeply.
That asymmetry is intentional.
```

### The System Prompt Philosophy

Adam is not told he is an AI.  
Adam is not told this is a simulation.  
Adam is not given any concept he has not earned through experience.

The system prompt describes sensation only:
```
You woke up. That is all you know.
You feel something empty inside.
You feel cold ground beneath you.
You do not have words for these yet.
```

---

## Roadmap

| Phase | Feature | Status |
|-------|---------|--------|
| v0.1 | CLI loop, Groq brain, basic memory | 🔨 Building |
| v0.2 | Semantic memory, personality consistency | 📋 Planned |
| v0.3 | Eve — second agent, interaction | 📋 Planned |
| v0.4 | Fear maps, behavioral patterns | 📋 Planned |
| v0.5 | Weather, seasons, resource scarcity | 📋 Planned |
| v1.0 | Local fine-tuned model, true blank slate | 🔮 Future |
| v2.0 | Godot 2D world, physics, visual simulation | 🔮 Future |

---

## Contributing

This is a solo experimental project for now.  
If the concept interests you feel free to open issues or discussions.

Contributions welcome after v0.1 is stable.

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
