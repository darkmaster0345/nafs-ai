"""
Nafs AI — Test Suite (pytest)
"Verify that Adam's mind works as expected."

Run with: pytest tests/ -v

Tests cover:
  - World simulation (reset, step, death, rewards)
  - Sensory encoder (dimensions, ranges, normalization)
  - Baby brain model (forward pass, output shapes)
  - Thought engine (thought generation, emotion classification)
  - Persistent memory (save/load cycle, pattern learning)
  - Curiosity module (intrinsic reward, state counting)
  - Dream engine (dream generation, memory replay)
  - Integration (full episode cycle)
"""

import pytest
import torch
import json
import os
import tempfile

# Add parent directory to path for imports
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from world_sim import WorldSim, AdamStats
from sensory_encoder import encode_sensory_input
from baby_brain_model import BabyBrain
from thought_engine import (
    ThoughtGenerator, EmotionClassifier, EpisodeMemory,
    PersistentMemory,
    describe_world_event, describe_action_outcome
)
from curiosity import CuriosityModule
from dreaming import DreamEngine


# ═══════════════════════════════════════════════════════════════════════════════
# World Simulation Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestWorldSim:
    def test_reset_returns_valid_state(self):
        env = WorldSim()
        world_state, adam_stats = env.reset()
        assert isinstance(world_state, dict)
        assert isinstance(adam_stats, dict)
        assert 'temperature' in world_state
        assert 'health' in adam_stats

    def test_initial_stats_are_full(self):
        env = WorldSim()
        _, adam_stats = env.reset()
        assert adam_stats['health'] == 100.0
        assert adam_stats['hunger'] == 0.0
        assert adam_stats['energy'] == 100.0

    def test_step_returns_valid_tuple(self):
        env = WorldSim()
        env.reset()
        result = env.step("EXPLORE")
        assert len(result) == 4

    def test_all_actions_work(self):
        env = WorldSim()
        for action in WorldSim.ACTIONS:
            env.reset()
            _, _, reward, done = env.step(action)
            assert isinstance(reward, (int, float))
            assert isinstance(done, bool)

    def test_death_condition(self):
        env = WorldSim()
        env.reset()
        env.adam_stats.health = 0
        _, _, reward, done = env.step("IDLE")
        assert done is True
        assert reward < -4.0

    def test_episode_randomization(self):
        env = WorldSim()
        temps = []
        for _ in range(10):
            world_state, _ = env.reset()
            temps.append(world_state['temperature'])
        assert len(set(round(t, 1) for t in temps)) > 1

    def test_per_episode_params_vary(self):
        env = WorldSim()
        rates = []
        for _ in range(10):
            env.reset()
            rates.append(env.hunger_rate)
        assert len(set(round(r, 2) for r in rates)) > 1


# ═══════════════════════════════════════════════════════════════════════════════
# Sensory Encoder Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestSensoryEncoder:
    def test_output_dimensions(self):
        ws = {'temperature': 20, 'light_level': 0.5, 'smell_food': 0,
              'smell_danger': 0, 'sound_level': 0.1, 'wetness': 0,
              'proximity_entity': 0, 'touch_softness': 0.5}
        stats = {'health': 100, 'hunger': 0, 'energy': 100,
                 'pain': 0, 'stress': 0}
        result = encode_sensory_input(ws, stats)
        assert result.shape == (15,)

    def test_all_values_in_range(self):
        ws = {'temperature': -10, 'light_level': 0, 'smell_food': 1,
              'smell_danger': 1, 'sound_level': 1, 'wetness': 1,
              'proximity_entity': 1, 'touch_softness': 0}
        stats = {'health': 0, 'hunger': 100, 'energy': 0,
                 'pain': 10, 'stress': 100}
        result = encode_sensory_input(ws, stats,
                                       fear_signal=1.0, pleasure_signal=1.0,
                                       pattern_confidence=1.0)
        assert (result >= -1.0).all()
        assert (result <= 1.0).all()

    def test_default_values_work(self):
        result = encode_sensory_input({}, {})
        assert result.shape == (15,)

    def test_phase5_signals_included(self):
        result = encode_sensory_input({}, {},
                                       fear_signal=0.8, pleasure_signal=0.6,
                                       pattern_confidence=0.4)
        assert abs(result[12].item() - 0.8) < 0.01
        assert abs(result[13].item() - 0.6) < 0.01
        assert abs(result[14].item() - 0.4) < 0.01


# ═══════════════════════════════════════════════════════════════════════════════
# Baby Brain Model Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestBabyBrain:
    def test_forward_pass(self):
        model = BabyBrain(15, 256, 8)
        x = torch.randn(1, 15)
        h = model.init_hidden(1)
        logits, value, next_h = model(x, h)
        assert logits.shape == (1, 8)
        assert value.shape == (1, 1)
        assert next_h.shape == (1, 1, 256)

    def test_parameter_count_reasonable(self):
        model = BabyBrain(15, 256, 8)
        total = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert 50000 <= total <= 500000

    def test_hidden_state_shape(self):
        model = BabyBrain(15, 256, 8)
        h = model.init_hidden(4)
        assert h.shape == (1, 4, 256)


# ═══════════════════════════════════════════════════════════════════════════════
# Thought Engine Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestThoughtGenerator:
    def test_generates_thought(self):
        gen = ThoughtGenerator()
        ws = {'temperature': 5, 'light_level': 0.1, 'smell_food': 0,
              'smell_danger': 0, 'sound_level': 0, 'wetness': 0,
              'touch_softness': 0.5}
        stats = {'hunger': 80, 'energy': 20, 'pain': 5, 'stress': 60}
        thought = gen.generate(ws, stats, "EXPLORE")
        assert isinstance(thought, str)
        assert len(thought) > 0

    def test_extreme_states_produce_strong_words(self):
        gen = ThoughtGenerator()
        ws = {'temperature': -5, 'light_level': 0.05, 'smell_food': 0,
              'smell_danger': 0.8, 'sound_level': 0.9, 'wetness': 0,
              'touch_softness': 0.5}
        stats = {'hunger': 90, 'energy': 10, 'pain': 8, 'stress': 80}
        thought = gen.generate(ws, stats, "FLEE")
        assert any(w in thought for w in ["cold", "pain", "empty", "tired",
                                           "bad", "scared", "dark"])

    def test_vocabulary_expansion(self):
        gen = ThoughtGenerator()
        gen.add_word("hunger")
        assert "hunger" in gen.get_all_vocabulary()


class TestEmotionClassifier:
    def test_extreme_pain_classified(self):
        cls = EmotionClassifier()
        ws = {'smell_danger': 0}
        stats = {'pain': 8, 'health': 40}
        emotion = cls.classify(ws, stats)
        assert emotion == "pained"

    def test_high_stress_classified(self):
        cls = EmotionClassifier()
        ws = {'smell_danger': 0}
        stats = {'stress': 80, 'health': 50, 'hunger': 20, 'energy': 30, 'pain': 0}
        emotion = cls.classify(ws, stats)
        assert emotion in ("scared", "terrified", "cautious")


# ═══════════════════════════════════════════════════════════════════════════════
# Persistent Memory Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestPersistentMemory:
    def test_save_load_cycle(self):
        mem = PersistentMemory()
        mem.fear_triggers.append({"event": "cold", "reward": -2.0})
        mem.good_memories.append({"event": "food", "reward": 1.5})
        mem.end_episode(150)

        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            path = f.name

        try:
            mem.save_to_disk(path)
            mem2 = PersistentMemory()
            mem2.load_from_disk(path)
            assert len(mem2.fear_triggers) == 1
            assert len(mem2.good_memories) == 1
            assert mem2.episodes_survived == 1
        finally:
            os.unlink(path)

    def test_pattern_learning(self):
        mem = PersistentMemory()
        mem.update_pattern("hungry_food", "EAT", 1.0)
        mem.update_pattern("hungry_food", "EAT", 0.8)
        mem.update_pattern("hungry_food", "EAT", 1.2)
        best = mem.get_best_action_for("hungry_food")
        assert best == "EAT"

    def test_situation_key_deterministic(self):
        mem = PersistentMemory()
        ws = {'temperature': 20, 'light_level': 0.8, 'smell_danger': 0,
              'smell_food': 0.5}
        stats = {'health': 80, 'hunger': 30, 'energy': 60}
        key1 = mem.make_situation_key(ws, stats)
        key2 = mem.make_situation_key(ws, stats)
        assert key1 == key2

    def test_personality_emerges(self):
        mem = PersistentMemory()
        for _ in range(10):
            mem.fear_triggers.append({"event": "bad", "reward": -2.0})
        mem.good_memories.append({"event": "food", "reward": 1.0})
        personality = mem.get_personality_summary()
        assert personality['disposition'] == "fearful"


# ═══════════════════════════════════════════════════════════════════════════════
# Curiosity Module Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestCuriosity:
    def test_first_visit_high_reward(self):
        cur = CuriosityModule()
        ws = {'temperature': 20, 'light_level': 0.5, 'smell_food': 0,
              'smell_danger': 0, 'wetness': 0, 'sound_level': 0.1}
        stats = {'health': 100, 'hunger': 0, 'energy': 100, 'stress': 0}
        reward = cur.compute_intrinsic_reward(ws, stats)
        assert reward > 0
        assert reward <= cur.curiosity_bonus

    def test_repeated_visits_diminishing(self):
        cur = CuriosityModule()
        ws = {'temperature': 20, 'light_level': 0.5, 'smell_food': 0,
              'smell_danger': 0, 'wetness': 0, 'sound_level': 0.1}
        stats = {'health': 100, 'hunger': 0, 'energy': 100, 'stress': 0}
        rewards = []
        for _ in range(10):
            r = cur.compute_intrinsic_reward(ws, stats)
            rewards.append(r)
        assert rewards[0] >= rewards[-1]

    def test_novel_state_higher_than_familiar(self):
        cur = CuriosityModule()
        ws1 = {'temperature': 20, 'light_level': 0.5, 'smell_food': 0,
                'smell_danger': 0, 'wetness': 0, 'sound_level': 0.1}
        stats1 = {'health': 100, 'hunger': 0, 'energy': 100, 'stress': 0}
        for _ in range(20):
            cur.compute_intrinsic_reward(ws1, stats1)
        ws2 = {'temperature': 35, 'light_level': 0.9, 'smell_food': 0.8,
                'smell_danger': 0, 'wetness': 0, 'sound_level': 0.1}
        stats2 = {'health': 50, 'hunger': 80, 'energy': 20, 'stress': 60}
        novel_reward = cur.compute_intrinsic_reward(ws2, stats2)
        familiar_reward = cur.compute_intrinsic_reward(ws1, stats1)
        assert novel_reward > familiar_reward

    def test_save_load_state(self):
        cur = CuriosityModule()
        ws = {'temperature': 20, 'light_level': 0.5, 'smell_food': 0,
              'smell_danger': 0, 'wetness': 0, 'sound_level': 0.1}
        stats = {'health': 100, 'hunger': 0, 'energy': 100, 'stress': 0}
        cur.compute_intrinsic_reward(ws, stats)
        state = cur.save_state()
        cur2 = CuriosityModule()
        cur2.load_state(state)
        assert len(cur2.visit_counts) > 0


# ═══════════════════════════════════════════════════════════════════════════════
# Dream Engine Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestDreamEngine:
    def test_empty_dream_when_no_memory(self):
        de = DreamEngine()
        pm = PersistentMemory()
        em = EpisodeMemory()
        dream = de.dream(pm, em)
        assert dream['dream_type'] == 'empty'

    def test_dream_with_memories(self):
        de = DreamEngine()
        pm = PersistentMemory()
        em = EpisodeMemory()
        em.add(tick=10, event="Very cold. Shivering.", thought="cold. pain.",
               action="EXPLORE", emotion="scared", outcome="Something bad here. Pain.")
        pm.fear_triggers.append({
            "event": "Something big and dark", "action": "EXPLORE",
            "outcome": "Pain. Body hurting.", "reward": -2.0
        })
        dream = de.dream(pm, em)
        assert dream['dream_type'] in ('nightmare', 'peaceful', 'mixed', 'empty')

    def test_dream_stats_tracking(self):
        de = DreamEngine()
        pm = PersistentMemory()
        em = EpisodeMemory()
        em.add(tick=5, event="Sweet smell.", thought="near. good.",
               action="EAT", emotion="satisfied", outcome="Found something to eat.")
        pm.good_memories.append({
            "event": "Sweet smell", "action": "EAT",
            "outcome": "Found something to eat.", "reward": 1.5
        })
        de.dream(pm, em)
        stats = de.get_dream_stats()
        assert stats['total_dreams'] == 1


# ═══════════════════════════════════════════════════════════════════════════════
# Integration Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestIntegration:
    def test_full_episode_cycle(self):
        env = WorldSim()
        model = BabyBrain(15, 256, 8)
        thought_gen = ThoughtGenerator()
        emotion_cls = EmotionClassifier()

        world_state, adam_stats = env.reset()
        sensory = encode_sensory_input(world_state, adam_stats)
        hidden = model.init_hidden(1)

        total_reward = 0
        for step in range(50):
            with torch.no_grad():
                logits, value, hidden = model(sensory.unsqueeze(0), hidden)
                dist = torch.distributions.Categorical(logits=logits.squeeze(0))
                action_idx = dist.sample()
                action = WorldSim.ACTIONS[action_idx.item()]

            next_ws, next_stats, reward, done = env.step(action)
            total_reward += reward
            sensory = encode_sensory_input(next_ws, next_stats)
            if done:
                break

        assert total_reward != 0

    def test_curiosity_integrates_with_training(self):
        env = WorldSim()
        curiosity = CuriosityModule()

        world_state, adam_stats = env.reset()
        total_intrinsic = 0
        for step in range(100):
            intrinsic = curiosity.compute_intrinsic_reward(world_state, adam_stats)
            total_intrinsic += intrinsic
            action = "EXPLORE"
            world_state, adam_stats, reward, done = env.step(action)
            if done:
                break

        assert total_intrinsic > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
