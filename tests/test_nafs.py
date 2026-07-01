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

    def test_world_has_biome(self):
        env = WorldSim()
        world_state, _ = env.reset()
        assert 'biome' in world_state
        assert world_state['biome'] in ['desert', 'forest', 'tundra', 'plains',
                                         'mountain', 'swamp', 'ocean', 'jungle',
                                         'cave', 'volcano']

    def test_world_has_weather(self):
        env = WorldSim()
        world_state, _ = env.reset()
        assert 'weather' in world_state

    def test_world_map_exists(self):
        env = WorldSim()
        env.reset()
        assert env.world_map.width > 0
        assert env.world_map.height > 0


# ═══════════════════════════════════════════════════════════════════════════════
# Sensory Encoder Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestSensoryEncoder:
    def test_output_dimensions(self):
        from sensory_encoder import INPUT_DIM
        ws = {'temperature': 20, 'light_level': 0.5, 'smell_food': 0,
              'smell_danger': 0, 'sound_level': 0.1, 'wetness': 0,
              'proximity_entity': 0, 'touch_softness': 0.5,
              'biome': 'plains', 'weather': 'clear', 'time_of_day': 12}
        stats = {'health': 100, 'hunger': 0, 'energy': 100,
                 'pain': 0, 'stress': 0}
        result = encode_sensory_input(ws, stats)
        assert result.shape == (INPUT_DIM,)
        assert INPUT_DIM == 21

    def test_all_values_in_range(self):
        ws = {'temperature': -10, 'light_level': 0, 'smell_food': 1,
              'smell_danger': 1, 'sound_level': 1, 'wetness': 1,
              'proximity_entity': 1, 'touch_softness': 0,
              'biome': 'volcano', 'weather': 'storm', 'time_of_day': 0}
        stats = {'health': 0, 'hunger': 100, 'energy': 0,
                 'pain': 10, 'stress': 100}
        result = encode_sensory_input(ws, stats,
                                       fear_signal=1.0, pleasure_signal=1.0,
                                       pattern_confidence=1.0)
        assert (result >= -1.0).all()
        assert (result <= 1.0).all()

    def test_default_values_work(self):
        from sensory_encoder import INPUT_DIM
        result = encode_sensory_input({}, {})
        assert result.shape == (INPUT_DIM,)

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
        from sensory_encoder import INPUT_DIM
        model = BabyBrain(INPUT_DIM, 256, 8)
        x = torch.randn(1, INPUT_DIM)
        h = model.init_hidden(1)
        logits, value, next_h = model(x, h)
        assert logits.shape == (1, 8)
        assert value.shape == (1, 1)
        assert next_h.shape == (1, 1, 256)

    def test_parameter_count_reasonable(self):
        from sensory_encoder import INPUT_DIM
        model = BabyBrain(INPUT_DIM, 256, 8)
        total = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert 50000 <= total <= 500000

    def test_hidden_state_shape(self):
        from sensory_encoder import INPUT_DIM
        model = BabyBrain(INPUT_DIM, 256, 8)
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
    def test_full_life_cycle(self):
        from sensory_encoder import INPUT_DIM
        env = WorldSim()
        model = BabyBrain(INPUT_DIM, 256, 8)

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

    def test_curiosity_integrates_with_life(self):
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

    def test_move_changes_position(self):
        env = WorldSim()
        env.reset()
        old_x, old_y = env.adam_x, env.adam_y
        env.step("MOVE")
        # Position should change (wrapping is possible so check not always same)
        # After many MOVEs, position should definitely change
        positions = set()
        for _ in range(10):
            env.step("MOVE")
            positions.add((env.adam_x, env.adam_y))
        assert len(positions) > 1  # Adam should have moved


if __name__ == '__main__':
    pytest.main([__file__, '-v'])


# ═══════════════════════════════════════════════════════════════════════════════
# Learned Thinking Module Tests (v1.0 experimental)
# ═══════════════════════════════════════════════════════════════════════════════

class TestLearnedThinking:
    def test_tokenizer_encode_decode_roundtrip(self):
        from learned_thinking import ThoughtTokenizer
        text = "cold. dark."
        tokens = ThoughtTokenizer.encode(text)
        assert tokens.shape[0] == ThoughtTokenizer.MAX_THOUGHT_LEN
        decoded = ThoughtTokenizer.decode(tokens)
        assert decoded == "cold. dark."

    def test_tokenizer_has_special_tokens(self):
        from learned_thinking import ThoughtTokenizer
        assert ThoughtTokenizer.PAD_IDX != ThoughtTokenizer.EOT_IDX
        assert ThoughtTokenizer.VOCAB_SIZE == 31

    def test_tokenizer_handles_unknown_chars(self):
        from learned_thinking import ThoughtTokenizer
        # Capital letters and punctuation not in vocab should become space
        tokens = ThoughtTokenizer.encode("COLD!!!")
        decoded = ThoughtTokenizer.decode(tokens)
        assert "cold" in decoded

    def test_transformer_forward_pass(self):
        from learned_thinking import ThoughtTransformer, ThoughtTokenizer
        model = ThoughtTransformer(sensory_dim=21)
        sensory = torch.randn(2, 21)  # batch of 2
        chars = torch.randint(0, ThoughtTokenizer.VOCAB_SIZE, (2, 10))
        logits = model(sensory, chars)
        assert logits.shape == (2, 10, ThoughtTokenizer.VOCAB_SIZE)

    def test_transformer_generate_returns_string(self):
        from learned_thinking import ThoughtTransformer
        model = ThoughtTransformer(sensory_dim=21)
        sensory = torch.randn(21)
        result = model.generate(sensory, max_len=20, temperature=0.5)
        assert isinstance(result, str)

    def test_learned_thinker_records_experiences(self):
        from learned_thinking import LearnedThinker
        thinker = LearnedThinker(sensory_dim=21, buffer_size=100, train_interval=5, batch_size=4)
        thinker.start()
        for i in range(20):
            sensory = torch.randn(21)
            thinker.record_experience(sensory, f"thought {i}")
        assert len(thinker.buffer) == 20
        assert thinker.tick_count == 20

    def test_learned_thinker_trains(self):
        from learned_thinking import LearnedThinker
        thinker = LearnedThinker(sensory_dim=21, buffer_size=200, train_interval=5, batch_size=8)
        thinker.start()
        for i in range(100):
            sensory = torch.randn(21)
            thinker.record_experience(sensory, "cold. dark." if i % 2 == 0 else "warm. good.")
        assert thinker.train_steps > 0
        assert thinker.stats['avg_loss'] > 0

    def test_learned_thinker_not_ready_initially(self):
        from learned_thinking import LearnedThinker
        thinker = LearnedThinker(sensory_dim=21)
        assert not thinker.is_ready()
        assert thinker.confidence == 0.0

    def test_blend_thoughts_uses_rule_based_when_low_confidence(self):
        from learned_thinking import blend_thoughts
        result = blend_thoughts("rule thought", "learned", 0.1)
        assert result == "rule thought"

    def test_blend_thoughts_uses_learned_when_high_confidence(self):
        from learned_thinking import blend_thoughts
        # Force deterministic choice
        import random
        random.seed(42)
        # With confidence=1.0, should always use learned
        result = blend_thoughts("rule", "learned", 1.0)
        assert result == "learned"

    def test_blend_thoughts_rejects_gibberish(self):
        from learned_thinking import blend_thoughts
        # Learned thought with too few alpha chars should be rejected
        result = blend_thoughts("rule thought", "...", 1.0)
        assert result == "rule thought"

    def test_learned_thinker_save_load(self):
        from learned_thinking import LearnedThinker
        import tempfile
        thinker1 = LearnedThinker(sensory_dim=21)
        thinker1.start()
        # Record some experiences to train
        for i in range(50):
            thinker1.record_experience(torch.randn(21), "test thought")
        # Save
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            path = f.name
        thinker1.save(path)
        # Load into new thinker
        thinker2 = LearnedThinker(sensory_dim=21)
        thinker2.load(path)
        assert thinker2.train_steps == thinker1.train_steps
        os.unlink(path)

    def test_parameter_count_reasonable(self):
        from learned_thinking import ThoughtTransformer
        model = ThoughtTransformer(sensory_dim=21)
        params = model.parameter_count()
        # Should be between 100K and 2M
        assert 100_000 < params < 2_000_000


# ═══════════════════════════════════════════════════════════════════════════════
# Eve Agent Tests (v0.3)
# ═══════════════════════════════════════════════════════════════════════════════

class TestEveAgent:
    def test_eve_can_be_created(self):
        from eve_agent import EveAgent
        from world_sim import WorldSim
        env = WorldSim()
        env.reset()
        eve = EveAgent(env, device='cpu')
        assert eve.name == "Eve"
        assert eve.model is not None
        assert eve.thought_engine is not None

    def test_eve_birth_sets_position(self):
        from eve_agent import EveAgent
        from world_sim import WorldSim
        env = WorldSim()
        env.reset()
        eve = EveAgent(env, device='cpu')
        eve.birth(adam_x=env.adam_x, adam_y=env.adam_y)
        assert eve.alive is True
        assert 0 <= eve.x < env.world_map.width
        assert 0 <= eve.y < env.world_map.height

    def test_eve_spawns_far_from_adam(self):
        from eve_agent import EveAgent
        from world_sim import WorldSim
        env = WorldSim()
        env.reset()
        eve = EveAgent(env, device='cpu')
        eve.birth(adam_x=env.adam_x, adam_y=env.adam_y)
        dist = abs(eve.x - env.adam_x) + abs(eve.y - env.adam_y)
        assert dist >= 10, f"Eve spawned too close to Adam (dist={dist})"

    def test_eve_has_separate_brain(self):
        from eve_agent import EveAgent
        from world_sim import WorldSim
        env = WorldSim()
        env.reset()
        eve = EveAgent(env, device='cpu')
        # Eve's model should be a different object than a fresh one
        assert eve.model is not None
        # Should have its own parameters
        params1 = list(eve.model.parameters())
        assert len(params1) > 0

    def test_eve_has_separate_thought_engine(self):
        from eve_agent import EveAgent
        from thought_engine import ThoughtEngine
        from world_sim import WorldSim
        env = WorldSim()
        env.reset()
        eve = EveAgent(env, device='cpu')
        assert isinstance(eve.thought_engine, ThoughtEngine)

    def test_eve_can_take_actions(self):
        from eve_agent import EveAgent
        from world_sim import WorldSim
        env = WorldSim()
        env.reset()
        eve = EveAgent(env, device='cpu')
        eve.birth(adam_x=env.adam_x, adam_y=env.adam_y)
        # Take a few actions
        for _ in range(5):
            action, _, _, _ = eve.choose_action(eve.get_world_state())
            result = eve.step(action, env.adam_x, env.adam_y)
            assert 'reward' in result
            assert 'done' in result
            assert 'thought' in result

    def test_eve_has_own_vocabulary(self):
        from eve_agent import EveAgent
        from world_sim import WorldSim
        env = WorldSim()
        env.reset()
        eve = EveAgent(env, device='cpu')
        eve.birth(adam_x=env.adam_x, adam_y=env.adam_y)
        # Eve should have starting vocabulary
        vocab = eve.thought_engine.get_vocabulary()
        assert len(vocab) > 0

    def test_eve_stats_are_independent(self):
        from eve_agent import EveAgent
        from world_sim import WorldSim
        env = WorldSim()
        env.reset()
        eve = EveAgent(env, device='cpu')
        eve.birth(adam_x=env.adam_x, adam_y=env.adam_y)
        # Eve's stats should be initialized
        assert eve.stats['health'] == 100.0
        assert eve.stats['energy'] == 100.0

    def test_eve_get_stats_for_dashboard(self):
        from eve_agent import EveAgent
        from world_sim import WorldSim
        env = WorldSim()
        env.reset()
        eve = EveAgent(env, device='cpu')
        eve.birth(adam_x=env.adam_x, adam_y=env.adam_y)
        stats = eve.get_stats()
        assert stats['name'] == 'Eve'
        assert stats['alive'] is True
        assert 'position' in stats
        assert 'thought' in stats
        assert 'emotion' in stats
        assert 'learned_thinking' in stats

    def test_eve_can_die(self):
        from eve_agent import EveAgent
        from world_sim import WorldSim
        env = WorldSim()
        env.reset()
        eve = EveAgent(env, device='cpu')
        eve.birth(adam_x=env.adam_x, adam_y=env.adam_y)
        # Force death
        eve.stats['health'] = 0
        result = eve.step("IDLE", env.adam_x, env.adam_y)
        assert result['done'] is True
        assert eve.alive is False
        assert eve.death_cause is not None

    def test_two_eves_have_different_minds(self):
        """Two Eve agents should have different initial brain weights."""
        from eve_agent import EveAgent
        from world_sim import WorldSim
        env = WorldSim()
        env.reset()
        eve1 = EveAgent(env, device='cpu', name="Eve1")
        eve1.birth()
        eve2 = EveAgent(env, device='cpu', name="Eve2")
        eve2.birth()
        # Their brain weights should be different
        params1 = list(eve1.model.parameters())
        params2 = list(eve2.model.parameters())
        # At least one parameter should differ
        any_diff = False
        for p1, p2 in zip(params1, params2):
            if not torch.equal(p1, p2):
                any_diff = True
                break
        assert any_diff, "Two agents should have different brain weights"

    def test_eve_senses_adam_when_nearby(self):
        """Eve should sense Adam's presence when he's nearby."""
        from eve_agent import EveAgent
        from world_sim import WorldSim
        env = WorldSim()
        env.reset()
        eve = EveAgent(env, device='cpu')
        eve.birth()
        # Place Adam right next to Eve
        adam_x = eve.x + 1
        adam_y = eve.y
        world_state = eve.get_world_state(adam_x, adam_y)
        assert world_state['other_presence'] > 0
        assert world_state['other_direction'] != "none"


# ═══════════════════════════════════════════════════════════════════════════════
# v0.3 Tests — Learned Thinking, Vocab Divergence, Multi-Agent
# ═══════════════════════════════════════════════════════════════════════════════

class TestLearnedThinking:
    """Test the tiny transformer that learns to generate thoughts."""

    def test_tokenizer_encode_decode(self):
        """Tokenizer should round-trip simple strings."""
        from learned_thinking import ThoughtTokenizer
        text = "cold. dark."
        encoded = ThoughtTokenizer.encode(text)
        decoded = ThoughtTokenizer.decode(encoded)
        assert decoded == "cold. dark.", f"Round-trip failed: '{decoded}'"

    def test_tokenizer_vocab_size(self):
        """Tokenizer should have ~31 tokens (29 chars + pad + eot)."""
        from learned_thinking import ThoughtTokenizer
        assert ThoughtTokenizer.VOCAB_SIZE == 31

    def test_tokenizer_handles_unknown_chars(self):
        """Unknown characters should map to space, not crash."""
        from learned_thinking import ThoughtTokenizer
        encoded = ThoughtTokenizer.encode("cold! @#$")
        decoded = ThoughtTokenizer.decode(encoded)
        # Special chars should be replaced with space
        assert "!" not in decoded
        assert "@" not in decoded

    def test_transformer_parameter_count(self):
        """Transformer should be tiny (< 1M params)."""
        from learned_thinking import ThoughtTransformer
        model = ThoughtTransformer(sensory_dim=21)
        params = model.parameter_count()
        assert 50000 < params < 1000000, f"Expected <1M params, got {params}"

    def test_transformer_forward_shape(self):
        """Forward pass should produce correct output shape."""
        from learned_thinking import ThoughtTransformer
        model = ThoughtTransformer(sensory_dim=21)
        sensory = torch.randn(2, 21)  # batch=2
        chars = torch.randint(0, 29, (2, 16))  # batch=2, seq=16
        logits = model(sensory, chars)
        assert logits.shape == (2, 16, 31), f"Got {logits.shape}"

    def test_learned_thinker_record_and_train(self):
        """LearnedThinker should record experiences and train without error."""
        from learned_thinking import LearnedThinker
        thinker = LearnedThinker(sensory_dim=21, device='cpu',
                                  train_interval=10, batch_size=8)
        thinker.start()
        # Record 20 experiences
        for i in range(20):
            sensory = torch.randn(21) * 0.5 + 0.5
            sensory = torch.clamp(sensory, 0, 1)
            thinker.record_experience(sensory, "cold. dark.")
        # Should have trained at least once
        assert thinker.train_steps > 0, "Should have trained"
        assert thinker.stats['total_experiences'] == 20

    def test_blend_thoughts_low_confidence_uses_rule(self):
        """At low confidence, blend should use rule-based thought."""
        from learned_thinking import blend_thoughts
        result = blend_thoughts("rule thought", "learned thought", confidence=0.1)
        assert result == "rule thought"

    def test_blend_thoughts_high_confidence_uses_learned(self):
        """At high confidence with valid learned thought, blend should use it."""
        from learned_thinking import blend_thoughts
        import random
        random.seed(42)
        # High confidence — should usually use learned
        uses_learned = 0
        for _ in range(100):
            result = blend_thoughts("rule thought", "learned thought", confidence=0.95)
            if result == "learned thought":
                uses_learned += 1
        assert uses_learned > 80, f"Expected >80/100 learned, got {uses_learned}"


class TestVocabDivergence:
    """Test vocabulary divergence logging between Adam and Eve."""

    def test_jaccard_identical_sets(self):
        """Identical vocabularies should have Jaccard=1.0."""
        from vocab_divergence import jaccard_similarity
        assert jaccard_similarity({"a", "b"}, {"a", "b"}) == 1.0

    def test_jaccard_disjoint_sets(self):
        """Disjoint vocabularies should have Jaccard=0.0."""
        from vocab_divergence import jaccard_similarity
        assert jaccard_similarity({"a", "b"}, {"c", "d"}) == 0.0

    def test_jaccard_partial_overlap(self):
        """Half-overlapping sets should have Jaccard=1/3."""
        from vocab_divergence import jaccard_similarity
        # |{a,b}|=2, |{b,c}|=2, |union|=3, |intersection|=1 → 1/3
        assert abs(jaccard_similarity({"a", "b"}, {"b", "c"}) - 1/3) < 1e-6

    def test_logger_creates_file(self, tmp_path):
        """Logger should create a JSONL log file."""
        from vocab_divergence import VocabDivergenceLogger
        log_path = str(tmp_path / "vocab.jsonl")
        logger = VocabDivergenceLogger(log_path=log_path, log_interval=10)
        logger.log(tick=10, adam_vocab=["a", "b"], eve_vocab=["b", "c"])
        assert os.path.exists(log_path)
        # File should have one line
        with open(log_path) as f:
            lines = f.readlines()
        assert len(lines) == 1
        import json
        entry = json.loads(lines[0])
        assert entry["tick"] == 10
        assert entry["adam_vocab_size"] == 2
        assert entry["shared_count"] == 1
        assert abs(entry["jaccard"] - 1/3) < 1e-4

    def test_logger_tracks_new_words(self, tmp_path):
        """Logger should track words added since last log."""
        from vocab_divergence import VocabDivergenceLogger
        logger = VocabDivergenceLogger(log_path=str(tmp_path / "v.jsonl"), log_interval=10)
        # First log
        logger.log(10, ["a", "b"], ["a", "b"])
        # Second log with new words
        entry = logger.log(20, ["a", "b", "c"], ["a", "b", "d"])
        assert entry["adam_new"] == ["c"]
        assert entry["eve_new"] == ["d"]

    def test_logger_should_log_interval(self, tmp_path):
        """should_log should only return True at interval boundaries."""
        from vocab_divergence import VocabDivergenceLogger
        logger = VocabDivergenceLogger(log_path=str(tmp_path / "v.jsonl"), log_interval=50)
        assert not logger.should_log(49)
        assert logger.should_log(50)
        assert not logger.should_log(51)
        assert logger.should_log(100)


class TestMultiAgentIntegration:
    """Test multi-agent infrastructure (without full training loop)."""

    def test_compute_other_presence_far(self):
        """Agents far apart should have presence=0."""
        from train_multi_agent import _compute_other_presence
        presence, direction = _compute_other_presence(0, 0, 10, 10)
        assert presence == 0.0
        assert direction == "none"

    def test_compute_other_presence_adjacent(self):
        """Adjacent agents should have high presence."""
        from train_multi_agent import _compute_other_presence
        presence, direction = _compute_other_presence(5, 5, 6, 5)
        assert presence > 0.5
        assert direction == "east"

    def test_compute_other_presence_same_tile(self):
        """Agents on same tile should have direction='here'."""
        from train_multi_agent import _compute_other_presence
        presence, direction = _compute_other_presence(5, 5, 5, 5)
        assert direction == "here"
        assert presence > 0.7

    def test_agent_runtime_init(self):
        """AgentRuntime should initialize with separate brains."""
        from train_multi_agent import AgentRuntime
        import torch
        adam = AgentRuntime("Adam", torch.device("cpu"), is_adam=True)
        eve = AgentRuntime("Eve", torch.device("cpu"), is_adam=False)
        # Brains should be separate objects
        assert adam.model is not eve.model
        # Brain weights should be different (random init)
        adam_w = next(adam.model.parameters())
        eve_w = next(eve.model.parameters())
        assert not torch.equal(adam_w, eve_w)

    def test_multi_agent_short_run(self, tmp_path, monkeypatch):
        """Run multi-agent loop for 30 ticks — should not crash."""
        from train_multi_agent import run_multi_agent_life
        # Change to tmp dir so we don't pollute the repo
        monkeypatch.chdir(tmp_path)
        try:
            run_multi_agent_life(
                learned_only=False,
                max_ticks=30,
                tick_delay=0.0,
            )
        except Exception as e:
            pytest.fail(f"Multi-agent run crashed: {e}")
        # Should have created life logs
        assert (tmp_path / "life_log_adam.json").exists()
        assert (tmp_path / "life_log_eve.json").exists()
        assert (tmp_path / "vocab_divergence.jsonl").exists()

    def test_multi_agent_vocab_diverges(self, tmp_path, monkeypatch):
        """Over 100 ticks, Adam and Eve vocabularies should diverge."""
        from train_multi_agent import run_multi_agent_life
        import json
        monkeypatch.chdir(tmp_path)
        run_multi_agent_life(
            learned_only=False,
            max_ticks=100,
            tick_delay=0.0,
        )
        # Read vocab divergence log
        with open(tmp_path / "vocab_divergence.jsonl") as f:
            entries = [json.loads(line) for line in f]
        assert len(entries) >= 1
        # Final entry should show some divergence (jaccard < 1.0)
        final = entries[-1]
        assert final["jaccard"] < 1.0, "Vocab should diverge over 100 ticks"


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 0.3 — Per-word vocab_log.jsonl + Convergence detection
# ═══════════════════════════════════════════════════════════════════════════════

class TestVocabWordLog:
    """Tests for the per-word vocab_log.jsonl (MD Phase 0.3 spec)."""

    def test_word_log_format_matches_md_spec(self, tmp_path):
        """vocab_log.jsonl entries must contain {tick, agent, word, context_of_discovery}."""
        from vocab_divergence import VocabDivergenceLogger
        logger = VocabDivergenceLogger(
            log_path=str(tmp_path / "v.jsonl"),
            word_log_path=str(tmp_path / "vocab_log.jsonl"),
            convergence_log_path=str(tmp_path / "vocab_conv.jsonl"),
        )
        logger.log_word_discovery(
            tick=42, agent="adam", word="cactus sand",
            meaning="a dry place", trigger="ENTERED_DESERT",
            context={"biome": "desert", "temperature": 38},
        )
        import json
        with open(tmp_path / "vocab_log.jsonl") as f:
            entry = json.loads(f.readline())
        # MD Phase 0.3 spec required keys
        assert entry["tick"] == 42
        assert entry["agent"] == "adam"
        assert entry["word"] == "cactus sand"
        assert "context_of_discovery" in entry
        assert entry["context_of_discovery"]["biome"] == "desert"

    def test_convergence_detection_same_trigger(self, tmp_path):
        """When both agents discover a word for the same trigger, flag as convergence."""
        from vocab_divergence import VocabDivergenceLogger
        import json
        logger = VocabDivergenceLogger(
            log_path=str(tmp_path / "v.jsonl"),
            word_log_path=str(tmp_path / "vocab_log.jsonl"),
            convergence_log_path=str(tmp_path / "vocab_conv.jsonl"),
        )
        # Adam discovers first
        logger.log_word_discovery(
            tick=10, agent="adam", word="cactus sand",
            meaning="dry place", trigger="ENTERED_DESERT",
            context={"biome": "desert"},
        )
        # Eve discovers a different word for the same trigger later
        logger.log_word_discovery(
            tick=50, agent="eve", word="sun hot",
            meaning="dry place", trigger="ENTERED_DESERT",
            context={"biome": "desert"},
        )
        with open(tmp_path / "vocab_conv.jsonl") as f:
            conv = json.loads(f.readline())
        assert conv["convergence_type"] == "SAME_TRIGGER"
        assert conv["adam_word"] == "cactus sand"
        assert conv["eve_word"] == "sun hot"
        assert conv["first_discoverer"] == "adam"
        assert conv["time_gap_ticks"] == 40

    def test_convergence_detection_exact_match(self, tmp_path):
        """EXACT_MATCH convergence when both agents invent the same word."""
        from vocab_divergence import VocabDivergenceLogger
        import json
        logger = VocabDivergenceLogger(
            log_path=str(tmp_path / "v.jsonl"),
            word_log_path=str(tmp_path / "vocab_log.jsonl"),
            convergence_log_path=str(tmp_path / "vocab_conv.jsonl"),
        )
        logger.log_word_discovery(
            tick=10, agent="adam", word="cold dark",
            meaning="a cave", trigger="ENTERED_CAVE",
            context={"biome": "cave"},
        )
        logger.log_word_discovery(
            tick=20, agent="eve", word="cold dark",
            meaning="a cave", trigger="ENTERED_CAVE",
            context={"biome": "cave"},
        )
        with open(tmp_path / "vocab_conv.jsonl") as f:
            conv = json.loads(f.readline())
        assert conv["convergence_type"] == "EXACT_MATCH"

    def test_no_convergence_when_only_one_agent_discovers(self, tmp_path):
        """Convergence log stays empty if only one agent discovers."""
        from vocab_divergence import VocabDivergenceLogger
        logger = VocabDivergenceLogger(
            log_path=str(tmp_path / "v.jsonl"),
            word_log_path=str(tmp_path / "vocab_log.jsonl"),
            convergence_log_path=str(tmp_path / "vocab_conv.jsonl"),
        )
        logger.log_word_discovery(
            tick=10, agent="adam", word="cactus sand",
            meaning="dry", trigger="ENTERED_DESERT",
            context={},
        )
        # No file or empty file
        import os
        assert not os.path.exists(tmp_path / "vocab_conv.jsonl") or \
               os.path.getsize(tmp_path / "vocab_conv.jsonl") == 0

    def test_render_dashboard_html(self, tmp_path):
        """Dashboard HTML should be created and contain expected sections."""
        from vocab_divergence import VocabDivergenceLogger
        out = tmp_path / "dashboard.html"
        logger = VocabDivergenceLogger(
            log_path=str(tmp_path / "v.jsonl"),
            word_log_path=str(tmp_path / "vocab_log.jsonl"),
            convergence_log_path=str(tmp_path / "vocab_conv.jsonl"),
        )
        logger.log_word_discovery(
            tick=1, agent="adam", word="hot",
            meaning="warm", trigger="HOT",
            context={},
        )
        logger.log_word_discovery(
            tick=2, agent="eve", word="warm",
            meaning="warm", trigger="HOT",
            context={},
        )
        path = logger.render_dashboard_html(str(out))
        assert os.path.exists(path)
        html = open(path).read()
        assert "Adam's Vocabulary" in html
        assert "Eve's Vocabulary" in html
        assert "Convergence" in html
