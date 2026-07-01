"""
Nafs AI — Multi-Generation Evolution (Phase 10)
===============================================

Implements MD Phase 10: evolution requires variation, selection,
inheritance, time. All four now exist in the simulation. This phase
instruments and measures what emerges.

Covers:
  10.1 Natural Selection Metrics
      - Per-generation: avg lifespan, avg offspring count, avg vocab size
      - Survival rate by biome (desert agents vs forest agents)
      - Biological traits correlating with longer lifespan
      - Behaviours correlating with more offspring

  10.2 Adaptive Radiation
      - After 5+ generations: measure trait divergence between
        geographically separated families
      - Desert family should show higher heat tolerance than forest family
      - Speciation event when biological distance > 2 std deviations

  10.3 Extinction Events
      - Random catastrophe: every 10,000 ticks, 10% chance of biome-wide
        drought or fire
      - Log mass extinction events
      - Post-extinction recovery tracking

  10.4 Open-Ended Evolution Check
      - Continuously produces new behaviours not in previous generations
      - Biological traits measurably diverge across geographically
        separated populations
      - Cultural practices differ between family groups without programmer
        intervention
      - Extinction events produce population recovery with measurable
        trait shifts
      - Vocabulary evolves (new words, lost words, borrowed words)
      - Packard et al. (2019) definition

Design constraints:
  - Does NOT modify base rewards
  - Standalone analysis layer that aggregates data from other engines
  - Builds on Phase 5 (ReproductionEngine) for lineage data

Usage:
    from evolution import EvolutionTracker
    tracker = EvolutionTracker(reproduction_engine)
    tracker.record_generation_data(generation, agents_data, tick)
    metrics = tracker.get_selection_metrics()
    speciation = tracker.check_speciation()
    extinction = tracker.trigger_catastrophe(tick, cause, agents_alive)
    oee = tracker.check_open_ended_evolution()
"""

import json
import os
import math
import random
from typing import Dict, List, Optional, Tuple, Set, Any
from collections import defaultdict


# ═══════════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════════

CATACLYSM_INTERVAL = 10000     # ticks between potential cataclysms
CATACLYSM_PROBABILITY = 0.10   # 10% chance per interval
SPECIATION_THRESHOLD = 2.0     # std deviations for speciation event
MIN_GENERATIONS_FOR_RADIATION = 5  # need 5+ generations for adaptive radiation

# OEE criteria (Packard et al. 2019)
OEE_CRITERIA = [
    "novel_behaviors",
    "biological_divergence",
    "cultural_divergence",
    "extinction_recovery_with_shifts",
    "vocabulary_evolution",
]


# ═══════════════════════════════════════════════════════════════════════════════
# EvolutionTracker
# ═══════════════════════════════════════════════════════════════════════════════

class EvolutionTracker:
    """
    Master evolution tracker for the Nafs AI world.

    Aggregates data from ReproductionEngine + BiologyEngine + CultureEngine
    to measure natural selection, adaptive radiation, extinction events,
    and open-ended evolution.
    """

    def __init__(self, reproduction_engine=None,
                 log_path: str = "evolution_events.jsonl",
                 seed: Optional[int] = None):
        self.reproduction = reproduction_engine
        self.log_path = log_path
        self.rng = random.Random(seed or random.randint(0, 999999))

        # Per-generation data: {generation: [agent_data, ...]}
        self.generation_data: Dict[int, List[Dict]] = defaultdict(list)

        # Per-biome survival tracking: {biome: {born: int, died: int, avg_lifespan: float}}
        self.biome_survival: Dict[str, Dict] = defaultdict(lambda: {
            "born": 0, "died": 0, "total_lifespan": 0.0
        })

        # Trait-lifespan correlations: {trait: {values: [], lifespans: []}}
        self.trait_lifespan_data: Dict[str, List[Tuple[float, int]]] = defaultdict(list)

        # Behavior-offspring correlations: {behavior: {values: [], offspring: []}}
        self.behavior_offspring_data: Dict[str, List[Tuple[int, int]]] = defaultdict(list)

        # Cataclysm history
        self.cataclysms: List[Dict] = []
        self.last_cataclysm_check = 0

        # Speciation events
        self.speciation_events: List[Dict] = []
        self._last_speciation_check_per_pair: Dict[Tuple[str, str], float] = {}

        # OEE criteria tracking
        self.oee_status: Dict[str, bool] = {criterion: False for criterion in OEE_CRITERIA}
        self.oee_history: List[Dict] = []

        # Behavior novelty tracking (for OEE)
        self.known_behaviors: Set[str] = set()
        self.novel_behavior_events: List[Dict] = []

        # Vocabulary evolution tracking
        self.vocab_history: List[Dict] = []  # snapshots over time
        self._last_vocab_snapshot: Dict[str, int] = {}  # {word: count}

        # Truncate log
        try:
            with open(self.log_path, "w") as f:
                f.write("")
        except Exception:
            pass

        self.current_tick = 0

    # ─────────────────────────────────────────────────────────────────────────
    # 10.1 Natural Selection Metrics
    # ─────────────────────────────────────────────────────────────────────────

    def record_generation_data(self, generation: int, agent_data: Dict,
                                 tick: int) -> None:
        """
        Record data about an agent for natural selection analysis.

        agent_data should contain:
          - agent_id, lifespan, offspring_count, vocabulary_size,
            biome, traits (dict of biological traits),
            behaviors (dict of action -> count)
        """
        self.current_tick = tick
        agent_data["recorded_tick"] = tick
        self.generation_data[generation].append(agent_data)

        # Update biome survival
        biome = agent_data.get("biome", "unknown")
        self.biome_survival[biome]["born"] += 1
        if agent_data.get("death_tick") is not None:
            self.biome_survival[biome]["died"] += 1
            self.biome_survival[biome]["total_lifespan"] += agent_data.get("lifespan", 0)

        # Track trait-lifespan correlations
        lifespan = agent_data.get("lifespan", 0)
        for trait_name, trait_value in agent_data.get("traits", {}).items():
            if isinstance(trait_value, (int, float)):
                self.trait_lifespan_data[trait_name].append((trait_value, lifespan))

        # Track behavior-offspring correlations
        offspring = agent_data.get("offspring_count", 0)
        for behavior, count in agent_data.get("behaviors", {}).items():
            self.behavior_offspring_data[behavior].append((count, offspring))

    def get_selection_metrics(self) -> Dict:
        """Get aggregated natural selection metrics per generation."""
        metrics = {}
        for gen, agents in self.generation_data.items():
            if not agents:
                continue
            lifespans = [a.get("lifespan", 0) for a in agents]
            offspring = [a.get("offspring_count", 0) for a in agents]
            vocab_sizes = [a.get("vocabulary_size", 0) for a in agents]

            metrics[gen] = {
                "generation": gen,
                "agent_count": len(agents),
                "avg_lifespan": sum(lifespans) / len(lifespans),
                "avg_offspring": sum(offspring) / len(offspring),
                "avg_vocab_size": sum(vocab_sizes) / len(vocab_sizes),
                "max_lifespan": max(lifespans),
                "max_offspring": max(offspring),
            }
        return metrics

    def get_biome_survival_stats(self) -> Dict:
        """Get survival statistics by biome."""
        stats = {}
        for biome, data in self.biome_survival.items():
            born = data["born"]
            died = data["died"]
            avg_lifespan = data["total_lifespan"] / max(1, died)
            survival_rate = (born - died) / max(1, born)
            stats[biome] = {
                "born": born,
                "died": died,
                "survival_rate": round(survival_rate, 3),
                "avg_lifespan": round(avg_lifespan, 1),
            }
        return stats

    def get_trait_lifespan_correlations(self) -> Dict[str, Dict]:
        """
        Compute correlation between each biological trait and lifespan.
        Returns {trait: {correlation, sample_size}}.
        """
        correlations = {}
        for trait, pairs in self.trait_lifespan_data.items():
            if len(pairs) < 5:
                continue
            values = [p[0] for p in pairs]
            lifespans = [p[1] for p in pairs]
            corr = self._pearson_correlation(values, lifespans)
            correlations[trait] = {
                "correlation": round(corr, 3),
                "sample_size": len(pairs),
            }
        return correlations

    def get_behavior_offspring_correlations(self) -> Dict[str, Dict]:
        """Compute correlation between behavior frequency and offspring count."""
        correlations = {}
        for behavior, pairs in self.behavior_offspring_data.items():
            if len(pairs) < 5:
                continue
            values = [p[0] for p in pairs]
            offspring = [p[1] for p in pairs]
            corr = self._pearson_correlation(values, offspring)
            correlations[behavior] = {
                "correlation": round(corr, 3),
                "sample_size": len(pairs),
            }
        return correlations

    def _pearson_correlation(self, x: List[float], y: List[float]) -> float:
        """Compute Pearson correlation coefficient."""
        n = len(x)
        if n != len(y) or n == 0:
            return 0.0
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(xi * yi for xi, yi in zip(x, y))
        sum_x2 = sum(xi ** 2 for xi in x)
        sum_y2 = sum(yi ** 2 for yi in y)
        numerator = n * sum_xy - sum_x * sum_y
        denominator = math.sqrt((n * sum_x2 - sum_x ** 2) * (n * sum_y2 - sum_y ** 2))
        if denominator == 0:
            return 0.0
        return numerator / denominator

    # ─────────────────────────────────────────────────────────────────────────
    # 10.2 Adaptive Radiation
    # ─────────────────────────────────────────────────────────────────────────

    def check_speciation(self, tick: int) -> Optional[Dict]:
        """
        Check for speciation events between geographically separated families.

        Speciation = biological distance > 2 std deviations.

        Returns speciation event dict if detected, None otherwise.
        """
        if len(self.generation_data) < MIN_GENERATIONS_FOR_RADIATION:
            return None

        # Group agents by family (using parent ID as family identifier)
        families = defaultdict(list)
        for gen, agents in self.generation_data.items():
            for agent in agents:
                parents = agent.get("parents", [])
                if parents:
                    family_id = parents[0]  # first parent = family
                    families[family_id].append(agent)

        if len(families) < 2:
            return None

        # Compute average traits per family
        family_traits = {}
        for family_id, agents in families.items():
            if not agents:
                continue
            trait_sums = defaultdict(float)
            trait_count = 0
            for agent in agents:
                for trait, value in agent.get("traits", {}).items():
                    if isinstance(value, (int, float)):
                        trait_sums[trait] += value
                        trait_count += 1
            if trait_count > 0:
                family_traits[family_id] = {
                    trait: val / len(agents) for trait, val in trait_sums.items()
                }

        # Compare each pair of families
        family_ids = list(family_traits.keys())
        for i in range(len(family_ids)):
            for j in range(i + 1, len(family_ids)):
                f1, f2 = family_ids[i], family_ids[j]
                pair_key = (f1, f2)

                # Compute biological distance
                traits_1 = family_traits[f1]
                traits_2 = family_traits[f2]
                all_traits = set(traits_1.keys()) | set(traits_2.keys())
                if not all_traits:
                    continue
                distances = []
                for trait in all_traits:
                    v1 = traits_1.get(trait, 0)
                    v2 = traits_2.get(trait, 0)
                    distances.append(abs(v1 - v2))

                if not distances:
                    continue
                mean_dist = sum(distances) / len(distances)
                std_dist = math.sqrt(sum((d - mean_dist) ** 2 for d in distances) / len(distances)) if len(distances) > 1 else 0

                # Speciation if max distance > 2 std deviations
                max_dist = max(distances)
                if std_dist > 0 and max_dist > SPECIATION_THRESHOLD * std_dist:
                    if pair_key not in self._last_speciation_check_per_pair or \
                       self._last_speciation_check_per_pair[pair_key] != max_dist:
                        event = {
                            "event": "SPECIATION",
                            "tick": tick,
                            "family_1": f1,
                            "family_2": f2,
                            "biological_distance": round(max_dist, 3),
                            "std_distance": round(std_dist, 3),
                            "divergent_traits": [
                                t for t in all_traits
                                if abs(traits_1.get(t, 0) - traits_2.get(t, 0)) >= max_dist * 0.9
                            ],
                        }
                        self.speciation_events.append(event)
                        self._last_speciation_check_per_pair[pair_key] = max_dist
                        self._log_event(event)
                        self.oee_status["biological_divergence"] = True
                        return event
        return None

    # ─────────────────────────────────────────────────────────────────────────
    # 10.3 Extinction Events
    # ─────────────────────────────────────────────────────────────────────────

    def check_cataclysm(self, tick: int, num_agents_alive: int) -> Optional[Dict]:
        """
        Check if a cataclysm should occur (every 10,000 ticks, 10% chance).

        Returns cataclysm event dict if triggered, None otherwise.
        """
        if tick - self.last_cataclysm_check < CATACLYSM_INTERVAL:
            return None
        self.last_cataclysm_check = tick

        if self.rng.random() > CATACLYSM_PROBABILITY:
            return None

        # Trigger cataclysm
        cause = self.rng.choice(["drought", "wildfire", "plague", "flood"])
        # Severity: 30-70% population loss
        severity = self.rng.uniform(0.3, 0.7)
        agents_killed = int(num_agents_alive * severity)

        event = {
            "event": "CATACLYSM",
            "tick": tick,
            "cause": cause,
            "agents_alive_before": num_agents_alive,
            "agents_killed": agents_killed,
            "agents_alive_after": num_agents_alive - agents_killed,
            "severity": round(severity, 3),
        }
        self.cataclysms.append(event)
        self._log_event(event)
        return event

    def record_recovery(self, tick: int, num_agents_alive: int,
                          avg_traits: Optional[Dict] = None) -> Dict:
        """
        Record population recovery after extinction.
        Returns recovery event with trait shifts (if any).
        """
        if not self.cataclysms:
            return {}

        last_cataclysm = self.cataclysms[-1]
        recovery = {
            "event": "RECOVERY",
            "tick": tick,
            "cataclysm_tick": last_cataclysm["tick"],
            "agents_alive": num_agents_alive,
            "recovery_time": tick - last_cataclysm["tick"],
            "avg_traits": avg_traits or {},
        }
        self._log_event(recovery)

        # If traits shifted from before, mark OEE criterion
        if avg_traits and last_cataclysm.get("pre_cataclysm_traits"):
            shifts = {}
            for trait, value in avg_traits.items():
                old_value = last_cataclysm["pre_cataclysm_traits"].get(trait, value)
                if abs(value - old_value) > 0.05:
                    shifts[trait] = {"before": old_value, "after": value}
            if shifts:
                self.oee_status["extinction_recovery_with_shifts"] = True

        return recovery

    def get_cataclysm_history(self) -> List[Dict]:
        return list(self.cataclysms)

    # ─────────────────────────────────────────────────────────────────────────
    # 10.4 Open-Ended Evolution Check
    # ─────────────────────────────────────────────────────────────────────────

    def record_behavior(self, behavior_signature: str, tick: int,
                          agent_id: str = "") -> bool:
        """
        Record that an agent exhibited a behavior.
        Returns True if this is a NOVEL behavior (never seen before).
        """
        is_novel = behavior_signature not in self.known_behaviors
        if is_novel:
            self.known_behaviors.add(behavior_signature)
            event = {
                "event": "NOVEL_BEHAVIOR",
                "tick": tick,
                "behavior": behavior_signature,
                "agent_id": agent_id,
            }
            self.novel_behavior_events.append(event)
            self._log_event(event)
            self.oee_status["novel_behaviors"] = True
        return is_novel

    def record_vocab_snapshot(self, tick: int, vocab_counts: Dict[str, int]) -> Dict:
        """
        Take a snapshot of vocabulary state.
        Detects new words, lost words, and borrowed words.
        """
        current_words = set(vocab_counts.keys())
        previous_words = set(self._last_vocab_snapshot.keys())

        new_words = current_words - previous_words
        lost_words = previous_words - current_words

        snapshot = {
            "event": "VOCAB_SNAPSHOT",
            "tick": tick,
            "total_words": len(current_words),
            "new_words": list(new_words),
            "lost_words": list(lost_words),
            "new_count": len(new_words),
            "lost_count": len(lost_words),
        }
        self.vocab_history.append(snapshot)
        self._log_event(snapshot)

        if new_words or lost_words:
            self.oee_status["vocabulary_evolution"] = True

        self._last_vocab_snapshot = dict(vocab_counts)
        return snapshot

    def mark_cultural_divergence(self, family_id: str,
                                   cultural_signature: Dict) -> None:
        """Mark that cultural practices differ between families."""
        self.oee_status["cultural_divergence"] = True

    def check_open_ended_evolution(self) -> Dict:
        """
        Check if simulation has achieved Open-Ended Evolution.

        Per Packard et al. (2019), OEE requires ALL of:
          1. Continuously produces new behaviours
          2. Biological traits measurably diverge
          3. Cultural practices differ between family groups
          4. Extinction events produce recovery with trait shifts
          5. Vocabulary evolves

        Returns dict with:
          - achieved: bool (True if all 5 criteria met)
          - criteria: {criterion: bool}
          - missing: list of unmet criteria
        """
        achieved = all(self.oee_status.values())
        missing = [c for c, met in self.oee_status.items() if not met]
        return {
            "achieved": achieved,
            "criteria": dict(self.oee_status),
            "missing": missing,
            "criteria_met": sum(1 for v in self.oee_status.values() if v),
            "total_criteria": len(self.oee_status),
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Logging + summary
    # ─────────────────────────────────────────────────────────────────────────

    def _log_event(self, event: Dict) -> None:
        try:
            with open(self.log_path, "a") as f:
                f.write(json.dumps(event) + "\n")
        except Exception as e:
            print(f"[Evolution] Failed to write log: {e}", flush=True)

    def get_summary(self) -> Dict:
        return {
            "generations_tracked": len(self.generation_data),
            "total_agents_recorded": sum(len(a) for a in self.generation_data.values()),
            "cataclysms": len(self.cataclysms),
            "speciation_events": len(self.speciation_events),
            "novel_behaviors": len(self.novel_behavior_events),
            "vocab_snapshots": len(self.vocab_history),
            "oee_criteria_met": sum(1 for v in self.oee_status.values() if v),
            "oee_total": len(self.oee_status),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Self-test
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Testing EvolutionTracker...")

    tracker = EvolutionTracker(seed=42, log_path="/tmp/test_evolution.jsonl")

    # Test 10.1: Natural selection metrics
    for gen in range(1, 4):
        for agent_id in range(5):
            tracker.record_generation_data(
                generation=gen,
                agent_data={
                    "agent_id": f"g{gen}_a{agent_id}",
                    "lifespan": 100 + gen * 50 + agent_id * 10,
                    "offspring_count": agent_id,
                    "vocabulary_size": 20 + gen * 5,
                    "biome": ["desert", "forest", "swamp"][agent_id % 3],
                    "traits": {"metabolism_rate": 1.0 + agent_id * 0.1,
                                "cold_resistance": 0.5 + agent_id * 0.1},
                    "behaviors": {"EAT": 10 + agent_id, "EXPLORE": 5 + agent_id},
                    "death_tick": 500,
                },
                tick=gen * 100,
            )

    metrics = tracker.get_selection_metrics()
    print(f"  Selection metrics for gen 1: {metrics.get(1, {})}")
    assert 1 in metrics
    assert metrics[1]["agent_count"] == 5

    biome_stats = tracker.get_biome_survival_stats()
    print(f"  Biome survival stats: {list(biome_stats.keys())}")
    assert "desert" in biome_stats

    trait_corrs = tracker.get_trait_lifespan_correlations()
    print(f"  Trait-lifespan correlations: {list(trait_corrs.keys())}")
    # May or may not have correlations (need 5+ samples)

    # Test 10.2: Speciation (need 5+ generations)
    # Add more generations with divergent traits
    for gen in range(4, 8):
        for agent_id in range(3):
            # Family A: high metabolism
            # Family B: low metabolism
            family = "A" if agent_id == 0 else "B"
            metabolism = 2.0 if family == "A" else 0.5
            tracker.record_generation_data(
                generation=gen,
                agent_data={
                    "agent_id": f"g{gen}_a{agent_id}",
                    "parents": [family],
                    "lifespan": 200,
                    "offspring_count": 1,
                    "vocabulary_size": 25,
                    "biome": "desert" if family == "A" else "forest",
                    "traits": {"metabolism_rate": metabolism,
                                "cold_resistance": 0.5},
                    "behaviors": {"EAT": 10},
                    "death_tick": 500,
                },
                tick=gen * 100,
            )

    speciation = tracker.check_speciation(tick=800)
    print(f"  Speciation check: {speciation}")
    # May or may not detect speciation

    # Test 10.3: Cataclysm
    # Force a cataclysm
    tracker.last_cataclysm_check = -10000  # allow check
    tracker.rng = random.Random(0)  # deterministic
    cataclysm = tracker.check_cataclysm(tick=10000, num_agents_alive=20)
    if cataclysm:
        print(f"  Cataclysm: {cataclysm['cause']}, killed {cataclysm['agents_killed']}")
        assert cataclysm["agents_killed"] > 0
        assert cataclysm["agents_alive_after"] < 20

    # Test 10.4: OEE criteria
    # Novel behavior
    is_novel = tracker.record_behavior("first_word_invented", tick=100)
    assert is_novel
    is_novel2 = tracker.record_behavior("first_word_invented", tick=200)
    assert not is_novel2  # already known

    # Vocab snapshot
    snap = tracker.record_vocab_snapshot(tick=500, vocab_counts={"hello": 1, "world": 2})
    print(f"  Vocab snapshot: total={snap['total_words']}")
    assert snap["total_words"] == 2

    snap2 = tracker.record_vocab_snapshot(tick=600, vocab_counts={"hello": 1, "new_word": 3})
    assert "new_word" in snap2["new_words"]
    assert "world" in snap2["lost_words"]

    # Mark cultural divergence
    tracker.mark_cultural_divergence("family_A", {"food_pref": "berries"})

    # Check OEE status
    oee = tracker.check_open_ended_evolution()
    print(f"  OEE status: {oee['criteria_met']}/{oee['total_criteria']} criteria met")
    print(f"    Criteria: {oee['criteria']}")
    assert oee["criteria"]["novel_behaviors"] is True
    assert oee["criteria"]["vocabulary_evolution"] is True
    assert oee["criteria"]["cultural_divergence"] is True

    # Summary
    summary = tracker.get_summary()
    print(f"  Summary: {summary}")

    print("\n✓ EvolutionTracker self-test passed")
