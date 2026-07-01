"""
Nafs AI — Social Engine (Phase 8)
=================================

Implements MD Phase 8: with reproduction comes family. With family comes
social structure. With social structure comes everything humanity has
ever built.

Covers:
  8.1 Relationship Memory
      - relationship_memory: {agent_id: {trust, familiarity, last_seen_tick,
        relation_type}}
      - Relation types: STRANGER / FAMILIAR / TRUSTED / FAMILY / MATE
      - Trust increases with: food sharing, proximity without aggression,
        following successfully
      - Trust decreases with: aggression, stealing food tile, sudden flee

  8.2 Family Bonds
      - Parent-child bond: automatic FAMILY relation at birth
      - Parents receive reward signal tied to child's survival
      - Child follows parent passively for first 200 ticks (attachment)
      - Parent teaches implicitly: child observes parent eat safe food

  8.3 Group Behaviour
      - 3+ agents with TRUSTED relations in 10-tile radius → GROUP state
      - Group hunting: multiple agents pursuing same prey tile
      - Group warmth: 3+ agents sleeping in same tile → lower body_temp drain
      - Group defence: danger → FAMILY agents move toward each other

  8.4 Territory & Resources
      - Agents track most-visited tiles as 'home territory'
      - CLAIM action: mark tile as territory (fades after 500 ticks)
      - Intruder sensory signal: stress boost when stranger in your territory
      - Conflict resolution: higher trust group usually wins disputes

  8.5 Population Dynamics
      - Food density naturally limits population (Malthusian pressure)
      - Disease spreads between agents in close proximity
      - Population > 10: resource competition forces territorial behavior
      - Population < 3: reproduction incentive increases

Design constraints:
  - Does NOT modify base rewards
  - Standalone module
  - Builds on Phase 7 (FirstContactEngine) for trust scores

Usage:
    from social import SocialEngine
    social = SocialEngine()
    social.register_family(parent_id, child_id, tick)
    social.update_relationship(agent_a, agent_b, 'share', tick)
    group = social.detect_group(agent_positions, tick)
    social.claim_territory(agent_id, x, y, tick)
    social.check_population_dynamics(num_agents, food_density)
"""

import random
from typing import Dict, List, Optional, Tuple, Set, Any
from collections import defaultdict


# ═══════════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════════

# Relation types
RELATION_STRANGER = "STRANGER"
RELATION_FAMILIAR = "FAMILIAR"
RELATION_TRUSTED = "TRUSTED"
RELATION_FAMILY = "FAMILY"
RELATION_MATE = "MATE"

# Trust thresholds for relation type
FAMILIAR_TRUST_THRESHOLD = 0.1
TRUSTED_TRUST_THRESHOLD = 0.5
MATE_TRUST_THRESHOLD = 0.8

# Familiarity increases per tick in proximity
FAMILIARITY_PER_TICK = 0.01
FAMILIARITY_DECAY = 0.995  # decays when not in proximity

# Family
ATTACHMENT_DURATION = 200  # ticks child follows parent
PARENT_CHILD_SURVIVAL_REWARD = 0.1  # parent gets this per tick child is alive

# Group behavior
GROUP_MIN_AGENTS = 3
GROUP_RADIUS = 10  # tiles
GROUP_WARMTH_MIN_AGENTS = 3
GROUP_WARMTH_BENEFIT = 0.5  # body_temp drain reduction

# Territory
TERRITORY_FADE_TICKS = 500  # claim fades after 500 ticks without presence
INTRUDER_STRESS_BOOST = 0.5

# Population dynamics
POPULATION_HIGH_THRESHOLD = 10  # resource competition
POPULATION_LOW_THRESHOLD = 3   # reproduction incentive
DISEASE_SPREAD_DISTANCE = 2    # tiles
DISEASE_SPREAD_PROB = 0.05     # per tick in proximity


# ═══════════════════════════════════════════════════════════════════════════════
# SocialEngine
# ═══════════════════════════════════════════════════════════════════════════════

class SocialEngine:
    """
    Master social engine for the Nafs AI world.

    Holds:
      - relationships: {(agent_a, agent_b): {trust, familiarity, last_seen, type}}
      - family_registry: {child_id: (parent1, parent2, birth_tick)}
      - territories: {agent_id: {tile: claim_tick}}
      - groups: detected groups per tick
      - population history
    """

    def __init__(self, first_contact_engine=None, seed: Optional[int] = None):
        self.first_contact = first_contact_engine
        self.rng = random.Random(seed or random.randint(0, 999999))

        # Relationships: {(a, b): {trust, familiarity, last_seen_tick, type}}
        self.relationships: Dict[Tuple[str, str], Dict] = {}

        # Family registry: {child_id: {parents: [p1, p2], birth_tick: int}}
        self.family_registry: Dict[str, Dict] = {}

        # Territories: {agent_id: {tile_tuple: last_claimed_tick}}
        self.territories: Dict[str, Dict[Tuple[int, int], int]] = defaultdict(dict)

        # Active groups (updated per tick)
        self.active_groups: List[Dict] = []

        # Population tracking
        self.population_history: List[Tuple[int, int]] = []  # (tick, count)

        # Disease tracking (which agents are infected)
        self.infected_agents: Set[str] = set()

        self.current_tick = 0

    def _pair_key(self, a: str, b: str) -> Tuple[str, str]:
        return (a, b) if a <= b else (b, a)

    # ─────────────────────────────────────────────────────────────────────────
    # 8.1 Relationship Memory
    # ─────────────────────────────────────────────────────────────────────────

    def get_relationship(self, agent_a: str, agent_b: str) -> Dict:
        """Get relationship info between two agents."""
        pair = self._pair_key(agent_a, agent_b)
        if pair not in self.relationships:
            self.relationships[pair] = {
                "trust": 0.0,
                "familiarity": 0.0,
                "last_seen_tick": None,
                "relation_type": RELATION_STRANGER,
                "interactions": 0,
            }
        return self.relationships[pair]

    def update_relationship(self, agent_a: str, agent_b: str,
                              event: str, tick: int,
                              value: float = 0.0) -> Dict:
        """
        Update relationship based on an event.

        Events:
          - 'share': trust + (positive)
          - 'proximity': familiarity +
          - 'follow_success': trust + small
          - 'aggression': trust - large
          - 'steal_food': trust - large
          - 'flee': trust - small
        """
        rel = self.get_relationship(agent_a, agent_b)
        rel["last_seen_tick"] = tick
        rel["interactions"] += 1

        if event == "share":
            rel["trust"] = min(1.0, rel["trust"] + 0.1)
        elif event == "proximity":
            rel["familiarity"] = min(1.0, rel["familiarity"] + FAMILIARITY_PER_TICK)
        elif event == "follow_success":
            rel["trust"] = min(1.0, rel["trust"] + 0.05)
            rel["familiarity"] = min(1.0, rel["familiarity"] + 0.02)
        elif event == "aggression":
            rel["trust"] = max(-1.0, rel["trust"] - 0.3)
        elif event == "steal_food":
            rel["trust"] = max(-1.0, rel["trust"] - 0.4)
        elif event == "flee":
            rel["trust"] = max(-1.0, rel["trust"] - 0.1)

        # Update relation type based on trust
        rel["relation_type"] = self._compute_relation_type(rel, agent_a, agent_b)
        return rel

    def _compute_relation_type(self, rel: Dict, agent_a: str, agent_b: str) -> str:
        """Determine relation type from trust + family status."""
        # Check family first (highest priority)
        if self._are_family(agent_a, agent_b):
            return RELATION_FAMILY
        # Check mate (very high trust)
        if rel["trust"] >= MATE_TRUST_THRESHOLD:
            return RELATION_MATE
        if rel["trust"] >= TRUSTED_TRUST_THRESHOLD:
            return RELATION_TRUSTED
        if rel["familiarity"] >= FAMILIAR_TRUST_THRESHOLD or rel["trust"] >= FAMILIAR_TRUST_THRESHOLD:
            return RELATION_FAMILIAR
        return RELATION_STRANGER

    def _are_family(self, agent_a: str, agent_b: str) -> bool:
        """Check if two agents are family (parent-child or siblings)."""
        for child_id, info in self.family_registry.items():
            parents = info["parents"]
            # Parent-child
            if (agent_a == child_id and agent_b in parents) or \
               (agent_b == child_id and agent_a in parents):
                return True
            # Siblings (share at least one parent)
            if agent_a in parents and agent_b in parents:
                return True
        return False

    def decay_familiarity(self) -> None:
        """Slowly decay familiarity for agents not in proximity."""
        for rel in self.relationships.values():
            rel["familiarity"] *= FAMILIARITY_DECAY

    # ─────────────────────────────────────────────────────────────────────────
    # 8.2 Family Bonds
    # ─────────────────────────────────────────────────────────────────────────

    def register_family(self, child_id: str, parents: List[str],
                          birth_tick: int) -> None:
        """Register a parent-child family relationship."""
        self.family_registry[child_id] = {
            "parents": list(parents),
            "birth_tick": birth_tick,
        }
        # Set FAMILY relation type for each parent-child pair
        for parent_id in parents:
            pair = self._pair_key(parent_id, child_id)
            if pair not in self.relationships:
                self.relationships[pair] = {
                    "trust": 0.5,  # family starts with baseline trust
                    "familiarity": 1.0,
                    "last_seen_tick": birth_tick,
                    "relation_type": RELATION_FAMILY,
                    "interactions": 0,
                }
            else:
                self.relationships[pair]["relation_type"] = RELATION_FAMILY
                self.relationships[pair]["trust"] = max(0.5, self.relationships[pair]["trust"])

    def is_in_attachment_period(self, child_id: str, tick: int) -> bool:
        """Check if child is still in attachment period (first 200 ticks)."""
        info = self.family_registry.get(child_id)
        if not info:
            return False
        return (tick - info["birth_tick"]) < ATTACHMENT_DURATION

    def get_parent_survival_reward(self, child_id: str, tick: int) -> Dict:
        """
        Returns reward signal for parents based on child's survival.
        Called each tick the child is alive.
        """
        info = self.family_registry.get(child_id)
        if not info:
            return {"parents": [], "reward": 0.0}

        return {
            "parents": info["parents"],
            "reward": PARENT_CHILD_SURVIVAL_REWARD,
            "child_id": child_id,
            "tick": tick,
        }

    def get_attachment_target(self, child_id: str) -> Optional[str]:
        """Get the parent the child should follow (first parent)."""
        info = self.family_registry.get(child_id)
        if not info:
            return None
        return info["parents"][0] if info["parents"] else None

    # ─────────────────────────────────────────────────────────────────────────
    # 8.3 Group Behaviour
    # ─────────────────────────────────────────────────────────────────────────

    def detect_groups(self, agent_positions: Dict[str, Tuple[int, int]],
                        tick: int) -> List[Dict]:
        """
        Detect groups of 3+ trusted agents within 10-tile radius.

        Returns list of group dicts: {members: [ids], center: (x, y), size: int}
        """
        self.active_groups = []
        agent_ids = list(agent_positions.keys())

        # For each agent, find trusted agents within GROUP_RADIUS
        for i, agent_a in enumerate(agent_ids):
            members = [agent_a]
            pos_a = agent_positions[agent_a]
            for agent_b in agent_ids[i+1:]:
                pos_b = agent_positions[agent_b]
                dist = abs(pos_a[0] - pos_b[0]) + abs(pos_a[1] - pos_b[1])
                if dist <= GROUP_RADIUS:
                    rel = self.get_relationship(agent_a, agent_b)
                    if rel["relation_type"] in (RELATION_TRUSTED, RELATION_FAMILY, RELATION_MATE):
                        members.append(agent_b)

            if len(members) >= GROUP_MIN_AGENTS:
                # Compute center
                cx = sum(agent_positions[m][0] for m in members) // len(members)
                cy = sum(agent_positions[m][1] for m in members) // len(members)
                group = {
                    "members": members,
                    "center": (cx, cy),
                    "size": len(members),
                    "tick_detected": tick,
                }
                # Avoid duplicates (don't add if same members already in a group)
                already = False
                for existing in self.active_groups:
                    if set(existing["members"]) == set(members):
                        already = True
                        break
                if not already:
                    self.active_groups.append(group)

        return self.active_groups

    def get_group_warmth_benefit(self, agent_id: str,
                                   agent_positions: Dict[str, Tuple[int, int]]
                                   ) -> float:
        """
        Returns body_temp drain reduction if 3+ agents are sleeping in same tile.
        """
        if agent_id not in agent_positions:
            return 0.0
        pos = agent_positions[agent_id]
        # Count agents on same tile
        same_tile_count = sum(1 for p in agent_positions.values() if p == pos)
        if same_tile_count >= GROUP_WARMTH_MIN_AGENTS:
            return GROUP_WARMTH_BENEFIT
        return 0.0

    def get_family_defense_targets(self, agent_id: str,
                                     agent_positions: Dict[str, Tuple[int, int]]
                                     ) -> List[str]:
        """
        Returns list of family members that should move toward agent_id
        when danger is present.
        """
        targets = []
        for other_id in agent_positions:
            if other_id == agent_id:
                continue
            rel = self.get_relationship(agent_id, other_id)
            if rel["relation_type"] == RELATION_FAMILY:
                targets.append(other_id)
        return targets

    # ─────────────────────────────────────────────────────────────────────────
    # 8.4 Territory & Resources
    # ─────────────────────────────────────────────────────────────────────────

    def claim_territory(self, agent_id: str, x: int, y: int, tick: int) -> None:
        """Agent claims a tile as territory."""
        self.territories[agent_id][(x, y)] = tick

    def is_territory(self, agent_id: str, x: int, y: int) -> bool:
        """Check if tile is agent's territory (not faded)."""
        tile = (x, y)
        if tile not in self.territories.get(agent_id, {}):
            return False
        last_claimed = self.territories[agent_id][tile]
        return (self.current_tick - last_claimed) < TERRITORY_FADE_TICKS

    def get_territory_owner(self, x: int, y: int) -> Optional[str]:
        """Get the agent whose territory this tile belongs to (if any)."""
        tile = (x, y)
        for agent_id, claims in self.territories.items():
            if tile in claims:
                last_claimed = claims[tile]
                if (self.current_tick - last_claimed) < TERRITORY_FADE_TICKS:
                    return agent_id
        return None

    def check_intruder(self, agent_id: str, x: int, y: int) -> Dict:
        """
        Check if agent is intruding on another agent's territory.

        Returns dict with:
          - is_intruder: bool
          - territory_owner: Optional[str]
          - stress_boost: float
        """
        owner = self.get_territory_owner(x, y)
        if owner is None or owner == agent_id:
            return {"is_intruder": False, "territory_owner": None, "stress_boost": 0.0}

        # Check if they're family — family can enter each other's territory
        if self._are_family(agent_id, owner):
            return {"is_intruder": False, "territory_owner": owner, "stress_boost": 0.0}

        return {
            "is_intruder": True,
            "territory_owner": owner,
            "stress_boost": INTRUDER_STRESS_BOOST,
        }

    def cleanup_expired_claims(self) -> int:
        """Remove territory claims older than TERRITORY_FADE_TICKS.
        Returns count of removed claims."""
        removed = 0
        for agent_id in list(self.territories.keys()):
            for tile in list(self.territories[agent_id].keys()):
                if (self.current_tick - self.territories[agent_id][tile]) >= TERRITORY_FADE_TICKS:
                    del self.territories[agent_id][tile]
                    removed += 1
            if not self.territories[agent_id]:
                del self.territories[agent_id]
        return removed

    def resolve_territory_dispute(self, agent_a: str, agent_b: str) -> str:
        """
        Resolve a territory dispute between two agents.
        Higher trust group wins.
        Returns the winning agent_id.
        """
        # Simple version: agent with more total trust wins
        trust_a = sum(rel["trust"] for pair, rel in self.relationships.items()
                       if agent_a in pair)
        trust_b = sum(rel["trust"] for pair, rel in self.relationships.items()
                       if agent_b in pair)
        return agent_a if trust_a >= trust_b else agent_b

    # ─────────────────────────────────────────────────────────────────────────
    # 8.5 Population Dynamics
    # ─────────────────────────────────────────────────────────────────────────

    def update_population(self, tick: int, num_agents: int) -> Dict:
        """Track population and return dynamics signals."""
        self.population_history.append((tick, num_agents))

        signals = {
            "population": num_agents,
            "overpopulated": num_agents > POPULATION_HIGH_THRESHOLD,
            "underpopulated": num_agents < POPULATION_LOW_THRESHOLD,
            "reproduction_incentive": 1.0 if num_agents < POPULATION_LOW_THRESHOLD else 0.0,
            "resource_competition": 1.0 if num_agents > POPULATION_HIGH_THRESHOLD else 0.0,
        }
        return signals

    def spread_disease(self, agent_positions: Dict[str, Tuple[int, int]],
                         tick: int) -> List[Dict]:
        """
        Spread disease between agents in close proximity.

        Returns list of newly infected agents: [{agent_id, source_id, tick}]
        """
        new_infections = []
        agent_ids = list(agent_positions.keys())
        for i, agent_a in enumerate(agent_ids):
            if agent_a not in self.infected_agents:
                continue
            for agent_b in agent_ids[i+1:]:
                if agent_b in self.infected_agents:
                    continue
                pos_a = agent_positions[agent_a]
                pos_b = agent_positions[agent_b]
                dist = abs(pos_a[0] - pos_b[0]) + abs(pos_a[1] - pos_b[1])
                if dist <= DISEASE_SPREAD_DISTANCE:
                    if self.rng.random() < DISEASE_SPREAD_PROB:
                        self.infected_agents.add(agent_b)
                        new_infections.append({
                            "agent_id": agent_b,
                            "source_id": agent_a,
                            "tick": tick,
                        })
        return new_infections

    def infect_agent(self, agent_id: str) -> None:
        """Mark an agent as infected."""
        self.infected_agents.add(agent_id)

    def cure_agent(self, agent_id: str) -> None:
        """Mark an agent as cured."""
        self.infected_agents.discard(agent_id)

    def is_infected(self, agent_id: str) -> bool:
        return agent_id in self.infected_agents

    # ─────────────────────────────────────────────────────────────────────────
    # Main step
    # ─────────────────────────────────────────────────────────────────────────

    def step(self, tick: int,
              agent_positions: Optional[Dict[str, Tuple[int, int]]] = None) -> Dict:
        """Per-tick social engine update."""
        self.current_tick = tick

        # Decay familiarity for absent agents
        if tick % 100 == 0:
            self.decay_familiarity()

        # Cleanup expired territory claims
        if tick % 200 == 0:
            self.cleanup_expired_claims()

        # Detect groups if positions provided
        groups = []
        if agent_positions:
            groups = self.detect_groups(agent_positions, tick)

        return {
            "active_groups": groups,
            "current_tick": tick,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Summary + serialization
    # ─────────────────────────────────────────────────────────────────────────

    def get_summary(self) -> Dict:
        return {
            "total_relationships": len(self.relationships),
            "family_members": len(self.family_registry),
            "active_groups": len(self.active_groups),
            "territory_claims": sum(len(t) for t in self.territories.values()),
            "infected_agents": len(self.infected_agents),
            "population_history_length": len(self.population_history),
        }

    def to_dict(self) -> Dict:
        return {
            'relationships': {
                f"{k[0]}|{k[1]}": v for k, v in self.relationships.items()
            },
            'family_registry': self.family_registry,
            'territories': {
                agent: {f"{k[0]},{k[1]}": v for k, v in claims.items()}
                for agent, claims in self.territories.items()
            },
            'infected_agents': list(self.infected_agents),
            'current_tick': self.current_tick,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Self-test
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Testing SocialEngine...")

    social = SocialEngine(seed=42)

    # Test 8.1: Relationship memory
    rel = social.get_relationship("adam", "eve")
    print(f"  Initial relationship: {rel}")
    assert rel["relation_type"] == "STRANGER"

    # Build trust through sharing
    for _ in range(5):
        social.update_relationship("adam", "eve", "share", tick=100)
    rel = social.get_relationship("adam", "eve")
    print(f"  After 5 shares: trust={rel['trust']:.2f}, type={rel['relation_type']}")
    assert rel["trust"] > 0
    assert rel["relation_type"] in ("FAMILIAR", "TRUSTED")

    # Test aggression drops trust
    trust_before = social.get_relationship("adam", "eve")["trust"]
    social.update_relationship("adam", "eve", "aggression", tick=110)
    trust_after = social.get_relationship("adam", "eve")["trust"]
    print(f"  After aggression: trust={trust_after:.2f}")
    assert trust_after < trust_before

    # Test 8.2: Family bonds
    social2 = SocialEngine(seed=42)
    social2.register_family("baby_1", ["adam", "eve"], birth_tick=100)
    rel = social2.get_relationship("adam", "baby_1")
    print(f"  Parent-child relation: {rel['relation_type']}")
    assert rel["relation_type"] == "FAMILY"

    # Attachment period
    assert social2.is_in_attachment_period("baby_1", tick=150)  # 50 < 200
    assert not social2.is_in_attachment_period("baby_1", tick=400)  # 300 > 200

    # Parent survival reward
    reward = social2.get_parent_survival_reward("baby_1", tick=200)
    print(f"  Parent survival reward: {reward}")
    assert reward["reward"] > 0
    assert "adam" in reward["parents"]

    # Test 8.3: Group behavior
    social3 = SocialEngine(seed=42)
    # Make adam, eve, baby all trust each other
    for _ in range(10):
        social3.update_relationship("adam", "eve", "share", tick=100)
        social3.update_relationship("adam", "carol", "share", tick=100)
        social3.update_relationship("eve", "carol", "share", tick=100)
    positions = {
        "adam": (5, 5), "eve": (6, 5), "carol": (7, 5),
        "dave": (50, 50),  # far away
    }
    groups = social3.detect_groups(positions, tick=200)
    print(f"  Detected groups: {len(groups)}")
    assert len(groups) >= 1
    assert groups[0]["size"] >= 3

    # Group warmth benefit
    positions_same_tile = {
        "adam": (5, 5), "eve": (5, 5), "carol": (5, 5),
    }
    warmth = social3.get_group_warmth_benefit("adam", positions_same_tile)
    print(f"  Group warmth benefit: {warmth}")
    assert warmth > 0

    # Test 8.4: Territory
    social4 = SocialEngine(seed=42)
    social4.current_tick = 100
    social4.claim_territory("adam", 5, 5, tick=100)
    assert social4.is_territory("adam", 5, 5)
    assert not social4.is_territory("adam", 6, 6)

    # Intruder check
    intruder = social4.check_intruder("eve", 5, 5)
    print(f"  Intruder check: {intruder}")
    assert intruder["is_intruder"]
    assert intruder["territory_owner"] == "adam"
    assert intruder["stress_boost"] > 0

    # Territory fades
    social4.current_tick = 700  # 600 > 500 (fade time)
    assert not social4.is_territory("adam", 5, 5)

    # Test 8.5: Population dynamics
    social5 = SocialEngine(seed=42)
    signals = social5.update_population(tick=100, num_agents=2)
    print(f"  Population 2: {signals}")
    assert signals["underpopulated"]
    assert signals["reproduction_incentive"] > 0

    signals = social5.update_population(tick=200, num_agents=15)
    print(f"  Population 15: {signals}")
    assert signals["overpopulated"]
    assert signals["resource_competition"] > 0

    # Disease spread
    social5.infect_agent("adam")
    positions = {"adam": (5, 5), "eve": (5, 6)}  # adjacent
    infections = social5.spread_disease(positions, tick=300)
    print(f"  Disease spread attempts: {len(infections)} new infections")
    # Note: probabilistic, may or may not spread on a single call

    # Test step
    social6 = SocialEngine(seed=42)
    result = social6.step(tick=100, agent_positions={"adam": (5, 5), "eve": (5, 5)})
    print(f"  Step result: {result}")

    # Summary
    summary = social3.get_summary()
    print(f"  Summary: {summary}")

    print("\n✓ SocialEngine self-test passed")
