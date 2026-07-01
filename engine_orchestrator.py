"""
Nafs AI — Engine Orchestrator (Phase 4-13 wiring)
==================================================

Bundles all phase-3+ engines into a single object that train_multi_agent.py
can call at the right hooks. This is the file that *actually wires* Phases
4-13 into the training loop:

    Phase 4  → GrowingBrain          (loss tracking + adaptive growth)
    Phase 5  → ReproductionEngine    (fertility, pregnancy, baby spawn)
    Phase 6  → MathIntuitionEngine   (food density, spatial memory, time sense)
    Phase 7  → FirstContactEngine    (first contact event, interactions, trust)
    Phase 8  → SocialEngine          (relationships, groups, territory)
    Phase 9  → CultureEngine         (observational learning, proto-tools, vocab lineage)
    Phase 10 → EvolutionTracker      (selection metrics, speciation, OEE check)
    Phase 11 → GodotBridge           (state serialization for Godot client)
    Phase 12 → EventLogSystem + LineageDatabase (event log + lineage DB)
    Phase 13 → WorldEvolution + DiseaseEvolution + NoveltyDetector + WorldSeeding

The orchestrator is *defensive* — every engine call is wrapped in try/except
so a bug in one engine never crashes the training loop. Failed calls are
logged to orchestrator_errors.jsonl.

Design constraints:
  - Does NOT modify base rewards (world_sim.py untouched)
  - All engines are optional — orchestrator works even if some engines fail
    to instantiate
  - Per-tick overhead is small (no engine does heavy compute on the hot path)
  - Heavy operations (speciation check, OEE check, snapshot) run periodically,
    not every tick

Usage (from train_multi_agent.py):

    from engine_orchestrator import EngineOrchestrator
    orch = EngineOrchestrator(env, device=str(DEVICE))
    orch.register_agent("adam", parents=[], generation=1, birth_tick=0, biome=...)
    orch.register_agent("eve",  parents=[], generation=1, birth_tick=0, biome=...)

    # Each tick:
    orch.on_tick(tick, adam_rt, eve_rt, adam_action, eve_action,
                 adam_reward, eve_reward, world_state)

    # On PPO update:
    orch.on_ppo_update("adam", loss, tick)
    orch.on_ppo_update("eve",  loss, tick)

    # On EAT:
    orch.on_eat("adam", food_type, x, y, tick)

    # On new word:
    orch.on_new_word("adam", word, meaning, tick)

    # On death:
    orch.on_death("adam", tick, cause)

    # Periodic:
    oee_status = orch.check_oee()           # every 1000 ticks
    orch.snapshot("auto", tick)             # every 5000 ticks
"""

import os
import json
import time
import random
import traceback
from collections import deque
from typing import Dict, List, Optional, Any, Tuple

# ── Phase 4-13 engine imports (all optional) ────────────────────────────────
try:
    from growing_brain import GrowingBrain
    GROWING_BRAIN_AVAILABLE = True
except Exception:
    GROWING_BRAIN_AVAILABLE = False

try:
    from reproduction import ReproductionEngine
    REPRODUCTION_AVAILABLE = True
except Exception:
    REPRODUCTION_AVAILABLE = False

try:
    from math_intuition import MathIntuitionEngine
    MATH_AVAILABLE = True
except Exception:
    MATH_AVAILABLE = False

try:
    from first_contact import FirstContactEngine
    FIRST_CONTACT_AVAILABLE = True
except Exception:
    FIRST_CONTACT_AVAILABLE = False

try:
    from social import SocialEngine
    SOCIAL_AVAILABLE = True
except Exception:
    SOCIAL_AVAILABLE = False

try:
    from culture import CultureEngine
    CULTURE_AVAILABLE = True
except Exception:
    CULTURE_AVAILABLE = False

try:
    from evolution import EvolutionTracker
    EVOLUTION_AVAILABLE = True
except Exception:
    EVOLUTION_AVAILABLE = False

try:
    from events import EventLogSystem, LineageDatabase
    EVENTS_AVAILABLE = True
except Exception:
    EVENTS_AVAILABLE = False

try:
    from open_ended import WorldEvolution, DiseaseEvolution, NoveltyDetector, WorldSeeding
    OPEN_ENDED_AVAILABLE = True
except Exception:
    OPEN_ENDED_AVAILABLE = False

try:
    from godot_bridge import GodotBridge
    GODOT_BRIDGE_AVAILABLE = True
except Exception:
    GODOT_BRIDGE_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════════════════════
# EngineOrchestrator
# ═══════════════════════════════════════════════════════════════════════════════

class EngineOrchestrator:
    """Bundles Phase 4-13 engines and exposes simple hooks for the training loop."""

    # Periodic cadence (in ticks)
    OEE_CHECK_INTERVAL = 1000          # check_open_ended_evolution()
    SNAPSHOT_INTERVAL = 5000           # WorldSeeding.snapshot()
    SPECIATION_CHECK_INTERVAL = 500    # evolution.check_speciation()
    CATACLYSM_CHECK_INTERVAL = 500     # evolution.check_cataclysm()
    CULTURE_DIGEST_INTERVAL = 200      # update_cultural_signature for each agent
    BEHAVIOR_RECORD_INTERVAL = 50      # record behavior signature

    def __init__(self, env, device: str = "cpu", seed: Optional[int] = None,
                 godot_state_path: str = "godot_state.json"):
        self.env = env
        self.device = device
        self.seed = seed
        self.godot_state_path = godot_state_path

        # Error log (append-only)
        self._error_log_path = "orchestrator_errors.jsonl"
        self._error_count = 0

        # Get physics engine from env (world_sim already instantiates one)
        self.physics = getattr(env, "physics", None)
        world_map = getattr(env, "world_map", None)

        # ── Phase 5: Reproduction ──────────────────────────────────────────
        self.reproduction = None
        if REPRODUCTION_AVAILABLE and self.physics is not None:
            try:
                self.reproduction = ReproductionEngine(
                    physics_engine=self.physics, seed=seed,
                    log_path="generations.jsonl",
                )
            except Exception as e:
                self._log_error("init", "reproduction", e)

        # ── Phase 6: Math Intuition ────────────────────────────────────────
        self.math = None
        if MATH_AVAILABLE and world_map is not None:
            try:
                self.math = MathIntuitionEngine(world_map, seed=seed)
            except Exception as e:
                self._log_error("init", "math_intuition", e)

        # ── Phase 7: First Contact ─────────────────────────────────────────
        self.first_contact = None
        if FIRST_CONTACT_AVAILABLE:
            try:
                self.first_contact = FirstContactEngine(
                    log_path="first_contact.jsonl", seed=seed,
                )
            except Exception as e:
                self._log_error("init", "first_contact", e)

        # ── Phase 8: Social ────────────────────────────────────────────────
        self.social = None
        if SOCIAL_AVAILABLE and self.first_contact is not None:
            try:
                self.social = SocialEngine(
                    first_contact_engine=self.first_contact, seed=seed,
                )
            except Exception as e:
                self._log_error("init", "social", e)

        # ── Phase 9: Culture ───────────────────────────────────────────────
        self.culture = None
        if CULTURE_AVAILABLE:
            try:
                self.culture = CultureEngine(
                    log_path="culture_events.jsonl", seed=seed,
                )
            except Exception as e:
                self._log_error("init", "culture", e)

        # ── Phase 10: Evolution ────────────────────────────────────────────
        self.evolution = None
        if EVOLUTION_AVAILABLE and self.reproduction is not None:
            try:
                self.evolution = EvolutionTracker(
                    reproduction_engine=self.reproduction,
                    log_path="evolution_events.jsonl", seed=seed,
                )
            except Exception as e:
                self._log_error("init", "evolution", e)

        # ── Phase 12: Events + Lineage ─────────────────────────────────────
        self.events = None
        self.lineage = None
        if EVENTS_AVAILABLE:
            try:
                self.events = EventLogSystem(log_path="events.jsonl")
            except Exception as e:
                self._log_error("init", "events", e)
            try:
                self.lineage = LineageDatabase(db_path="lineage.db")
            except Exception as e:
                self._log_error("init", "lineage", e)

        # ── Phase 13: Open-ended systems ───────────────────────────────────
        self.world_evol = None
        self.disease_evol = None
        self.novelty = None
        self.world_seeding = None
        if OPEN_ENDED_AVAILABLE:
            if world_map is not None:
                try:
                    self.world_evol = WorldEvolution(world_map, seed=seed)
                except Exception as e:
                    self._log_error("init", "world_evol", e)
            try:
                self.disease_evol = DiseaseEvolution(seed=seed)
            except Exception as e:
                self._log_error("init", "disease_evol", e)
            try:
                self.novelty = NoveltyDetector()
            except Exception as e:
                self._log_error("init", "novelty", e)
            try:
                self.world_seeding = WorldSeeding(snapshot_dir="snapshots")
            except Exception as e:
                self._log_error("init", "world_seeding", e)

        # ── Phase 11: Godot bridge ─────────────────────────────────────────
        self.godot = None
        if GODOT_BRIDGE_AVAILABLE:
            try:
                biology = getattr(env, "biology", None)
                chemistry = getattr(env, "chemistry", None)
                self.godot = GodotBridge(
                    world_sim=env, physics=self.physics,
                    chemistry=chemistry, biology=biology,
                    reproduction=self.reproduction,
                    first_contact=self.first_contact,
                    social=self.social, evolution=self.evolution,
                )
            except Exception as e:
                self._log_error("init", "godot_bridge", e)

        # ── State ──────────────────────────────────────────────────────────
        self.agent_ids: List[str] = []          # ordered list of registered agent IDs
        self.agent_generation: Dict[str, int] = {}
        self.baby_runtimes: Dict[str, Dict] = {} # baby_id → lightweight runtime info
        self.last_oee_status: Optional[Dict] = None
        self.first_contact_fired = False
        self.first_birth_fired = False
        self.first_death_fired = False
        self.first_word_fired = False
        self.recent_behaviors: deque = deque(maxlen=1000)
        self.recent_oee_checks: deque = deque(maxlen=50)

        # Summary printed at end of training
        self.tick_count = 0

    # ─────────────────────────────────────────────────────────────────────────
    # Error logging
    # ─────────────────────────────────────────────────────────────────────────
    def _log_error(self, hook: str, engine: str, exc: Exception):
        """Log an engine error to orchestrator_errors.jsonl (best-effort)."""
        self._error_count += 1
        try:
            with open(self._error_log_path, "a") as f:
                f.write(json.dumps({
                    "ts": time.time(),
                    "hook": hook,
                    "engine": engine,
                    "error": str(exc)[:500],
                    "traceback": traceback.format_exc()[-1500:],
                }) + "\n")
        except Exception:
            pass  # never let error logging crash the loop

    # ─────────────────────────────────────────────────────────────────────────
    # Agent registration
    # ─────────────────────────────────────────────────────────────────────────
    def register_agent(self, agent_id: str, parents: List[str], generation: int,
                        birth_tick: int, biome: str = "", traits: Optional[Dict] = None):
        """Register an agent with reproduction engine + lineage DB."""
        self.agent_ids.append(agent_id)
        self.agent_generation[agent_id] = generation

        if self.reproduction is not None:
            try:
                self.reproduction.register_agent(
                    agent_id=agent_id, parents=parents, generation=generation,
                    birth_tick=birth_tick, traits=traits,
                )
            except Exception as e:
                self._log_error("register_agent", "reproduction", e)

        if self.lineage is not None:
            try:
                self.lineage.insert_agent(
                    agent_id=agent_id, parents=parents, generation=generation,
                    birth_tick=birth_tick, biome=biome, traits=traits,
                )
            except Exception as e:
                self._log_error("register_agent", "lineage", e)

        if self.events is not None:
            try:
                if generation == 1:
                    # Adam & Eve — not a "birth" event in the usual sense
                    self.events.record(
                        event_type="BIRTH", tick=birth_tick, agent_id=agent_id,
                        details={"generation": generation, "parents": parents,
                                 "biome": biome, "is_founder": True},
                        generation=generation,
                    )
                else:
                    self.events.record(
                        event_type="BIRTH", tick=birth_tick, agent_id=agent_id,
                        details={"generation": generation, "parents": parents,
                                 "biome": biome},
                        generation=generation,
                    )
            except Exception as e:
                self._log_error("register_agent", "events", e)

    # ─────────────────────────────────────────────────────────────────────────
    # Per-tick hook (called every tick from training loop)
    # ─────────────────────────────────────────────────────────────────────────
    def on_tick(self, tick: int, adam_rt, eve_rt, babies: List,
                adam_action: str, eve_action: str,
                adam_reward: float, eve_reward: float,
                world_state: Dict) -> Dict:
        """
        Main per-tick hook. Calls all engines that need per-tick updates.

        Returns a dict of notable events that happened this tick (for the
        training loop to log / display).
        """
        self.tick_count = tick
        events_this_tick = []

        # Build agent positions dict (for social engine)
        agent_positions = {}
        if adam_rt is not None:
            agent_positions["adam"] = (adam_rt.x, adam_rt.y)
        if eve_rt is not None:
            agent_positions["eve"] = (eve_rt.x, eve_rt.y)
        for baby in babies:
            agent_positions[baby["id"]] = (baby["x"], baby["y"])

        # ── Phase 6: Math intuition step ─────────────────────────────────
        if self.math is not None:
            try:
                self.math.step(tick)
                if adam_rt is not None:
                    self.math.record_visit(adam_rt.x, adam_rt.y, tick)
                if eve_rt is not None:
                    self.math.record_visit(eve_rt.x, eve_rt.y, tick)
            except Exception as e:
                self._log_error("on_tick", "math", e)

        # ── Phase 7: First Contact check ─────────────────────────────────
        if (self.first_contact is not None and adam_rt is not None
                and eve_rt is not None and not self.first_contact_fired):
            try:
                fc_event = self.first_contact.check_first_contact(
                    adam_pos=(adam_rt.x, adam_rt.y),
                    eve_pos=(eve_rt.x, eve_rt.y),
                    adam_vocab_size=len(adam_rt.thought_engine.get_vocabulary()),
                    eve_vocab_size=len(eve_rt.thought_engine.get_vocabulary()),
                    tick=tick, adam_id="adam", eve_id="eve",
                )
                if fc_event is not None:
                    self.first_contact_fired = True
                    events_this_tick.append({
                        "type": "FIRST_CONTACT",
                        "details": fc_event,
                    })
                    if self.events is not None:
                        self.events.record(
                            event_type="FIRST_CONTACT", tick=tick,
                            details=fc_event,
                        )
                    if self.godot is not None:
                        self.godot.fire_milestone(
                            "FIRST_CONTACT",
                            details={"tick": tick, **fc_event},
                        )
                    print(f"  \U0001f91d FIRST CONTACT at tick {tick}! "
                          f"Adam at {fc_event.get('adam_position')}, "
                          f"Eve at {fc_event.get('eve_position')}", flush=True)
            except Exception as e:
                self._log_error("on_tick", "first_contact", e)

        # ── Phase 7: Vocabulary contact (when within 2 tiles) ────────────
        if (self.first_contact is not None and adam_rt is not None
                and eve_rt is not None):
            try:
                adam_dialogue = adam_rt.latest_thought or ""
                eve_dialogue = eve_rt.latest_thought or ""
                self.first_contact.check_vocabulary_contact(
                    adam_pos=(adam_rt.x, adam_rt.y),
                    eve_pos=(eve_rt.x, eve_rt.y),
                    adam_dialogue=adam_dialogue,
                    eve_dialogue=eve_dialogue,
                    tick=tick, adam_id="adam", eve_id="eve",
                )
            except Exception as e:
                self._log_error("on_tick", "vocab_contact", e)

        # ── Phase 8: Social engine step ──────────────────────────────────
        if self.social is not None:
            try:
                self.social.step(tick, agent_positions=agent_positions)
            except Exception as e:
                self._log_error("on_tick", "social", e)

        # ── Phase 9: Culture — record observations when agents near each other
        if (self.culture is not None and self.first_contact is not None
                and adam_rt is not None and eve_rt is not None):
            try:
                dist = abs(adam_rt.x - eve_rt.x) + abs(adam_rt.y - eve_rt.y)
                if dist <= 5:
                    trust = self.first_contact.get_trust("adam", "eve")
                    self.culture.record_observation(
                        observer_id="eve", observed_id="adam",
                        action=adam_action, target=world_state.get("biome", ""),
                        reward=adam_reward, trust=trust, tick=tick,
                    )
                    self.culture.record_observation(
                        observer_id="adam", observed_id="eve",
                        action=eve_action, target=world_state.get("biome", ""),
                        reward=eve_reward, trust=trust, tick=tick,
                    )
            except Exception as e:
                self._log_error("on_tick", "culture_obs", e)

        # ── Phase 9: Proto-tool detection (shelter, fire use, cooking) ────
        if self.culture is not None and self.physics is not None:
            try:
                biome = world_state.get("biome", "plains")
                # Adam shelter/fire detection
                if adam_rt is not None:
                    self.culture.detect_shelter_use(
                        agent_id="adam", biome=biome,
                        action=adam_action, tick=tick,
                    )
                    fire_count = self.physics._count_adjacent_fires(adam_rt.x, adam_rt.y)
                    if fire_count > 0:
                        self.culture.detect_fire_use(
                            agent_id="adam", adjacent_fire_count=fire_count, tick=tick,
                        )
                if eve_rt is not None:
                    self.culture.detect_shelter_use(
                        agent_id="eve", biome=biome,
                        action=eve_action, tick=tick,
                    )
                    fire_count = self.physics._count_adjacent_fires(eve_rt.x, eve_rt.y)
                    if fire_count > 0:
                        self.culture.detect_fire_use(
                            agent_id="eve", adjacent_fire_count=fire_count, tick=tick,
                        )
            except Exception as e:
                self._log_error("on_tick", "culture_proto", e)

        # ── Phase 9: Cultural signature update (periodic) ────────────────
        if (self.culture is not None
                and tick % self.CULTURE_DIGEST_INTERVAL == 0):
            try:
                for aid in self.agent_ids:
                    self.culture.update_cultural_signature(aid)
            except Exception as e:
                self._log_error("on_tick", "culture_signature", e)

        # ── Phase 5: Reproduction — fertility check ──────────────────────
        if (self.reproduction is not None and adam_rt is not None
                and eve_rt is not None
                and adam_rt.alive and eve_rt.alive):
            try:
                repro_event = self._check_reproduction(tick, adam_rt, eve_rt)
                if repro_event:
                    events_this_tick.append(repro_event)
            except Exception as e:
                self._log_error("on_tick", "reproduction", e)

        # ── Phase 5: Pregnancy update ────────────────────────────────────
        if self.reproduction is not None and eve_rt is not None:
            try:
                preg_event = self._update_pregnancy_if_any(tick, eve_rt, adam_rt)
                if preg_event:
                    events_this_tick.append(preg_event)
            except Exception as e:
                self._log_error("on_tick", "pregnancy", e)

        # ── Phase 10: Evolution — behavior record + periodic checks ──────
        if self.evolution is not None:
            try:
                if tick % self.BEHAVIOR_RECORD_INTERVAL == 0:
                    sig = self._build_behavior_signature(
                        adam_rt, eve_rt, adam_action, eve_action,
                    )
                    self.evolution.record_behavior(sig, tick, agent_id="system")
                    if self.novelty is not None:
                        self.novelty.record_behavior(sig, tick)
            except Exception as e:
                self._log_error("on_tick", "evolution_behavior", e)

            try:
                if tick % self.SPECIATION_CHECK_INTERVAL == 0:
                    spec = self.evolution.check_speciation(tick)
                    if spec:
                        events_this_tick.append({"type": "SPECIATION", "details": spec})
                        if self.events is not None:
                            self.events.record(
                                event_type="SPECIATION", tick=tick, details=spec,
                            )
            except Exception as e:
                self._log_error("on_tick", "speciation", e)

            try:
                if tick % self.CATACLYSM_CHECK_INTERVAL == 0:
                    n_alive = sum(1 for aid in self.agent_ids
                                   if self._is_agent_alive(aid, adam_rt, eve_rt, babies))
                    cat = self.evolution.check_cataclysm(tick, n_alive)
                    if cat:
                        events_this_tick.append({"type": "EXTINCTION", "details": cat})
                        if self.events is not None:
                            self.events.record(
                                event_type="EXTINCTION", tick=tick, details=cat,
                            )
                        if self.godot is not None:
                            self.godot.fire_milestone(
                                "EXTINCTION", details={"tick": tick, **cat},
                            )
                        print(f"  \U0001f480 EXTINCTION EVENT at tick {tick}: "
                              f"{cat.get('cause', 'unknown')}", flush=True)
            except Exception as e:
                self._log_error("on_tick", "cataclysm", e)

            # Vocab snapshot for evolution tracker
            if tick % 500 == 0:
                try:
                    vocab_counts: Dict[str, int] = {}
                    if adam_rt is not None:
                        vocab_counts["adam"] = len(adam_rt.thought_engine.get_vocabulary())
                    if eve_rt is not None:
                        vocab_counts["eve"] = len(eve_rt.thought_engine.get_vocabulary())
                    self.evolution.record_vocab_snapshot(tick, vocab_counts)
                except Exception as e:
                    self._log_error("on_tick", "vocab_snapshot", e)

        # ── Phase 13: World evolution + disease evolution ────────────────
        if self.world_evol is not None:
            try:
                avg_temp = 20.0
                if self.physics is not None:
                    try:
                        avg_temp = float(getattr(self.physics, "body_temp", 20.0))
                    except Exception:
                        pass
                self.world_evol.step(tick, avg_temp=avg_temp)
            except Exception as e:
                self._log_error("on_tick", "world_evol", e)

        if self.disease_evol is not None:
            try:
                gen = max(self.agent_generation.values(), default=1)
                avg_imm = 0.0
                if (hasattr(self.env, "biology") and self.env.biology is not None):
                    try:
                        avg_imm = self.env.biology.get_average_immunity()
                    except Exception:
                        pass
                self.disease_evol.step(tick, generation=gen, avg_immunity=avg_imm)
            except Exception as e:
                self._log_error("on_tick", "disease_evol", e)

        # ── Phase 10: OEE check (periodic) ───────────────────────────────
        if (self.evolution is not None
                and tick > 0 and tick % self.OEE_CHECK_INTERVAL == 0):
            try:
                self.last_oee_status = self.check_oee()
                self.recent_oee_checks.append({
                    "tick": tick, "status": self.last_oee_status,
                })
                # Write to disk for offline inspection
                with open("oee_status.json", "w") as f:
                    json.dump(self.last_oee_status, f, indent=2)
                events_this_tick.append({
                    "type": "OEE_CHECK",
                    "details": self.last_oee_status,
                })
            except Exception as e:
                self._log_error("on_tick", "oee_check", e)

        # ── Phase 13: Periodic snapshot ──────────────────────────────────
        if (self.world_seeding is not None
                and tick > 0 and tick % self.SNAPSHOT_INTERVAL == 0):
            try:
                self.snapshot(f"auto_tick{tick}", tick)
            except Exception as e:
                self._log_error("on_tick", "snapshot", e)

        # ── Phase 11: Update Godot state on disk ─────────────────────────
        if self.godot is not None and tick % 10 == 0:
            try:
                self._update_godot_state(tick, adam_rt, eve_rt, babies,
                                          adam_action, eve_action, world_state)
            except Exception as e:
                self._log_error("on_tick", "godot_state", e)

        return {"events": events_this_tick}

    # ─────────────────────────────────────────────────────────────────────────
    # Reproduction
    # ─────────────────────────────────────────────────────────────────────────
    def _check_reproduction(self, tick: int, adam_rt, eve_rt) -> Optional[Dict]:
        """Check fertility conditions and start pregnancy if met."""
        # Need both agents to have biology engines (we use env.biology for Adam;
        # Eve's biology lives on eve_agent, but for simplicity we use a minimal
        # proxy object exposing the methods reproduction.check_fertility needs).
        if not hasattr(self.env, "biology") or self.env.biology is None:
            return None

        # Only start a new pregnancy if Eve isn't already pregnant
        if self.reproduction.get_pregnancy_status("eve") is not None:
            return None

        # Don't trigger if Eve is currently ill or critically injured
        try:
            if (hasattr(eve_rt, "stats") and
                    eve_rt.stats.get("health", 100) < 70):
                return None
        except Exception:
            pass

        fertility = self.reproduction.check_fertility(
            adam_bio=self.env.biology,
            eve_bio=self.env.biology,  # shared proxy for now
            adam_pos=(adam_rt.x, adam_rt.y),
            eve_pos=(eve_rt.x, eve_rt.y),
            tick=tick,
            danger_present=False,
        )

        if fertility.get("can_reproduce"):
            try:
                self.reproduction.start_pregnancy("eve", "adam", tick)
                if self.events is not None:
                    self.events.record(
                        event_type="FAMILY_FORMED", tick=tick,
                        details={"parents": ["adam", "eve"], **fertility},
                    )
                print(f"  \U0001f495 Pregnancy started at tick {tick} "
                      f"(fertile_window={fertility.get('fertile_window')})",
                      flush=True)
                return {"type": "PREGNANCY_STARTED", "details": fertility}
            except Exception as e:
                self._log_error("_check_reproduction", "reproduction", e)
        return None

    def _update_pregnancy_if_any(self, tick: int, eve_rt, adam_rt) -> Optional[Dict]:
        """Update pregnancy; if gestation complete, spawn baby."""
        status = self.reproduction.get_pregnancy_status("eve")
        if status is None:
            return None

        update = self.reproduction.update_pregnancy("eve", self.env.biology, tick)
        if update.get("baby_born"):
            try:
                baby = self.reproduction.spawn_baby(
                    eve_pos=(eve_rt.x, eve_rt.y),
                    eve_id="eve", adam_id="adam",
                    eve_bio=self.env.biology, adam_bio=self.env.biology,
                    tick=tick,
                    world_width=self.env.world_map.width,
                    world_height=self.env.world_map.height,
                )
                baby_id = baby.get("agent_id", f"baby_{tick}")
                # Register with lineage + events
                self.register_agent(
                    agent_id=baby_id, parents=["adam", "eve"],
                    generation=2, birth_tick=tick,
                    biome=getattr(eve_rt, "current_biome",
                                   self.env.current_biome),
                    traits=baby.get("traits", {}),
                )
                # Track in baby_runtimes so training loop can render it
                self.baby_runtimes[baby_id] = {
                    "id": baby_id,
                    "x": baby.get("x", eve_rt.x),
                    "y": baby.get("y", eve_rt.y),
                    "birth_tick": tick,
                    "parents": ["adam", "eve"],
                    "traits": baby.get("traits", {}),
                }
                if not self.first_birth_fired:
                    self.first_birth_fired = True
                    if self.events is not None:
                        self.events.record(
                            event_type="FIRST_BIRTH", tick=tick,
                            details={"baby_id": baby_id, **baby},
                        )
                    if self.godot is not None:
                        self.godot.fire_milestone(
                            "FIRST_BIRTH",
                            details={"tick": tick, "baby_id": baby_id, **baby},
                        )
                    print(f"  \U0001f476 FIRST BIRTH at tick {tick}! "
                          f"Baby '{baby_id}' born at "
                          f"({baby.get('x')}, {baby.get('y')})", flush=True)
                else:
                    print(f"  \U0001f476 Birth at tick {tick}: baby '{baby_id}'",
                          flush=True)
                return {"type": "BIRTH", "details": baby}
            except Exception as e:
                self._log_error("_update_pregnancy", "spawn_baby", e)
        return None

    # ─────────────────────────────────────────────────────────────────────────
    # PPO update hook — Phase 4 (growing brain)
    # ─────────────────────────────────────────────────────────────────────────
    def on_ppo_update(self, agent_id: str, loss: float, tick: int):
        """Called after a PPO update with the loss value. Records to GrowingBrain
        and triggers growth if loss has plateaued."""
        # GrowingBrain is opt-in per agent (default off — BabyBrain stays fixed).
        # Only fire if the agent has a GrowingBrain attached.
        # (The training loop can attach one by setting agent.growing_brain.)
        # For now, we just record the loss in the evolution tracker.
        if self.evolution is not None:
            try:
                # We don't have a per-agent loss tracker in EvolutionTracker;
                # this is a no-op stub for future expansion.
                pass
            except Exception:
                pass

    def check_growing_brain(self, agent_id: str, agent_rt, tick: int):
        """If agent has a growing_brain attribute, check whether it should grow."""
        gb = getattr(agent_rt, "growing_brain", None)
        if gb is None:
            return None
        try:
            if gb.should_grow(tick):
                info = gb.grow(tick, reason="loss_plateau",
                                vocab_size=len(agent_rt.thought_engine.get_vocabulary())
                                if hasattr(agent_rt, "thought_engine") else 0)
                if self.events is not None:
                    self.events.record(
                        event_type="BRAIN_GROWTH", tick=tick, agent_id=agent_id,
                        details=info,
                    )
                print(f"  \U0001f9e0 BRAIN GROWTH for {agent_id} at tick {tick}: "
                      f"{info.get('old_param_count')} → {info.get('new_param_count')} "
                      f"params ({info.get('reason')})", flush=True)
                return info
        except Exception as e:
            self._log_error("check_growing_brain", "growing_brain", e)
        return None

    # ─────────────────────────────────────────────────────────────────────────
    # EAT / DRINK / SLEEP hooks
    # ─────────────────────────────────────────────────────────────────────────
    def on_eat(self, agent_id: str, food_type: str, x: int, y: int, tick: int):
        """Called when an agent successfully eats. Records to math/culture/evolution."""
        if self.math is not None:
            try:
                self.math.consume_food(x, y)
            except Exception as e:
                self._log_error("on_eat", "math", e)
        if self.culture is not None:
            try:
                # Detect cooking discovery — if food is cooked, fire event
                self.culture.detect_cooking(
                    agent_id=agent_id, food_type=food_type, tick=tick,
                )
            except Exception as e:
                self._log_error("on_eat", "culture", e)

    def on_drink(self, agent_id: str, water_source: str, tick: int):
        """Called when an agent drinks."""
        pass  # no specific engine hook needed yet

    def on_new_word(self, agent_id: str, word: str, meaning: str, tick: int):
        """Called when an agent invents a new word."""
        if not word:
            return
        if self.culture is not None:
            try:
                self.culture.record_word_invention(
                    word=word, originator_id=agent_id, tick=tick, meaning=meaning,
                )
            except Exception as e:
                self._log_error("on_new_word", "culture", e)
        if self.events is not None:
            try:
                self.events.record(
                    event_type="FIRST_WORD", tick=tick, agent_id=agent_id,
                    details={"word": word, "meaning": meaning},
                )
            except Exception as e:
                self._log_error("on_new_word", "events", e)
        if self.godot is not None and not self.first_word_fired:
            self.first_word_fired = True
            try:
                self.godot.fire_milestone(
                    "FIRST_WORD",
                    details={"tick": tick, "agent": agent_id,
                             "word": word, "meaning": meaning},
                )
            except Exception as e:
                self._log_error("on_new_word", "godot", e)

    # ─────────────────────────────────────────────────────────────────────────
    # Death hook
    # ─────────────────────────────────────────────────────────────────────────
    def on_death(self, agent_id: str, tick: int, cause: str = "unknown",
                  stats: Optional[Dict] = None):
        """Called when an agent dies."""
        if self.reproduction is not None:
            try:
                self.reproduction.record_death(
                    agent_id=agent_id, death_tick=tick,
                    cause=cause, final_stats=stats or {},
                )
            except Exception as e:
                self._log_error("on_death", "reproduction", e)
        if self.lineage is not None:
            try:
                self.lineage.record_death(
                    agent_id=agent_id, death_tick=tick, cause=cause,
                )
                if stats:
                    self.lineage.update_peak_stats(agent_id, stats)
            except Exception as e:
                self._log_error("on_death", "lineage", e)
        if self.events is not None:
            try:
                if not self.first_death_fired:
                    self.first_death_fired = True
                    self.events.record(
                        event_type="FIRST_DEATH", tick=tick, agent_id=agent_id,
                        details={"cause": cause, "stats": stats or {}},
                    )
                    if self.godot is not None:
                        self.godot.fire_milestone(
                            "FIRST_DEATH",
                            details={"tick": tick, "agent": agent_id, "cause": cause},
                        )
                    print(f"  \u26b0\ufe0f FIRST DEATH at tick {tick}: "
                          f"{agent_id} ({cause})", flush=True)
                else:
                    self.events.record(
                        event_type="DEATH", tick=tick, agent_id=agent_id,
                        details={"cause": cause, "stats": stats or {}},
                    )
            except Exception as e:
                self._log_error("on_death", "events", e)

    # ─────────────────────────────────────────────────────────────────────────
    # OEE check (Phase 10)
    # ─────────────────────────────────────────────────────────────────────────
    def check_oee(self) -> Dict:
        """Check Open-Ended Evolution criteria (Packard et al. 2019)."""
        if self.evolution is None:
            return {
                "available": False,
                "criteria_met": 0,
                "criteria_total": 5,
                "oee_achieved": False,
                "message": "EvolutionTracker not instantiated",
            }
        try:
            status = self.evolution.check_open_ended_evolution()
            status["available"] = True
            status["tick"] = self.tick_count
            # Persist
            with open("oee_status.json", "w") as f:
                json.dump(status, f, indent=2)
            return status
        except Exception as e:
            self._log_error("check_oee", "evolution", e)
            return {
                "available": True,
                "error": str(e),
                "criteria_met": 0,
                "criteria_total": 5,
                "oee_achieved": False,
            }

    # ─────────────────────────────────────────────────────────────────────────
    # Snapshot (Phase 13)
    # ─────────────────────────────────────────────────────────────────────────
    def snapshot(self, name: str, tick: int) -> Optional[str]:
        """Save a snapshot of all engine state."""
        if self.world_seeding is None:
            return None
        try:
            engines = {
                "physics": self.physics,
                "math": self.math,
                "first_contact": self.first_contact,
                "social": self.social,
                "culture": self.culture,
                "evolution": self.evolution,
                "world_evol": self.world_evol,
                "disease_evol": self.disease_evol,
            }
            # Filter None
            engines = {k: v for k, v in engines.items() if v is not None}
            path = self.world_seeding.snapshot(name, engines, tick)
            return path
        except Exception as e:
            self._log_error("snapshot", "world_seeding", e)
            return None

    # ─────────────────────────────────────────────────────────────────────────
    # Godot state
    # ─────────────────────────────────────────────────────────────────────────
    def _update_godot_state(self, tick: int, adam_rt, eve_rt, babies: List,
                             adam_action: str, eve_action: str,
                             world_state: Dict):
        """Build Godot state and write to disk for the HTTP server to serve."""
        if self.godot is None:
            return
        agent_positions = {}
        agent_data = []
        if adam_rt is not None:
            agent_positions["adam"] = (adam_rt.x, adam_rt.y)
            agent_data.append({
                "id": "adam", "x": adam_rt.x, "y": adam_rt.y,
                "type": "adam",
                "action": adam_action,
                "thought": adam_rt.latest_thought,
                "emotion": adam_rt.latest_emotion,
                "life_stage": "adult",
                "health": adam_rt.stats.get("health", 100),
                "hunger": adam_rt.stats.get("hunger", 0),
                "energy": adam_rt.stats.get("energy", 100),
                "vocab_size": len(adam_rt.thought_engine.get_vocabulary()),
                "generation": 1,
            })
        if eve_rt is not None:
            agent_positions["eve"] = (eve_rt.x, eve_rt.y)
            agent_data.append({
                "id": "eve", "x": eve_rt.x, "y": eve_rt.y,
                "type": "eve",
                "action": eve_action,
                "thought": eve_rt.latest_thought,
                "emotion": eve_rt.latest_emotion,
                "life_stage": "adult",
                "health": eve_rt.stats.get("health", 100),
                "hunger": eve_rt.stats.get("hunger", 0),
                "energy": eve_rt.stats.get("energy", 100),
                "vocab_size": len(eve_rt.thought_engine.get_vocabulary()),
                "generation": 1,
            })
        for baby in babies:
            agent_positions[baby["id"]] = (baby["x"], baby["y"])
            agent_data.append({
                "id": baby["id"], "x": baby["x"], "y": baby["y"],
                "type": "baby",
                "action": "IDLE",
                "thought": "...",
                "emotion": "quiet",
                "life_stage": "newborn",
                "health": 100, "hunger": 0, "energy": 100,
                "vocab_size": 0,
                "generation": baby.get("generation", 2),
            })

        state = self.godot.build_state(
            tick=tick, agent_positions=agent_positions, agent_data=agent_data,
        )
        with open(self.godot_state_path, "w") as f:
            json.dump(state, f)

    # ─────────────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────────────
    def _is_agent_alive(self, agent_id: str, adam_rt, eve_rt, babies: List) -> bool:
        if agent_id == "adam":
            return adam_rt is not None and adam_rt.alive
        if agent_id == "eve":
            return eve_rt is not None and eve_rt.alive
        return any(b["id"] == agent_id for b in babies)

    def _build_behavior_signature(self, adam_rt, eve_rt,
                                    adam_action: str, eve_action: str) -> str:
        """Build a coarse behavior signature for novelty detection."""
        sig_parts = []
        if adam_rt is not None:
            sig_parts.append(f"a:{adam_action}")
        if eve_rt is not None:
            sig_parts.append(f"e:{eve_action}")
        return "|".join(sig_parts)

    def get_babies(self) -> List[Dict]:
        """Return list of currently tracked babies."""
        return list(self.baby_runtimes.values())

    def get_summary(self) -> Dict:
        """Return a summary of orchestrator state for end-of-life logging."""
        summary = {
            "tick_count": self.tick_count,
            "error_count": self._error_count,
            "agent_ids": list(self.agent_ids),
            "first_contact_fired": self.first_contact_fired,
            "first_birth_fired": self.first_birth_fired,
            "first_death_fired": self.first_death_fired,
            "first_word_fired": self.first_word_fired,
            "babies_tracked": len(self.baby_runtimes),
            "oee_checks": list(self.recent_oee_checks),
            "last_oee_status": self.last_oee_status,
            "engines": {
                "reproduction": self.reproduction is not None,
                "math": self.math is not None,
                "first_contact": self.first_contact is not None,
                "social": self.social is not None,
                "culture": self.culture is not None,
                "evolution": self.evolution is not None,
                "events": self.events is not None,
                "lineage": self.lineage is not None,
                "world_evol": self.world_evol is not None,
                "disease_evol": self.disease_evol is not None,
                "novelty": self.novelty is not None,
                "world_seeding": self.world_seeding is not None,
                "godot_bridge": self.godot is not None,
            },
        }
        # Add per-engine summaries (best-effort)
        for name, engine in [
            ("reproduction", self.reproduction),
            ("first_contact", self.first_contact),
            ("social", self.social),
            ("culture", self.culture),
            ("evolution", self.evolution),
            ("events", self.events),
            ("world_evol", self.world_evol),
            ("disease_evol", self.disease_evol),
            ("novelty", self.novelty),
            ("godot_bridge", self.godot),
        ]:
            if engine is not None and hasattr(engine, "get_summary"):
                try:
                    summary[f"{name}_summary"] = engine.get_summary()
                except Exception as e:
                    summary[f"{name}_summary"] = {"error": str(e)[:200]}
        return summary

    def finalize(self):
        """Called at end of training. Persists final OEE status + summary."""
        try:
            if self.evolution is not None:
                self.last_oee_status = self.check_oee()
        except Exception as e:
            self._log_error("finalize", "oee", e)

        try:
            summary = self.get_summary()
            with open("orchestrator_summary.json", "w") as f:
                json.dump(summary, f, indent=2, default=str)
            print(f"  \U0001f4ca Orchestrator summary written to orchestrator_summary.json",
                  flush=True)
            if self.last_oee_status:
                print(f"  \U0001f9ee Final OEE status: "
                      f"{self.last_oee_status.get('criteria_met', 0)}/"
                      f"{self.last_oee_status.get('criteria_total', 5)} criteria met "
                      f"(OEE achieved: "
                      f"{self.last_oee_status.get('oee_achieved', False)})",
                      flush=True)
        except Exception as e:
            self._log_error("finalize", "summary", e)
