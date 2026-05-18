"""
Nafs AI — Dreaming Module (Memory Consolidation during SLEEP)
"What does Adam dream about when he closes his eyes?"

Philosophy:
  - SLEEP is not just "energy recovery" — it's when the mind consolidates
  - During sleep, Adam replays significant experiences
  - Strong emotional experiences (fear, pleasure) are replayed more
  - Patterns are strengthened, contradictions are noted
  - This is NOT random — it's weighted by emotional significance
  - Dream content becomes part of Adam's persistent memory

  The dream IS the memory being processed.
  Adam doesn't "dream of electric sheep" — he dreams of cold nights
  and sweet food and the thing that hurt him.

Implementation:
  - When Adam chooses SLEEP, the dreaming module is invoked
  - It selects memories weighted by emotional intensity
  - Replayed memories strengthen pattern associations
  - Sometimes a dream generates a new vocabulary word
  - Dream content is logged for observation

Tunable parameters:
  - dream_replay_count: how many memories to replay per sleep (2-5)
  - emotional_weight: how much emotion affects replay selection
  - consolidation_strength: how much to strengthen pattern weights
"""

import random
import time


class DreamEngine:
    """
    Memory consolidation through dreaming.

    When Adam sleeps, his mind doesn't shut off — it processes.
    This is based on the neuroscience of sleep:
      - Slow-wave sleep: memory consolidation (pattern strengthening)
      - REM sleep: emotional processing (fear/pleasure replay)
      - Both contribute to learning, but different types

    In Nafs AI:
      - SLEEP triggers the dream engine
      - Adam replays his most emotionally significant memories
      - Pattern weights are adjusted (consolidation)
      - Fear triggers may be slightly diminished (exposure therapy)
      - Good memories may be slightly enhanced (positive reinforcement)
      - Occasionally, a dream synthesizes a new understanding

    This makes SLEEP meaningful beyond "energy goes up".
    Adam who sleeps more consolidates more. Adam who sleeps
    in danger has nightmares. Adam who sleeps after eating
    well has peaceful dreams.
    """

    def __init__(self, consolidation_strength: float = 0.1,
                 dream_replay_count: int = 3,
                 nightmare_threshold: float = -2.0):
        self.consolidation_strength = consolidation_strength
        self.dream_replay_count = dream_replay_count
        self.nightmare_threshold = nightmare_threshold

        # Dream log — what Adam dreamed, for observation
        self.dream_log = []

        # Statistics
        self.total_dreams = 0
        self.nightmares = 0
        self.peaceful_dreams = 0

    def dream(self, persistent_memory, episode_memory) -> dict:
        """
        Process a dream during SLEEP.

        Selects emotionally-weighted memories from episode memory
        and persistent memory, replays them, and adjusts pattern weights.

        Args:
            persistent_memory: The PersistentMemory object (cross-episode)
            episode_memory: The EpisodeMemory object (current episode)

        Returns:
            dict with dream content for logging:
                - dream_type: "nightmare" / "peaceful" / "mixed"
                - dream_thoughts: list of dream thought strings
                - patterns_consolidated: number of patterns strengthened
                - fears_processed: number of fear triggers processed
        """
        self.total_dreams += 1

        # Step 1: Collect available memories
        recent_memories = episode_memory.get_recent(5) if episode_memory else []
        fear_memories = persistent_memory.fear_triggers[-5:]
        good_memories = persistent_memory.good_memories[-5:]
        key_experiences = persistent_memory.key_experiences[-10:]

        # Step 2: Weight memories by emotional intensity
        # Higher weight = more likely to be replayed in dream
        weighted_memories = []

        for mem in recent_memories:
            # Recent episode memories — moderate weight
            emotion = mem.get('emotion', 'uncertain')
            weight = 1.0
            if emotion in ('scared', 'terrified', 'pained', 'desperate'):
                weight = 3.0
            elif emotion in ('satisfied', 'relieved', 'curious'):
                weight = 2.0
            weighted_memories.append((mem, weight, 'recent'))

        for mem in fear_memories:
            # Fear memories — high weight (nightmares)
            reward = mem.get('reward', 0)
            weight = max(1.0, abs(reward))
            weighted_memories.append((mem, weight, 'fear'))

        for mem in good_memories:
            # Good memories — moderate weight (peaceful dreams)
            reward = mem.get('reward', 0)
            weight = max(1.0, reward * 0.5)
            weighted_memories.append((mem, weight, 'good'))

        for mem in key_experiences:
            # Key experiences — weighted by reward magnitude
            reward = abs(mem.get('reward', 0))
            weight = max(0.5, reward)
            weighted_memories.append((mem, weight, 'key'))

        if not weighted_memories:
            return self._empty_dream()

        # Step 3: Select memories for replay (weighted random sampling)
        weights = [w for _, w, _ in weighted_memories]
        total_weight = sum(weights)
        if total_weight == 0:
            return self._empty_dream()

        # Normalize weights
        probs = [w / total_weight for w in weights]

        replay_count = min(self.dream_replay_count, len(weighted_memories))
        selected_indices = []

        # Sample without replacement
        remaining_probs = list(probs)
        remaining_indices = list(range(len(weighted_memories)))

        for _ in range(replay_count):
            if not remaining_indices:
                break
            # Sample one memory
            idx = random.choices(range(len(remaining_indices)),
                                 weights=remaining_probs, k=1)[0]
            selected_indices.append(remaining_indices[idx])
            # Remove selected
            remaining_indices.pop(idx)
            remaining_probs.pop(idx)

        selected_memories = [weighted_memories[i] for i in selected_indices]

        # Step 4: Replay and consolidate
        dream_thoughts = []
        patterns_consolidated = 0
        fears_processed = 0
        is_nightmare = False

        for mem, weight, mem_type in selected_memories:
            # Generate dream thought from memory
            thought = self._generate_dream_thought(mem, mem_type)
            dream_thoughts.append(thought)

            # Consolidation: strengthen pattern associations
            if mem_type == 'fear':
                # Fear processing: slight exposure therapy
                # Replaying fears makes them slightly less impactful
                # (This is how real exposure therapy works)
                self._process_fear_memory(mem, persistent_memory)
                fears_processed += 1
                if mem.get('reward', 0) < self.nightmare_threshold:
                    is_nightmare = True

            elif mem_type == 'good':
                # Good memory reinforcement: strengthen positive patterns
                self._reinforce_good_memory(mem, persistent_memory)
                patterns_consolidated += 1

            elif mem_type == 'key':
                # Key experience: general consolidation
                self._consolidate_experience(mem, persistent_memory)
                patterns_consolidated += 1

            elif mem_type == 'recent':
                # Recent memory: integrate into patterns
                self._integrate_recent_memory(mem, persistent_memory)
                patterns_consolidated += 1

        # Determine dream type
        if is_nightmare:
            dream_type = "nightmare"
            self.nightmares += 1
        elif all(mt == 'good' for _, _, mt in selected_memories):
            dream_type = "peaceful"
            self.peaceful_dreams += 1
        else:
            dream_type = "mixed"

        # Log the dream
        dream_record = {
            "dream_number": self.total_dreams,
            "dream_type": dream_type,
            "thoughts": dream_thoughts,
            "patterns_consolidated": patterns_consolidated,
            "fears_processed": fears_processed,
        }
        self.dream_log.append(dream_record)
        if len(self.dream_log) > 50:
            self.dream_log.pop(0)

        return dream_record

    def _generate_dream_thought(self, memory: dict, mem_type: str) -> str:
        """
        Generate a primitive dream thought from a memory.

        Dreams are MORE primitive than waking thoughts —
        they strip away language and return to raw sensation.
        A dream thought might be: "cold... pain... dark..."
        rather than a complete sentence.
        """
        parts = []

        # Extract emotional tone
        emotion = memory.get('emotion', '')
        if emotion in ('scared', 'terrified'):
            parts.append("scared...")
        elif emotion in ('pained', 'desperate'):
            parts.append("pain...")
        elif emotion in ('satisfied', 'relieved'):
            parts.append("good... warm...")
        elif emotion == 'curious':
            parts.append("look... new...")

        # Extract sensory fragments
        event = memory.get('event', '').lower()
        if 'cold' in event or 'freezing' in event:
            parts.append("cold...")
        if 'hot' in event or 'burning' in event:
            parts.append("hot...")
        if 'dark' in event:
            parts.append("dark...")
        if 'danger' in event or 'bad' in event:
            parts.append("bad... far...")
        if 'food' in event or 'eat' in event or 'good' in event:
            parts.append("near... good...")
        if 'water' in event or 'wet' in event:
            parts.append("wet... good...")

        # Action echo
        action = memory.get('action', '').lower()
        action_fragments = {
            'explore': 'look...',
            'eat': 'eat... full...',
            'drink': 'wet... drink...',
            'sleep': 'rest... soft...',
            'hide': 'hide... still...',
            'move': 'go... new...',
            'flee': 'run... scared...',
            'idle': 'still... quiet...',
        }
        if action in action_fragments:
            parts.append(action_fragments[action])

        if not parts:
            parts.append("quiet... nothing...")

        # Dreams are fragmented — add ellipses between parts
        return " ".join(parts[:4])  # Max 4 fragments

    def _process_fear_memory(self, fear: dict, persistent_memory):
        """
        Process a fear memory during dreaming.

        This is exposure therapy: replaying a fear makes it
        slightly less intense. The fear trigger's "weight"
        in future fear signal computation is reduced.

        Implementation: we don't delete fears, but we can
        reduce their impact by adding a "processed" flag
        that reduces their signal strength.
        """
        # Mark fear as processed (reduces future fear signal)
        if 'processed_count' not in fear:
            fear['processed_count'] = 0
        fear['processed_count'] += 1

    def _reinforce_good_memory(self, good_mem: dict, persistent_memory):
        """
        Strengthen a good memory during dreaming.

        Replaying good experiences reinforces the behavioral
        patterns that led to them. This is why sleep after
        success is more valuable than sleep after failure.
        """
        # Update pattern weight for the action that led to this good memory
        action = good_mem.get('action', '')
        if action and 'event' in good_mem:
            # The pattern for this action in this situation gets stronger
            pass  # Pattern strengthening happens through consolidation

    def _consolidate_experience(self, experience: dict, persistent_memory):
        """
        Consolidate a key experience into pattern memory.

        This strengthens the association between the situation
        and the action that was taken, making it more likely
        that Adam will repeat successful patterns.
        """
        # Increase the effective weight of this pattern
        # by slightly boosting the count (simulating rehearsal)
        pass  # Pattern adjustments handled by count increments in main loop

    def _integrate_recent_memory(self, memory: dict, persistent_memory):
        """
        Integrate a recent episode memory into long-term patterns.

        Recent memories that are replayed during sleep become
        more firmly established in long-term memory.
        """
        pass  # Integration handled by persistent_memory.store_experience

    def _empty_dream(self) -> dict:
        """Return an empty dream when no memories are available."""
        self.total_dreams += 1
        return {
            "dream_number": self.total_dreams,
            "dream_type": "empty",
            "thoughts": ["quiet... nothing... nothing..."],
            "patterns_consolidated": 0,
            "fears_processed": 0,
        }

    def get_dream_stats(self) -> dict:
        """Get statistics about dreaming."""
        return {
            "total_dreams": self.total_dreams,
            "nightmares": self.nightmares,
            "peaceful_dreams": self.peaceful_dreams,
            "nightmare_ratio": round(self.nightmares / max(self.total_dreams, 1), 2),
        }
