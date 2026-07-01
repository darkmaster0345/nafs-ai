"""
Nafs AI — Vocabulary Divergence Logger (v0.3)
==============================================

Tracks how Adam's and Eve's vocabularies diverge over time.

Philosophy:
    Adam and Eve start with the same seed vocabulary (config.STARTING_VOCABULARY).
    As they live their separate lives in the same world, they discover new words
    based on their own experiences. Adam might discover "cactus" in the desert
    while Eve discovers "moss" in a cave.

    This module logs the divergence:
      - Words only Adam knows
      - Words only Eve knows
      - Words they both know (shared)
      - Jaccard similarity (shared / union) — 1.0 = identical, 0.0 = no overlap

    Over time, we expect the Jaccard similarity to decrease as they explore
    different biomes and have different experiences. This is the emergence of
    distinct "languages" — or at least distinct lexicons.

Output:
    - vocab_divergence.jsonl  (append-only, one JSON object per log entry)
    - Each entry: { tick, adam_vocab_size, eve_vocab_size, shared, only_adam,
                    only_eve, jaccard, adam_new, eve_new }

Usage:
    from vocab_divergence import VocabDivergenceLogger

    logger = VocabDivergenceLogger(log_path="vocab_divergence.jsonl")
    logger.log(tick=100, adam_vocab=adam.thought_engine.get_vocabulary(),
                             eve_vocab=eve.thought_engine.get_vocabulary())
    logger.summary()  # prints final divergence stats
"""

import json
import os
import time
from typing import List, Set, Dict, Any, Optional


def jaccard_similarity(a: Set[str], b: Set[str]) -> float:
    """Jaccard similarity between two sets: |intersection| / |union|."""
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 1.0
    return len(a & b) / len(union)


class VocabDivergenceLogger:
    """
    Logs vocabulary divergence between two agents over time.

    Call .log() periodically (e.g., every 50 ticks) to record a snapshot.
    Call .summary() at the end to print final stats.
    """

    def __init__(
        self,
        log_path: str = "vocab_divergence.jsonl",
        log_interval: int = 50,
    ):
        self.log_path = log_path
        self.log_interval = log_interval
        self.entries: List[Dict[str, Any]] = []
        self._prev_adam: Set[str] = set()
        self._prev_eve: Set[str] = set()
        self._start_time = time.time()

        # Truncate existing log file at start (fresh each run)
        try:
            with open(self.log_path, "w") as f:
                f.write("")  # truncate
        except Exception:
            pass

    def should_log(self, tick: int) -> bool:
        """Check if this tick should be logged."""
        return tick > 0 and tick % self.log_interval == 0

    def log(
        self,
        tick: int,
        adam_vocab: List[str],
        eve_vocab: List[str],
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Record a vocabulary snapshot.

        Args:
            tick: Current simulation tick
            adam_vocab: List of Adam's vocabulary words
            eve_vocab: List of Eve's vocabulary words
            extra: Optional extra fields to include (e.g., learned_thinking stats)

        Returns:
            The log entry dict
        """
        adam_set = set(adam_vocab)
        eve_set = set(eve_vocab)

        shared = adam_set & eve_set
        only_adam = adam_set - eve_set
        only_eve = eve_set - adam_set
        jaccard = jaccard_similarity(adam_set, eve_set)

        # New words since last log
        adam_new = adam_set - self._prev_adam
        eve_new = eve_set - self._prev_eve
        self._prev_adam = adam_set
        self._prev_eve = eve_set

        entry = {
            "tick": tick,
            "elapsed_sec": round(time.time() - self._start_time, 1),
            "adam_vocab_size": len(adam_set),
            "eve_vocab_size": len(eve_set),
            "shared_count": len(shared),
            "only_adam_count": len(only_adam),
            "only_eve_count": len(only_eve),
            "jaccard": round(jaccard, 4),
            "adam_new": sorted(list(adam_new)),
            "eve_new": sorted(list(eve_new)),
            "only_adam_sample": sorted(list(only_adam))[:10],
            "only_eve_sample": sorted(list(only_eve))[:10],
        }
        if extra:
            entry.update(extra)

        self.entries.append(entry)

        # Append to JSONL file
        try:
            with open(self.log_path, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            print(f"[VocabDivergence] Failed to write log: {e}", flush=True)

        # Print significant divergence events
        if adam_new or eve_new:
            print(
                f"  \U0001f4dd VOCAB @ tick {tick}: Adam={len(adam_set)} Eve={len(eve_set)} "
                f"shared={len(shared)} jaccard={jaccard:.3f}",
                flush=True,
            )
            if adam_new:
                print(f"     Adam +{sorted(adam_new)}", flush=True)
            if eve_new:
                print(f"     Eve  +{sorted(eve_new)}", flush=True)

        return entry

    def summary(self) -> Dict[str, Any]:
        """Print and return final divergence summary."""
        if not self.entries:
            print("\n[VocabDivergence] No entries logged.", flush=True)
            return {}

        first = self.entries[0]
        last = self.entries[-1]

        print(f"\n{'=' * 70}", flush=True)
        print(f"  \U0001f4da VOCABULARY DIVERGENCE SUMMARY", flush=True)
        print(f"{'─' * 70}", flush=True)
        print(f"  Total log entries: {len(self.entries)}", flush=True)
        print(f"  Span: tick {first['tick']} → {last['tick']}", flush=True)
        print(f"  Duration: {last['elapsed_sec']:.1f}s", flush=True)
        print(f"{'─' * 70}", flush=True)
        print(f"  Adam final vocab: {last['adam_vocab_size']} words", flush=True)
        print(f"  Eve  final vocab: {last['eve_vocab_size']} words", flush=True)
        print(f"  Shared words:     {last['shared_count']}", flush=True)
        print(f"  Only Adam:        {last['only_adam_count']} words", flush=True)
        print(f"  Only Eve:         {last['only_eve_count']} words", flush=True)
        print(f"{'─' * 70}", flush=True)
        print(f"  Jaccard similarity over time:", flush=True)
        # Sample up to 5 entries to show the trend (no duplicates)
        n = len(self.entries)
        if n <= 5:
            sample_idxs = list(range(n))
        else:
            sample_idxs = [0, n // 4, n // 2, 3 * n // 4, n - 1]
        seen = set()
        for idx in sample_idxs:
            if idx in seen:
                continue
            seen.add(idx)
            e = self.entries[idx]
            bar_len = int(e['jaccard'] * 20)
            bar = '█' * bar_len + '░' * (20 - bar_len)
            print(f"    tick {e['tick']:>5}: [{bar}] {e['jaccard']:.3f}", flush=True)
        print(f"{'─' * 70}", flush=True)
        print(f"  Only-Adam sample: {last['only_adam_sample']}", flush=True)
        print(f"  Only-Eve  sample: {last['only_eve_sample']}", flush=True)
        print(f"{'=' * 70}\n", flush=True)

        return {
            "entries": len(self.entries),
            "first_tick": first['tick'],
            "last_tick": last['tick'],
            "duration_sec": last['elapsed_sec'],
            "final_adam_vocab": last['adam_vocab_size'],
            "final_eve_vocab": last['eve_vocab_size'],
            "final_shared": last['shared_count'],
            "final_jaccard": last['jaccard'],
            "jaccard_trend": [e['jaccard'] for e in self.entries],
        }

    def get_latest(self) -> Optional[Dict[str, Any]]:
        """Return the latest log entry, or None if no entries."""
        return self.entries[-1] if self.entries else None


# ═══════════════════════════════════════════════════════════════════════════════
# Self-test
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Testing VocabDivergenceLogger...")

    logger = VocabDivergenceLogger(log_path="/tmp/test_vocab.jsonl", log_interval=10)

    # Simulate divergence over 100 ticks
    base = ["hot", "cold", "pain", "good", "bad"]
    for tick in range(0, 101, 10):
        adam_vocab = base + (["cactus", "sand", "sun"] if tick >= 30 else []) + \
                     (["drought"] if tick >= 60 else [])
        eve_vocab = base + (["moss", "drip", "shadow"] if tick >= 30 else []) + \
                    (["echo"] if tick >= 60 else [])
        logger.log(tick, adam_vocab, eve_vocab)

    logger.summary()
    print(f"\n✓ VocabDivergenceLogger works — log at /tmp/test_vocab.jsonl")
