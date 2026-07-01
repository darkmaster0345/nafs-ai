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

Outputs:
    - vocab_divergence.jsonl  (periodic snapshots)
      Each entry: { tick, adam_vocab_size, eve_vocab_size, shared, only_adam,
                    only_eve, jaccard, adam_new, eve_new }

    - vocab_log.jsonl         (per-word discovery events — MD Phase 0.3 spec)
      Each entry: { tick, agent, word, meaning, trigger, context_of_discovery }
      One row per word discovered, by either agent.

    - vocab_convergence.jsonl (convergence events — MD Phase 0.3 highlight)
      Each entry: { tick, adam_word, eve_word, shared_meaning, shared_trigger,
                    convergence_type }
      Flags when both agents independently discovered a word for the same
      trigger / meaning (linguistic convergence).

Usage:
    from vocab_divergence import VocabDivergenceLogger

    logger = VocabDivergenceLogger(log_path="vocab_divergence.jsonl")
    logger.log(tick=100, adam_vocab=adam.thought_engine.get_vocabulary(),
                             eve_vocab=eve.thought_engine.get_vocabulary())
    logger.log_word_discovery(tick=42, agent="adam", word="cactus sand",
                              meaning="a dry place", trigger="ENTERED_DESERT",
                              context={"biome": "desert", "temperature": 38})
    logger.summary()  # prints final divergence stats + convergence highlights
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
        word_log_path: Optional[str] = None,
        convergence_log_path: Optional[str] = None,
    ):
        self.log_path = log_path
        self.log_interval = log_interval
        self.entries: List[Dict[str, Any]] = []
        self._prev_adam: Set[str] = set()
        self._prev_eve: Set[str] = set()
        self._start_time = time.time()

        # Per-word discovery log (MD Phase 0.3 spec)
        # Each entry: {tick, agent, word, meaning, trigger, context_of_discovery}
        self.word_log_path = word_log_path or "vocab_log.jsonl"
        # Convergence log: when both agents independently discovered a word
        # for the same trigger / meaning.
        self.convergence_log_path = convergence_log_path or "vocab_convergence.jsonl"

        # In-memory indexes for convergence detection
        # key = (trigger or normalized meaning), value = {agent: {word, tick, meaning, trigger, context}}
        self._by_trigger: Dict[str, Dict[str, Dict[str, Any]]] = {}
        # All discoveries by agent, for the dashboard / summary
        self._discoveries: Dict[str, List[Dict[str, Any]]] = {"adam": [], "eve": []}
        # Convergence events we've already flagged (avoid duplicates)
        self._flagged_convergences: Set[str] = set()

        # Truncate log files at start (fresh each run)
        for path in (self.log_path, self.word_log_path, self.convergence_log_path):
            try:
                with open(path, "w") as f:
                    f.write("")
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

    # ════════════════════════════════════════════════════════════════════════════
    # Per-word discovery log (MD Phase 0.3 spec)
    # ════════════════════════════════════════════════════════════════════════════

    def log_word_discovery(
        self,
        tick: int,
        agent: str,
        word: str,
        meaning: str = "",
        trigger: str = "",
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Log a single word discovery event.

        This implements the MD Phase 0.3 spec:
            vocab_log.jsonl --- per tick: {tick, agent, word, context_of_discovery}

        We extend the schema with meaning + trigger so we can later detect
        convergence (both agents independently inventing a word for the same
        trigger / meaning).

        Args:
            tick: Simulation tick when the word was discovered
            agent: "adam" or "eve"
            word: The discovered word (may be a compound like "cactus sand")
            meaning: Human-readable meaning / description
            trigger: The trigger name that fired the discovery (e.g. ENTERED_DESERT)
            context: Dict of contextual state at discovery time
                     (biome, temperature, health, action, etc.)

        Returns:
            The log entry dict
        """
        agent = (agent or "").lower()
        if agent not in ("adam", "eve"):
            return {}

        entry = {
            "tick": tick,
            "agent": agent,
            "word": word,
            "meaning": meaning,
            "trigger": trigger,
            "context_of_discovery": context or {},
            "elapsed_sec": round(time.time() - self._start_time, 1),
        }

        # Persist to JSONL
        try:
            with open(self.word_log_path, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            print(f"[VocabDivergence] Failed to write word log: {e}", flush=True)

        # Index for convergence detection
        self._discoveries[agent].append(entry)
        if trigger:
            self._by_trigger.setdefault(trigger, {})
            self._by_trigger[trigger][agent] = entry
            # Check for convergence
            self._maybe_flag_convergence(trigger)

        return entry

    def _maybe_flag_convergence(self, trigger: str) -> None:
        """
        If both Adam and Eve have discovered a word for the same trigger,
        flag it as a linguistic convergence event.

        Convergence types:
          - EXACT_MATCH       : both invented the SAME word string
          - SAME_TRIGGER      : different words, but same trigger / meaning
                                (independent invention of distinct labels for
                                 the same concept — true linguistic divergence
                                 with semantic convergence)
        """
        if trigger in self._flagged_convergences:
            return
        bucket = self._by_trigger.get(trigger, {})
        a = bucket.get("adam")
        e = bucket.get("eve")
        if not a or not e:
            return

        convergence_type = "EXACT_MATCH" if a["word"] == e["word"] else "SAME_TRIGGER"
        # Time-gap: who discovered first?
        first_agent = "adam" if a["tick"] <= e["tick"] else "eve"
        time_gap = abs(a["tick"] - e["tick"])

        entry = {
            "tick": max(a["tick"], e["tick"]),
            "trigger": trigger,
            "shared_meaning": a.get("meaning") or e.get("meaning", ""),
            "adam_word": a["word"],
            "adam_tick": a["tick"],
            "eve_word": e["word"],
            "eve_tick": e["tick"],
            "convergence_type": convergence_type,
            "first_discoverer": first_agent,
            "time_gap_ticks": time_gap,
            "elapsed_sec": round(time.time() - self._start_time, 1),
        }

        try:
            with open(self.convergence_log_path, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as ex:
            print(f"[VocabDivergence] Failed to write convergence log: {ex}", flush=True)

        self._flagged_convergences.add(trigger)
        print(
            f"  \U0001f9e0 CONVERGENCE @ trigger={trigger}: "
            f"Adam='{a['word']}' (t{a['tick']}) vs Eve='{e['word']}' (t{e['tick']}) "
            f"— {convergence_type}, gap={time_gap}t",
            flush=True,
        )

    def get_discoveries(self, agent: str) -> List[Dict[str, Any]]:
        """Return all word discoveries logged for an agent."""
        return list(self._discoveries.get(agent.lower(), []))

    def get_convergences(self) -> List[Dict[str, Any]]:
        """Return all flagged convergence events."""
        out = []
        for trigger, bucket in self._by_trigger.items():
            if "adam" in bucket and "eve" in bucket:
                a, e = bucket["adam"], bucket["eve"]
                out.append({
                    "trigger": trigger,
                    "adam_word": a["word"],
                    "eve_word": e["word"],
                    "convergence_type": "EXACT_MATCH" if a["word"] == e["word"] else "SAME_TRIGGER",
                })
        return out

    def render_dashboard_html(self, output_path: str = "docs/vocab_dashboard.html") -> str:
        """
        Build a static HTML dashboard panel showing side-by-side Adam vs Eve
        vocabulary with convergence highlights.

        This satisfies MD Phase 0.3:
            - Dashboard panel: side-by-side Adam vocabulary vs Eve vocabulary
            - Highlight words both discovered independently for same thing
              (linguistic convergence)

        The HTML is fully self-contained (no external deps) and reads
        vocab_log.jsonl + vocab_convergence.jsonl via fetch() when served
        from the same directory.
        """
        html = '''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>Nafs AI — Vocabulary Divergence Dashboard</title>
<style>
  * { box-sizing: border-box; }
  body {
    margin: 0; padding: 24px;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background: #0d1117; color: #c9d1d9;
  }
  h1 { color: #58a6ff; margin: 0 0 4px; font-size: 22px; }
  .sub { color: #8b949e; margin-bottom: 24px; font-size: 13px; }
  .panel {
    background: #161b22; border: 1px solid #30363d; border-radius: 8px;
    padding: 16px; margin-bottom: 16px;
  }
  .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
  .agent-name { font-size: 16px; font-weight: 600; margin-bottom: 8px; }
  .adam-color { color: #58a6ff; }
  .eve-color  { color: #ff7b72; }
  .word-list {
    list-style: none; padding: 0; margin: 0;
    max-height: 360px; overflow-y: auto;
    font-family: "SF Mono", Menlo, monospace; font-size: 12px;
  }
  .word-list li {
    padding: 6px 8px; border-bottom: 1px solid #21262d;
    display: flex; justify-content: space-between; align-items: center;
  }
  .word { font-weight: 600; }
  .meaning { color: #8b949e; font-size: 11px; max-width: 200px; text-align: right; }
  .tick { color: #6e7681; font-size: 10px; margin-left: 6px; }
  .convergence {
    background: #1f2937; border-left: 3px solid #3fb950;
    padding: 8px 12px; margin-bottom: 8px; border-radius: 4px;
    font-size: 12px;
  }
  .convergence .label { color: #3fb950; font-weight: 600; }
  .convergence.exact { border-left-color: #bc8cff; }
  .convergence.exact .label { color: #bc8cff; }
  .stats {
    display: flex; gap: 24px; flex-wrap: wrap;
    margin-bottom: 16px; font-size: 13px;
  }
  .stat { background: #161b22; padding: 8px 12px; border-radius: 6px; border: 1px solid #30363d; }
  .stat .num { font-size: 20px; font-weight: 700; color: #58a6ff; }
  .stat .lbl { color: #8b949e; font-size: 11px; text-transform: uppercase; letter-spacing: 0.5px; }
  .empty { color: #6e7681; font-style: italic; padding: 12px; text-align: center; }
  button { background: #238636; color: white; border: 0; padding: 6px 12px;
           border-radius: 4px; cursor: pointer; font-size: 12px; }
  button:hover { background: #2ea043; }
</style>
</head>
<body>
  <h1>NAFS AI — Vocabulary Divergence Dashboard</h1>
  <div class="sub">Phase 0.3 — Side-by-side Adam vs Eve vocabulary with convergence highlights.</div>

  <div class="stats" id="stats"></div>

  <div class="panel">
    <div class="agent-name adam-color">Convergence Highlights</div>
    <div id="convergences"><div class="empty">No convergence events yet.</div></div>
  </div>

  <div class="grid">
    <div class="panel">
      <div class="agent-name adam-color">Adam's Vocabulary</div>
      <ul class="word-list" id="adam-vocab"><div class="empty">No discoveries logged yet.</div></ul>
    </div>
    <div class="panel">
      <div class="agent-name eve-color">Eve's Vocabulary</div>
      <ul class="word-list" id="eve-vocab"><div class="empty">No discoveries logged yet.</div></ul>
    </div>
  </div>

  <button onclick="location.reload()">Refresh</button>

<script>
async function load_jsonl(path) {
  try {
    const r = await fetch(path);
    if (!r.ok) return [];
    const text = await r.text();
    return text.trim().split('\\n').filter(Boolean).map(l => JSON.parse(l));
  } catch (e) { return []; }
}

function render_word_list(elemId, discoveries) {
  const el = document.getElementById(elemId);
  if (!discoveries.length) {
    el.innerHTML = '<div class="empty">No discoveries logged yet.</div>';
    return;
  }
  el.innerHTML = discoveries.map(d => `
    <li>
      <span><span class="word">${d.word}</span><span class="tick">@t${d.tick}</span></span>
      <span class="meaning">${d.meaning || d.trigger || ''}</span>
    </li>
  `).join('');
}

function render_convergences(events) {
  const el = document.getElementById('convergences');
  if (!events.length) {
    el.innerHTML = '<div class="empty">No convergence events yet.</div>';
    return;
  }
  el.innerHTML = events.map(e => `
    <div class="convergence ${e.convergence_type === 'EXACT_MATCH' ? 'exact' : ''}">
      <span class="label">${e.convergence_type.replace('_', ' ')}</span> —
      trigger <code>${e.trigger}</code>:
      Adam "<span class="adam-color">${e.adam_word}</span>" (t${e.adam_tick}) vs
      Eve "<span class="eve-color">${e.eve_word}</span>" (t${e.eve_tick})
      — gap ${e.time_gap_ticks}t, first: ${e.first_discoverer}
    </div>
  `).join('');
}

function render_stats(adam, eve, conv) {
  const shared_triggers = new Set(
    conv.map(c => c.trigger)
  );
  const adamWords = new Set(adam.map(d => d.word));
  const eveWords  = new Set(eve.map(d => d.word));
  const shared    = [...adamWords].filter(w => eveWords.has(w));
  const jaccard   = (adamWords.size + eveWords.size) === 0 ? 1.0 :
                    shared.length / (new Set([...adamWords, ...eveWords])).size;

  document.getElementById('stats').innerHTML = `
    <div class="stat"><div class="num">${adam.length}</div><div class="lbl">Adam words</div></div>
    <div class="stat"><div class="num">${eve.length}</div><div class="lbl">Eve words</div></div>
    <div class="stat"><div class="num">${shared.length}</div><div class="lbl">Shared words</div></div>
    <div class="stat"><div class="num">${conv.length}</div><div class="lbl">Convergences</div></div>
    <div class="stat"><div class="num">${jaccard.toFixed(3)}</div><div class="lbl">Jaccard</div></div>
  `;
}

(async () => {
  const [adamAll, eveAll, conv] = await Promise.all([
    load_jsonl('vocab_log.jsonl').then(arr => arr.filter(d => d.agent === 'adam')),
    load_jsonl('vocab_log.jsonl').then(arr => arr.filter(d => d.agent === 'eve')),
    load_jsonl('vocab_convergence.jsonl'),
  ]);
  render_word_list('adam-vocab', adamAll.reverse());
  render_word_list('eve-vocab',  eveAll.reverse());
  render_convergences(conv);
  render_stats(adamAll, eveAll, conv);
})();
</script>
</body>
</html>
'''
        try:
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            with open(output_path, "w") as f:
                f.write(html)
        except Exception as e:
            print(f"[VocabDivergence] Failed to write dashboard HTML: {e}", flush=True)
        return output_path


# ═══════════════════════════════════════════════════════════════════════════════
# Self-test
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Testing VocabDivergenceLogger...")

    logger = VocabDivergenceLogger(
        log_path="/tmp/test_vocab.jsonl",
        word_log_path="/tmp/test_vocab_log.jsonl",
        convergence_log_path="/tmp/test_vocab_conv.jsonl",
        log_interval=10,
    )

    # Simulate divergence over 100 ticks
    base = ["hot", "cold", "pain", "good", "bad"]
    for tick in range(0, 101, 10):
        adam_vocab = base + (["cactus", "sand", "sun"] if tick >= 30 else []) + \
                     (["drought"] if tick >= 60 else [])
        eve_vocab = base + (["moss", "drip", "shadow"] if tick >= 30 else []) + \
                    (["echo"] if tick >= 60 else [])
        logger.log(tick, adam_vocab, eve_vocab)

    # Per-word discovery events (MD Phase 0.3 spec)
    logger.log_word_discovery(
        tick=42, agent="adam", word="cactus sand",
        meaning="a dry place", trigger="ENTERED_DESERT",
        context={"biome": "desert", "temperature": 38, "action": "move"},
    )
    logger.log_word_discovery(
        tick=58, agent="eve", word="moss drip",
        meaning="a wet shady place", trigger="ENTERED_CAVE",
        context={"biome": "cave", "temperature": 12, "action": "move"},
    )
    # Convergence! Both independently discover a word for "ENTERED_DESERT"
    logger.log_word_discovery(
        tick=71, agent="eve", word="sun hot",
        meaning="a dry place", trigger="ENTERED_DESERT",
        context={"biome": "desert", "temperature": 41, "action": "move"},
    )
    # EXACT_MATCH convergence
    logger.log_word_discovery(
        tick=80, agent="adam", word="cold dark",
        meaning="a cold dark place", trigger="ENTERED_CAVE",
        context={"biome": "cave"},
    )
    logger.log_word_discovery(
        tick=85, agent="eve", word="cold dark",
        meaning="a cold dark place", trigger="ENTERED_CAVE",
        context={"biome": "cave"},
    )

    logger.summary()

    # Print per-word log + convergence log
    print("\n--- Per-word discoveries (vocab_log.jsonl) ---")
    with open("/tmp/test_vocab_log.jsonl") as f:
        for line in f:
            print("  " + line.rstrip())

    print("\n--- Convergence events (vocab_convergence.jsonl) ---")
    with open("/tmp/test_vocab_conv.jsonl") as f:
        for line in f:
            print("  " + line.rstrip())

    # Render the dashboard HTML
    dashboard_path = logger.render_dashboard_html("/tmp/test_vocab_dashboard.html")
    print(f"\n✓ Dashboard HTML written to: {dashboard_path}")
    print(f"✓ VocabDivergenceLogger works — logs at /tmp/test_vocab*.jsonl")
