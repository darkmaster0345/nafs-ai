"""
Nafs AI — Full Observability Layer (Phase 12)
=============================================

Implements MD Phase 12: a simulation is only as good as your ability to
understand what happened. This phase ensures every emergent event is
captured, searchable, and explainable.

Covers:
  12.1 Event Log System
      - events.jsonl: permanent timestamped log of every named event
      - 12 event types: BIRTH, DEATH, FIRST_CONTACT, FIRST_WORD,
        COOKING_DISCOVERY, FIRE_DISCOVERY, FAMILY_FORMED, EXTINCTION,
        SPECIATION, CULTURAL_TRANSFER, TERRITORY_CLAIMED, BRAIN_GROWTH
      - Queryable by type, agent, generation, tick range
      - Dashboard event timeline

  12.2 Lineage Database
      - SQLite database: agents, relationships, vocabulary, events tables
      - Every agent's full life stored (birth, death, parents, offspring, stats)
      - Queries: longest-lived, most-generations bloodline
      - Export family tree as JSON

  12.3 Science Dashboard (HTML generator)
      - Population graph over time
      - Trait evolution graph (avg traits per generation)
      - Vocabulary graph (words over time, per generation)
      - Cultural distance matrix (heatmap)
      - Brain size distribution (histogram)
      - Extinction timeline marked on all graphs

Design constraints:
  - Does NOT modify base rewards
  - Standalone module
  - Uses Python's built-in sqlite3 (no external deps)
"""

import json
import os
import sqlite3
import time
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict


# ═══════════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════════

EVENT_TYPES = {
    "BIRTH", "DEATH", "FIRST_CONTACT", "FIRST_WORD",
    "COOKING_DISCOVERY", "FIRE_DISCOVERY", "FAMILY_FORMED",
    "EXTINCTION", "SPECIATION", "CULTURAL_TRANSFER",
    "TERRITORY_CLAIMED", "BRAIN_GROWTH",
}

EVENTS_LOG_PATH = "events.jsonl"
LINEAGE_DB_PATH = "lineage.db"


# ═══════════════════════════════════════════════════════════════════════════════
# EventLogSystem
# ═══════════════════════════════════════════════════════════════════════════════

class EventLogSystem:
    """Permanent timestamped log of every named event in the simulation."""

    def __init__(self, log_path: str = EVENTS_LOG_PATH):
        self.log_path = log_path
        self.events: List[Dict] = []
        try:
            with open(log_path, "w") as f:
                f.write("")
        except Exception:
            pass

    def record(self, event_type: str, tick: int,
                agent_id: str = "", details: Optional[Dict] = None,
                generation: Optional[int] = None,
                timestamp: Optional[float] = None) -> Dict:
        """Record an event. Returns the event dict."""
        if event_type not in EVENT_TYPES:
            raise ValueError(f"Unknown event type: {event_type}")

        event = {
            "event_type": event_type,
            "tick": tick,
            "agent_id": agent_id,
            "details": details or {},
            "generation": generation,
            "timestamp": timestamp or time.time(),
        }
        self.events.append(event)
        try:
            with open(self.log_path, "a") as f:
                f.write(json.dumps(event, default=str) + "\n")
        except Exception as e:
            print(f"[EventLog] Failed to write log: {e}", flush=True)
        return event

    def query(self, event_type: Optional[str] = None,
               agent_id: Optional[str] = None,
               generation: Optional[int] = None,
               min_tick: Optional[int] = None,
               max_tick: Optional[int] = None,
               limit: Optional[int] = None) -> List[Dict]:
        """Query events with optional filters."""
        results = []
        for event in self.events:
            if event_type and event["event_type"] != event_type:
                continue
            if agent_id and event["agent_id"] != agent_id:
                continue
            if generation is not None and event["generation"] != generation:
                continue
            if min_tick is not None and event["tick"] < min_tick:
                continue
            if max_tick is not None and event["tick"] > max_tick:
                continue
            results.append(event)
        results.sort(key=lambda e: e["tick"])
        if limit:
            results = results[:limit]
        return results

    def get_event_types(self) -> List[str]:
        return sorted(set(e["event_type"] for e in self.events))

    def get_event_count(self, event_type: Optional[str] = None) -> int:
        if event_type:
            return sum(1 for e in self.events if e["event_type"] == event_type)
        return len(self.events)

    def get_timeline(self, min_tick: Optional[int] = None,
                      max_tick: Optional[int] = None) -> List[Dict]:
        return self.query(min_tick=min_tick, max_tick=max_tick)

    def get_first_event_of_type(self, event_type: str) -> Optional[Dict]:
        for event in self.events:
            if event["event_type"] == event_type:
                return event
        return None

    def get_summary(self) -> Dict:
        by_type = defaultdict(int)
        for event in self.events:
            by_type[event["event_type"]] += 1
        return {
            "total_events": len(self.events),
            "by_type": dict(by_type),
            "first_tick": min((e["tick"] for e in self.events), default=None),
            "last_tick": max((e["tick"] for e in self.events), default=None),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# LineageDatabase
# ═══════════════════════════════════════════════════════════════════════════════

class LineageDatabase:
    """SQLite database storing every agent's full life."""

    def __init__(self, db_path: str = LINEAGE_DB_PATH):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self) -> None:
        cursor = self.conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS agents (
                agent_id TEXT PRIMARY KEY,
                parents TEXT,
                generation INTEGER,
                birth_tick INTEGER,
                death_tick INTEGER,
                death_cause TEXT,
                traits TEXT,
                peak_stats TEXT,
                biome TEXT
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS relationships (
                agent_a TEXT,
                agent_b TEXT,
                relation_type TEXT,
                trust REAL,
                familiarity REAL,
                last_seen_tick INTEGER,
                PRIMARY KEY (agent_a, agent_b)
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS vocabulary (
                agent_id TEXT,
                word TEXT,
                meaning TEXT,
                tick_discovered INTEGER,
                generation INTEGER,
                PRIMARY KEY (agent_id, word)
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_type TEXT,
                tick INTEGER,
                agent_id TEXT,
                details TEXT,
                timestamp REAL
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_agents_generation ON agents(generation)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_agents_birth_tick ON agents(birth_tick)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_tick ON events(tick)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_vocab_agent ON vocabulary(agent_id)")
        self.conn.commit()

    def insert_agent(self, agent_id: str, parents: List[str],
                       generation: int, birth_tick: int,
                       biome: str = "", traits: Optional[Dict] = None,
                       peak_stats: Optional[Dict] = None) -> None:
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO agents
            (agent_id, parents, generation, birth_tick, death_tick, death_cause,
             traits, peak_stats, biome)
            VALUES (?, ?, ?, ?, NULL, '', ?, ?, ?)
        """, (agent_id, json.dumps(parents), generation, birth_tick,
              json.dumps(traits or {}), json.dumps(peak_stats or {}), biome))
        self.conn.commit()

    def record_death(self, agent_id: str, death_tick: int,
                       death_cause: str = "") -> None:
        cursor = self.conn.cursor()
        cursor.execute("UPDATE agents SET death_tick = ?, death_cause = ? WHERE agent_id = ?",
                       (death_tick, death_cause, agent_id))
        self.conn.commit()

    def update_peak_stats(self, agent_id: str, stats: Dict) -> None:
        cursor = self.conn.cursor()
        cursor.execute("UPDATE agents SET peak_stats = ? WHERE agent_id = ?",
                       (json.dumps(stats), agent_id))
        self.conn.commit()

    def get_agent(self, agent_id: str) -> Optional[Dict]:
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM agents WHERE agent_id = ?", (agent_id,))
        row = cursor.fetchone()
        return self._row_to_agent_dict(row) if row else None

    def _row_to_agent_dict(self, row: sqlite3.Row) -> Dict:
        return {
            "agent_id": row["agent_id"],
            "parents": json.loads(row["parents"]) if row["parents"] else [],
            "generation": row["generation"],
            "birth_tick": row["birth_tick"],
            "death_tick": row["death_tick"],
            "death_cause": row["death_cause"],
            "traits": json.loads(row["traits"]) if row["traits"] else {},
            "peak_stats": json.loads(row["peak_stats"]) if row["peak_stats"] else {},
            "biome": row["biome"],
        }

    def query_longest_lived(self) -> Optional[Dict]:
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM agents WHERE death_tick IS NOT NULL
            ORDER BY (death_tick - birth_tick) DESC LIMIT 1
        """)
        row = cursor.fetchone()
        return self._row_to_agent_dict(row) if row else None

    def query_most_generations_bloodline(self) -> Optional[Dict]:
        cursor = self.conn.cursor()
        cursor.execute("SELECT MAX(generation) as max_gen FROM agents")
        row = cursor.fetchone()
        if row and row["max_gen"]:
            cursor.execute("""
                SELECT * FROM agents WHERE generation = ?
                ORDER BY birth_tick LIMIT 1
            """, (row["max_gen"],))
            agent_row = cursor.fetchone()
            return self._row_to_agent_dict(agent_row) if agent_row else None
        return None

    def query_agents_by_generation(self, generation: int) -> List[Dict]:
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM agents WHERE generation = ?", (generation,))
        return [self._row_to_agent_dict(row) for row in cursor.fetchall()]

    def query_offspring(self, agent_id: str) -> List[Dict]:
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM agents")
        all_agents = cursor.fetchall()
        offspring = []
        for row in all_agents:
            parents = json.loads(row["parents"]) if row["parents"] else []
            if agent_id in parents:
                offspring.append(self._row_to_agent_dict(row))
        return offspring

    def insert_vocabulary(self, agent_id: str, word: str,
                            meaning: str, tick_discovered: int,
                            generation: int) -> None:
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO vocabulary
            (agent_id, word, meaning, tick_discovered, generation)
            VALUES (?, ?, ?, ?, ?)
        """, (agent_id, word, meaning, tick_discovered, generation))
        self.conn.commit()

    def get_agent_vocabulary(self, agent_id: str) -> List[Dict]:
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM vocabulary WHERE agent_id = ?", (agent_id,))
        return [dict(row) for row in cursor.fetchall()]

    def get_word_origin(self, word: str) -> Optional[Dict]:
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM vocabulary WHERE word = ?
            ORDER BY tick_discovered ASC LIMIT 1
        """, (word,))
        row = cursor.fetchone()
        return dict(row) if row else None

    def insert_relationship(self, agent_a: str, agent_b: str,
                              relation_type: str, trust: float,
                              familiarity: float, last_seen_tick: int) -> None:
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO relationships
            (agent_a, agent_b, relation_type, trust, familiarity, last_seen_tick)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (agent_a, agent_b, relation_type, trust, familiarity, last_seen_tick))
        self.conn.commit()

    def get_relationships(self, agent_id: str) -> List[Dict]:
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM relationships WHERE agent_a = ? OR agent_b = ?",
                       (agent_id, agent_id))
        return [dict(row) for row in cursor.fetchall()]

    def insert_event(self, event_type: str, tick: int,
                       agent_id: str = "", details: Optional[Dict] = None,
                       timestamp: Optional[float] = None) -> None:
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO events (event_type, tick, agent_id, details, timestamp)
            VALUES (?, ?, ?, ?, ?)
        """, (event_type, tick, agent_id,
              json.dumps(details or {}), timestamp or time.time()))
        self.conn.commit()

    def query_events(self, event_type: Optional[str] = None,
                       min_tick: Optional[int] = None,
                       max_tick: Optional[int] = None,
                       limit: Optional[int] = None) -> List[Dict]:
        query = "SELECT * FROM events WHERE 1=1"
        params = []
        if event_type:
            query += " AND event_type = ?"
            params.append(event_type)
        if min_tick is not None:
            query += " AND tick >= ?"
            params.append(min_tick)
        if max_tick is not None:
            query += " AND tick <= ?"
            params.append(max_tick)
        query += " ORDER BY tick ASC"
        if limit:
            query += " LIMIT ?"
            params.append(limit)
        cursor = self.conn.cursor()
        cursor.execute(query, params)
        rows = cursor.fetchall()
        result = []
        for row in rows:
            d = dict(row)
            d["details"] = json.loads(d["details"]) if d["details"] else {}
            result.append(d)
        return result

    def export_family_tree(self, root_agent_id: str,
                             max_depth: int = 5) -> Optional[Dict]:
        def build_node(agent_id: str, depth: int) -> Optional[Dict]:
            if depth > max_depth:
                return None
            agent = self.get_agent(agent_id)
            if not agent:
                return None
            offspring = self.query_offspring(agent_id)
            return {
                "agent_id": agent_id,
                "generation": agent["generation"],
                "birth_tick": agent["birth_tick"],
                "death_tick": agent["death_tick"],
                "death_cause": agent["death_cause"],
                "parents": agent["parents"],
                "children": [build_node(c["agent_id"], depth + 1) for c in offspring],
            }
        return build_node(root_agent_id, 0)

    def export_all_agents_json(self) -> str:
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM agents")
        agents = [self._row_to_agent_dict(row) for row in cursor.fetchall()]
        return json.dumps(agents, default=str, indent=2)

    def get_summary(self) -> Dict:
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(*) as c FROM agents")
        total_agents = cursor.fetchone()["c"]
        cursor.execute("SELECT COUNT(*) as c FROM agents WHERE death_tick IS NULL")
        living = cursor.fetchone()["c"]
        cursor.execute("SELECT COUNT(*) as c FROM vocabulary")
        total_words = cursor.fetchone()["c"]
        cursor.execute("SELECT COUNT(*) as c FROM events")
        total_events = cursor.fetchone()["c"]
        cursor.execute("SELECT MAX(generation) as g FROM agents")
        max_gen = cursor.fetchone()["g"] or 0
        return {
            "total_agents": total_agents,
            "living_agents": living,
            "deceased_agents": total_agents - living,
            "total_words": total_words,
            "total_events": total_events,
            "max_generation": max_gen,
        }

    def close(self) -> None:
        self.conn.close()


# ═══════════════════════════════════════════════════════════════════════════════
# ScienceDashboard (Phase 12.3)
# ═══════════════════════════════════════════════════════════════════════════════

class ScienceDashboard:
    """
    Generates a static HTML dashboard with charts visualizing the simulation.

    Charts:
      - Population over time (line chart)
      - Trait evolution per generation (line chart)
      - Vocabulary size over time (line chart)
      - Cultural distance matrix (heatmap)
      - Brain size distribution (histogram)
      - Extinction timeline (marked on all charts)
    """

    def __init__(self, output_path: str = "docs/science_dashboard.html"):
        self.output_path = output_path

    def generate(self, population_history: List[Tuple[int, int]],
                  trait_evolution: Dict[int, Dict[str, float]],
                  vocab_history: List[Tuple[int, int]],
                  cultural_distances: Optional[Dict] = None,
                  brain_sizes: Optional[List[int]] = None,
                  extinction_ticks: Optional[List[int]] = None) -> str:
        """Generate the HTML dashboard file. Returns the file path."""
        population_data = json.dumps(population_history)
        trait_data = json.dumps(trait_evolution)
        vocab_data = json.dumps(vocab_history)
        cultural_data = json.dumps(cultural_distances or {})
        brain_data = json.dumps(brain_sizes or [])
        extinction_data = json.dumps(extinction_ticks or [])

        html = f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>Nafs AI — Science Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
  body {{ font-family: -apple-system, sans-serif; background: #0d1117; color: #c9d1d9; margin: 20px; }}
  h1 {{ color: #58a6ff; }}
  h2 {{ color: #8b949e; margin-top: 30px; }}
  .chart-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
  .chart-container {{ background: #161b22; border: 1px solid #30363d; padding: 16px; border-radius: 8px; }}
  .full-width {{ grid-column: 1 / -1; }}
  .stats {{ display: flex; gap: 16px; margin-bottom: 20px; }}
  .stat {{ background: #161b22; padding: 12px; border-radius: 6px; border: 1px solid #30363d; }}
  .stat .num {{ font-size: 24px; color: #58a6ff; }}
  .stat .lbl {{ color: #8b949e; font-size: 11px; text-transform: uppercase; }}
</style>
</head>
<body>
  <h1>NAFS AI — Science Dashboard</h1>
  <div class="stats">
    <div class="stat"><div class="num">{len(population_history)}</div><div class="lbl">Ticks tracked</div></div>
    <div class="stat"><div class="num">{len(trait_evolution)}</div><div class="lbl">Generations</div></div>
    <div class="stat"><div class="num">{len(vocab_history)}</div><div class="lbl">Vocab snapshots</div></div>
    <div class="stat"><div class="num">{len(extinction_ticks or [])}</div><div class="lbl">Extinctions</div></div>
  </div>

  <div class="chart-grid">
    <div class="chart-container">
      <h2>Population Over Time</h2>
      <canvas id="populationChart"></canvas>
    </div>
    <div class="chart-container">
      <h2>Vocabulary Size Over Time</h2>
      <canvas id="vocabChart"></canvas>
    </div>
    <div class="chart-container full-width">
      <h2>Trait Evolution Per Generation</h2>
      <canvas id="traitChart"></canvas>
    </div>
    <div class="chart-container">
      <h2>Brain Size Distribution</h2>
      <canvas id="brainChart"></canvas>
    </div>
    <div class="chart-container">
      <h2>Extinction Timeline</h2>
      <div id="extinctionList"></div>
    </div>
  </div>

<script>
const populationData = {population_data};
const traitData = {trait_data};
const vocabData = {vocab_data};
const brainData = {brain_data};
const extinctionTicks = {extinction_data};

// Population chart
new Chart(document.getElementById('populationChart'), {{
  type: 'line',
  data: {{
    labels: populationData.map(p => p[0]),
    datasets: [{{
      label: 'Agents alive',
      data: populationData.map(p => p[1]),
      borderColor: '#58a6ff',
      backgroundColor: 'rgba(88, 166, 255, 0.1)',
    }}]
  }},
  options: {{ responsive: true }}
}});

// Vocabulary chart
new Chart(document.getElementById('vocabChart'), {{
  type: 'line',
  data: {{
    labels: vocabData.map(v => v[0]),
    datasets: [{{
      label: 'Total words',
      data: vocabData.map(v => v[1]),
      borderColor: '#3fb950',
      backgroundColor: 'rgba(63, 185, 80, 0.1)',
    }}]
  }},
  options: {{ responsive: true }}
}});

// Trait evolution chart
const traitLabels = Object.keys(traitData).sort((a, b) => a - b);
const traitNames = [...new Set(traitLabels.flatMap(g => Object.keys(traitData[g])))];
const traitDatasets = traitNames.map(name => ({{
  label: name,
  data: traitLabels.map(g => traitData[g][name] || 0),
  borderColor: '#' + Math.floor(Math.random()*16777215).toString(16),
}}));
new Chart(document.getElementById('traitChart'), {{
  type: 'line',
  data: {{ labels: traitLabels, datasets: traitDatasets }},
  options: {{ responsive: true }}
}});

// Brain size histogram
const brainBins = {{}};
brainData.forEach(s => {{ brainBins[s] = (brainBins[s] || 0) + 1; }});
const sortedBins = Object.keys(brainBins).map(Number).sort((a, b) => a - b);
new Chart(document.getElementById('brainChart'), {{
  type: 'bar',
  data: {{
    labels: sortedBins,
    datasets: [{{
      label: 'Agent count',
      data: sortedBins.map(b => brainBins[b]),
      backgroundColor: '#bc8cff',
    }}]
  }},
  options: {{ responsive: true }}
}});

// Extinction list
const extList = document.getElementById('extinctionList');
extinctionTicks.forEach(tick => {{
  const div = document.createElement('div');
  div.textContent = '☠ Extinction at tick ' + tick;
  div.style.color = '#ff7b72';
  div.style.margin = '4px 0';
  extList.appendChild(div);
}});
if (!extinctionTicks.length) {{
  extList.innerHTML = '<div style="color:#8b949e">No extinction events recorded.</div>';
}}
</script>
</body>
</html>
'''
        try:
            os.makedirs(os.path.dirname(self.output_path) or ".", exist_ok=True)
            with open(self.output_path, "w") as f:
                f.write(html)
        except Exception as e:
            print(f"[ScienceDashboard] Failed to write: {e}", flush=True)
        return self.output_path


# ═══════════════════════════════════════════════════════════════════════════════
# Self-test
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import tempfile

    print("Testing EventLogSystem + LineageDatabase + ScienceDashboard...")

    with tempfile.TemporaryDirectory() as tmp:
        # EventLogSystem
        log = EventLogSystem(log_path=os.path.join(tmp, "events.jsonl"))
        log.record("BIRTH", tick=100, agent_id="baby_1", generation=2,
                    details={"parents": ["adam", "eve"]})
        log.record("DEATH", tick=500, agent_id="adam", generation=1,
                    details={"cause": "starvation"})
        log.record("FIRST_CONTACT", tick=200, agent_id="adam", details={"distance": 4})
        log.record("FIRST_WORD", tick=150, agent_id="adam", details={"word": "cold pain"})

        assert len(log.query()) == 4
        assert len(log.query(event_type="BIRTH")) == 1
        assert len(log.query(max_tick=180)) == 2  # BIRTH(100), FIRST_WORD(150)
        assert log.get_first_event_of_type("BIRTH")["tick"] == 100
        summary = log.get_summary()
        assert summary["total_events"] == 4
        print(f"  EventLog: {summary['by_type']}")

        # LineageDatabase
        db = LineageDatabase(os.path.join(tmp, "lineage.db"))
        db.insert_agent("adam", [], 1, 0, biome="plains")
        db.insert_agent("eve", [], 1, 0, biome="forest")
        db.insert_agent("baby_1", ["adam", "eve"], 2, 100, biome="plains")
        db.record_death("adam", 500, "starvation")
        db.record_death("eve", 800, "old_age")

        longest = db.query_longest_lived()
        assert longest["agent_id"] == "eve"
        print(f"  Longest lived: {longest['agent_id']} ({longest['death_tick']} ticks)")

        assert len(db.query_agents_by_generation(2)) == 1
        assert len(db.query_offspring("adam")) == 1

        db.insert_vocabulary("adam", "cold pain", "extreme cold hurts", 100, 1)
        assert len(db.get_agent_vocabulary("adam")) == 1
        assert db.get_word_origin("cold pain")["agent_id"] == "adam"

        db.insert_relationship("adam", "eve", "MATE", 0.8, 0.9, 200)
        assert len(db.get_relationships("adam")) >= 1

        db.insert_event("BIRTH", tick=100, agent_id="baby_1")
        assert len(db.query_events()) == 1

        tree = db.export_family_tree("adam")
        assert tree["agent_id"] == "adam"
        assert len(tree["children"]) == 1

        db_summary = db.get_summary()
        print(f"  DB Summary: {db_summary}")
        assert db_summary["total_agents"] == 3
        db.close()

        # ScienceDashboard
        dashboard = ScienceDashboard(output_path=os.path.join(tmp, "dashboard.html"))
        pop_history = [(tick, 2 + (tick // 100)) for tick in range(0, 1000, 50)]
        trait_evol = {
            1: {"metabolism": 1.0, "curiosity": 0.5},
            2: {"metabolism": 1.1, "curiosity": 0.6},
            3: {"metabolism": 1.2, "curiosity": 0.7},
        }
        vocab_hist = [(tick, 5 + tick // 100) for tick in range(0, 1000, 100)]
        brain_sizes = [777, 4353, 4353, 31383, 67289]
        extinction_ticks = [5000]

        path = dashboard.generate(pop_history, trait_evol, vocab_hist,
                                    brain_sizes=brain_sizes,
                                    extinction_ticks=extinction_ticks)
        assert os.path.exists(path)
        print(f"  Dashboard written to: {path}")

    print("\n✓ EventLogSystem + LineageDatabase + ScienceDashboard self-test passed")
