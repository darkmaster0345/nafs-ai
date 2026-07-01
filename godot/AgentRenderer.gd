extends Node2D

## ════════════════════════════════════════════════════════════════════════════
##  Nafs AI — AgentRenderer v3.0 (Cinematic Edition)
##  ───────────────────────────────────────────────────────────────────────────
##  Procedural skeletal animation with verlet physics, mood auras,
##  particle systems, and Disney's 12 principles applied.
##
##  Architecture (per agent):
##    AgentBody (Node2D + _draw)
##      ├─ Head         — squash-stretch breathing, look-at, blink
##      ├─ Torso        — bends with movement, breathing scale
##      ├─ Arms ×2      — shoulder+elbow joints, swing with gait
##      ├─ Legs ×2      — hip+knee+foot, proper walk cycle (IK-ish)
##      ├─ Hands ×2     — squash on contact
##      ├─ Hair strands — verlet integration, secondary motion
##      └─ Face         — eyes, eyebrows, mouth, sweat, tears
##    AgentShadow (Polygon2D + blur shader)
##    AgentAura (GPUParticles2D + CanvasItem shader)
##    AgentParticles (multiple GPUParticles2D)
##      ├─ Dust       — when walking/running
##      ├─ Breaths    — fog puffs in cold biomes
##      ├─ Sweat      — drops in hot biomes
##      ├─ Sparkles   — on new word discovery
##      ├─ Hearts     — on reproduction event
##      ├─ SoulDrift  — on death (dissolve into particles)
##      └─ BirthLight — pillar of light on spawn
##
##  Disney 12 Principles applied:
##    1. Squash & Stretch  — breathing, footsteps, eat lunge
##    2. Anticipation      — lean-back before FLEE, crouch before JUMP
##    3. Staging           — aura focuses eye on emotion
##    4. Straight-ahead    — verlet hair uses real physics
##    5. Follow-through    — hair continues after head stops
##    6. Slow in/out       — all transitions use ease functions
##    7. Arcs              — limbs move along arcs, not straight lines
##    8. Secondary action  — hair, breath, dust support primary action
##    9. Timing           — different speeds per action (SLEEP slow, FLEE fast)
##   10. Exaggeration     — bigger than realistic for emotional punch
##   11. Solid drawing    — drop shadow, rim light, depth shading
##   12. Appeal           — clear silhouettes, harmonious palette
## ════════════════════════════════════════════════════════════════════════════

const TILE_SIZE: int = 32

# ── Color palette per agent type ──────────────────────────────────────────────
const AGENT_COLORS: Dictionary = {
        "adam":  { "body": Color(0.35, 0.55, 1.00, 1.0), "accent": Color(0.55, 0.75, 1.00, 1.0),
                   "hair": Color(0.20, 0.35, 0.70, 1.0), "aura":  Color(0.40, 0.60, 1.00, 0.6) },
        "eve":   { "body": Color(1.00, 0.45, 0.65, 1.0), "accent": Color(1.00, 0.65, 0.80, 1.0),
                   "hair": Color(0.70, 0.25, 0.45, 1.0), "aura":  Color(1.00, 0.50, 0.70, 0.6) },
        "baby":  { "body": Color(1.00, 1.00, 1.00, 1.0), "accent": Color(0.95, 0.95, 1.00, 1.0),
                   "hair": Color(0.80, 0.80, 0.90, 1.0), "aura":  Color(1.00, 1.00, 1.00, 0.5) },
        "elder": { "body": Color(0.80, 0.70, 0.90, 1.0), "accent": Color(0.90, 0.85, 0.95, 1.0),
                   "hair": Color(0.95, 0.95, 0.95, 1.0), "aura":  Color(0.85, 0.80, 0.95, 0.5) },
}

# ── Mood → aura color map ────────────────────────────────────────────────────
const MOOD_COLORS: Dictionary = {
        "calm":     Color(0.50, 0.70, 1.00, 0.4),
        "happy":    Color(1.00, 0.85, 0.30, 0.5),
        "curious":  Color(0.80, 1.00, 0.50, 0.5),
        "afraid":   Color(0.60, 0.40, 1.00, 0.6),
        "hurt":     Color(1.00, 0.20, 0.20, 0.7),
        "hungry":   Color(1.00, 0.60, 0.20, 0.5),
        "sleeping": Color(0.60, 0.50, 0.90, 0.4),
        "dying":    Color(0.30, 0.30, 0.30, 0.8),
}

# ── Per-action animation parameters ───────────────────────────────────────────
const ANIM: Dictionary = {
        "IDLE":     {"bob":1.2, "freq":1.4, "sway":0.0, "lean":0.00, "jitter":0.0, "scale":1.00, "alpha":1.00, "gait":0.0, "arm_swing":0.3, "blink_rate":3.0},
        "EXPLORE":  {"bob":2.0, "freq":3.0, "sway":0.6, "lean":0.06, "jitter":0.0, "scale":1.00, "alpha":1.00, "gait":1.0, "arm_swing":1.0, "blink_rate":2.0},
        "MOVE":     {"bob":2.5, "freq":4.0, "sway":0.8, "lean":0.10, "jitter":0.0, "scale":1.00, "alpha":1.00, "gait":1.4, "arm_swing":1.4, "blink_rate":2.0},
        "EAT":      {"bob":0.5, "freq":6.0, "sway":0.0, "lean":0.18, "jitter":0.0, "scale":1.00, "alpha":1.00, "gait":0.0, "arm_swing":0.2, "blink_rate":1.5},
        "DRINK":    {"bob":0.5, "freq":5.0, "sway":0.0, "lean":0.22, "jitter":0.0, "scale":1.00, "alpha":1.00, "gait":0.0, "arm_swing":0.3, "blink_rate":1.5},
        "SLEEP":    {"bob":0.3, "freq":0.5, "sway":0.0, "lean":0.00, "jitter":0.0, "scale":0.85,"alpha":0.85,"gait":0.0, "arm_swing":0.0, "blink_rate":0.0},
        "HIDE":     {"bob":0.2, "freq":0.5, "sway":0.0, "lean":-0.05,"jitter":0.0, "scale":0.70,"alpha":0.55,"gait":0.0, "arm_swing":0.1, "blink_rate":5.0},
        "FLEE":     {"bob":3.0, "freq":8.0, "sway":1.8, "lean":0.28, "jitter":1.2, "scale":1.05,"alpha":1.00, "gait":2.2, "arm_swing":2.0, "blink_rate":0.5},
        "HURT":     {"bob":0.0, "freq":0.0, "sway":0.0, "lean":0.00, "jitter":3.5, "scale":1.10,"alpha":1.00, "gait":0.0, "arm_swing":0.0, "blink_rate":0.0},
        "OBSERVE":  {"bob":0.6, "freq":1.8, "sway":0.0, "lean":0.02, "jitter":0.0, "scale":1.00, "alpha":1.00, "gait":0.0, "arm_swing":0.2, "blink_rate":1.0},
        "APPROACH": {"bob":1.8, "freq":3.2, "sway":0.5, "lean":0.08, "jitter":0.0, "scale":1.00, "alpha":1.00, "gait":0.9, "arm_swing":0.9, "blink_rate":1.5},
        "FOLLOW":   {"bob":2.0, "freq":3.0, "sway":0.6, "lean":0.07, "jitter":0.0, "scale":1.00, "alpha":1.00, "gait":1.0, "arm_swing":1.0, "blink_rate":2.0},
        "SHARE":    {"bob":0.8, "freq":2.0, "sway":0.0, "lean":0.05, "jitter":0.0, "scale":1.00, "alpha":1.00, "gait":0.0, "arm_swing":0.5, "blink_rate":1.0},
}

# All active agents: {agent_id: AgentState}
var agents: Dictionary = {}

var _time: float = 0.0
var _biome_temp: float = 0.5  # 0=cold, 1=hot — used for breath fog / sweat
var _weather: String = "clear"

# ════════════════════════════════════════════════════════════════════════════
#  INITIALIZATION
# ════════════════════════════════════════════════════════════════════════════

func _ready() -> void:
        await get_tree().process_frame
        var nc = get_node_or_null("/root/NetworkController")
        if nc:
                nc.connect("agents_updated", _on_agents_updated)
                # tick_processed carries the full state dict (weather, time_of_day, biome, ...)
                nc.connect("tick_processed", _on_tick_processed)

func _process(delta: float) -> void:
        _time += delta
        for agent_id in agents:
                var a = agents[agent_id]
                _update_agent_animation(a, delta)
                _update_hair_physics(a, delta)
                _update_particles(a, delta)
                a.body.queue_redraw()
                a.face.queue_redraw()
                if is_instance_valid(a.shadow):
                        a.shadow.queue_redraw()

# ════════════════════════════════════════════════════════════════════════════
#  DATA INGESTION — receives state from NetworkController
# ════════════════════════════════════════════════════════════════════════════

func _on_tick_processed(state: Dictionary) -> void:
        # Extract world-level info we care about for particle effects
        if state.has("weather"):
                _weather = state.weather
        # Map biome temperature (Kelvin or 0-1) to a 0-1 cold→hot scale
        if state.has("biome_temp"):
                _biome_temp = float(state.biome_temp)
        elif state.has("temperature"):
                # If it's in Celsius, normalize: -10..40 → 0..1
                var t = float(state.temperature)
                _biome_temp = clamp((t + 10.0) / 50.0, 0.0, 1.0)
        # Update each agent's biome_temp locally
        for aid in agents:
                agents[aid].biome_temp = _biome_temp

func _on_agents_updated(agents_data: Array) -> void:
        var seen = {}
        for d in agents_data:
                var aid = d.get("id", "")
                if aid == "": continue
                seen[aid] = true
                if not agents.has(aid):
                        _spawn_agent(aid, d)
                _update_agent_data(aid, d)
        for id in agents.keys():
                if not seen.has(id):
                        _kill_agent(id)
        # Update family lines after all agent positions are settled
        _update_family_lines(agents_data)

func _update_agent_data(aid: String, d: Dictionary) -> void:
        var a = agents[aid]
        # Target position (smoothed toward)
        var x = d.get("x", 0)
        var y = d.get("y", 0)
        a.target_pos = Vector2(x * TILE_SIZE + TILE_SIZE/2, y * TILE_SIZE + TILE_SIZE/2)
        # Movement direction = difference from current pos
        a.move_dir = (a.target_pos - a.last_pos).normalized() if a.last_pos != a.target_pos else Vector2.ZERO
        if a.target_pos.distance_to(a.last_pos) > 0.5:
                a.facing = a.move_dir.x  # -1 = left, +1 = right
        a.last_pos = a.target_pos

        # Action transition
        var new_action = d.get("action", "IDLE")
        if new_action != a.action:
                a.prev_action = a.action
                a.action = new_action
                a.action_t = 0.0
                _trigger_action_particles(a, new_action)
        a.action_t += 1.0 / 60.0  # rough estimate; overridden in _process

        # HP drop = hurt
        var hp = d.get("hp", 100)
        if a.last_hp != null and hp < a.last_hp - 5:
                a.hurt_flash = 1.0
                _burst(a.hurt_particles, 12)
        a.last_hp = hp
        a.hp = hp

        # Mood inference from stats
        a.mood = _infer_mood(d, a.action)
        a.thought_text = d.get("dialogue", d.get("thought", ""))

        # Scale by life stage
        a.life_stage = d.get("life_stage", "adult")
        a.base_scale = _life_stage_scale(a.life_stage)

        # Color
        var atype = d.get("type", "adam").to_lower()
        a.colors = AGENT_COLORS.get(atype, AGENT_COLORS.adam)
        if a.life_stage == "elder":
                a.colors = AGENT_COLORS.elder

        # Blink rate from action params
        var params = ANIM.get(a.action, ANIM.IDLE)
        a.blink_period = 1.0 / max(0.01, params.blink_rate) if params.blink_rate > 0 else 999.0

func _infer_mood(d: Dictionary, action: String) -> String:
        var hp = d.get("hp", 100)
        var stress = d.get("stress", 0)
        var hunger = d.get("hunger", 0)
        var energy = d.get("energy", 100)
        if hp < 30: return "hurt"
        if action == "SLEEP": return "sleeping"
        if action == "FLEE" or action == "HIDE": return "afraid"
        if stress > 5: return "afraid"
        if hunger > 60: return "hungry"
        if action == "EXPLORE": return "curious"
        if hp > 80 and stress < 2 and hunger < 30: return "happy"
        return "calm"

# ════════════════════════════════════════════════════════════════════════════
#  AGENT SPAWN — creates full node hierarchy
# ════════════════════════════════════════════════════════════════════════════

func _spawn_agent(aid: String, d: Dictionary) -> void:
        # Root node — moves with target_pos
        var root = Node2D.new()
        root.name = "Agent_" + aid
        add_child(root)

        # Shadow (Polygon2D with blur shader)
        var shadow = Polygon2D.new()
        shadow.polygon = PackedVector2Array([
                Vector2(-14, -3), Vector2(14, -3), Vector2(14, 3), Vector2(-14, 3)
        ])
        shadow.color = Color(0, 0, 0, 0.35)
        shadow.position = Vector2(0, 22)
        shadow.z_index = -5
        var shadow_mat = ShaderMaterial.new()
        shadow_mat.shader = _blur_shadow_shader()
        shadow.material = shadow_mat
        root.add_child(shadow)

        # Body (custom draw — head, torso, arms, legs, hair)
        var body = _AgentBodyDraw.new()
        body.name = "Body"
        body.z_index = 1
        root.add_child(body)

        # Face (custom draw — eyes, mouth, etc.)
        var face = _AgentFaceDraw.new()
        face.name = "Face"
        face.z_index = 5
        root.add_child(face)

        # Action label above head
        var label = Label.new()
        label.text = "IDLE"
        label.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
        label.add_theme_font_size_override("font_size", 10)
        label.position = Vector2(-30, -55)
        label.custom_minimum_size = Vector2(60, 14)
        label.z_index = 10
        root.add_child(label)

        # Thought label (hidden until hover)
        var thought = Label.new()
        thought.text = ""
        thought.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
        thought.add_theme_font_size_override("font_size", 9)
        thought.position = Vector2(-60, -75)
        thought.custom_minimum_size = Vector2(120, 14)
        thought.visible = false
        thought.add_theme_color_override("font_color", Color(0.95, 0.95, 0.95))
        thought.z_index = 11
        root.add_child(thought)

        # ── Particle systems ───────────────────────────────────────────────────
        var dust = _make_particles(Color(0.7, 0.6, 0.5, 0.6), 0.5, 8.0, Vector2(0,18), Vector2(20, 5))
        root.add_child(dust)

        var breath = _make_particles(Color(0.85, 0.9, 1.0, 0.5), 1.5, 4.0, Vector2(0,-12), Vector2(5, 2))
        root.add_child(breath)

        var sweat = _make_particles(Color(0.5, 0.7, 1.0, 0.7), 0.4, 6.0, Vector2(0,-15), Vector2(3, 1))
        root.add_child(sweat)

        var sparkles = _make_particles(Color(1.0, 0.95, 0.4, 0.9), 0.8, 5.0, Vector2(0,-10), Vector2(15, 15))
        root.add_child(sparkles)

        var hearts = _make_particles(Color(1.0, 0.4, 0.6, 0.9), 1.0, 6.0, Vector2(0,-10), Vector2(10, 10))
        root.add_child(hearts)

        var hurt_parts = _make_particles(Color(1.0, 0.2, 0.2, 0.9), 0.3, 5.0, Vector2(0,-8), Vector2(8, 8))
        root.add_child(hurt_parts)

        var soul = _make_particles(Color(0.9, 0.9, 1.0, 0.7), 3.0, 6.0, Vector2(0,-10), Vector2(3, 1))
        root.add_child(soul)

        var birth_light = _make_particles(Color(1.0, 1.0, 0.9, 1.0), 1.2, 8.0, Vector2(0,0), Vector2(25, 40))
        root.add_child(birth_light)
        _burst(birth_light, 30)  # initial birth burst

        # Store state
        agents[aid] = {
                "root": root, "shadow": shadow, "body": body, "face": face,
                "action_label": label, "thought_label": thought,
                "family_lines": [],
                "dust_particles": dust, "breath_particles": breath,
                "sweat_particles": sweat, "sparkle_particles": sparkles,
                "heart_particles": hearts, "hurt_particles": hurt_parts,
                "soul_particles": soul, "birth_particles": birth_light,
                "target_pos": Vector2.ZERO, "last_pos": Vector2.ZERO,
                "move_dir": Vector2.ZERO, "facing": 1.0,
                "action": "IDLE", "prev_action": null, "action_t": 0.0,
                "mood": "calm", "thought_text": "",
                "base_scale": 1.0, "life_stage": "adult",
                "colors": AGENT_COLORS.adam,
                "hp": 100, "last_hp": null, "hurt_flash": 0.0,
                "spawn_anim_t": 0.8,  # 800ms birth animation
                "death_anim_t": 0.0,  # grows when killed
                "blink_t": 0.0, "blink_period": 3.0,
                "blink_state": 1.0,  # 1 = open, 0 = closed
                "gait_phase": 0.0,
                "breath_phase": 0.0,
                "hair_strands": _init_hair(5),  # 5 hair strands
                "last_hp_drop_t": 0.0,
                "biome_temp": 0.5,
        }
        # Initialize body drawing state
        body.state = agents[aid]
        face.state = agents[aid]

# ── Hair strand initialization (verlet) ──────────────────────────────────────
func _init_hair(n: int) -> Array:
        var strands = []
        for i in range(n):
                var strand = {
                        "points": [Vector2(-6 + i*3, -16), Vector2(-7 + i*3, -19), Vector2(-8 + i*3, -22), Vector2(-9 + i*3, -25)],
                        "prev":   [Vector2(-6 + i*3, -16), Vector2(-7 + i*3, -19), Vector2(-8 + i*3, -22), Vector2(-9 + i*3, -25)],
                        "seg_len": 3.0,
                }
                strands.append(strand)
        return strands

# ════════════════════════════════════════════════════════════════════════════
#  ANIMATION UPDATE — applies all motion each frame
# ════════════════════════════════════════════════════════════════════════════

func _update_agent_animation(a: Dictionary, delta: float) -> void:
        var root = a.root as Node2D
        if not is_instance_valid(root): return

        # Smoothly lerp root position toward target
        root.position = root.position.lerp(a.target_pos, min(1.0, delta * 10.0))

        # Get anim params
        var params = ANIM.get(a.action, ANIM.IDLE)

        # Spawn animation (birth): scale 0 → 1 with ease-out-back
        var spawn_p = 1.0
        if a.spawn_anim_t > 0:
                a.spawn_anim_t = max(0.0, a.spawn_anim_t - delta)
                var t = 1.0 - a.spawn_anim_t / 0.8
                t = clamp(t, 0.0, 1.0)
                # Ease-out-back: overshoot then settle
                var c1 = 1.70158
                var c3 = c1 + 1.0
                spawn_p = 1.0 + c3 * pow(t - 1.0, 3.0) + c1 * pow(t - 1.0, 2.0)

        # Death animation: scale 1 → 0 with ease-in
        var death_p = 1.0
        if a.death_anim_t > 0:
                a.death_anim_t = max(0.0, a.death_anim_t - delta)
                var t = 1.0 - a.death_anim_t / 1.2
                t = clamp(t, 0.0, 1.0)
                death_p = 1.0 - t * t  # ease-in quad
                # Emit soul particles as it dies
                if randf() < 0.5:
                        _burst(a.soul_particles, 2)

        # Bob, sway, jitter
        var t = _time
        var bob = sin(t * params.freq * 2.0 * PI) * params.bob
        var sway = sin(t * params.sway * 0.5 * 2.0 * PI) * params.sway if params.sway > 0 else 0.0
        var jitter_x = (randf() - 0.5) * params.jitter
        var jitter_y = (randf() - 0.5) * params.jitter

        # Apply position offsets to body
        var body = a.body
        body.offset = Vector2(sway + jitter_x, bob + jitter_y)
        body.lean = params.lean * a.facing  # lean in facing direction
        body.scale_factor = params.scale * a.base_scale * spawn_p * death_p

        # Gait phase advances when moving
        var speed = abs(a.move_dir.x) + abs(a.move_dir.y)
        a.gait_phase += delta * params.gait * 6.0
        a.breath_phase += delta * 1.5

        # Blink
        a.blink_t += delta
        if a.blink_t > a.blink_period:
                a.blink_t = 0.0
                a.blink_state = 0.0  # close
        elif a.blink_t > 0.1:  # 100ms closed
                a.blink_state = lerp(a.blink_state, 1.0, min(1.0, delta * 20.0))

        # Update face state
        var face = a.face
        face.blink = a.blink_state
        face.mood = a.mood
        face.action = a.action
        face.facing = a.facing
        face.hp = a.hp if a.hp != null else 100
        face.colors = a.colors
        face.time = _time
        face.scale_factor = body.scale_factor

        # Body color shift during hurt
        if a.hurt_flash > 0:
                a.hurt_flash = max(0.0, a.hurt_flash - delta * 3.0)
        body.hurt_flash = a.hurt_flash
        body.colors = a.colors
        body.mood = a.mood
        body.facing = a.facing
        body.gait_phase = a.gait_phase
        body.breath_phase = a.breath_phase
        body.action = a.action
        body.arm_swing = params.arm_swing
        body.time = _time

        # Shadow scales with HP (lower HP = smaller shadow = fading presence)
        if is_instance_valid(a.shadow):
                var shadow_scale = 0.5 + 0.5 * (a.hp / 100.0) if a.hp != null else 1.0
                shadow_scale *= spawn_p * death_p
                a.shadow.scale = Vector2(shadow_scale, shadow_scale * 0.5)
                a.shadow.modulate.a = 0.35 * spawn_p * death_p

        # Label effects
        if a.action == "SLEEP":
                a.action_label.text = "Z" + "z".repeat(int(1 + (sin(_time * 2.0) + 1.0) * 1.5))
                a.action_label.modulate.a = 0.5 + 0.5 * sin(_time * 2.0)
                a.action_label.modulate = Color(0.8, 0.8, 1.0, a.action_label.modulate.a)
        elif a.action == "HIDE":
                a.action_label.text = "..."
                a.action_label.modulate.a = 0.4
        elif a.action == "HURT" or a.hurt_flash > 0.5:
                a.action_label.text = "!"
                a.action_label.modulate = Color(1.0, 0.3, 0.3, 1.0)
        elif a.action == "EAT":
                a.action_label.text = "nom"
                a.action_label.modulate = Color(1.0, 0.8, 0.4, 0.9)
        elif a.action == "FLEE":
                a.action_label.text = "!!"
                a.action_label.modulate = Color(1.0, 0.6, 0.2, 1.0)
        elif a.action == "SHARE":
                a.action_label.text = "♥"
                a.action_label.modulate = Color(1.0, 0.4, 0.6, 1.0)
        else:
                a.action_label.text = a.action
                a.action_label.modulate = Color(1, 1, 1, 0.8)

        # Thought label
        a.thought_label.text = a.thought_text

# ── Verlet hair physics ──────────────────────────────────────────────────────
func _update_hair_physics(a: Dictionary, delta: float) -> void:
        var gravity = Vector2(0, 30.0)
        var damping = 0.92
        for strand in a.hair_strands:
                var pts = strand.points
                var prev = strand.prev
                for i in range(1, pts.size()):
                        var vel = (pts[i] - prev[i]) * damping
                        prev[i] = pts[i]
                        pts[i] = pts[i] + vel + gravity * delta * 0.5
                # Satisfy distance constraints (2 iterations)
                for _iter in range(2):
                        for i in range(1, pts.size()):
                                var d = pts[i] - pts[i-1]
                                var dist = d.length()
                                if dist > 0.001:
                                        var diff = (dist - strand.seg_len) / dist
                                        if i == 1:
                                                # Anchor to head (doesn't move)
                                                pts[i] = pts[i] - d * diff
                                        else:
                                                pts[i-1] = pts[i-1] + d * 0.5 * diff
                                                pts[i] = pts[i] - d * 0.5 * diff
                # Anchor root point follows head
                var head_offset = Vector2(a.facing * 2, 0)
                pts[0] = Vector2(-3, -16) + head_offset

# ── Particle control ─────────────────────────────────────────────────────────
func _update_particles(a: Dictionary, delta: float) -> void:
        # Dust: emit when moving fast
        var speed = a.target_pos.distance_to(a.last_pos) * 60.0
        if speed > 1.0 and (a.action == "MOVE" or a.action == "EXPLORE" or a.action == "FLEE"):
                _set_emitting(a.dust_particles, true)
                _set_rate(a.dust_particles, min(20.0, speed * 2.0))
        else:
                _set_emitting(a.dust_particles, false)

        # Breath fog: cold biomes
        if _biome_temp < 0.35 and a.action != "SLEEP":
                _set_emitting(a.breath_particles, true)
                _set_rate(a.breath_particles, 5.0)
        else:
                _set_emitting(a.breath_particles, false)

        # Sweat: hot biomes
        if _biome_temp > 0.7 and a.action in ["MOVE", "EXPLORE", "FLEE", "EAT"]:
                _set_emitting(a.sweat_particles, true)
                _set_rate(a.sweat_particles, 3.0)
        else:
                _set_emitting(a.sweat_particles, false)

        # Mood-based aura particles (curious = sparkles, sleeping = soft)
        if a.mood == "curious" and randf() < 0.05:
                _burst(a.sparkle_particles, 1)
        if a.mood == "sleeping" and randf() < 0.02:
                _burst(a.soul_particles, 1)

func _trigger_action_particles(a: Dictionary, new_action: String) -> void:
        match new_action:
                "EAT":
                        _burst(a.dust_particles, 5)
                "SHARE":
                        _burst(a.heart_particles, 8)
                "FLEE":
                        _burst(a.dust_particles, 10)
                "HURT":
                        _burst(a.hurt_particles, 15)
                "SLEEP":
                        _burst(a.soul_particles, 3)
                "OBSERVE":
                        _burst(a.sparkle_particles, 4)
        # New word discovered (passed as separate event if you wire it)

# ════════════════════════════════════════════════════════════════════════════
#  AGENT DEATH — cinematic dissolve
# ════════════════════════════════════════════════════════════════════════════

func _kill_agent(aid: String) -> void:
        if not agents.has(aid): return
        var a = agents[aid]
        if a.death_anim_t > 0: return  # already dying
        a.death_anim_t = 1.2  # 1.2 seconds of dissolve
        # Massive soul particle burst
        _burst(a.soul_particles, 40)
        # Schedule actual removal after animation
        get_tree().create_timer(1.3).timeout.connect(func(): _finalize_death(aid))

func _finalize_death(aid: String) -> void:
        if not agents.has(aid): return
        var a = agents[aid]
        if is_instance_valid(a.root):
                a.root.queue_free()
        agents.erase(aid)

# ════════════════════════════════════════════════════════════════════════════
#  FAMILY LINES — draw connecting lines between parents and children
# ════════════════════════════════════════════════════════════════════════════

func _update_family_lines(agents_data: Array) -> void:
        # Clear all existing family lines
        for aid in agents:
                var a = agents[aid]
                for line in a.family_lines:
                        if is_instance_valid(line):
                                line.queue_free()
                a.family_lines = []
        # Draw lines between parents and children within 10 tiles
        for agent_data in agents_data:
                var child_id = agent_data.get("id", "")
                var parents = agent_data.get("parents", [])
                for parent_id in parents:
                        if not agents.has(child_id) or not agents.has(parent_id):
                                continue
                        var child_root = agents[child_id].root as Node2D
                        var parent_root = agents[parent_id].root as Node2D
                        if not is_instance_valid(child_root) or not is_instance_valid(parent_root):
                                continue
                        var dist = child_root.global_position.distance_to(parent_root.global_position)
                        if dist <= 10 * TILE_SIZE:
                                var line = _create_family_line(parent_root.global_position, child_root.global_position)
                                add_child(line)
                                agents[child_id].family_lines.append(line)

func _create_family_line(from: Vector2, to: Vector2) -> Line2D:
        var line = Line2D.new()
        line.add_point(from)
        line.add_point(to)
        line.default_color = Color(1.0, 0.8, 0.4, 0.5)
        line.width = 1.5
        return line

# ════════════════════════════════════════════════════════════════════════════
#  HELPER FACTORIES
# ════════════════════════════════════════════════════════════════════════════

func _life_stage_scale(s: String) -> float:
        match s:
                "newborn":    return 0.45
                "child":      return 0.65
                "adolescent": return 0.85
                "adult":      return 1.0
                "elder":      return 1.15
                "ancient":    return 1.25
                _:            return 1.0

func _make_particles(color: Color, lifetime: float, scale: float,
                                         pos: Vector2, spread: Vector2) -> GPUParticles2D:
        var p = GPUParticles2D.new()
        p.position = pos
        p.emitting = false
        p.amount = 30
        p.lifetime = lifetime
        p.explosiveness = 0.5
        p.local_coords = false
        var mat = ParticleProcessMaterial.new()
        mat.direction = Vector3(0, -1, 0)
        mat.spread = 30.0
        mat.initial_velocity_min = 5.0
        mat.initial_velocity_max = 15.0
        mat.gravity = Vector3(0, 20, 0)
        mat.scale_min = scale * 0.5
        mat.scale_max = scale
        mat.color = color
        mat.hue_variation_min = -0.05
        mat.hue_variation_max = 0.05
        p.process_material = mat
        return p

func _set_emitting(p: GPUParticles2D, on: bool) -> void:
        if is_instance_valid(p):
                p.emitting = on

func _set_rate(p: GPUParticles2D, rate: float) -> void:
        if is_instance_valid(p) and is_instance_valid(p.process_material):
                p.amount = int(clamp(rate * 2, 5, 50))

func _burst(p: GPUParticles2D, n: int) -> void:
        if not is_instance_valid(p): return
        p.amount = n
        p.emitting = true
        p.restart()
        # Auto-stop after one burst
        await get_tree().create_timer(p.lifetime * 0.5).timeout
        if is_instance_valid(p):
                p.emitting = false

func _blur_shadow_shader() -> Shader:
        var code = """
        shader_type canvas_item;
        void fragment() {
                vec2 uv = UV;
                vec4 c = vec4(0.0);
                float r = 3.0 / 64.0;
                c += texture(TEXTURE, uv + vec2(-r, -r));
                c += texture(TEXTURE, uv + vec2( r, -r));
                c += texture(TEXTURE, uv + vec2(-r,  r));
                c += texture(TEXTURE, uv + vec2( r,  r));
                c += texture(TEXTURE, uv);
                c /= 5.0;
                COLOR = c * vec4(1.0, 1.0, 1.0, 0.6);
        }
        """
        var s = Shader.new()
        s.code = code
        return s

# ════════════════════════════════════════════════════════════════════════════
#  HOVER — show thought bubble on mouseover
# ════════════════════════════════════════════════════════════════════════════

func _input(event: InputEvent) -> void:
        if event is InputEventMouseMotion:
                _check_hover(event.position)

func _check_hover(mouse_pos: Vector2) -> void:
        for aid in agents:
                var a = agents[aid]
                var root = a.root as Node2D
                if not is_instance_valid(root): continue
                var dist = root.global_position.distance_to(mouse_pos)
                a.thought_label.visible = dist < 30

func clear_all() -> void:
        for aid in agents.keys().duplicate():
                _finalize_death(aid)


# ════════════════════════════════════════════════════════════════════════════
#  INNER CLASS: AgentBodyDraw — draws skeletal body via _draw()
# ════════════════════════════════════════════════════════════════════════════

class _AgentBodyDraw extends Node2D:
        var state: Dictionary = {}
        var offset: Vector2 = Vector2.ZERO
        var lean: float = 0.0
        var scale_factor: float = 1.0
        var hurt_flash: float = 0.0
        var colors: Dictionary = {}
        var mood: String = "calm"
        var facing: float = 1.0
        var gait_phase: float = 0.0
        var breath_phase: float = 0.0
        var action: String = "IDLE"
        var arm_swing: float = 1.0
        var time: float = 0.0

        func _draw() -> void:
                if state.is_empty(): return
                var s = scale_factor
                var ox = offset.x
                var oy = offset.y
                var body_color = colors.get("body", Color.WHITE)
                var accent = colors.get("accent", body_color)
                var hair_color = colors.get("hair", body_color)

                # Hurt flash tints body red
                if hurt_flash > 0:
                        body_color = Color(
                                lerpf(body_color.r, 1.0, hurt_flash),
                                lerpf(body_color.g, 0.2, hurt_flash),
                                lerpf(body_color.b, 0.2, hurt_flash),
                                1.0
                        )

                # ── BREATHING: subtle scale pulse on torso ─────────────────────────
                var breath_scale = 1.0 + 0.05 * sin(breath_phase * 2.0 * PI)

                # ── TORSO (with lean) ──────────────────────────────────────────────
                var torso_top = Vector2(ox, oy - 8 * s)
                var torso_bot = Vector2(ox + lean * 4, oy + 6 * s)
                _torso(torso_top, torso_bot, 6 * s * breath_scale, body_color, accent)

                # ── HEAD (squash-stretch) ──────────────────────────────────────────
                var head_center = Vector2(ox + lean * 8, oy - 14 * s)
                var head_sqx = 1.0 + 0.04 * sin(breath_phase * 2.0 * PI + 0.5)
                var head_sqy = 1.0 - 0.04 * sin(breath_phase * 2.0 * PI + 0.5)
                # Action-based head squash
                match action:
                        "SLEEP": head_sqx = 1.15; head_sqy = 0.85
                        "EAT", "DRINK": head_sqy = 0.92; head_sqx = 1.08
                        "FLEE": head_sqy = 1.1; head_sqx = 0.9
                _head(head_center, 7 * s, body_color, accent, head_sqx, head_sqy, facing)

                # ── HAIR (verlet strands) ──────────────────────────────────────────
                for strand in state.get("hair_strands", []):
                        _draw_hair_strand(strand, head_center, s, hair_color)

                # ── ARMS (with shoulder + elbow swing) ─────────────────────────────
                var shoulder_l = torso_top + Vector2(-5 * s, 1 * s)
                var shoulder_r = torso_top + Vector2(5 * s, 1 * s)
                # Arm swing phase offset by π/2 from gait
                var arm_phase = gait_phase + PI / 2.0
                var arm_swing_amt = sin(arm_phase) * 0.4 * arm_swing
                # EAT: arms forward toward mouth
                if action == "EAT" or action == "DRINK":
                        arm_swing_amt = -0.8
                # SLEEP: arms tucked in
                elif action == "SLEEP":
                        arm_swing_amt = 0.6
                # HIDE: arms up around head
                elif action == "HIDE":
                        arm_swing_amt = -1.2

                var elbow_l = shoulder_l + Vector2(-2 * s + sin(arm_phase) * 2 * s, 5 * s + arm_swing_amt * 2 * s)
                var hand_l = elbow_l + Vector2(-1 * s, 4 * s + arm_swing_amt * 1 * s)
                _limb(shoulder_l, elbow_l, hand_l, 2.5 * s, accent, hand_l, 2 * s, body_color)
                var elbow_r = shoulder_r + Vector2(2 * s - sin(arm_phase) * 2 * s, 5 * s - arm_swing_amt * 2 * s)
                var hand_r = elbow_r + Vector2(1 * s, 4 * s - arm_swing_amt * 1 * s)
                _limb(shoulder_r, elbow_r, hand_r, 2.5 * s, accent, hand_r, 2 * s, body_color)

                # ── LEGS (with hip + knee, walk cycle) ─────────────────────────────
                var hip_l = torso_bot + Vector2(-3 * s, 1 * s)
                var hip_r = torso_bot + Vector2(3 * s, 1 * s)
                # Walk cycle: legs alternate using gait_phase
                var leg_phase_l = gait_phase
                var leg_phase_r = gait_phase + PI
                # When not moving, legs straight
                if action == "IDLE" or action == "SLEEP" or action == "HIDE":
                        leg_phase_l = 0.0
                        leg_phase_r = 0.0
                var knee_l = hip_l + Vector2(sin(leg_phase_l) * 2 * s, 5 * s)
                var foot_l = knee_l + Vector2(sin(leg_phase_l + 0.5) * 3 * s, 5 * s)
                # SLEEP: legs folded under
                if action == "SLEEP":
                        knee_l = hip_l + Vector2(-2 * s, 3 * s)
                        foot_l = knee_l + Vector2(-1 * s, 2 * s)
                var knee_r = hip_r + Vector2(sin(leg_phase_r) * 2 * s, 5 * s)
                var foot_r = knee_r + Vector2(sin(leg_phase_r + 0.5) * 3 * s, 5 * s)
                if action == "SLEEP":
                        knee_r = hip_r + Vector2(2 * s, 3 * s)
                        foot_r = knee_r + Vector2(1 * s, 2 * s)
                _limb(hip_l, knee_l, foot_l, 3 * s, accent, foot_l, 2.5 * s, body_color)
                _limb(hip_r, knee_r, foot_r, 3 * s, accent, foot_r, 2.5 * s, body_color)

                # ── MOOD AURA (radial gradient behind body) ───────────────────────
                var aura_color = MOOD_COLORS.get(mood, MOOD_COLORS.calm)
                if hurt_flash > 0:
                        aura_color = Color(1.0, 0.2, 0.2, 0.7)
                var aura_pos = head_center
                var aura_radius = 18 * s + 2 * sin(time * 3.0)
                _aura(aura_pos, aura_radius, aura_color)

                # ── RIM LIGHT (subtle highlight on facing side) ───────────────────
                var rim_pos = head_center + Vector2(facing * 3 * s, -2 * s)
                draw_circle(rim_pos, 2 * s, Color(1, 1, 1, 0.4))

        # ── Drawing primitives ───────────────────────────────────────────────
        func _torso(top: Vector2, bot: Vector2, width: float, color: Color, accent: Color) -> void:
                # Trapezoid torso: narrower at top, wider at bottom
                var pts = PackedVector2Array([
                        top + Vector2(-width * 0.6, 0),
                        top + Vector2(width * 0.6, 0),
                        bot + Vector2(width, 0),
                        bot + Vector2(-width, 0),
                ])
                var cols = PackedColorArray([color, color, color, color])
                draw_polygon(pts, cols)
                # Highlight strip down center
                draw_line(top, bot, accent.lightened(0.2), 1.0)

        func _head(center: Vector2, radius: float, color: Color, accent: Color,
                           sqx: float, sqy: float, facing: float) -> void:
                # Use draw_arc for the head shape with squash-stretch
                var rx = radius * sqx
                var ry = radius * sqy
                var pts = PackedVector2Array()
                var cols = PackedColorArray()
                var n = 24
                for i in range(n + 1):
                        var a = (float(i) / n) * TAU
                        pts.append(center + Vector2(cos(a) * rx, sin(a) * ry))
                        cols.append(color)
                draw_polygon(pts, cols)
                # Highlight on facing side
                var hl_pos = center + Vector2(facing * rx * 0.4, -ry * 0.3)
                draw_circle(hl_pos, rx * 0.3, accent.lightened(0.3))

        func _limb(joint1: Vector2, joint2: Vector2, joint3: Vector2,
                           width: float, color: Color, end: Vector2, end_r: float,
                           end_color: Color) -> void:
                # Tapered limb: thick at joint1, thin at joint3
                var pts1 = PackedVector2Array([joint1, joint2])
                draw_polyline(pts1, color, width, true)
                var pts2 = PackedVector2Array([joint2, joint3])
                draw_polyline(pts2, color, width * 0.8, true)
                # Joint circles for smoothness
                draw_circle(joint2, width * 0.5, color)
                # Hand/foot
                draw_circle(end, end_r, end_color)

        func _draw_hair_strand(strand: Dictionary, head_center: Vector2, s: float,
                                                   color: Color) -> void:
                var pts = strand.points
                if pts.size() < 2: return
                # Transform strand points to head_center space
                var transformed = PackedVector2Array()
                for p in pts:
                        transformed.append(head_center + p * s)
                # Fade color along strand
                for i in range(transformed.size() - 1):
                        var a = 1.0 - (float(i) / float(transformed.size()))
                        var c = Color(color.r, color.g, color.b, a)
                        draw_line(transformed[i], transformed[i+1], c, 1.5 * s)

        func _aura(center: Vector2, radius: float, color: Color) -> void:
                # Multi-layer radial aura
                var n = 16
                for layer in range(3):
                        var r = radius * (1.0 + layer * 0.3)
                        var a = color.a * (0.3 - layer * 0.08)
                        var c = Color(color.r, color.g, color.b, a)
                        var pts = PackedVector2Array()
                        var cols = PackedColorArray()
                        for i in range(n + 1):
                                var ang = (float(i) / n) * TAU
                                pts.append(center + Vector2(cos(ang) * r, sin(ang) * r))
                                cols.append(c)
                        draw_polygon(pts, cols)


# ════════════════════════════════════════════════════════════════════════════
#  INNER CLASS: AgentFaceDraw — draws procedural face
# ════════════════════════════════════════════════════════════════════════════

class _AgentFaceDraw extends Node2D:
        var state: Dictionary = {}
        var blink: float = 1.0
        var mood: String = "calm"
        var action: String = "IDLE"
        var facing: float = 1.0
        var hp: float = 100.0
        var colors: Dictionary = {}
        var time: float = 0.0
        var scale_factor: float = 1.0

        func _draw() -> void:
                if state.is_empty(): return
                var s = scale_factor
                var cx = facing * 1.5  # slight facing offset
                var head_y = -14 * s
                var head_center = Vector2(cx, head_y)

                # Sleeping = closed eyes (single line)
                if action == "SLEEP":
                        # Closed-eye arcs
                        _eye_closed(head_center + Vector2(-3 * s, -1 * s), s)
                        _eye_closed(head_center + Vector2(3 * s, -1 * s), s)
                        # Mouth: small 'o'
                        _mouth_arc(head_center + Vector2(0, 3 * s), 1 * s, 0.3, Color(0.2, 0.1, 0.1, 0.6))
                        # Sweat drop (sleeping = peaceful, no sweat)
                        return

                # Eyes (with blink)
                var eye_color = Color(0.1, 0.1, 0.15, 1.0)
                if hp < 30:
                        eye_color = Color(0.5, 0.2, 0.2, 1.0)  # pain-strained
                # Look direction (eyes track facing)
                var pupil_offset = Vector2(facing * 0.8, 0)
                if mood == "curious":
                        pupil_offset = Vector2(facing * 1.2, -0.5)
                elif mood == "afraid":
                        pupil_offset = Vector2(0, -1.0)
                _eye(head_center + Vector2(-3 * s, -1 * s), 1.5 * s, blink, eye_color, pupil_offset, s)
                _eye(head_center + Vector2(3 * s, -1 * s), 1.5 * s, blink, eye_color, pupil_offset, s)

                # Eyebrows (mood-driven)
                _eyebrow(head_center + Vector2(-3 * s, -3.5 * s), mood, facing, s, true)
                _eyebrow(head_center + Vector2(3 * s, -3.5 * s), mood, facing, s, false)

                # Mouth (action/mood-driven)
                var mouth_y = 3 * s
                match action:
                        "EAT":
                                _mouth_arc(head_center + Vector2(0, mouth_y), 2.5 * s, 0.5, Color(0.3, 0.1, 0.1, 1.0))
                        "DRINK":
                                _mouth_arc(head_center + Vector2(0, mouth_y), 2 * s, 0.7, Color(0.3, 0.1, 0.1, 1.0))
                        "FLEE":
                                # Open wide in fear
                                draw_circle(head_center + Vector2(0, mouth_y), 1.5 * s, Color(0.2, 0.05, 0.05, 1.0))
                        "HURT":
                                # Gritted teeth — zigzag
                                _teeth(head_center + Vector2(0, mouth_y), s)
                        "SLEEP":
                                _mouth_arc(head_center + Vector2(0, mouth_y), 1 * s, 0.4, Color(0.2, 0.1, 0.1, 0.5))
                        _:
                                # Mood-based mouth
                                match mood:
                                        "happy":
                                                _mouth_smile(head_center + Vector2(0, mouth_y), 3 * s, Color(0.3, 0.1, 0.1, 1.0))
                                        "afraid":
                                                _mouth_arc(head_center + Vector2(0, mouth_y), 1.5 * s, 0.6, Color(0.2, 0.05, 0.05, 1.0))
                                        "hungry":
                                                # Slight frown + drool
                                                _mouth_frown(head_center + Vector2(0, mouth_y), 2 * s, Color(0.3, 0.1, 0.1, 1.0))
                                                # Drool drop
                                                draw_circle(head_center + Vector2(2 * s, mouth_y + 2 * s), 0.5 * s, Color(0.5, 0.7, 1.0, 0.6))
                                        "curious":
                                                _mouth_arc(head_center + Vector2(0, mouth_y), 1.2 * s, 0.3, Color(0.3, 0.1, 0.1, 1.0))
                                        _:
                                                # Neutral straight line
                                                draw_line(head_center + Vector2(-1.5 * s, mouth_y),
                                                                  head_center + Vector2(1.5 * s, mouth_y),
                                                                  Color(0.3, 0.1, 0.1, 1.0), 1.0)

                # Sweat drop on hot/hurt
                if mood == "afraid" or mood == "hurt" or hp < 50:
                        var sweat_x = facing * 4 * s
                        var sweat_y = -3 * s + sin(time * 8.0) * 0.5
                        draw_circle(head_center + Vector2(sweat_x, sweat_y), 0.8 * s, Color(0.5, 0.7, 1.0, 0.7))

                # Tears when very hurt
                if hp < 30:
                        draw_circle(head_center + Vector2(-3 * s, 1 * s), 0.7 * s, Color(0.4, 0.6, 1.0, 0.8))
                        draw_circle(head_center + Vector2(3 * s, 1 * s), 0.7 * s, Color(0.4, 0.6, 1.0, 0.8))

        # ── Face primitives ──────────────────────────────────────────────────
        func _eye(pos: Vector2, r: float, blink_amt: float, color: Color,
                          pupil_offset: Vector2, s: float) -> void:
                # White
                var ry = r * blink_amt
                if ry < 0.1:
                        # Closed eye = line
                        draw_line(pos + Vector2(-r, 0), pos + Vector2(r, 0), Color(0.1, 0.1, 0.15, 1.0), 1.0)
                        return
                # Eye white (ellipse)
                var pts = PackedVector2Array()
                var cols = PackedColorArray()
                var n = 12
                for i in range(n + 1):
                        var a = (float(i) / n) * TAU
                        pts.append(pos + Vector2(cos(a) * r, sin(a) * ry))
                        cols.append(Color(1, 1, 1, 1))
                draw_polygon(pts, cols)
                # Pupil
                draw_circle(pos + pupil_offset, r * 0.5, color)
                # Highlight
                draw_circle(pos + pupil_offset + Vector2(-r * 0.2, -r * 0.2), r * 0.2, Color(1, 1, 1, 0.8))

        func _eye_closed(pos: Vector2, s: float) -> void:
                # Downward arc (closed, peaceful)
                var pts = PackedVector2Array()
                for i in range(7):
                        var t = float(i) / 6.0
                        var x = lerp(-2 * s, 2 * s, t)
                        var y = sin(t * PI) * 0.5
                        pts.append(pos + Vector2(x, y))
                draw_polyline(pts, Color(0.1, 0.1, 0.15, 1.0), 1.0, true)

        func _eyebrow(pos: Vector2, mood: String, facing: float, s: float, left: bool) -> void:
                var dir = 1 if left else -1
                var pts = PackedVector2Array()
                var col = Color(0.1, 0.1, 0.15, 1.0)
                match mood:
                        "afraid":
                                # Raised — high on outer side
                                pts.append(pos + Vector2(-2 * s, 1 * s * dir))
                                pts.append(pos + Vector2(2 * s, -1 * s * dir))
                        "hurt":
                                # Inner-down (anguish)
                                pts.append(pos + Vector2(-2 * s, -1 * s * dir))
                                pts.append(pos + Vector2(2 * s, 1 * s * dir))
                        "happy":
                                # Slight raise, relaxed
                                pts.append(pos + Vector2(-2 * s, 0))
                                pts.append(pos + Vector2(2 * s, -0.5))
                        "curious":
                                # One raised (outer)
                                pts.append(pos + Vector2(-2 * s, 0.5))
                                pts.append(pos + Vector2(2 * s, -1 * s))
                        "hungry":
                                # Flat, focused
                                pts.append(pos + Vector2(-2 * s, 0))
                                pts.append(pos + Vector2(2 * s, 0))
                        _:
                                pts.append(pos + Vector2(-2 * s, 0))
                                pts.append(pos + Vector2(2 * s, 0))
                draw_polyline(pts, col, 1.2 * s, true)

        func _mouth_smile(pos: Vector2, width: float, color: Color) -> void:
                var pts = PackedVector2Array()
                for i in range(7):
                        var t = float(i) / 6.0
                        var x = lerp(-width, width, t)
                        var y = sin(t * PI) * width * 0.4
                        pts.append(pos + Vector2(x, y))
                draw_polyline(pts, color, 1.5, true)

        func _mouth_frown(pos: Vector2, width: float, color: Color) -> void:
                var pts = PackedVector2Array()
                for i in range(7):
                        var t = float(i) / 6.0
                        var x = lerp(-width, width, t)
                        var y = -sin(t * PI) * width * 0.4
                        pts.append(pos + Vector2(x, y))
                draw_polyline(pts, color, 1.5, true)

        func _mouth_arc(pos: Vector2, r: float, open_amt: float, color: Color) -> void:
                # Open mouth = filled circle scaled by open_amt
                draw_circle(pos, r * open_amt, color)

        func _teeth(pos: Vector2, s: float) -> void:
                # Zigzag for gritted teeth
                var pts = PackedVector2Array()
                for i in range(5):
                        var x = (i - 2) * s
                        var y = sin(i * 2.0) * 0.5 * s
                        pts.append(pos + Vector2(x, y))
                draw_polyline(pts, Color(1, 1, 1, 1), 1.0, true)
                draw_line(pos + Vector2(-2 * s, -0.5 * s), pos + Vector2(2 * s, -0.5 * s), Color(0.2, 0.05, 0.05, 1.0), 1.0)
