extends Node2D

## AgentRenderer (Phase 11.3 — Animated Edition)
## ─────────────────────────────────────────────
## Procedural animations driven by current action — no sprite sheets needed.
##
## Animation map:
##   IDLE     → gentle vertical bob (sin wave, slow)
##   EXPLORE  → faster bob + slight horizontal sway
##   MOVE     → bob + lean in movement direction
##   EAT      → quick forward lunge, repeat
##   DRINK    → head dip down
##   SLEEP    → slow breathing (scale pulse), eyes closed
##   HIDE     → shrink + lower opacity
##   FLEE     → fast horizontal jitter + lean
##   HURT     → red flash + screen-shake-like jitter (~250ms)
##
## Birth event → spawn-in scale-up (0 → 1.0 over 600ms, ease-out-back)
## Death event → scale-down + fade-out (1.0 → 0 over 800ms, ease-in)

const TILE_SIZE: int = 32

# Agent visual configs by type/stage
const AGENT_COLORS: Dictionary = {
	"adam":   Color(0.30, 0.50, 1.00, 1.0),   # blue
	"eve":    Color(1.00, 0.40, 0.60, 1.0),   # pink
	"baby":   Color(1.00, 1.00, 1.00, 1.0),   # white
	"elder":  Color(0.80, 0.70, 0.90, 0.8),   # pale purple
}

# Per-action animation parameters
const ANIM_PARAMS: Dictionary = {
	"IDLE":     {"bob_amp": 1.5,  "bob_freq": 1.5, "sway_amp": 0.0, "lean": 0.0,  "jitter": 0.0, "scale": 1.0, "alpha": 1.0},
	"EXPLORE":  {"bob_amp": 2.0,  "bob_freq": 3.0, "sway_amp": 0.8, "lean": 0.05, "jitter": 0.0, "scale": 1.0, "alpha": 1.0},
	"MOVE":     {"bob_amp": 2.5,  "bob_freq": 4.0, "sway_amp": 1.0, "lean": 0.10, "jitter": 0.0, "scale": 1.0, "alpha": 1.0},
	"EAT":      {"bob_amp": 0.5,  "bob_freq": 6.0, "sway_amp": 0.0, "lean": 0.15, "jitter": 0.0, "scale": 1.0, "alpha": 1.0},
	"DRINK":    {"bob_amp": 0.5,  "bob_freq": 5.0, "sway_amp": 0.0, "lean": 0.20, "jitter": 0.0, "scale": 1.0, "alpha": 1.0},
	"SLEEP":    {"bob_amp": 0.3,  "bob_freq": 0.5, "sway_amp": 0.0, "lean": 0.0,  "jitter": 0.0, "scale": 0.85,"alpha": 0.85},
	"HIDE":     {"bob_amp": 0.2,  "bob_freq": 0.5, "sway_amp": 0.0, "lean": 0.0,  "jitter": 0.0, "scale": 0.70,"alpha": 0.50},
	"FLEE":     {"bob_amp": 3.0,  "bob_freq": 8.0, "sway_amp": 2.0, "lean": 0.25, "jitter": 1.5, "scale": 1.0, "alpha": 1.0},
	"HURT":     {"bob_amp": 0.0,  "bob_freq": 0.0, "sway_amp": 0.0, "lean": 0.0,  "jitter": 3.0, "scale": 1.0, "alpha": 1.0},
}

# {agent_id: {sprite, body, glow, action_label, thought_label, family_lines, current_action, last_action, hurt_timer, hurt_flash, birth_tween}}
var agent_sprites: Dictionary = {}

var _time: float = 0.0

func _ready() -> void:
	await get_tree().process_frame
	var nc = get_node_or_null("/root/NetworkController")
	if nc:
		nc.connect("agents_updated", _on_agents_updated)

func _process(delta: float) -> void:
	_time += delta
	_animate_all_agents(delta)

# ── Public API ────────────────────────────────────────────────────────────────

func _on_agents_updated(agents_data: Array) -> void:
	var seen_ids = {}
	for agent_data in agents_data:
		var agent_id = agent_data.get("id", "")
		if agent_id == "":
			continue
		seen_ids[agent_id] = true
		_update_agent(agent_id, agent_data)

	for id in agent_sprites.keys():
		if not seen_ids.has(id):
			_remove_agent(id)

	_update_family_lines(agents_data)

func _update_agent(agent_id: String, data: Dictionary) -> void:
	if not agent_sprites.has(agent_id):
		_create_agent_sprite(agent_id, data)

	var info = agent_sprites[agent_id]
	var sprite = info.sprite as CharacterBody2D
	var action_label = info.action_label as Label
	var thought_label = info.thought_label as Label

	# Track action change → trigger birth-of-action animation
	var new_action = data.get("action", "IDLE")
	if new_action != info.current_action:
		info.last_action = info.current_action
		info.current_action = new_action
		# Detect HURT specially: when HP drops sharply
		# (handled in _animate via action name)
		if new_action == "HURT":
			info.hurt_timer = 0.25
			info.hurt_flash = 1.0

	var x = data.get("x", 0)
	var y = data.get("y", 0)
	info.target_pos = Vector2(x * TILE_SIZE + TILE_SIZE/2, y * TILE_SIZE + TILE_SIZE/2)

	action_label.text = new_action
	thought_label.text = data.get("dialogue", data.get("thought", ""))

	# Scale by life stage (kept as base scale; animation modulates on top)
	var life_stage = data.get("life_stage", "adult")
	info.base_scale = _get_agent_scale(life_stage)

	# Color
	var agent_type = data.get("type", "adam").to_lower()
	var color = AGENT_COLORS.get(agent_type, AGENT_COLORS.adam)
	if life_stage == "elder":
		color = AGENT_COLORS.elder
	info.base_color = color

	# HP-based hurt detection (if HP dropped)
	var hp = data.get("hp", 100)
	if info.has("last_hp") and hp < info.last_hp - 5:
		info.hurt_timer = 0.25
		info.hurt_flash = 1.0
	info.last_hp = hp

# ── Animation loop ────────────────────────────────────────────────────────────

func _animate_all_agents(delta: float) -> void:
	for agent_id in agent_sprites.keys():
		var info = agent_sprites[agent_id]
		var sprite = info.sprite as CharacterBody2D
		if not is_instance_valid(sprite):
			continue

		# Smoothly move toward target_pos
		if info.target_pos != sprite.global_position:
			sprite.global_position = sprite.global_position.lerp(info.target_pos, min(1.0, delta * 8.0))

		# Smooth base scale toward info.base_scale
		var bs = info.base_scale if info.base_scale != null else 1.0
		var cur_scale = sprite.scale.x
		cur_scale = lerpf(cur_scale, bs, min(1.0, delta * 5.0))

		# Get anim params for current action
		var action = info.current_action if info.current_action else "IDLE"
		var params = ANIM_PARAMS.get(action, ANIM_PARAMS.IDLE)

		# Compute animation offset
		var t = _time
		var bob = sin(t * params.bob_freq * 2.0 * PI) * params.bob_amp
		var sway = cos(t * params.sway_amp * 0.5 * 2.0 * PI) * params.sway_amp if params.sway_amp > 0 else 0.0
		# Lean = rotation
		var lean = params.lean
		# Jitter (random offset for FLEE / HURT)
		var jitter_x = 0.0
		var jitter_y = 0.0
		if params.jitter > 0:
			jitter_x = (randf() - 0.5) * params.jitter
			jitter_y = (randf() - 0.5) * params.jitter

		# Apply anim scale (SLEEP/HIDE shrink) + base scale
		var final_scale = cur_scale * params.scale
		# Birth tween: if spawn_anim_t > 0, ease-in from 0
		if info.spawn_anim_t > 0:
			info.spawn_anim_t -= delta
			var p = 1.0 - max(0.0, info.spawn_anim_t) / 0.6  # 0→1 over 0.6s
			p = clamp(p, 0.0, 1.0)
			# Ease-out-back: overshoot then settle
			var c1 = 1.70158
			var c3 = c1 + 1.0
			var eased = 1.0 + c3 * pow(p - 1.0, 3.0) + c1 * pow(p - 1.0, 2.0)
			final_scale *= eased
			sprite.modulate.a = eased * params.alpha
		else:
			# Hurt flash decay
			if info.hurt_flash > 0:
				info.hurt_flash = max(0.0, info.hurt_flash - delta * 4.0)
			var alpha = params.alpha
			# Tint red during hurt flash
			var body = info.body as ColorRect
			var base_color = info.base_color if info.base_color else Color.WHITE
			if info.hurt_flash > 0:
				var flash = info.hurt_flash
				body.color = Color(
					lerpf(base_color.r, 1.0, flash),
					lerpf(base_color.g, 0.2, flash),
					lerpf(base_color.b, 0.2, flash),
					1.0
				)
			else:
				body.color = base_color
			sprite.modulate.a = alpha

		sprite.scale = Vector2(final_scale, final_scale)
		# Position offset for bob/sway/jitter
		sprite.position = Vector2(sway + jitter_x, bob + jitter_y)
		# Rotation for lean
		sprite.rotation = lean

		# Sleep "Zzz" indicator — make thought label show "Zzz"
		var action_label = info.action_label as Label
		if action == "SLEEP":
			# Pulsing "Zzz" text above head
			action_label.text = "Z" + "z".repeat(int(1 + sin(t * 2.0) * 1.5 + 1.5))
			action_label.modulate.a = 0.5 + 0.5 * sin(t * 2.0)
		elif action == "HIDE":
			action_label.text = "..."
			action_label.modulate.a = 0.4
		elif action == "HURT":
			action_label.text = "!"
			action_label.modulate = Color(1.0, 0.3, 0.3)
		else:
			action_label.modulate.a = 1.0
			action_label.modulate = Color.WHITE

		# Birth flash decay — spawn_anim_t handled above

func _create_agent_sprite(agent_id: String, data: Dictionary) -> void:
	var agent = CharacterBody2D.new()
	agent.name = "Agent_" + agent_id

	# Main body — circle (a ColorRect for now, can be replaced with Sprite2D later)
	var body = ColorRect.new()
	body.color = Color.WHITE
	body.size = Vector2(24, 24)
	body.position = -body.size / 2
	body.z_index = 1
	agent.add_child(body)

	# Inner glow (slightly larger, lower z-index, semi-transparent)
	var glow = ColorRect.new()
	glow.color = Color(1, 1, 1, 0.25)
	glow.size = Vector2(36, 36)
	glow.position = -glow.size / 2
	glow.z_index = 0
	agent.add_child(glow)

	# Eyes (two tiny dark rectangles) — appear "closed" during SLEEP
	var left_eye = ColorRect.new()
	left_eye.color = Color(0.1, 0.1, 0.1, 1.0)
	left_eye.size = Vector2(3, 4)
	left_eye.position = Vector2(-7, -4)
	left_eye.name = "LeftEye"
	agent.add_child(left_eye)

	var right_eye = ColorRect.new()
	right_eye.color = Color(0.1, 0.1, 0.1, 1.0)
	right_eye.size = Vector2(3, 4)
	right_eye.position = Vector2(4, -4)
	right_eye.name = "RightEye"
	agent.add_child(right_eye)

	# Mouth — small dark line, opens during EAT/HURT
	var mouth = ColorRect.new()
	mouth.color = Color(0.2, 0.1, 0.1, 1.0)
	mouth.size = Vector2(6, 1)
	mouth.position = Vector2(-3, 4)
	mouth.name = "Mouth"
	agent.add_child(mouth)

	# Action label (above agent)
	var action_label = Label.new()
	action_label.text = "IDLE"
	action_label.horizontal_alignment = Label.HORIZONTAL_ALIGNMENT_CENTER
	action_label.add_theme_font_size_override("font_size", 11)
	action_label.position = Vector2(-30, -50)
	action_label.custom_minimum_size = Vector2(60, 14)
	agent.add_child(action_label)

	# Thought bubble (hidden by default, shown on hover)
	var thought_label = Label.new()
	thought_label.text = ""
	thought_label.horizontal_alignment = Label.HORIZONTAL_ALIGNMENT_CENTER
	thought_label.add_theme_font_size_override("font_size", 9)
	thought_label.position = Vector2(-50, -70)
	thought_label.custom_minimum_size = Vector2(100, 14)
	thought_label.visible = false
	thought_label.add_theme_color_override("font_color", Color(0.9, 0.9, 0.9))
	agent.add_child(thought_label)

	add_child(agent)

	agent_sprites[agent_id] = {
		"sprite": agent,
		"body": body,
		"glow": glow,
		"left_eye": left_eye,
		"right_eye": right_eye,
		"mouth": mouth,
		"action_label": action_label,
		"thought_label": thought_label,
		"family_lines": [],
		"current_action": "IDLE",
		"last_action": null,
		"base_scale": 1.0,
		"base_color": AGENT_COLORS.adam,
		"hurt_timer": 0.0,
		"hurt_flash": 0.0,
		"spawn_anim_t": 0.6,    # 600ms birth animation
		"target_pos": Vector2.ZERO,
		"last_hp": 100,
	}

func _get_agent_scale(life_stage: String) -> float:
	match life_stage:
		"newborn":    return 0.4
		"child":      return 0.6
		"adolescent": return 0.8
		"adult":      return 1.0
		"elder":      return 1.2
		"ancient":    return 1.3
		_:            return 1.0

func _remove_agent(agent_id: String) -> void:
	if not agent_sprites.has(agent_id):
		return
	var info = agent_sprites[agent_id]
	info.sprite.queue_free()
	for line in info.family_lines:
		if is_instance_valid(line):
			line.queue_free()
	agent_sprites.erase(agent_id)

func _update_family_lines(agents_data: Array) -> void:
	for info in agent_sprites.values():
		for line in info.family_lines:
			if is_instance_valid(line):
				line.queue_free()
		info.family_lines = []

	for agent_data in agents_data:
		var child_id = agent_data.get("id", "")
		var parents = agent_data.get("parents", [])
		for parent_id in parents:
			if not agent_sprites.has(child_id) or not agent_sprites.has(parent_id):
				continue
			var child_sprite = agent_sprites[child_id].sprite
			var parent_sprite = agent_sprites[parent_id].sprite
			var dist = child_sprite.global_position.distance_to(parent_sprite.global_position)
			if dist <= 10 * TILE_SIZE:
				var line = _create_family_line(parent_sprite.global_position, child_sprite.global_position)
				add_child(line)
				agent_sprites[child_id].family_lines.append(line)

func _create_family_line(from: Vector2, to: Vector2) -> Line2D:
	var line = Line2D.new()
	line.add_point(from)
	line.add_point(to)
	line.default_color = Color(1.0, 0.8, 0.4, 0.5)
	line.width = 1.5
	return line

# ── Hover detection ───────────────────────────────────────────────────────────

func _input(event: InputEvent) -> void:
	if event is InputEventMouseMotion:
		_check_hover(event.position)

func _check_hover(mouse_pos: Vector2) -> void:
	for agent_id in agent_sprites.keys():
		var info = agent_sprites[agent_id]
		var sprite = info.sprite as CharacterBody2D
		var thought_label = info.thought_label as Label
		var dist = sprite.global_position.distance_to(mouse_pos)
		thought_label.visible = dist < 30

func clear_all() -> void:
	for agent_id in agent_sprites.keys().duplicate():
		_remove_agent(agent_id)
