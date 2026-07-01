extends Node2D

## AgentRenderer (Phase 11.2)
## Renders Adam, Eve, babies, and elders with distinct visual styles.
## Includes action indicators, thought bubbles, and family lines.

const TILE_SIZE: int = 32

# Agent visual configs by type/stage
const AGENT_COLORS: Dictionary = {
	"adam":   Color(0.30, 0.50, 1.00, 1.0),   # blue
	"eve":    Color(1.00, 0.40, 0.60, 1.0),   # pink
	"baby":   Color(1.00, 1.00, 1.00, 1.0),   # white
	"elder":  Color(0.80, 0.70, 0.90, 0.8),   # pale purple
}

# All active agent sprites
# {agent_id: {sprite, action_label, thought_label, family_lines: []}}
var agent_sprites: Dictionary = {}

func _ready() -> void:
	# Connect to NetworkController for updates
	await get_tree().process_frame
	var nc = get_node_or_null("/root/NetworkController")
	if nc:
		nc.connect("agents_updated", _on_agents_updated)

# ── Public API ────────────────────────────────────────────────────────────────

func _on_agents_updated(agents_data: Array) -> void:
	# agents_data: [{id, x, y, type, action, thought, dialogue, life_stage, parents}, ...]
	var seen_ids = {}
	for agent_data in agents_data:
		var agent_id = agent_data.get("id", "")
		if agent_id == "":
			continue
		seen_ids[agent_id] = true
		_update_agent(agent_id, agent_data)

	# Remove sprites for agents no longer present
	for id in agent_sprites.keys():
		if not seen_ids.has(id):
			_remove_agent(id)

	# Update family lines after all agents are positioned
	_update_family_lines(agents_data)

func _update_agent(agent_id: String, data: Dictionary) -> void:
	if not agent_sprites.has(agent_id):
		_create_agent_sprite(agent_id, data)

	var agent_info = agent_sprites[agent_id]
	var sprite = agent_info.sprite as CharacterBody2D
	var action_label = agent_info.action_label as Label
	var thought_label = agent_info.thought_label as Label

	# Update position
	var x = data.get("x", 0)
	var y = data.get("y", 0)
	sprite.global_position = Vector2(x * TILE_SIZE + TILE_SIZE/2, y * TILE_SIZE + TILE_SIZE/2)

	# Update action label
	action_label.text = data.get("action", "IDLE")

	# Update thought bubble (visible on hover only)
	thought_label.text = data.get("dialogue", data.get("thought", ""))

	# Update size based on life stage
	var life_stage = data.get("life_stage", "adult")
	var scale = _get_agent_scale(life_stage)
	sprite.scale = Vector2(scale, scale)

	# Update color based on type and life stage
	var agent_type = data.get("type", "adam").to_lower()
	var color = AGENT_COLORS.get(agent_type, AGENT_COLORS.adam)
	if life_stage == "elder":
		color = AGENT_COLORS.elder
	sprite.modulate = color

func _create_agent_sprite(agent_id: String, data: Dictionary) -> void:
	var agent = CharacterBody2D.new()
	agent.name = "Agent_" + agent_id

	# Main body — circle
	var body = _create_circle(Color.WHITE, 12.0)
	agent.add_child(body)

	# Inner glow
	var glow = _create_circle(Color(1, 1, 1, 0.3), 18.0)
	glow.z_index = -1
	agent.add_child(glow)

	# Action label (above agent)
	var action_label = Label.new()
	action_label.text = "IDLE"
	action_label.horizontal_alignment = Label.HORIZONTAL_ALIGNMENT_CENTER
	action_label.add_theme_font_size_override("font_size", 10)
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
		"action_label": action_label,
		"thought_label": thought_label,
		"body": body,
		"glow": glow,
		"family_lines": [],
	}

func _create_circle(color: Color, radius: float) -> Node2D:
	var circle = Node2D.new()
	var rect = ColorRect.new()
	rect.color = color
	rect.size = Vector2(radius * 2, radius * 2)
	rect.position = -rect.size / 2
	# Make it round by using a circle mask (simplified: just a ColorRect)
	circle.add_child(rect)
	return circle

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
	# Remove family lines
	for line in info.family_lines:
		if is_instance_valid(line):
			line.queue_free()
	agent_sprites.erase(agent_id)

func _update_family_lines(agents_data: Array) -> void:
	# Clear all existing family lines
	for info in agent_sprites.values():
		for line in info.family_lines:
			if is_instance_valid(line):
				line.queue_free()
		info.family_lines = []

	# Draw lines between parents and children within 10 tiles
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
		thought_label.visible = dist < 30  # show thought when hovering within 30px

func clear_all() -> void:
	for agent_id in agent_sprites.keys().duplicate():
		_remove_agent(agent_id)
