extends CanvasLayer

## ObserverControls (Phase 11.3)
## Provides UI controls for the observer:
## - Click agent → sidebar shows full stats
## - Time controls: pause, 1x, 5x, 20x speed
## - Follow mode: camera tracks selected agent
## - History mode: scrub back through last 1000 ticks
## - Vocabulary panel: toggle to see any agent's word list
## - Family tree view: click to open full lineage graph

signal speed_changed(speed: float)
signal follow_toggled(enabled: bool)
agent_selected(agent_id: String)

var sidebar: Panel = null
var stats_label: Label = null
var vocab_panel: Panel = null
var vocab_label: Label = null
var family_tree_panel: Panel = null
var family_tree_label: Label = null

var selected_agent_id: String = ""
var follow_mode: bool = false
var simulation_speed: float = 1.0
var history_ticks: Array = []  # last 1000 tick states
var current_history_index: int = -1
var camera: Camera2D = null

func _ready() -> void:
	_create_sidebar()
	_create_vocab_panel()
	_create_family_tree_panel()
	_create_time_controls()

	await get_tree().process_frame
	camera = get_viewport().get_camera_2d()

	# Connect to NetworkController
	var nc = get_node_or_null("/root/NetworkController")
	if nc:
		nc.connect("tick_processed", _on_tick_processed)

# ── Sidebar (agent stats) ─────────────────────────────────────────────────────

func _create_sidebar() -> void:
	sidebar = Panel.new()
	sidebar.size = Vector2(280, 400)
	sidebar.position = Vector2(10, 10)
	sidebar.visible = false
	add_child(sidebar)

	stats_label = Label.new()
	stats_label.text = ""
	stats_label.position = Vector2(10, 10)
	stats_label.custom_minimum_size = Vector2(260, 380)
	stats_label.add_theme_font_size_override("font_size", 11)
	stats_label.vertical_alignment = Label.VERTICAL_ALIGNMENT_TOP
	sidebar.add_child(stats_label)

func _on_tick_processed(tick_data: Dictionary) -> void:
	# Store in history (max 1000)
	history_ticks.append(tick_data)
	if len(history_ticks) > 1000:
		history_ticks.pop_front()

	# Update sidebar if an agent is selected
	if selected_agent_id != "" and tick_data.has("agents"):
		for agent in tick_data.agents:
			if agent.get("id") == selected_agent_id:
				_update_sidebar(agent)
				break

func _update_sidebar(agent_data: Dictionary) -> void:
	var text = "Agent: " + agent_data.get("id", "") + "\n"
	text += "─────────────────\n"
	text += "Life Stage: " + agent_data.get("life_stage", "") + "\n"
	text += "Age: " + str(agent_data.get("age_ticks", 0)) + " ticks\n"
	text += "─────────────────\n"
	text += "Health: " + str(agent_data.get("health", 0)) + "\n"
	text += "Glucose: " + str(agent_data.get("glucose", 0)) + "\n"
	text += "Hydration: " + str(agent_data.get("hydration", 0)) + "\n"
	text += "Body Temp: " + str(agent_data.get("body_temp", 0)) + "°C\n"
	text += "─────────────────\n"
	text += "Vocabulary: " + str(agent_data.get("vocabulary_size", 0)) + " words\n"
	text += "Generation: " + str(agent_data.get("generation", 1)) + "\n"
	text += "─────────────────\n"
	text += "Parents: " + str(agent_data.get("parents", [])) + "\n"
	text += "─────────────────\n"
	text += "Thought: " + agent_data.get("thought", "") + "\n"
	text += "Dialogue: " + agent_data.get("dialogue", "") + "\n"
	text += "─────────────────\n"
	text += "Action: " + agent_data.get("action", "IDLE") + "\n"
	text += "Injury: " + agent_data.get("injury_name", "NONE") + "\n"
	text += "Sleep Debt: " + str(agent_data.get("sleep_debt", 0)) + "\n"

	stats_label.text = text
	sidebar.visible = true

# ── Vocabulary panel ──────────────────────────────────────────────────────────

func _create_vocab_panel() -> void:
	vocab_panel = Panel.new()
	vocab_panel.size = Vector2(300, 400)
	vocab_panel.position = Vector2(310, 10)
	vocab_panel.visible = false
	add_child(vocab_panel)

	var title = Label.new()
	title.text = "Vocabulary"
	title.add_theme_font_size_override("font_size", 16)
	title.position = Vector2(10, 10)
	vocab_panel.add_child(title)

	vocab_label = Label.new()
	vocab_label.text = ""
	vocab_label.position = Vector2(10, 40)
	vocab_label.custom_minimum_size = Vector2(280, 350)
	vocab_label.add_theme_font_size_override("font_size", 10)
	vocab_label.vertical_alignment = Label.VERTICAL_ALIGNMENT_TOP
	vocab_panel.add_child(vocab_label)

func show_vocabulary(vocab_list: Array) -> void:
	vocab_label.text = ""
	for word in vocab_list:
		vocab_label.text += word + "\n"
	vocab_panel.visible = true

func hide_vocabulary() -> void:
	vocab_panel.visible = false

# ── Family tree panel ─────────────────────────────────────────────────────────

func _create_family_tree_panel() -> void:
	family_tree_panel = Panel.new()
	family_tree_panel.size = Vector2(500, 400)
	family_tree_panel.position = Vector2(620, 10)
	family_tree_panel.visible = false
	add_child(family_tree_panel)

	var title = Label.new()
	title.text = "Family Tree"
	title.add_theme_font_size_override("font_size", 16)
	title.position = Vector2(10, 10)
	family_tree_panel.add_child(title)

	family_tree_label = Label.new()
	family_tree_label.text = ""
	family_tree_label.position = Vector2(10, 40)
	family_tree_label.custom_minimum_size = Vector2(480, 350)
	family_tree_label.add_theme_font_size_override("font_size", 10)
	family_tree_label.vertical_alignment = Label.VERTICAL_ALIGNMENT_TOP
	family_tree_panel.add_child(family_tree_label)

func show_family_tree(family_data: Dictionary) -> void:
	family_tree_label.text = _format_family_tree(family_data, 0)
	family_tree_panel.visible = true

func _format_family_tree(node: Dictionary, depth: int) -> String:
	var indent = "  ".repeat(depth)
	var text = indent + "- " + node.get("agent_id", "?")
	text += " (Gen " + str(node.get("generation", 1)) + ")"
	if node.get("death_tick") != null:
		text += " [DEAD]"
	text += "\n"
	for child in node.get("children", []):
		text += _format_family_tree(child, depth + 1)
	return text

func hide_family_tree() -> void:
	family_tree_panel.visible = false

# ── Time controls ─────────────────────────────────────────────────────────────

func _create_time_controls() -> void:
	var container = HBoxContainer.new()
	container.position = Vector2(10, get_viewport().get_visible_rect().size.y - 60)
	container.size = Vector2(400, 50)
	add_child(container)

	var pause_btn = Button.new()
	pause_btn.text = "⏸ Pause"
	pause_btn.pressed.connect(func(): _set_speed(0.0))
	container.add_child(pause_btn)

	var speed_1x = Button.new()
	speed_1x.text = "▶ 1x"
	speed_1x.pressed.connect(func(): _set_speed(1.0))
	container.add_child(speed_1x)

	var speed_5x = Button.new()
	speed_5x.text = "▶▶ 5x"
	speed_5x.pressed.connect(func(): _set_speed(5.0))
	container.add_child(speed_5x)

	var speed_20x = Button.new()
	speed_20x.text = "⏩ 20x"
	speed_20x.pressed.connect(func(): _set_speed(20.0))
	container.add_child(speed_20x)

	var follow_btn = Button.new()
	follow_btn.text = "Follow"
	follow_btn.toggle_mode = true
	follow_btn.toggled.connect(func(pressed): _toggle_follow(pressed))
	container.add_child(follow_btn)

	var vocab_btn = Button.new()
	vocab_btn.text = "Vocab"
	vocab_btn.toggle_mode = true
	vocab_btn.toggled.connect(func(pressed):
		if pressed: vocab_panel.visible = true
		else: vocab_panel.visible = false
	)
	container.add_child(vocab_btn)

	var tree_btn = Button.new()
	tree_btn.text = "Family"
	tree_btn.toggle_mode = true
	tree_btn.toggled.connect(func(pressed):
		if pressed: family_tree_panel.visible = true
		else: family_tree_panel.visible = false
	)
	container.add_child(tree_btn)

func _set_speed(speed: float) -> void:
	simulation_speed = speed
	speed_changed.emit(speed)

func _toggle_follow(enabled: bool) -> void:
	follow_mode = enabled
	follow_toggled.emit(enabled)

# ── Agent selection ───────────────────────────────────────────────────────────

func _input(event: InputEvent) -> void:
	if event is InputEventMouseButton and event.pressed and event.button_index == MOUSE_BUTTON_LEFT:
		_select_agent_at(event.position)

func _select_agent_at(pos: Vector2) -> void:
	# Find agent at this position
	var agent_renderer = get_node_or_null("../AgentRenderer")
	if not agent_renderer:
		return

	for agent_id in agent_renderer.agent_sprites.keys():
		var sprite = agent_renderer.agent_sprites[agent_id].sprite as CharacterBody2D
		var dist = sprite.global_position.distance_to(pos)
		if dist < 20:
			selected_agent_id = agent_id
			agent_selected.emit(agent_id)
			return

	# Clicked empty space — deselect
	selected_agent_id = ""
	sidebar.visible = false

# ── Follow mode ───────────────────────────────────────────────────────────────

func _process(delta: float) -> void:
	if follow_mode and selected_agent_id != "" and camera:
		var agent_renderer = get_node_or_null("../AgentRenderer")
		if agent_renderer and agent_renderer.agent_sprites.has(selected_agent_id):
			var sprite = agent_renderer.agent_sprites[selected_agent_id].sprite as CharacterBody2D
			camera.global_position = camera.global_position.lerp(sprite.global_position, 0.1)

# ── History scrubbing ─────────────────────────────────────────────────────────

func seek_to_history(index: int) -> void:
	if index < 0 or index >= len(history_ticks):
		return
	current_history_index = index
	# Emit signal so other components can render the historical state
	var state = history_ticks[index]
	var nc = get_node_or_null("/root/NetworkController")
	if nc and nc.has_method("render_historical_state"):
		nc.render_historical_state(state)
