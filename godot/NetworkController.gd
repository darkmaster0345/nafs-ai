extends Node

## NetworkController (The Bridge) — Phase 11 enhanced
## Handles HTTP polling loop and command synchronization.
##
## Connects to a Python backend (server.py) that exposes the simulation
## state via Socket.IO / HTTP.
##
## Phase 11 additions:
##   - Receives full world state (biome map, agents, fires, water)
##   - Emits signals for milestone events
##   - Provides history for scrubbing

signal tick_processed(tick_data: Dictionary)
signal agents_updated(agents_data: Array)
signal milestone_reached(type: String, details: Dictionary)

@export var server_url: String = "http://localhost:5000/state"
@export var poll_interval: float = 0.5

@onready var http_request: HTTPRequest = HTTPRequest.new()

var _is_requesting: bool = false
var last_tick_data: Dictionary = {}

# Cached world state
var biome_map: Dictionary = {}
var agent_states: Array = []
var fire_tiles: Array = []
var water_tiles: Array = []
var current_weather: String = "clear"
var current_time_of_day: int = 12

func _ready() -> void:
	add_child(http_request)
	http_request.request_completed.connect(_on_request_completed)
	_start_poll_timer()

func _start_poll_timer() -> void:
	var timer = get_tree().create_timer(poll_interval)
	timer.timeout.connect(_on_poll_timer_timeout)

func _on_poll_timer_timeout() -> void:
	_fetch_state()
	_start_poll_timer()

func _fetch_state() -> void:
	if _is_requesting:
		return
	_is_requesting = true

	var error = http_request.request(server_url)
	if error != OK:
		_is_requesting = false
		push_error("HTTP request failed: %d" % error)

func _on_request_completed(_result: int, response_code: int, _headers: PackedStringArray, body: PackedByteArray) -> void:
	_is_requesting = false

	if response_code != 200:
		push_warning("Server returned non-200 code: %d" % response_code)
		return

	var json = JSON.new()
	var error = json.parse(body.get_string_from_utf8())
	if error != OK:
		push_error("JSON parse error: %s at line %d" % [json.get_error_message(), json.get_error_line()])
		return

	var response = json.get_data()
	if not response is Dictionary:
		return

	last_tick_data = response
	_process_state(response)

func _process_state(state: Dictionary) -> void:
	# Update biome map
	if state.has("biome_map"):
		biome_map = state.biome_map
		var world_renderer = get_node_or_null("../WorldRenderer")
		if world_renderer:
			world_renderer.call("update_biome_map", biome_map)

	# Update agents
	if state.has("agents"):
		agent_states = state.agents
		var agent_renderer = get_node_or_null("../AgentRenderer")
		if agent_renderer:
			agent_renderer.call("_on_agents_updated", agent_states)
		agents_updated.emit(agent_states)

	# Update fire tiles
	if state.has("fire_tiles"):
		fire_tiles = state.fire_tiles
		var world_renderer = get_node_or_null("../WorldRenderer")
		if world_renderer:
			world_renderer.call("update_fire_tiles", fire_tiles)

	# Update water tiles
	if state.has("water_tiles"):
		water_tiles = state.water_tiles
		var world_renderer = get_node_or_null("../WorldRenderer")
		if world_renderer:
			world_renderer.call("update_water_tiles", water_tiles)

	# Update time of day
	if state.has("time_of_day"):
		current_time_of_day = state.time_of_day
		var world_renderer = get_node_or_null("../WorldRenderer")
		if world_renderer:
			world_renderer.call("update_day_night", current_time_of_day)

	# Update weather
	if state.has("weather"):
		current_weather = state.weather
		var world_renderer = get_node_or_null("../WorldRenderer")
		if world_renderer:
			world_renderer.call("update_weather", current_weather)

	# Check for milestones
	if state.has("milestones"):
		for milestone in state.milestones:
			_handle_milestone(milestone)

	# Emit tick processed
	tick_processed.emit(state)

func _handle_milestone(milestone: Dictionary) -> void:
	var type = milestone.get("type", "")
	var details = milestone.get("details", {})
	milestone_reached.emit(type, details)

	# Also trigger death tombstone if applicable
	if type == "FIRST_DEATH" and details.has("position"):
		var milestone_banner = get_node_or_null("../MilestoneBanner")
		if milestone_banner:
			milestone_banner.call("show_tombstone",
				int(details.position[0]), int(details.position[1]),
				details.get("agent_id", ""))

## Public method for Adam.gd to report events (legacy)
func report_event(event: Dictionary) -> void:
	# Forward to server if needed
	pass

## Render a historical state (called by ObserverControls for scrubbing)
func render_historical_state(state: Dictionary) -> void:
	_process_state(state)
