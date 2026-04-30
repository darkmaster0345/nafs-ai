extends Node

## NetworkController (The Bridge)
## Handles HTTP polling loop and command synchronization.

@export var server_url: String = "http://localhost:5000/update"
@export var poll_interval: float = 0.5

@onready var http_request: HTTPRequest = HTTPRequest.new()

var _is_requesting: bool = false
var event_queue: Array = []
var adam_node: CharacterBody2D = null
var hud_node: CanvasLayer = null

func _ready() -> void:
    add_child(http_request)
    http_request.request_completed.connect(_on_request_completed)

    # Wait for scene to be ready before finding nodes
    await get_tree().process_frame
    adam_node = get_tree().get_first_node_in_group("adam") as CharacterBody2D
    hud_node = get_tree().get_first_node_in_group("hud") as CanvasLayer

    # Start the loop
    _start_poll_timer()

func _start_poll_timer() -> void:
    var timer = get_tree().create_timer(poll_interval)
    timer.timeout.connect(_on_poll_timer_timeout)

func _on_poll_timer_timeout() -> void:
    send_update()
    _start_poll_timer()

func send_update() -> void:
    if _is_requesting:
        return
    _is_requesting = true

    if hud_node:
        hud_node.call("set_thinking", true)

    # We send this so Python knows where Adam is and what he's experiencing
    var state = {
        "position": {
            "x": adam_node.global_position.x if adam_node else 0.0,
            "y": adam_node.global_position.y if adam_node else 0.0
        },
        "events": event_queue.duplicate()
    }

    # Clear queue after copying
    event_queue.clear()

    var json_query = JSON.stringify(state)
    var headers = ["Content-Type: application/json"]

    var error = http_request.request(server_url, headers, HTTPClient.METHOD_POST, json_query)
    if error != OK:
        _is_requesting = false
        if hud_node:
            hud_node.call("set_thinking", false)
        push_error("An error occurred in the HTTP request.")

func _on_request_completed(_result: int, response_code: int, _headers: PackedStringArray, body: PackedByteArray) -> void:
    _is_requesting = false

    if hud_node:
        hud_node.call("set_thinking", false)

    if response_code != 200:
        push_warning("Server returned non-200 code: %d" % response_code)
        return

    var json = JSON.new()
    var error = json.parse(body.get_string_from_utf8())

    if error == OK:
        var response = json.get_data()

        # Update UI with stats if available
        if response.has("stats") and hud_node:
            hud_node.call("update_stats", response["stats"])

        if response.has("world_status") and hud_node:
            hud_node.call("update_world_info", response["world_status"])

        if response.has("history") and hud_node:
            hud_node.call("update_history", response["history"])

        if response.has("command"):
            _apply_command(response["command"])
    else:
        push_error("JSON Parse Error: %s at line %d" % [json.get_error_message(), json.get_error_line()])

func _apply_command(command: Dictionary) -> void:
    if not adam_node:
        return

    var action = command.get("action", "IDLE")
    var target = command.get("target", "")
    var thought = command.get("thought", "")
    var emotion = command.get("emotion", "uncertain")
    var dialogue = command.get("dialogue", "")

    # Update HUD
    if hud_node:
        hud_node.call("update_thought", thought, emotion)
        hud_node.call("update_dialogue", dialogue)

    match action:
        "MOVE":
            if target is Dictionary and target.has("x") and target.has("y"):
                adam_node.call("set_move_target", Vector2(target["x"], target["y"]))
            elif target is String and target != "":
                # Future: Resolve name to position
                push_warning("MOVE target is string '%s' - resolving string targets not yet implemented." % target)
                # For now, just a random nudge
                adam_node.call("set_move_target", adam_node.global_position + Vector2(randf_range(-100, 100), randf_range(-100, 100)))
            else:
                # Default: Move to a random nearby point for visualization if no target
                adam_node.call("set_move_target", adam_node.global_position + Vector2(randf_range(-200, 200), randf_range(-200, 200)))
        "IDLE":
            adam_node.call("stop")
        _:
            # Other actions like EAT/DRINK/SLEEP are visualized by stopping
            adam_node.call("stop")

## Public method for Adam.gd to report events
func report_event(event: Dictionary) -> void:
    event_queue.append(event)
