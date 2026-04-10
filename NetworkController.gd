extends Node

## NetworkController (The Bridge)
## Handles HTTP polling loop and command synchronization.

@export var server_url: String = "http://localhost:5000/update"
@export var poll_interval: float = 0.5

@onready var http_request: HTTPRequest = HTTPRequest.new()

var event_queue: Array = []
var adam_node: CharacterBody2D = null

func _ready() -> void:
    add_child(http_request)
    http_request.request_completed.connect(_on_request_completed)

    # Wait for scene to be ready before finding Adam
    await get_tree().process_frame
    adam_node = get_tree().get_first_node_in_group("adam") as CharacterBody2D

    # Start the loop
    _start_poll_timer()

func _start_poll_timer() -> void:
    var timer = get_tree().create_timer(poll_interval)
    timer.timeout.connect(_on_poll_timer_timeout)

func _on_poll_timer_timeout() -> void:
    send_update()
    _start_poll_timer()

func send_update() -> void:
    if http_request.get_http_client_status() != HTTPClient.STATUS_DISCONNECTED:
        # Request still in progress, skip this tick to avoid overlapping
        return

    # We send this so Python knows where Adam is and what he's experiencing
    var state = {
        "position": {
            "x": adam_node.global_position.x if adam_node else 0,
            "y": adam_node.global_position.y if adam_node else 0
        },
        "events": event_queue.duplicate()
    }

    # Clear queue after copying
    event_queue.clear()

    var json_query = JSON.stringify(state)
    var headers = ["Content-Type: application/json"]

    var error = http_request.request(server_url, headers, HTTPClient.METHOD_POST, json_query)
    if error != OK:
        push_error("An error occurred in the HTTP request.")

func _on_request_completed(_result: int, response_code: int, _headers: PackedStringArray, body: PackedByteArray) -> void:
    if response_code != 200:
        push_warning("Server returned non-200 code: %d" % response_code)
        return

    var json = JSON.new()
    var error = json.parse(body.get_string_from_utf8())

    if error == OK:
        var response = json.get_data()
        if response.has("command"):
            _apply_command(response["command"])
    else:
        push_error("JSON Parse Error: %s at line %d" % [json.get_error_message(), json.get_error_line()])

func _apply_command(command: Dictionary) -> void:
    if not adam_node:
        return

    var action = command.get("action", "IDLE")
    var target = command.get("target", "") # Python might send string name of target

    # Log the thought/emotion for UI or debugging
    # print("[Adam Thought] %s (%s)" % [command.get("thought", ""), command.get("emotion", "")])

    match action:
        "MOVE":
            # In a real world, 'target' might be coords or an object name.
            # For v1.0, we'll assume target coords or a random nudge if empty.
            if target is Dictionary and target.has("x") and target.has("y"):
                adam_node.call("set_move_target", Vector2(target["x"], target["y"]))
            else:
                # If target is string like "tree", logic to find tree would go here.
                # Default: Move to a random nearby point for visualization.
                pass
        "IDLE":
            adam_node.call("stop")
        _:
            # Other actions like EAT/DRINK are mostly internal simulations,
            # but can be visualized with animations/effects here.
            adam_node.call("stop")

## Public method for Adam.gd to report events
func report_event(event: Dictionary) -> void:
    event_queue.append(event)
