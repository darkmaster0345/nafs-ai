extends CharacterBody2D

## Adam Agent Controller
## Handles movement, navigation, and sensory feedback.

@export var speed: float = 200.0
@export var rotation_speed: float = 10.0

@onready var nav_agent: NavigationAgent2D = $NavigationAgent2D as NavigationAgent2D
@onready var sprite: Sprite2D = $Sprite2D as Sprite2D
@onready var interaction_area: Area2D = $InteractionArea as Area2D
@onready var food_area: Area2D = $FoodArea as Area2D

var target_position: Vector2 = Vector2.ZERO
var current_action: String = "IDLE"
var action_label: Label

func _ready() -> void:
    # Anti-Failure: Check for assets, create placeholder if missing
    if sprite.texture == null:
        _create_placeholder_sprite()

    # Setup Navigation
    nav_agent.path_desired_distance = 20.0
    nav_agent.target_desired_distance = 20.0
    nav_agent.avoidance_enabled = true

    # Signal for avoidance
    nav_agent.velocity_computed.connect(_on_velocity_computed)

    # Connect signals for interaction
    interaction_area.area_entered.connect(_on_interaction_area_entered)
    food_area.area_entered.connect(_on_food_area_entered)

    # Create Floating Action Label
    action_label = Label.new()
    action_label.text = "IDLE"
    action_label.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
    action_label.add_theme_font_size_override("font_size", 14)
    action_label.position = Vector2(-50, -60)
    action_label.custom_minimum_size = Vector2(100, 20)
    add_child(action_label)

func _physics_process(delta: float) -> void:
    # Update label rotation so it stays upright
    action_label.global_rotation = 0
    action_label.text = current_action

    if current_action == "IDLE":
        velocity = Vector2.ZERO
        move_and_slide()
        return

    if nav_agent.is_navigation_finished():
        velocity = Vector2.ZERO
        move_and_slide()
        return

    # Navigation Logic
    var current_pos: Vector2 = global_position
    var next_path_pos: Vector2 = nav_agent.get_next_path_position()

    var new_velocity: Vector2 = (next_path_pos - current_pos).normalized() * speed

    if nav_agent.avoidance_enabled:
        nav_agent.set_velocity(new_velocity)
    else:
        _on_velocity_computed(new_velocity)

    # Smooth Rotation
    if velocity.length() > 0:
        var target_rotation = velocity.angle()
        rotation = lerp_angle(rotation, target_rotation, rotation_speed * delta)

func _on_velocity_computed(safe_velocity: Vector2) -> void:
    velocity = safe_velocity
    move_and_slide()

func set_move_target(new_target: Vector2) -> void:
    target_position = new_target
    nav_agent.target_position = target_position
    current_action = "MOVE"

func stop() -> void:
    current_action = "IDLE"
    nav_agent.target_position = global_position

func _create_placeholder_sprite() -> void:
    # Create a simple ColorRect placeholder at runtime
    var rect = ColorRect.new()
    rect.color = Color.MEDIUM_AQUAMARINE
    rect.size = Vector2(32, 32)
    rect.position = -rect.size / 2
    add_child(rect)
    # Move behind or hide the invisible sprite node
    sprite.visible = false

# ── Sensory Feedback ──────────────────────────────────────────────────────────

func _on_interaction_area_entered(area: Area2D) -> void:
    # Notify NetworkController of collision
    var nc = NetworkController
    if nc:
        nc.report_event({"type": "collision", "name": area.name})

func _on_food_area_entered(area: Area2D) -> void:
    # Specific food detection
    if area.is_in_group("food"):
        var nc = NetworkController
        if nc:
            nc.report_event({"type": "food_found", "name": area.name})
