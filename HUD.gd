extends CanvasLayer

## HUD (Heads-Up Display)
## Manages UI elements for Adam's stats, thoughts, and world state.

@onready var health_bar: ProgressBar = $StatsContainer/Health/ProgressBar
@onready var hunger_bar: ProgressBar = $StatsContainer/Hunger/ProgressBar
@onready var energy_bar: ProgressBar = $StatsContainer/Energy/ProgressBar
@onready var stress_bar: ProgressBar = $StatsContainer/Stress/ProgressBar

@onready var thought_label: Label = $ThoughtPanel/VBoxContainer/ThoughtLabel
@onready var dialogue_label: Label = $DialoguePanel/VBoxContainer/DialogueLabel
@onready var emotion_label: Label = $StatusContainer/Emotion/ValueLabel
@onready var world_label: Label = $StatusContainer/World/ValueLabel
@onready var thinking_indicator: Control = $ThinkingIndicator

func _ready() -> void:
    # Reset UI
    update_stats({"health": 100, "hunger": 0, "energy": 100, "stress": 0})
    update_thought("Waking up...", "neutral")
    update_world_info("Day 1 - Dawn")
    set_thinking(false)

func update_stats(stats: Dictionary) -> void:
    if stats.has("health"): health_bar.value = stats["health"]
    if stats.has("hunger"): hunger_bar.value = stats["hunger"]
    if stats.has("energy"): energy_bar.value = stats["energy"]
    if stats.has("stress"): stress_bar.value = stats["stress"]

func update_thought(thought: String, emotion: String) -> void:
    thought_label.text = thought
    emotion_label.text = emotion.capitalize()

    # Optional: Change thought box color based on emotion?
    # var panel = $ThoughtPanel
    # ...

func update_dialogue(dialogue: String) -> void:
    if dialogue.strip_edges() == "":
        $DialoguePanel.hide()
    else:
        $DialoguePanel.show()
        dialogue_label.text = dialogue

func update_world_info(info: String) -> void:
    world_label.text = info

func set_thinking(is_thinking: bool) -> void:
    thinking_indicator.visible = is_thinking
