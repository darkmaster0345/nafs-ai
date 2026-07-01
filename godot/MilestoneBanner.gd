extends CanvasLayer

## MilestoneBanner (Phase 11.4)
## Displays full-screen banners for major events:
## - First Contact
## - First Birth
## - First Death
## - First Word Invented
## - New Generation
## - Extinction Event
## - First Cooking Discovery

signal milestone_dismissed()

var banner: ColorRect = null
var title_label: Label = null
var subtitle_label: Label = null
var dismiss_timer: Timer = null
var current_milestone: String = ""

# Milestone configurations
const MILESTONE_CONFIGS: Dictionary = {
	"FIRST_CONTACT": {
		"title": "FIRST CONTACT",
		"subtitle": "Adam and Eve have met for the first time",
		"color": Color(0.2, 0.5, 1.0, 0.9),
		"duration": 5.0,
	},
	"FIRST_BIRTH": {
		"title": "FIRST BIRTH",
		"subtitle": "A new baby has been born",
		"color": Color(1.0, 0.4, 0.7, 0.9),
		"duration": 4.0,
	},
	"FIRST_DEATH": {
		"title": "FIRST DEATH",
		"subtitle": "An agent has died for the first time",
		"color": Color(0.3, 0.3, 0.3, 0.9),
		"duration": 4.0,
	},
	"FIRST_WORD": {
		"title": "FIRST WORD INVENTED",
		"subtitle": "A new word has entered the world",
		"color": Color(0.9, 0.7, 0.2, 0.9),
		"duration": 3.0,
	},
	"NEW_GENERATION": {
		"title": "NEW GENERATION",
		"subtitle": "A new generation has begun",
		"color": Color(0.2, 0.8, 0.4, 0.9),
		"duration": 3.0,
	},
	"EXTINCTION": {
		"title": "EXTINCTION EVENT",
		"subtitle": "Cataclysm has struck the world",
		"color": Color(0.8, 0.1, 0.1, 0.9),
		"duration": 5.0,
	},
	"COOKING_DISCOVERY": {
		"title": "COOKING DISCOVERED",
		"subtitle": "An agent has discovered cooked food",
		"color": Color(1.0, 0.5, 0.2, 0.9),
		"duration": 4.0,
	},
}

func _ready() -> void:
	_create_banner()
	_create_timer()

	# Connect to NetworkController
	await get_tree().process_frame
	var nc = get_node_or_null("/root/NetworkController")
	if nc:
		nc.connect("milestone_reached", _on_milestone_reached)

func _create_banner() -> void:
	# Full-screen banner with semi-transparent background
	banner = ColorRect.new()
	banner.size = Vector2(get_viewport().get_visible_rect().size)
	banner.color = Color(0, 0, 0, 0)
	banner.mouse_filter = Control.MOUSE_FILTER_IGNORE
	banner.visible = false
	add_child(banner)

	# Title label (large, centered)
	title_label = Label.new()
	title_label.text = ""
	title_label.horizontal_alignment = Label.HORIZONTAL_ALIGNMENT_CENTER
	title_label.vertical_alignment = Label.VERTICAL_ALIGNMENT_CENTER
	title_label.add_theme_font_size_override("font_size", 48)
	title_label.add_theme_color_override("font_color", Color.WHITE)
	title_label.size = banner.size
	title_label.mouse_filter = Control.MOUSE_FILTER_IGNORE
	banner.add_child(title_label)

	# Subtitle label (smaller, below title)
	subtitle_label = Label.new()
	subtitle_label.text = ""
	subtitle_label.horizontal_alignment = Label.HORIZONTAL_ALIGNMENT_CENTER
	subtitle_label.vertical_alignment = Label.VERTICAL_ALIGNMENT_CENTER
	subtitle_label.add_theme_font_size_override("font_size", 20)
	subtitle_label.add_theme_color_override("font_color", Color(0.9, 0.9, 0.9))
	subtitle_label.size = banner.size
	subtitle_label.position = Vector2(0, 60)
	subtitle_label.mouse_filter = Control.MOUSE_FILTER_IGNORE
	banner.add_child(subtitle_label)

func _create_timer() -> void:
	dismiss_timer = Timer.new()
	dismiss_timer.one_shot = true
	dismiss_timer.timeout.connect(_dismiss_banner)
	add_child(dismiss_timer)

func _on_milestone_reached(milestone_type: String, details: Dictionary) -> void:
	if not MILESTONE_CONFIGS.has(milestone_type):
		return

	var config = MILESTONE_CONFIGS[milestone_type]
	current_milestone = milestone_type

	title_label.text = config.title
	var subtitle_text = config.subtitle
	# Append details if provided
	if details.has("word"):
		subtitle_text += ": \"" + str(details.word) + "\""
	elif details.has("generation"):
		subtitle_text += ": Generation " + str(details.generation)
	elif details.has("agent_id"):
		subtitle_text += " (" + str(details.agent_id) + ")"
	subtitle_label.text = subtitle_text

	banner.color = config.color
	banner.visible = true

	# Animate in
	var tween = create_tween()
	tween.tween_property(banner, "color:a", config.color.a, 0.3)

	dismiss_timer.start(config.duration)

func _dismiss_banner() -> void:
	var tween = create_tween()
	tween.tween_property(banner, "color:a", 0.0, 0.5)
	tween.tween_callback(func(): banner.visible = false)
	milestone_dismissed.emit()

# ── Death tombstone markers ──────────────────────────────────────────────────

var tombstones: Dictionary = {}  # {(x,y): ColorRect}

func show_tombstone(x: int, y: int, agent_id: String) -> void:
	# Tombstone marker on tile, fades over 500 ticks
	var key = str(x) + "," + str(y)
	if tombstones.has(key):
		return  # already a tombstone here

	var tombstone = ColorRect.new()
	tombstone.color = Color(0.4, 0.4, 0.4, 0.7)
	tombstone.size = Vector2(20, 24)
	tombstone.position = Vector2(x * 32 + 6, y * 32 + 4)
	tombstone.tooltip_text = "RIP " + agent_id
	add_child(tombstone)
	tombstones[key] = tombstone

	# Schedule fade (500 ticks at 1 tick/sec ≈ 500 seconds real-time, but use 60s for demo)
	var fade_timer = Timer.new()
	fade_timer.one_shot = true
	fade_timer.wait_time = 60.0
	fade_timer.timeout.connect(func():
		if is_instance_valid(tombstone):
			var tween = create_tween()
			tween.tween_property(tombstone, "color:a", 0.0, 5.0)
			tween.tween_callback(func(): tombstone.queue_free())
		tombstones.erase(key)
		fade_timer.queue_free()
	)
	add_child(fade_timer)
	fade_timer.start()
