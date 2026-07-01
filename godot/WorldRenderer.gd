extends Node2D

## WorldRenderer (Phase 11.1)
## Renders the 64x64 tile map with biome sprites, fire animation,
## water ripple, day/night cycle, and weather overlays.
##
## Connects to NetworkController to receive world_state updates.

const TILE_SIZE: int = 32  # pixels per tile
const MAP_WIDTH: int = 64
const MAP_HEIGHT: int = 64

# Biome colors (used when no sprite texture is available)
const BIOME_COLORS: Dictionary = {
	"desert":  Color(0.93, 0.79, 0.40, 1.0),   # yellow
	"forest":  Color(0.20, 0.50, 0.20, 1.0),   # green
	"tundra":  Color(0.85, 0.90, 0.95, 1.0),   # white
	"plains":  Color(0.55, 0.70, 0.35, 1.0),   # light green
	"mountain":Color(0.45, 0.40, 0.35, 1.0),   # brown
	"swamp":   Color(0.30, 0.35, 0.20, 1.0),   # dark green-brown
	"ocean":   Color(0.15, 0.30, 0.60, 1.0),   # blue
	"jungle":  Color(0.15, 0.40, 0.15, 1.0),   # dark green
	"cave":    Color(0.20, 0.15, 0.20, 1.0),   # dark
	"volcano": Color(0.60, 0.20, 0.10, 1.0),   # red
}

# Tile state containers
var biome_tiles: Array = []          # 2D array of ColorRect per tile
var fire_sprites: Dictionary = {}    # {(x,y): AnimatedSprite2D}
var water_sprites: Dictionary = {}   # {(x,y): AnimatedSprite2D}
var weather_overlay: ColorRect = null
var day_night_overlay: ColorRect = null

# Weather particle systems
var rain_particles: CPUParticles2D = null
var snow_particles: CPUParticles2D = null
var sandstorm_overlay: ColorRect = null

# Current state (updated by NetworkController)
var current_time_of_day: int = 12
var current_weather: String = "clear"
var current_biome_map: Dictionary = {}  # {"x,y": "biome_name"}

func _ready() -> void:
	_initialize_tile_map()
	_initialize_overlays()
	_initialize_weather_particles()

func _initialize_tile_map() -> void:
	# Create a ColorRect for each tile in the 64x64 grid
	for x in range(MAP_WIDTH):
		biome_tiles.append([])
		for y in range(MAP_HEIGHT):
			var tile = ColorRect.new()
			tile.size = Vector2(TILE_SIZE, TILE_SIZE)
			tile.position = Vector2(x * TILE_SIZE, y * TILE_SIZE)
			tile.color = BIOME_COLORS["plains"]  # default
			add_child(tile)
			biome_tiles[x].append(tile)

func _initialize_overlays() -> void:
	# Day/night overlay (full-screen, modulates brightness)
	day_night_overlay = ColorRect.new()
	day_night_overlay.size = Vector2(MAP_WIDTH * TILE_SIZE, MAP_HEIGHT * TILE_SIZE)
	day_night_overlay.color = Color(0.0, 0.0, 0.0, 0.0)  # transparent (day)
	day_night_overlay.mouse_filter = Control.MOUSE_FILTER_IGNORE
	add_child(day_night_overlay)

	# Weather overlay
	weather_overlay = ColorRect.new()
	weather_overlay.size = Vector2(MAP_WIDTH * TILE_SIZE, MAP_HEIGHT * TILE_SIZE)
	weather_overlay.color = Color(0.5, 0.5, 0.5, 0.0)  # transparent
	weather_overlay.mouse_filter = Control.MOUSE_FILTER_IGNORE
	add_child(weather_overlay)

	# Sandstorm overlay (separate, can be opaque)
	sandstorm_overlay = ColorRect.new()
	sandstorm_overlay.size = Vector2(MAP_WIDTH * TILE_SIZE, MAP_HEIGHT * TILE_SIZE)
	sandstorm_overlay.color = Color(0.85, 0.70, 0.40, 0.0)  # transparent
	sandstorm_overlay.mouse_filter = Control.MOUSE_FILTER_IGNORE
	add_child(sandstorm_overlay)

func _initialize_weather_particles() -> void:
	# Rain particles
	rain_particles = CPUParticles2D.new()
	rain_particles.amount = 200
	rain_particles.direction = Vector2(0.2, 1.0)  # slight angle
	rain_particles.initial_velocity = 300.0
	rain_particles.gravity = Vector2(0, 200)
	rain_particles.lifetime = 1.5
	rain_particles.color = Color(0.6, 0.7, 0.9, 0.7)
	rain_particles.scale = Vector2(0.05, 0.3)
	rain_particles.emitting = false
	add_child(rain_particles)

	# Snow particles
	snow_particles = CPUParticles2D.new()
	snow_particles.amount = 150
	snow_particles.direction = Vector2(0.1, 1.0)
	snow_particles.initial_velocity = 50.0
	snow_particles.gravity = Vector2(0, 30)
	snow_particles.lifetime = 4.0
	snow_particles.color = Color(1.0, 1.0, 1.0, 0.8)
	snow_particles.scale = Vector2(0.15, 0.15)
	snow_particles.emitting = false
	add_child(snow_particles)

# ── Public API (called by NetworkController) ──────────────────────────────────

func update_biome_map(biome_data: Dictionary) -> void:
	# biome_data: {"x,y": "forest", ...}
	current_biome_map = biome_data
	for key in biome_data.keys():
		var parts = key.split(",")
		if parts.size() != 2:
			continue
		var x = int(parts[0])
		var y = int(parts[1])
		if x < 0 or x >= MAP_WIDTH or y < 0 or y >= MAP_HEIGHT:
			continue
		var biome = biome_data[key]
		if BIOME_COLORS.has(biome):
			biome_tiles[x][y].color = BIOME_COLORS[biome]

func update_fire_tiles(fire_positions: Array) -> void:
	# fire_positions: [[x, y], ...]
	# Remove old fires not in new list
	var new_set = {}
	for pos in fire_positions:
		new_set[str(pos[0]) + "," + str(pos[1])] = true

	for key in fire_sprites.keys():
		if not new_set.has(key):
			fire_sprites[key].queue_free()
			fire_sprites.erase(key)

	# Add new fires
	for pos in fire_positions:
		var key = str(pos[0]) + "," + str(pos[1])
		if not fire_sprites.has(key):
			var fire = _create_fire_sprite(pos[0], pos[1])
			fire_sprites[key] = fire
			add_child(fire)

func _create_fire_sprite(x: int, y: int) -> Node2D:
	# Create a flickering fire effect using a ColorRect with animation
	var fire = Node2D.new()
	fire.position = Vector2(x * TILE_SIZE + TILE_SIZE/2, y * TILE_SIZE + TILE_SIZE/2)

	var fire_rect = ColorRect.new()
	fire_rect.color = Color(1.0, 0.5, 0.1, 0.9)
	fire_rect.size = Vector2(TILE_SIZE * 0.6, TILE_SIZE * 0.6)
	fire_rect.position = -fire_rect.size / 2
	fire.add_child(fire_rect)

	# Add a glow
	var glow = ColorRect.new()
	glow.color = Color(1.0, 0.7, 0.3, 0.4)
	glow.size = Vector2(TILE_SIZE, TILE_SIZE)
	glow.position = -glow.size / 2
	fire.add_child(glow)
	glow.z_index = -1

	return fire

func update_water_tiles(water_positions: Array) -> void:
	# Similar to fire but for temporary water (rain puddles)
	for key in water_sprites.keys():
		water_sprites[key].queue_free()
		water_sprites.erase(key)

	for pos in water_positions:
		var key = str(pos[0]) + "," + str(pos[1])
		var water = ColorRect.new()
		water.color = Color(0.3, 0.5, 0.8, 0.6)
		water.size = Vector2(TILE_SIZE * 0.8, TILE_SIZE * 0.8)
		water.position = Vector2(pos[0] * TILE_SIZE + TILE_SIZE * 0.1,
		                          pos[1] * TILE_SIZE + TILE_SIZE * 0.1)
		water_sprites[key] = water
		add_child(water)

func update_day_night(time_of_day: int) -> void:
	current_time_of_day = time_of_day
	# Compute darkness: 0 at noon, 0.7 at midnight
	var darkness: float = 0.0
	if time_of_day >= 18 or time_of_day < 6:
		# Night
		if time_of_day >= 18:
			darkness = lerp(0.0, 0.7, float(time_of_day - 18) / 6.0)
		else:
			darkness = lerp(0.7, 0.0, float(time_of_day) / 6.0)
	elif time_of_day >= 5 and time_of_day < 7:
		# Dawn
		darkness = lerp(0.7, 0.0, float(time_of_day - 5) / 2.0)
	elif time_of_day >= 17 and time_of_day < 19:
		# Dusk
		darkness = lerp(0.0, 0.7, float(time_of_day - 17) / 2.0)

	day_night_overlay.color = Color(0.0, 0.05, 0.15, darkness)

func update_weather(weather: String) -> void:
	current_weather = weather
	# Reset all
	rain_particles.emitting = false
	snow_particles.emitting = false
	sandstorm_overlay.color = Color(0.85, 0.70, 0.40, 0.0)
	weather_overlay.color = Color(0.5, 0.5, 0.5, 0.0)

	match weather:
		"rain":
			rain_particles.emitting = true
			weather_overlay.color = Color(0.4, 0.5, 0.6, 0.2)
		"storm":
			rain_particles.emitting = true
			rain_particles.amount = 400
			weather_overlay.color = Color(0.3, 0.3, 0.4, 0.4)
		"snow":
			snow_particles.emitting = true
			weather_overlay.color = Color(0.9, 0.9, 1.0, 0.2)
		"blizzard":
			snow_particles.emitting = true
			snow_particles.amount = 400
			weather_overlay.color = Color(0.7, 0.7, 0.8, 0.5)
		"sandstorm":
			sandstorm_overlay.color = Color(0.85, 0.70, 0.40, 0.7)
		"fog":
			weather_overlay.color = Color(0.8, 0.8, 0.8, 0.3)
		"heatwave":
			weather_overlay.color = Color(1.0, 0.8, 0.5, 0.15)

func _process(delta: float) -> void:
	# Animate fire flicker
	for fire in fire_sprites.values():
		if fire.get_child_count() > 0:
			var rect = fire.get_child(0) as ColorRect
			if rect:
				var flicker = sin(Time.get_ticks_msec() * 0.01 + fire.position.x) * 0.1
				rect.color = Color(1.0, 0.5 + flicker, 0.1, 0.9)

	# Animate water ripple
	for water in water_sprites.values():
		if water is ColorRect:
			var ripple = sin(Time.get_ticks_msec() * 0.005 + water.position.x) * 0.1
			water.color = Color(0.3, 0.5, 0.8, 0.6 + ripple)
