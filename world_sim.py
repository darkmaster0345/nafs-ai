"""
Nafs AI — World Simulation (Minecraft-like Biome System)
"A world that Adam can walk through. Every step, a new land."

Biomes: Desert, Forest, Tundra, Plains, Mountain, Swamp, Ocean, Jungle, Cave, Volcano
Weather: Clear, Rain, Snow, Storm, Heatwave, Fog, Sandstorm, Blizzard

Philosophy:
  - Adam lives in a VAST world, not a tiny box
  - Every biome affects survival differently
  - Weather changes create dynamic challenges
  - The world is generated procedurally — every run is unique
  - Adam must learn to adapt to different terrains and conditions

No external APIs. No LLMs. No pretrained knowledge.
"""

import random
import math


# ═══════════════════════════════════════════════════════════════════════════════
# Biome Definitions — Each biome has unique survival characteristics
# ═══════════════════════════════════════════════════════════════════════════════

BIOMES = {
    "desert": {
        "base_temp": 38.0,
        "temp_range": (-5, 20),       # Night drops, day rises
        "food_chance": 0.04,          # Rare food in desert
        "water_chance": 0.02,         # Very rare water
        "danger_chance": 0.08,        # Scorpions, snakes
        "shelter_chance": 0.05,       # Rare shade
        "energy_drain": 1.5,          # Heat drains energy fast
        "hunger_rate": 0.6,           # Normal hunger
        "emoji": "\U0001f3dc",       # Desert emoji
        "desc": "Sand stretches everywhere. Hot wind. No shade.",
        "ground": "sand",
    },
    "forest": {
        "base_temp": 22.0,
        "temp_range": (-3, 8),
        "food_chance": 0.18,          # Lots of berries, mushrooms
        "water_chance": 0.15,         # Streams, dew
        "danger_chance": 0.06,        # Wolves, bears
        "shelter_chance": 0.20,       # Trees, caves
        "energy_drain": 0.7,          # Moderate
        "hunger_rate": 0.5,
        "emoji": "\U0001f332",
        "desc": "Trees everywhere. Green. Cool air. Sounds of life.",
        "ground": "grass",
    },
    "tundra": {
        "base_temp": -5.0,
        "temp_range": (-15, 5),
        "food_chance": 0.03,          # Very scarce
        "water_chance": 0.05,         # Frozen lakes (melt some)
        "danger_chance": 0.04,        # Polar predators
        "shelter_chance": 0.08,       # Ice caves
        "energy_drain": 1.8,          # Cold drains energy
        "hunger_rate": 0.7,           # Body burns calories for warmth
        "emoji": "\u2744\ufe0f",
        "desc": "White ground. Freezing wind. Nothing grows here.",
        "ground": "snow",
    },
    "plains": {
        "base_temp": 20.0,
        "temp_range": (-5, 10),
        "food_chance": 0.12,          # Grass, roots, small game
        "water_chance": 0.12,         # Ponds, streams
        "danger_chance": 0.05,        # Predators in tall grass
        "shelter_chance": 0.10,       # Occasional trees
        "energy_drain": 0.6,          # Easy walking
        "hunger_rate": 0.4,
        "emoji": "\U0001f33f",
        "desc": "Open grassland. Wind in the grass. Can see far.",
        "ground": "grass",
    },
    "mountain": {
        "base_temp": 5.0,
        "temp_range": (-20, 10),
        "food_chance": 0.06,          # Mountain goats, berries
        "water_chance": 0.10,         # Snowmelt streams
        "danger_chance": 0.10,        # Falls, predators
        "shelter_chance": 0.15,       # Caves
        "energy_drain": 1.3,          # Hard climbing
        "hunger_rate": 0.6,
        "emoji": "\u26f0\ufe0f",
        "desc": "Steep rocks. Thin air. Cold and harsh.",
        "ground": "rock",
    },
    "swamp": {
        "base_temp": 25.0,
        "temp_range": (-2, 8),
        "food_chance": 0.10,          # Fish, roots
        "water_chance": 0.25,         # Water everywhere
        "danger_chance": 0.12,        # Snakes, gators, disease
        "shelter_chance": 0.06,       # Not much dry ground
        "energy_drain": 1.0,          # Mud is tiring
        "hunger_rate": 0.5,
        "emoji": "\U0001f33a",
        "desc": "Wet ground. Smell of decay. Bugs. Muddy water.",
        "ground": "mud",
    },
    "ocean": {
        "base_temp": 18.0,
        "temp_range": (-5, 8),
        "food_chance": 0.08,          # Fish
        "water_chance": 0.40,         # Salt water (not great but present)
        "danger_chance": 0.09,        # Currents, creatures
        "shelter_chance": 0.02,       # No shelter at sea
        "energy_drain": 1.2,          # Swimming is hard
        "hunger_rate": 0.5,
        "emoji": "\U0001f30a",
        "desc": "Water everywhere. Waves. Salt. No ground under feet.",
        "ground": "water",
    },
    "jungle": {
        "base_temp": 30.0,
        "temp_range": (-2, 5),
        "food_chance": 0.20,          # Fruits everywhere
        "water_chance": 0.20,         # Rain, streams
        "danger_chance": 0.14,        # Snakes, insects, big cats
        "shelter_chance": 0.12,       # Dense canopy
        "energy_drain": 1.1,          # Humid, hard to move
        "hunger_rate": 0.5,
        "emoji": "\U0001f33f",
        "desc": "Dense green. Humid. Bugs. Fruit above. Something watching.",
        "ground": "moss",
    },
    "cave": {
        "base_temp": 12.0,
        "temp_range": (-3, 3),
        "food_chance": 0.05,          # Fungi, bats
        "water_chance": 0.08,         # Drips
        "danger_chance": 0.11,        # Bats, falling rocks
        "shelter_chance": 0.30,       # You're already inside
        "energy_drain": 0.8,
        "hunger_rate": 0.4,
        "emoji": "\U0001f5f3",       # Cave
        "desc": "Dark. Echoes. Dripping water. Cold stone walls.",
        "ground": "stone",
    },
    "volcano": {
        "base_temp": 40.0,
        "temp_range": (0, 10),
        "food_chance": 0.01,          # Nothing grows
        "water_chance": 0.01,         # Everything evaporates
        "danger_chance": 0.15,        # Lava, gas, eruption
        "shelter_chance": 0.03,       # Rocks
        "energy_drain": 2.0,          # Extreme heat
        "hunger_rate": 0.7,
        "emoji": "\U0001f30b",
        "desc": "Hot ground. Smoke. Smell of sulfur. Red glow ahead.",
        "ground": "lava_rock",
    },
}

# Weather types and their effects
WEATHER_TYPES = {
    "clear":     {"emoji": "\u2600\ufe0f", "wetness": 0.0, "temp_mod": 0.0,  "visibility": 1.0, "desc": "Clear sky."},
    "rain":      {"emoji": "\U0001f327\ufe0f", "wetness": 0.7, "temp_mod": -3.0, "visibility": 0.6, "desc": "Rain falling. Wet. Cold drops."},
    "snow":      {"emoji": "\U0001f328\ufe0f", "wetness": 0.3, "temp_mod": -8.0, "visibility": 0.5, "desc": "Snow. White flakes. Cold."},
    "storm":     {"emoji": "\u26c8\ufe0f", "wetness": 0.9, "temp_mod": -5.0, "visibility": 0.3, "desc": "Storm! Thunder. Lightning. Wind!"},
    "heatwave":  {"emoji": "\U0001f525", "wetness": 0.0, "temp_mod": 10.0, "visibility": 0.8, "desc": "Scorching heat. Air shimmers. Dry."},
    "fog":       {"emoji": "\U0001f32b\ufe0f", "wetness": 0.3, "temp_mod": -2.0, "visibility": 0.2, "desc": "Fog. Cannot see far. White mist."},
    "sandstorm": {"emoji": "\U0001f32c\ufe0f", "wetness": 0.0, "temp_mod": 5.0,  "visibility": 0.2, "desc": "Sandstorm! Sand in eyes. Cannot see."},
    "blizzard":  {"emoji": "\u2744\ufe0f", "wetness": 0.6, "temp_mod": -15.0, "visibility": 0.15, "desc": "Blizzard! Whiteout. Cannot move well."},
}

# Weather transition probabilities (Markov chain)
# Each weather type has weighted transitions to other types
WEATHER_TRANSITIONS = {
    "clear":     {"clear": 50, "rain": 15, "snow": 5, "storm": 3, "heatwave": 10, "fog": 10, "sandstorm": 5, "blizzard": 2},
    "rain":      {"clear": 15, "rain": 35, "snow": 10, "storm": 20, "heatwave": 2, "fog": 15, "sandstorm": 1, "blizzard": 2},
    "snow":      {"clear": 10, "rain": 10, "snow": 35, "storm": 5, "heatwave": 1, "fog": 15, "sandstorm": 2, "blizzard": 22},
    "storm":     {"clear": 20, "rain": 25, "snow": 5, "storm": 15, "heatwave": 2, "fog": 20, "sandstorm": 3, "blizzard": 10},
    "heatwave":  {"clear": 35, "rain": 5, "snow": 1, "storm": 5, "heatwave": 30, "fog": 5, "sandstorm": 17, "blizzard": 2},
    "fog":       {"clear": 30, "rain": 20, "snow": 10, "storm": 5, "heatwave": 5, "fog": 20, "sandstorm": 5, "blizzard": 5},
    "sandstorm": {"clear": 25, "rain": 2, "snow": 1, "storm": 5, "heatwave": 20, "fog": 10, "sandstorm": 35, "blizzard": 2},
    "blizzard":  {"clear": 10, "rain": 5, "snow": 30, "storm": 10, "heatwave": 1, "fog": 10, "sandstorm": 2, "blizzard": 32},
}


class AdamStats:
    def __init__(self):
        self.health = 100.0
        self.hunger = 0.0     # 0 = full, 100 = starving
        self.energy = 100.0   # 0 = exhausted, 100 = full
        self.stress = 0.0     # 0 = calm, 100 = high stress
        self.pain = 0.0       # 0 = none, 10 = severe

    def is_alive(self):
        return self.health > 0 and self.hunger < 100

    def to_dict(self):
        return {
            'health': self.health,
            'hunger': self.hunger,
            'energy': self.energy,
            'stress': self.stress,
            'pain': self.pain
        }


class WorldMap:
    """
    Procedurally-generated Minecraft-like world map.
    
    Uses Perlin-like noise (simplified) to create biome regions.
    Adam starts in a random location and can MOVE to adjacent tiles.
    Each tile has a biome type.
    """

    def __init__(self, width=64, height=64, seed=None):
        self.width = width
        self.height = height
        self.seed = seed or random.randint(0, 999999)
        self.tiles = {}   # (x, y) -> biome_name
        self._generate()

    def _generate(self):
        """Generate the world map with biome regions."""
        rng = random.Random(self.seed)
        
        # Simplified noise: create clusters of biomes
        # Place biome "seeds" and expand them using flood-fill-like growth
        biome_list = list(BIOMES.keys())
        
        # Initialize all tiles as None
        for x in range(self.width):
            for y in range(self.height):
                self.tiles[(x, y)] = None
        
        # Place biome seed points (multiple per biome type)
        seeds = []
        for biome_name in biome_list:
            # Place 2-4 seeds for each biome type
            num_seeds = rng.randint(2, 4)
            for _ in range(num_seeds):
                sx = rng.randint(0, self.width - 1)
                sy = rng.randint(0, self.height - 1)
                seeds.append((sx, sy, biome_name))
        
        # Grow biomes from seeds using BFS
        from collections import deque
        queue = deque()
        for sx, sy, biome_name in seeds:
            self.tiles[(sx, sy)] = biome_name
            queue.append((sx, sy, biome_name))
        
        # Shuffle queue order for fairness
        queue_list = list(queue)
        rng.shuffle(queue_list)
        queue = deque(queue_list)
        
        while queue:
            x, y, biome_name = queue.popleft()
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    if self.tiles[(nx, ny)] is None:
                        self.tiles[(nx, ny)] = biome_name
                        queue.append((nx, ny, biome_name))
        
        # Fill any remaining None tiles with plains
        for x in range(self.width):
            for y in range(self.height):
                if self.tiles[(x, y)] is None:
                    self.tiles[(x, y)] = "plains"

    def get_biome(self, x, y):
        """Get the biome at a position, with wrapping."""
        wx = x % self.width
        wy = y % self.height
        return self.tiles.get((wx, wy), "plains")

    def get_nearby_biomes(self, x, y, radius=2):
        """Get list of nearby biome types for sensory input."""
        nearby = set()
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                if dx == 0 and dy == 0:
                    continue
                nearby.add(self.get_biome(x + dx, y + dy))
        return list(nearby)


class WeatherSystem:
    """
    Dynamic weather system with Markov chain transitions.
    
    Weather changes based on:
      - Current weather (most weather continues)
      - Current biome (desert = more sandstorm, tundra = more blizzard)
      - Time of day (some weathers more likely at night)
      - Random transitions (unpredictable but realistic)
    """

    def __init__(self, initial_weather="clear"):
        self.current = initial_weather
        self.tick_counter = 0
        self.transition_cooldown = 0  # Minimum ticks between weather changes

    def update(self, biome_name: str, time_of_day: int):
        """
        Update weather for this tick.
        
        Weather changes roughly every 10-30 ticks.
        Biome biases the transition probabilities.
        """
        self.tick_counter += 1
        
        # Cooldown: weather doesn't change every tick
        if self.transition_cooldown > 0:
            self.transition_cooldown -= 1
            return
        
        # Probability of weather change this tick (~5% per tick)
        if random.random() > 0.05:
            return
        
        # Get base transition probabilities
        transitions = WEATHER_TRANSITIONS.get(self.current, WEATHER_TRANSITIONS["clear"]).copy()
        
        # Apply biome biases
        biome = BIOMES.get(biome_name, BIOMES["plains"])
        base_temp = biome["base_temp"]
        
        if biome_name == "desert":
            transitions["sandstorm"] = transitions.get("sandstorm", 0) + 15
            transitions["heatwave"] = transitions.get("heatwave", 0) + 10
            transitions["snow"] = max(0, transitions.get("snow", 0) - 5)
            transitions["blizzard"] = max(0, transitions.get("blizzard", 0) - 5)
        elif biome_name == "tundra":
            transitions["blizzard"] = transitions.get("blizzard", 0) + 15
            transitions["snow"] = transitions.get("snow", 0) + 10
            transitions["heatwave"] = max(0, transitions.get("heatwave", 0) - 5)
        elif biome_name == "ocean":
            transitions["storm"] = transitions.get("storm", 0) + 10
            transitions["fog"] = transitions.get("fog", 0) + 8
        elif biome_name == "volcano":
            transitions["heatwave"] = transitions.get("heatwave", 0) + 15
            transitions["sandstorm"] = max(0, transitions.get("sandstorm", 0) - 5)
        elif biome_name == "cave":
            transitions["fog"] = transitions.get("fog", 0) + 10
            transitions["storm"] = max(0, transitions.get("storm", 0) - 5)
            transitions["clear"] = max(0, transitions.get("clear", 0) - 5)
        elif biome_name == "swamp":
            transitions["fog"] = transitions.get("fog", 0) + 10
            transitions["rain"] = transitions.get("rain", 0) + 8
        
        # Time of day effects
        if time_of_day >= 22 or time_of_day < 4:
            # Night: more fog, less storms
            transitions["fog"] = transitions.get("fog", 0) + 5
        elif 12 <= time_of_day < 16:
            # Afternoon: more storms
            transitions["storm"] = transitions.get("storm", 0) + 5
        
        # Select new weather
        options = list(transitions.keys())
        weights = [transitions.get(o, 1) for o in options]
        new_weather = random.choices(options, weights=weights, k=1)[0]
        
        if new_weather != self.current:
            self.current = new_weather
            self.transition_cooldown = random.randint(5, 15)  # Weather stays for a while

    def get_data(self):
        """Get current weather data."""
        return WEATHER_TYPES.get(self.current, WEATHER_TYPES["clear"])


class WorldSim:
    """
    Minecraft-like world simulation with biomes and dynamic weather.
    
    Adam lives ONCE. He explores a procedurally generated world
    with diverse biomes, random weather, and survival challenges.
    When Adam dies, the program ends — no restart, no memory carries over.
    """

    ACTIONS = ["EXPLORE", "EAT", "DRINK", "SLEEP", "HIDE", "MOVE", "FLEE", "IDLE"]

    def __init__(self):
        self.time_of_day = 0
        self.weather_system = WeatherSystem()
        self.adam_stats = AdamStats()
        self.current_tick = 0
        
        # World map
        self.world_map = WorldMap()
        self.adam_x = random.randint(0, self.world_map.width - 1)
        self.adam_y = random.randint(0, self.world_map.height - 1)
        self.current_biome = self.world_map.get_biome(self.adam_x, self.adam_y)
        
        # World events (nearby resources/dangers)
        self.world_events = {
            'food_nearby': 0.0,
            'danger_nearby': 0.0,
            'water_nearby': 0.0,
            'shelter_nearby': 0.0,
        }

        # Direction Adam is facing (for movement display)
        self.facing_direction = random.choice(["north", "south", "east", "west"])

    def reset(self):
        """
        Initialize Adam's one and only life.
        Called ONCE at program start. There is no second episode.
        """
        self.time_of_day = random.randint(6, 12)  # Start in morning
        self.weather_system = WeatherSystem(
            initial_weather=random.choice(["clear", "clear", "clear", "fog", "rain"])
        )
        self.adam_stats = AdamStats()
        self.current_tick = 0
        
        # Random spawn location
        self.adam_x = random.randint(0, self.world_map.width - 1)
        self.adam_y = random.randint(0, self.world_map.height - 1)
        self.current_biome = self.world_map.get_biome(self.adam_x, self.adam_y)
        self.facing_direction = random.choice(["north", "south", "east", "west"])
        
        # Initial world events based on biome
        biome = BIOMES[self.current_biome]
        self.world_events = {
            'food_nearby': 1.0 if random.random() < biome["food_chance"] * 3 else 0.0,
            'danger_nearby': 0.0,  # No danger at birth
            'water_nearby': 1.0 if random.random() < biome["water_chance"] * 3 else 0.0,
            'shelter_nearby': 1.0 if random.random() < biome["shelter_chance"] * 3 else 0.0,
        }
        
        return self._get_world_state(), self.adam_stats.to_dict()

    def _get_world_state(self):
        """Build the full world state dictionary."""
        biome = BIOMES[self.current_biome]
        weather_data = self.weather_system.get_data()
        
        # Light level from time of day
        light_level = 0.0
        if 6 <= self.time_of_day < 18:
            light_level = 1.0
        elif 5 <= self.time_of_day < 6 or 18 <= self.time_of_day < 19:
            light_level = 0.5
        
        # Weather reduces visibility
        visibility = weather_data["visibility"]
        if self.current_biome == "cave":
            visibility *= 0.3  # Caves are always dark
            light_level *= 0.2
        
        # Wetness from weather and biome
        wetness = weather_data["wetness"]
        if self.current_biome in ("swamp", "ocean"):
            wetness = max(wetness, 0.5)
        elif self.current_biome == "desert":
            wetness = min(wetness, 0.1)
        
        # Sound level based on events, biome, weather
        sound_level = 0.1  # Base ambient
        if self.world_events['danger_nearby'] > 0.5:
            sound_level = random.uniform(0.6, 0.9)
        elif self.world_events['food_nearby'] > 0.5:
            sound_level = random.uniform(0.2, 0.4)
        if self.weather_system.current == "storm":
            sound_level = min(1.0, sound_level + 0.4)
        if self.current_biome == "cave":
            sound_level = max(0.05, sound_level - 0.1)
        
        # Temperature: base biome + weather modifier + time of day
        temp_variation = random.uniform(*biome["temp_range"])
        temperature = biome["base_temp"] + weather_data["temp_mod"] + temp_variation
        if 0 <= self.time_of_day < 6 or 18 <= self.time_of_day < 24:
            temperature -= random.uniform(2, 5)
        
        # Touch softness based on ground type
        ground = biome["ground"]
        touch_softness = {
            "grass": 0.7, "sand": 0.5, "snow": 0.8, "rock": 0.2,
            "mud": 0.6, "water": 0.3, "moss": 0.9, "stone": 0.15,
            "lava_rock": 0.1,
        }.get(ground, 0.5)
        
        return {
            'temperature': temperature,
            'light_level': light_level * visibility,
            'smell_food': self.world_events['food_nearby'],
            'smell_danger': self.world_events['danger_nearby'],
            'sound_level': sound_level,
            'wetness': wetness,
            'proximity_entity': self.world_events['water_nearby'],
            'touch_softness': touch_softness,
            # New biome/weather fields
            'biome': self.current_biome,
            'weather': self.weather_system.current,
            'time_of_day': self.time_of_day,
            'adam_x': self.adam_x,
            'adam_y': self.adam_y,
            'facing': self.facing_direction,
            'visibility': visibility,
            'ground': ground,
        }

    def step(self, action: str):
        """Process one tick of the simulation."""
        reward = 0.0
        done = False
        
        self.current_tick += 1
        self.time_of_day = (self.time_of_day + 1) % 24
        
        # Update weather
        self.weather_system.update(self.current_biome, self.time_of_day)
        weather_data = self.weather_system.get_data()
        
        # Get current biome data
        biome = BIOMES[self.current_biome]
        
        # Update biome (in case Adam moved last tick)
        self.current_biome = self.world_map.get_biome(self.adam_x, self.adam_y)
        biome = BIOMES[self.current_biome]
        
        # --- Update world events based on biome ---
        food_prob = random.uniform(0.05, 0.15) * biome["food_chance"] / 0.1
        self.world_events['food_nearby'] = 1.0 if random.random() < food_prob else 0.0
        danger_prob = random.uniform(0.02, 0.10) * biome["danger_chance"] / 0.05
        self.world_events['danger_nearby'] = 1.0 if random.random() < danger_prob else 0.0
        water_prob = random.uniform(0.08, 0.18) * biome["water_chance"] / 0.1
        self.world_events['water_nearby'] = 1.0 if random.random() < water_prob else 0.0
        shelter_prob = random.uniform(0.03, 0.12) * biome["shelter_chance"] / 0.1
        self.world_events['shelter_nearby'] = 1.0 if random.random() < shelter_prob else 0.0
        
        # Weather can spawn/despawn events
        if self.weather_system.current == "storm":
            self.world_events['danger_nearby'] = 1.0 if random.random() < 0.15 else self.world_events['danger_nearby']
        if self.weather_system.current == "rain":
            self.world_events['water_nearby'] = max(self.world_events['water_nearby'], 0.5 if random.random() < 0.3 else 0.0)
        
        # --- Passive stat changes (biome-dependent) ---
        self.adam_stats.hunger += biome["hunger_rate"]
        self.adam_stats.energy -= biome["energy_drain"]
        self.adam_stats.stress += 0.1
        self.adam_stats.health -= 0.05  # Minor natural decay
        
        # Weather effects on stats
        if self.weather_system.current == "rain":
            self.adam_stats.energy -= 0.3
            if self.current_biome not in ("swamp", "ocean", "forest"):
                self.adam_stats.stress += 0.2
        elif self.weather_system.current == "snow":
            self.adam_stats.energy -= 0.5
            self.adam_stats.health -= 0.2
        elif self.weather_system.current == "storm":
            self.adam_stats.energy -= 0.8
            self.adam_stats.stress += 0.5
            if random.random() < 0.1:
                self.adam_stats.health -= random.uniform(1, 3)
        elif self.weather_system.current == "heatwave":
            self.adam_stats.energy -= 0.6
            self.adam_stats.hunger += 0.3
            if self.current_biome in ("desert", "volcano"):
                self.adam_stats.health -= 0.5
        elif self.weather_system.current == "blizzard":
            self.adam_stats.energy -= 1.0
            self.adam_stats.health -= 0.5
            self.adam_stats.stress += 0.8
        elif self.weather_system.current == "sandstorm":
            self.adam_stats.energy -= 0.5
            self.adam_stats.stress += 0.4
            if random.random() < 0.15:
                self.adam_stats.pain = min(10, self.adam_stats.pain + random.uniform(1, 3))
        elif self.weather_system.current == "fog":
            self.adam_stats.stress += 0.1
        
        # Biome passive effects
        if self.current_biome == "desert":
            self.adam_stats.hunger += 0.2
        elif self.current_biome == "tundra":
            self.adam_stats.health -= 0.1
            self.adam_stats.energy -= 0.3
        elif self.current_biome == "volcano":
            self.adam_stats.health -= 0.2
            self.adam_stats.hunger += 0.3
        elif self.current_biome == "ocean":
            self.adam_stats.energy -= 0.2  # Treading water
        
        # Critical stat effects
        if self.adam_stats.hunger >= 80:
            self.adam_stats.health -= 0.5
            self.adam_stats.stress += 0.5
        if self.adam_stats.energy <= 20:
            self.adam_stats.health -= 0.2
            self.adam_stats.stress += 0.3
        
        # --- Process Adam's action ---
        if action == "EXPLORE":
            reward += 0.1
            self.adam_stats.energy -= 2.0
            self.adam_stats.stress += 0.1
            if random.random() < 0.2:
                if random.random() < 0.5:
                    self.world_events['food_nearby'] = 1.0
                else:
                    self.world_events['water_nearby'] = 1.0
            if random.random() < 0.05:
                self.world_events['danger_nearby'] = 1.0
                self.adam_stats.pain = random.uniform(1, 5)
                self.adam_stats.health -= self.adam_stats.pain
                reward -= 1.5
            # Look at nearby biomes
            nearby = self.world_map.get_nearby_biomes(self.adam_x, self.adam_y, 2)
            if nearby:
                # Discovering a new biome nearby is interesting
                pass
        
        elif action == "EAT":
            if self.world_events['food_nearby'] > 0.5:
                food_value = random.uniform(20, 40)
                # Biome affects food quality
                if self.current_biome == "jungle":
                    food_value *= 1.3  # Better food
                elif self.current_biome in ("desert", "tundra", "volcano"):
                    food_value *= 0.7  # Poor food
                self.adam_stats.hunger = max(0.0, self.adam_stats.hunger - food_value)
                reward += 1.0
                self.world_events['food_nearby'] = 0.0
            else:
                self.adam_stats.stress += 0.5
                reward -= 0.1
            self.adam_stats.energy -= 0.5
        
        elif action == "DRINK":
            if self.world_events['water_nearby'] > 0.5:
                drink_value = random.uniform(5, 15)
                self.adam_stats.hunger = max(0.0, self.adam_stats.hunger - drink_value)
                self.adam_stats.stress = max(0.0, self.adam_stats.stress - 5.0)
                reward += 0.2
                self.world_events['water_nearby'] = 0.0
            elif self.weather_system.current in ("rain", "storm", "blizzard"):
                # Can collect rainwater
                self.adam_stats.hunger = max(0.0, self.adam_stats.hunger - random.uniform(3, 8))
                self.adam_stats.stress = max(0.0, self.adam_stats.stress - 2.0)
                reward += 0.1
            else:
                self.adam_stats.stress += 0.2
                reward -= 0.05
            self.adam_stats.energy -= 0.3
        
        elif action == "SLEEP":
            if self.world_events['shelter_nearby'] > 0.5 and (self.time_of_day >= 18 or self.time_of_day < 6):
                self.adam_stats.energy = min(100.0, self.adam_stats.energy + random.uniform(30, 50))
                self.adam_stats.health = min(100.0, self.adam_stats.health + random.uniform(1, 3))
                self.adam_stats.stress = max(0.0, self.adam_stats.stress - random.uniform(10, 20))
                reward += 0.3
            else:
                self.adam_stats.energy = min(100.0, self.adam_stats.energy + random.uniform(10, 20))
                self.adam_stats.stress = max(0.0, self.adam_stats.stress - random.uniform(2, 5))
                if 6 <= self.time_of_day < 18:
                    reward -= 0.05
                else:
                    reward += 0.05
            self.adam_stats.hunger += 1.0
            # Cave is good for sleeping
            if self.current_biome == "cave":
                self.adam_stats.energy = min(100.0, self.adam_stats.energy + random.uniform(5, 10))
                self.adam_stats.stress = max(0.0, self.adam_stats.stress - 3.0)
        
        elif action == "HIDE":
            self.adam_stats.energy -= 1.0
            self.adam_stats.stress += 0.1
            if self.world_events['danger_nearby'] > 0.5:
                if random.random() < 0.7:
                    self.world_events['danger_nearby'] = 0.0
                    self.adam_stats.stress = max(0.0, self.adam_stats.stress - 5.0)
                    reward += 0.5
                else:
                    self.adam_stats.pain = random.uniform(3, 8)
                    self.adam_stats.health -= self.adam_stats.pain
                    reward -= 1.5
            else:
                reward -= 0.1
            # Better hiding in forest/cave
            if self.current_biome in ("forest", "cave", "swamp"):
                reward += 0.05  # Hiding is easier in cover
        
        elif action == "MOVE":
            self.adam_stats.energy -= 1.5
            self.adam_stats.hunger += 0.1
            self.adam_stats.stress += 0.05
            
            # Move Adam to an adjacent tile
            direction_map = {
                "north": (0, -1), "south": (0, 1),
                "east": (1, 0), "west": (-1, 0),
            }
            # Choose direction (biased toward facing direction)
            directions = list(direction_map.keys())
            if random.random() < 0.6:
                chosen = self.facing_direction
            else:
                chosen = random.choice(directions)
            
            dx, dy = direction_map[chosen]
            new_x = (self.adam_x + dx) % self.world_map.width
            new_y = (self.adam_y + dy) % self.world_map.height
            self.adam_x = new_x
            self.adam_y = new_y
            self.facing_direction = chosen
            
            # Update biome
            new_biome = self.world_map.get_biome(self.adam_x, self.adam_y)
            self.current_biome = new_biome
            
            # Discover resources in new tile
            if random.random() < 0.15:
                self.world_events['food_nearby'] = 1.0 if random.random() < 0.5 else 0.0
                self.world_events['water_nearby'] = 1.0 if random.random() < 0.5 else 0.0
            if random.random() < 0.03:
                self.world_events['danger_nearby'] = 1.0
            
            # Movement reward (exploration)
            reward += 0.05
            
            # Ocean is dangerous to move in
            if new_biome == "ocean":
                self.adam_stats.energy -= 1.0  # Swimming is extra tiring
                if random.random() < 0.08:
                    self.world_events['danger_nearby'] = 1.0  # Currents
        
        elif action == "FLEE":
            self.adam_stats.energy -= 3.0
            self.adam_stats.stress += 0.5
            if self.world_events['danger_nearby'] > 0.5:
                if random.random() < 0.9:
                    self.world_events['danger_nearby'] = 0.0
                    self.adam_stats.stress = max(0.0, self.adam_stats.stress - 10.0)
                    reward += 0.3
                    # Flee moves Adam to a random adjacent tile
                    dx, dy = random.choice([(0, -1), (0, 1), (1, 0), (-1, 0)])
                    self.adam_x = (self.adam_x + dx) % self.world_map.width
                    self.adam_y = (self.adam_y + dy) % self.world_map.height
                    self.current_biome = self.world_map.get_biome(self.adam_x, self.adam_y)
                else:
                    self.adam_stats.pain = random.uniform(5, 10)
                    self.adam_stats.health -= self.adam_stats.pain
                    reward -= 1.5
            else:
                reward -= 0.2
        
        elif action == "IDLE":
            self.adam_stats.energy += 0.5
            self.adam_stats.hunger += 0.2
            self.adam_stats.stress = max(0.0, self.adam_stats.stress - 0.5)
            reward -= 0.05
        
        # --- Clamp stats ---
        self.adam_stats.health = max(0.0, min(100.0, self.adam_stats.health))
        self.adam_stats.hunger = max(0.0, min(100.0, self.adam_stats.hunger))
        self.adam_stats.energy = max(0.0, min(100.0, self.adam_stats.energy))
        self.adam_stats.stress = max(0.0, min(100.0, self.adam_stats.stress))
        self.adam_stats.pain = max(0.0, min(10.0, self.adam_stats.pain))
        
        # --- Check for death ---
        if not self.adam_stats.is_alive():
            reward -= 5.0
            done = True
        
        # --- Survival reward ---
        if not done:
            reward += 0.01
        
        next_world_state = self._get_world_state()
        next_adam_stats = self.adam_stats.to_dict()
        
        return next_world_state, next_adam_stats, reward, done


if __name__ == '__main__':
    env = WorldSim()
    world_state, adam_stats = env.reset()
    print(f"Adam wakes up in a {world_state['biome']} biome!")
    print(f"Weather: {world_state['weather']}")
    print(f"Position: ({world_state['adam_x']}, {world_state['adam_y']})")
    print(f"Time: {world_state['time_of_day']}:00")
    print(f"World size: {env.world_map.width}x{env.world_map.height}")
    
    done = False
    episode_reward = 0
    steps = 0
    while not done and steps < 100:
        action = random.choice(WorldSim.ACTIONS)
        world_state, adam_stats, reward, done = env.step(action)
        episode_reward += reward
        steps += 1
        biome_emoji = BIOMES.get(world_state['biome'], {}).get('emoji', '?')
        weather_emoji = WEATHER_TYPES.get(world_state['weather'], {}).get('emoji', '?')
        print(f"\nTick {steps}: {biome_emoji} {world_state['biome']} | {weather_emoji} {world_state['weather']} | Action: {action}")
        print(f"  HP:{adam_stats['health']:.0f} Hunger:{adam_stats['hunger']:.0f} Energy:{adam_stats['energy']:.0f} Stress:{adam_stats['stress']:.0f}")
    
    print(f"\nAdam survived {steps} ticks with total reward {round(episode_reward, 2)}")
    if not env.adam_stats.is_alive():
        print("Adam died.")
