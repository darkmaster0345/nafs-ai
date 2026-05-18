"""
Nafs AI — Sensory Encoder (Extended for Biome/Weather)
"Adam feels the sand between his toes, the snow on his skin."

Converts world state + Adam's stats into a numerical tensor for the neural network.

Input dimensions (21 total):
  0.  temperature       [-1, 1]     Scaled from Celsius
  1.  pain              [0, 1]      Physical pain level
  2.  hunger            [0, 1]      Hunger level
  3.  energy            [0, 1]      Energy level
  4.  light_level       [0, 1]      Ambient light (affected by weather/visibility)
  5.  smell_food        [0, 1]      Food nearby signal
  6.  smell_danger      [0, 1]      Danger nearby signal
  7.  sound_level       [0, 1]      Ambient sound
  8.  wetness           [0, 1]      Moisture in environment
  9.  touch_softness    [0, 1]      Ground texture
  10. proximity_entity  [0, 1]      Water proximity
  11. internal_stress   [0, 1]      Adam's stress
  12. fear_signal       [0, 1]      Phase 5 fear signal
  13. pleasure_signal   [0, 1]      Phase 5 pleasure signal
  14. pattern_confidence [0, 1]     Phase 5 pattern confidence
  15. biome_temp_offset [-1, 1]     How extreme this biome's temperature is
  16. biome_food_availability [0,1] How much food this biome typically provides
  17. biome_danger_level [0, 1]     How dangerous this biome is
  18. biome_shelter_available [0,1] How much shelter this biome provides
  19. weather_intensity [0, 1]      How intense the current weather is
  20. time_of_day       [0, 1]      Scaled time (0=midnight, 0.5=noon)
"""

import torch

# Biome feature encoding
BIOME_FEATURES = {
    "desert":  {"temp_offset": 0.8, "food_avail": 0.1, "danger_level": 0.4, "shelter_avail": 0.1},
    "forest":  {"temp_offset": 0.1, "food_avail": 0.8, "danger_level": 0.3, "shelter_avail": 0.7},
    "tundra":  {"temp_offset": -0.9, "food_avail": 0.1, "danger_level": 0.2, "shelter_avail": 0.2},
    "plains":  {"temp_offset": 0.0, "food_avail": 0.6, "danger_level": 0.2, "shelter_avail": 0.4},
    "mountain": {"temp_offset": -0.4, "food_avail": 0.3, "danger_level": 0.5, "shelter_avail": 0.5},
    "swamp":   {"temp_offset": 0.2, "food_avail": 0.5, "danger_level": 0.6, "shelter_avail": 0.2},
    "ocean":   {"temp_offset": -0.1, "food_avail": 0.4, "danger_level": 0.5, "shelter_avail": 0.0},
    "jungle":  {"temp_offset": 0.5, "food_avail": 0.9, "danger_level": 0.7, "shelter_avail": 0.5},
    "cave":    {"temp_offset": -0.2, "food_avail": 0.2, "danger_level": 0.5, "shelter_avail": 0.9},
    "volcano": {"temp_offset": 1.0, "food_avail": 0.0, "danger_level": 0.8, "shelter_avail": 0.1},
}

# Weather intensity encoding
WEATHER_INTENSITY = {
    "clear": 0.0, "rain": 0.3, "snow": 0.4, "storm": 0.8,
    "heatwave": 0.6, "fog": 0.3, "sandstorm": 0.7, "blizzard": 0.9,
}

INPUT_DIM = 21  # Updated from 15 to 21


def encode_sensory_input(world_state: dict, adam_stats: dict,
                        fear_signal: float = 0.0, pleasure_signal: float = 0.0,
                        pattern_confidence: float = 0.0) -> torch.Tensor:
    """
    Converts NAFS world state and Adam's stats into a 21-dimensional numerical vector.
    """
    # Default values
    default_world_state = {
        'temperature': 25.0, 'light_level': 0.5, 'smell_food': 0.0,
        'smell_danger': 0.0, 'sound_level': 0.0, 'wetness': 0.0,
        'proximity_entity': 0.0, 'touch_softness': 0.5,
        'biome': 'plains', 'weather': 'clear', 'time_of_day': 12,
    }
    default_adam_stats = {
        'health': 100.0, 'hunger': 0.0, 'energy': 100.0,
        'pain': 0.0, 'stress': 0.0
    }

    ws = {**default_world_state, **world_state}
    stats = {**default_adam_stats, **adam_stats}

    # --- Original 15 dimensions ---
    temperature_scaled = max(-1.0, min(1.0, (ws['temperature'] - 15.0) / 25.0))
    pain_scaled = max(0.0, min(1.0, stats['pain'] / 10.0))
    hunger_scaled = max(0.0, min(1.0, stats['hunger'] / 100.0))
    energy_scaled = max(0.0, min(1.0, stats['energy'] / 100.0))
    light_level_scaled = max(0.0, min(1.0, ws['light_level']))
    smell_food_scaled = max(0.0, min(1.0, ws['smell_food']))
    smell_danger_scaled = max(0.0, min(1.0, ws['smell_danger']))
    sound_level_scaled = max(0.0, min(1.0, ws['sound_level']))
    wetness_scaled = max(0.0, min(1.0, ws['wetness']))
    touch_softness_scaled = max(0.0, min(1.0, ws.get('touch_softness', 0.5)))
    proximity_entity_scaled = max(0.0, min(1.0, ws['proximity_entity']))
    internal_stress_scaled = max(0.0, min(1.0, stats['stress'] / 100.0))
    fear_signal_scaled = max(0.0, min(1.0, fear_signal))
    pleasure_signal_scaled = max(0.0, min(1.0, pleasure_signal))
    pattern_confidence_scaled = max(0.0, min(1.0, pattern_confidence))

    # --- New biome dimensions (16-18) ---
    biome_name = ws.get('biome', 'plains')
    biome_features = BIOME_FEATURES.get(biome_name, BIOME_FEATURES['plains'])
    biome_temp_offset = biome_features['temp_offset']
    biome_food_avail = biome_features['food_avail']
    biome_danger_level = biome_features['danger_level']
    biome_shelter_avail = biome_features['shelter_avail']

    # --- Weather dimension (19) ---
    weather_name = ws.get('weather', 'clear')
    weather_intensity = WEATHER_INTENSITY.get(weather_name, 0.0)

    # --- Time of day dimension (20) ---
    time_of_day = ws.get('time_of_day', 12)
    time_scaled = time_of_day / 24.0  # 0.0 = midnight, 0.5 = noon

    # Combine into 21-dimensional tensor
    sensory_vector = torch.tensor([
        temperature_scaled,
        pain_scaled,
        hunger_scaled,
        energy_scaled,
        light_level_scaled,
        smell_food_scaled,
        smell_danger_scaled,
        sound_level_scaled,
        wetness_scaled,
        touch_softness_scaled,
        proximity_entity_scaled,
        internal_stress_scaled,
        fear_signal_scaled,
        pleasure_signal_scaled,
        pattern_confidence_scaled,
        biome_temp_offset,
        biome_food_avail,
        biome_danger_level,
        biome_shelter_avail,
        weather_intensity,
        time_scaled,
    ], dtype=torch.float32)

    return sensory_vector
