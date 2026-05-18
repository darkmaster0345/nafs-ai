import random
import time

# ── World Event Templates ─────────────────────────────────────────────────────
# Described as pure sensation — no named objects yet.
# Adam will name these things himself over time.

WORLD_EVENTS = [
    # Hunger / Survival
    "Your stomach twists. Something is wrong inside. Empty feeling.",
    "You notice a sweet smell coming from low to the ground, nearby.",
    "A soft thing is on the ground in front of you. It is a berry.",
    "You see a tall thing with many arms reaching upward. Something hangs from it.",
    "Water moves fast nearby. Loud. Cold air comes from it.",
    "Still water. Flat. You can see yourself in it.",
    "You find a fruit on the ground. It has a sweet smell.",

    # Environment
    "The light above is bright and hot. Ground is warm.",
    "The light is fading. Air getting colder.",
    "No light. Cannot see. Sounds in the dark.",
    "Water falls from above. Cold. Everything getting wet.",
    "Strong air pushing against you. Hard to stand.",
    "Ground shakes slightly. Then stops.",
    "Smoke smell from far away. Orange glow in the distance.",

    # Danger
    "Something moves in the tall grass. You cannot see what.",
    "A loud crack sound from above. Very bright light. Then loud boom.",
    "Something large and dark is watching you from far away. It has not moved.",
    "You step on something sharp. Pain shoots through your foot.",
    "Something bit you. Small pain. Itching.",

    # Discovery
    "You find a dark opening in the rock wall. Cool air comes out of it.",
    "The ground here is softer. Easier to walk.",
    "You find a dry place under a large flat rock. Protected from wind.",
    "A small creature runs past your feet. Too fast to touch.",
    "You hear a sound like yours. But far away. Very far.",

    # Rest / Recovery
    "You are lying down. Ground is hard but you are still.",
    "Warmth on your face. The light above.",
    "You are very tired. Eyes heavy.",
]

WEATHER_POOL = [
    "clear", "clear", "clear", "clear",
    "cloudy", "cloudy",
    "raining",
    "cold_wind",
    "hot"
]

TIMES_OF_DAY = ["dawn", "morning", "midday", "afternoon", "dusk", "night", "deep_night"]


class World:
    food_keywords = [
        "berry", "berries", "hangs", "hanging",
        "sweet", "smell", "sweet smell", "soft thing",
        "fruit", "food", "ripe"
    ]

    def __init__(self):
        self.tick         = 0
        self.day          = 1
        self.time_of_day  = "dawn"
        self.weather      = "clear"
        self.temperature  = 18   # celsius feeling
        self._time_index  = 0
        self.weather_ticks = 0
        self.last_event   = ""

    def tick_forward(self):
        self.tick += 1
        self.weather_ticks += 1

        # Advance time of day
        self._time_index = (self._time_index + 1) % len(TIMES_OF_DAY)
        self.time_of_day = TIMES_OF_DAY[self._time_index]

        if self._time_index == 0:
            self.day += 1

        # Random weather shift (Only if minimum 3 ticks per weather type)
        if self.weather_ticks >= 3:
            if random.random() < 0.1:
                self.weather = random.choice(WEATHER_POOL)
                self.weather_ticks = 0

        # Temperature follows time of day loosely
        temp_map = {
            "dawn": 12, "morning": 16, "midday": 28,
            "afternoon": 25, "dusk": 18, "night": 10, "deep_night": 7
        }
        self.temperature = temp_map[self.time_of_day] + random.randint(-3, 3)

    def get_event(self) -> str:
        """Return a sensory description of what Adam currently experiences."""
        base_event = random.choice(WORLD_EVENTS)

        # Append environmental context
        context_parts = []

        if self.weather == "raining":
            context_parts.append("Water falls from above.")
        if self.weather == "cold_wind":
            context_parts.append("Cold air pushes hard.")
        if self.weather == "hot":
            context_parts.append("Air is heavy and very warm.")
        if self.time_of_day == "night" or self.time_of_day == "deep_night":
            context_parts.append("Cannot see much. Dark.")
        if self.time_of_day == "dawn":
            context_parts.append("Light slowly coming back.")

        event_str = base_event
        if context_parts:
            event_str = base_event + " " + " ".join(context_parts)

        self.last_event = event_str
        return self.last_event

    def apply_action(self, action: str, target: str, adam_state: dict) -> dict:
        """
        Apply Adam's action to the world.
        Returns outcome dict: {health_delta, hunger_delta, energy_delta, outcome_text}
        """
        outcome = {
            "health_delta":  0,
            "hunger_delta":  0,
            "energy_delta":  0,
            "outcome_text":  "",
        }

        action = action.upper()

        if action == "EAT":
            # Check if any keyword in food_keywords exists in the most recent world event text
            found_food = any(keyword in self.last_event.lower() for keyword in self.food_keywords)

            if found_food:
                roll = random.random()
                if roll < 0.9:
                    outcome["hunger_delta"] = -35
                    outcome["energy_delta"] = +10
                    outcome["outcome_text"] = "Something warm spreads inside. The empty feeling gets smaller."
                else:
                    outcome["health_delta"] = -15
                    outcome["hunger_delta"] = +5
                    outcome["outcome_text"] = "Pain. Stomach hurts badly now."
            else:
                outcome["hunger_delta"] = 0
                outcome["outcome_text"] = "Nothing here to eat."

        elif action == "SLEEP":
            if self.time_of_day in ["night", "deep_night", "dawn"]:
                outcome["energy_delta"]  = +40
                outcome["health_delta"]  = +5
                outcome["outcome_text"]  = "Darkness. Then light again. Body feels less heavy."
            else:
                outcome["energy_delta"]  = +15
                outcome["outcome_text"]  = "Short rest. Slightly better."

        elif action == "DRINK":
            outcome["hunger_delta"]  = -10
            outcome["energy_delta"]  = +5
            outcome["outcome_text"]  = "The wet feeling in mouth."

        elif action == "HIDE":
            if self.weather in ["raining", "cold_wind"]:
                outcome["health_delta"]  = +3
                outcome["outcome_text"]  = "The pushing cold stopped."
            else:
                outcome["outcome_text"]  = "You wait. Still. Nothing happens."

        elif action == "EXPLORE":
            # Chance of finding something
            if random.random() > 0.5:
                outcome["outcome_text"]  = random.choice(WORLD_EVENTS)
            else:
                outcome["outcome_text"]  = "You moved. New ground under feet. Nothing different yet."

        elif action == "MOVE":
            outcome["energy_delta"]  = -3
            outcome["outcome_text"]  = f"You moved toward {target}." if target else "You moved."

        elif action == "INTERACT":
            outcome["outcome_text"]  = f"You touched {target}. Not sure what happened." if target else "You reached out."

        elif action == "IDLE":
            outcome["outcome_text"]  = "You stayed still. Time passed."

        return outcome

    def status_line(self) -> str:
        return f"Day {self.day} — {self.time_of_day.replace('_', ' ').title()} — {self.weather.replace('_', ' ').title()}"
