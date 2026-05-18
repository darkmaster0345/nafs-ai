
import random

class AdamStats:
    def __init__(self):
        self.health = 100.0
        self.hunger = 0.0  # 0 = full, 100 = starving
        self.energy = 100.0 # 0 = exhausted, 100 = full
        self.stress = 0.0  # 0 = calm, 100 = high stress
        self.pain = 0.0    # 0 = none, 10 = severe

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

class WorldSim:
    ACTIONS = ["EXPLORE", "EAT", "DRINK", "SLEEP", "HIDE", "MOVE", "FLEE", "IDLE"]

    def __init__(self):
        self.time_of_day = 0 # 0-23, represents hours
        self.weather = 'clear'
        self.temperature = 20.0 # Celsius
        self.world_events = {
            'food_nearby': 0.0, # 0.0 to 1.0
            'danger_nearby': 0.0,
            'water_nearby': 0.0,
            'shelter_nearby': 0.0
        }
        self.adam_stats = AdamStats()
        self.current_tick = 0

        # Per-episode randomization parameters — vary each episode
        # so Adam encounters genuinely different worlds every time
        self.episode_seed = None  # Set in reset()
        self.hunger_rate = 0.5    # Base hunger increase per tick
        self.energy_drain = 0.7   # Base energy drain per tick
        self.danger_intensity = 1.0  # Multiplier for danger damage
        self.food_abundance = 1.0    # Multiplier for food spawn rate

    def reset(self):
        # Randomize initial conditions each episode — Adam wakes up in
        # different circumstances every time, preventing pattern collapse
        # and ensuring diverse experience for vocabulary discovery.

        # Per-episode world parameters — vary the "physics" of each world
        # so Adam can't overfit to a single static environment
        self.hunger_rate = random.uniform(0.3, 0.8)     # How fast Adam gets hungry
        self.energy_drain = random.uniform(0.4, 1.0)    # How fast Adam gets tired
        self.danger_intensity = random.uniform(0.5, 2.0)  # How much danger hurts
        self.food_abundance = random.uniform(0.6, 1.5)   # How often food appears

        self.time_of_day = random.randint(0, 23)  # Random time of day
        self.weather = random.choice(['clear', 'clear', 'clear', 'rainy', 'cold', 'hot'])  # Weighted toward clear
        self.temperature = random.uniform(5.0, 35.0)  # Random starting temperature
        self.world_events = {
            'food_nearby': 1.0 if random.random() < 0.3 else 0.0,     # 30% chance food nearby at start
            'danger_nearby': 1.0 if random.random() < 0.1 else 0.0,   # 10% chance danger nearby at start
            'water_nearby': 1.0 if random.random() < 0.25 else 0.0,   # 25% chance water nearby at start
            'shelter_nearby': 1.0 if random.random() < 0.15 else 0.0, # 15% chance shelter nearby at start
        }
        self.adam_stats = AdamStats()
        self.current_tick = 0
        return self._get_world_state(), self.adam_stats.to_dict()

    def _get_world_state(self):
        # Convert time_of_day to light_level (e.g., 6-18 day, rest night)
        light_level = 0.0
        if 6 <= self.time_of_day < 18:
            light_level = 1.0 # Day
        elif 5 <= self.time_of_day < 6 or 18 <= self.time_of_day < 19:
            light_level = 0.5 # Dawn/Dusk
        # Else 0.0 (Night)

        # Map weather to sensory inputs
        wetness = 0.0
        if self.weather == 'rainy':
            wetness = 0.8

        # Simulate sound level based on events/time
        sound_level = 0.1 # Base ambient sound
        if self.world_events['danger_nearby'] > 0.5:
            sound_level = random.uniform(0.6, 0.9)
        elif self.world_events['food_nearby'] > 0.5:
            sound_level = random.uniform(0.2, 0.4)

        # Simulate touch softness (e.g., if Adam is hiding or sleeping)
        touch_softness = 0.5 # Default
        # if action == 'SLEEP' or action == 'HIDE': # This needs to be based on previous action or current state
        #    touch_softness = 0.8

        return {
            'temperature': self.temperature,
            'light_level': light_level,
            'smell_food': self.world_events['food_nearby'],
            'smell_danger': self.world_events['danger_nearby'],
            'sound_level': sound_level,
            'wetness': wetness,
            'proximity_entity': self.world_events['water_nearby'], # Re-using for water for now
            'touch_softness': touch_softness # Placeholder, will be updated by actions
        }

    def step(self, action: str):
        reward = 0.0
        done = False

        # --- Update world state --- 
        self.current_tick += 1
        self.time_of_day = (self.time_of_day + 1) % 24 # Advance time by 1 hour

        # Random weather changes
        if random.random() < 0.05: # 5% chance to change weather
            self.weather = random.choice(['clear', 'rainy', 'cold', 'hot'])
        
        # Temperature fluctuation based on time of day and weather
        if 0 <= self.time_of_day < 6 or 18 <= self.time_of_day < 24: # Night/Evening
            self.temperature -= random.uniform(0.5, 2.0)
        else: # Day
            self.temperature += random.uniform(0.5, 2.0)
        self.temperature = max(-10.0, min(40.0, self.temperature)) # Keep temp in reasonable range

        # Random world events — probabilities vary to create diverse experiences
        # Food spawn rate is modified by per-episode food_abundance
        food_prob = random.uniform(0.08, 0.15) * self.food_abundance
        self.world_events['food_nearby'] = 1.0 if random.random() < food_prob else 0.0
        self.world_events['danger_nearby'] = 1.0 if random.random() < random.uniform(0.03, 0.10) else 0.0
        self.world_events['water_nearby'] = 1.0 if random.random() < random.uniform(0.10, 0.20) else 0.0
        self.world_events['shelter_nearby'] = 1.0 if random.random() < random.uniform(0.05, 0.12) else 0.0

        # --- Apply passive stats changes --- 
        self.adam_stats.hunger += self.hunger_rate  # Adam gets hungrier over time (varies per episode)
        self.adam_stats.energy -= self.energy_drain  # Adam loses energy over time (varies per episode)
        self.adam_stats.stress += 0.1 # Adam gets a little stressed over time
        self.adam_stats.health -= 0.05 # Minor health decay

        # Environmental effects
        if self.weather == 'cold':
            self.adam_stats.energy -= 0.5
            self.adam_stats.health -= 0.1
            self.adam_stats.stress += 0.2
        elif self.weather == 'hot':
            self.adam_stats.energy -= 0.3
            self.adam_stats.hunger += 0.2
            self.adam_stats.stress += 0.1
        if self.adam_stats.hunger >= 80: # Very hungry
            self.adam_stats.health -= 0.5
            self.adam_stats.stress += 0.5
        if self.adam_stats.energy <= 20: # Very tired
            self.adam_stats.health -= 0.2
            self.adam_stats.stress += 0.3

        # --- Process Adam's action --- 
        if action == "EXPLORE":
            reward += 0.1 # explore=+0.1
            self.adam_stats.energy -= 2.0
            self.adam_stats.stress += 0.1
            if random.random() < 0.2: # Chance to find food/water/shelter
                if random.random() < 0.5: self.world_events['food_nearby'] = 1.0
                else: self.world_events['water_nearby'] = 1.0
            if random.random() < 0.05: # Chance to encounter danger
                self.world_events['danger_nearby'] = 1.0
                self.adam_stats.pain = random.uniform(1, 5) * self.danger_intensity
                self.adam_stats.health -= self.adam_stats.pain
                reward -= 1.5 # pain=-1.5

        elif action == "EAT":
            if self.world_events['food_nearby'] > 0.5:
                self.adam_stats.hunger = max(0.0, self.adam_stats.hunger - random.uniform(20, 40))
                reward += 1.0 # eat_success=+1.0
                self.world_events['food_nearby'] = 0.0 # Food consumed
            else:
                self.adam_stats.stress += 0.5 # Failed to eat
                reward -= 0.1
            self.adam_stats.energy -= 0.5

        elif action == "DRINK":
            if self.world_events['water_nearby'] > 0.5:
                self.adam_stats.hunger = max(0.0, self.adam_stats.hunger - random.uniform(5, 15)) # Drinking helps with hunger a bit
                self.adam_stats.stress = max(0.0, self.adam_stats.stress - 5.0)
                reward += 0.2
                self.world_events['water_nearby'] = 0.0 # Water consumed
            else:
                self.adam_stats.stress += 0.2
                reward -= 0.05
            self.adam_stats.energy -= 0.3

        elif action == "SLEEP":
            if self.world_events['shelter_nearby'] > 0.5 and (self.time_of_day >= 18 or self.time_of_day < 6): # Better sleep with shelter at night
                self.adam_stats.energy = min(100.0, self.adam_stats.energy + random.uniform(30, 50))
                self.adam_stats.health = min(100.0, self.adam_stats.health + random.uniform(1, 3))
                self.adam_stats.stress = max(0.0, self.adam_stats.stress - random.uniform(10, 20))
                reward += 0.3 # sleep_restore=+0.3 (reduced from 0.5)
            else:
                self.adam_stats.energy = min(100.0, self.adam_stats.energy + random.uniform(10, 20))
                self.adam_stats.stress = max(0.0, self.adam_stats.stress - random.uniform(2, 5))
                # Sleeping during daytime without shelter is slightly wasteful
                if 6 <= self.time_of_day < 18:
                    reward -= 0.05 # Small penalty for sleeping during day
                else:
                    reward += 0.05 # Tiny reward for sleeping at night without shelter
            self.adam_stats.hunger += 1.0 # Still gets hungry while sleeping

        elif action == "HIDE":
            self.adam_stats.energy -= 1.0
            self.adam_stats.stress += 0.1
            if self.world_events['danger_nearby'] > 0.5:
                if random.random() < 0.7: # 70% chance to avoid danger
                    self.world_events['danger_nearby'] = 0.0
                    self.adam_stats.stress = max(0.0, self.adam_stats.stress - 5.0)
                    reward += 0.5 # Hiding from danger is important
                else:
                    self.adam_stats.pain = random.uniform(3, 8) * self.danger_intensity
                    self.adam_stats.health -= self.adam_stats.pain
                    reward -= 1.5 # pain=-1.5
            else:
                reward -= 0.1 # Small penalty for hiding when no danger (wastes time)

        elif action == "MOVE":
            self.adam_stats.energy -= 1.5
            self.adam_stats.hunger += 0.1
            self.adam_stats.stress += 0.05
            if random.random() < 0.1: # Chance to find something new
                self.world_events['food_nearby'] = 1.0 if random.random() < 0.5 else 0.0
                self.world_events['water_nearby'] = 1.0 if random.random() < 0.5 else 0.0
            if random.random() < 0.03: # Small chance to encounter danger
                self.world_events['danger_nearby'] = 1.0

        elif action == "FLEE":
            self.adam_stats.energy -= 3.0
            self.adam_stats.stress += 0.5
            if self.world_events['danger_nearby'] > 0.5:
                if random.random() < 0.9: # High chance to escape danger
                    self.world_events['danger_nearby'] = 0.0
                    self.adam_stats.stress = max(0.0, self.adam_stats.stress - 10.0)
                    reward += 0.3 # flee_danger=+0.3
                else:
                    self.adam_stats.pain = random.uniform(5, 10) * self.danger_intensity
                    self.adam_stats.health -= self.adam_stats.pain
                    reward -= 1.5 # pain=-1.5
            else:
                reward -= 0.2 # Penalty for fleeing when no danger

        elif action == "IDLE":
            self.adam_stats.energy += 0.5 # Rest a bit
            self.adam_stats.hunger += 0.2
            self.adam_stats.stress = max(0.0, self.adam_stats.stress - 0.5)
            reward -= 0.05 # Small penalty for doing nothing (encourages action)

        # --- Clamp stats to valid ranges --- 
        self.adam_stats.health = max(0.0, min(100.0, self.adam_stats.health))
        self.adam_stats.hunger = max(0.0, min(100.0, self.adam_stats.hunger))
        self.adam_stats.energy = max(0.0, min(100.0, self.adam_stats.energy))
        self.adam_stats.stress = max(0.0, min(100.0, self.adam_stats.stress))
        self.adam_stats.pain = max(0.0, min(10.0, self.adam_stats.pain))

        # --- Check for death condition --- 
        if not self.adam_stats.is_alive():
            reward -= 5.0 # death=-5.0
            done = True

        # --- Add survival reward --- 
        if not done:
            reward += 0.01 # survive_tick=+0.01

        next_world_state = self._get_world_state()
        next_adam_stats = self.adam_stats.to_dict()

        return next_world_state, next_adam_stats, reward, done

if __name__ == '__main__':
    env = WorldSim()
    world_state, adam_stats = env.reset()
    print("Initial World State:", world_state)
    print("Initial Adam Stats:", adam_stats)

    done = False
    episode_reward = 0
    steps = 0
    while not done and steps < 100:
        action = random.choice(WorldSim.ACTIONS)
        world_state, adam_stats, reward, done = env.step(action)
        episode_reward += reward
        steps += 1
        print(f"\nStep {steps}, Action: {action}")
        print("  World State:", {k: round(v, 2) for k, v in world_state.items()})
        print("  Adam Stats:", {k: round(v, 2) for k, v in adam_stats.items()})
        print("  Reward:", round(reward, 2))
        print("  Done:", done)

    print(f"\nEpisode finished after {steps} steps with total reward {round(episode_reward, 2)}")
    if not env.adam_stats.is_alive():
        print("Adam died.")
    else:
        print("Adam survived.")
