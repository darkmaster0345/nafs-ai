"""
Nafs AI — PPO Brain (Training Run 3)
A from-scratch neural network for Adam's primitive consciousness.

No pretrained models. No LLMs. No external knowledge.
Adam learns purely through sensory experience — pain, hunger, warmth, fear.

This brain implements Proximal Policy Optimization (PPO) with:
  - Entropy bonus to prevent mono-behavior collapse
  - Action diversity penalty to discourage repetitive actions
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical

from config import PPO_CONFIG, ACTION_NAMES


# ═══════════════════════════════════════════════════════════════════════════════
# Neural Network Architecture — Actor-Critic
# ═══════════════════════════════════════════════════════════════════════════════

class ActorCritic(nn.Module):
    """
    A small actor-critic network.

    The actor outputs action probabilities (policy).
    The critic outputs a state value estimate.

    Input:  observation vector (Adam's sensory state)
    Output: action logits (actor), state value (critic)

    Architecture is intentionally small (~55MB RAM constraint from prior runs).
    No pretrained weights. Random initialization only.
    """

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()

        # Shared feature extractor — processes raw sensory input
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )

        # Actor head — maps features to action probabilities
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, action_dim),
        )

        # Critic head — maps features to state value estimate
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Initialize weights with small values — no pretrained knowledge
        self._init_weights()

    def _init_weights(self):
        """Xavier initialization with small gains — start from ignorance."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.5)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor):
        features = self.shared(x)
        action_logits = self.actor(features)
        state_value = self.critic(features)
        return action_logits, state_value

    def get_action(self, obs: torch.Tensor):
        """Sample an action from the current policy and return info for training."""
        action_logits, state_value = self.forward(obs)

        # Create categorical distribution for sampling
        dist = Categorical(logits=action_logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        return action.item(), log_prob, entropy, state_value

    def evaluate(self, obs: torch.Tensor, actions: torch.Tensor):
        """Evaluate actions for given observations (used during PPO update)."""
        action_logits, state_value = self.forward(obs)

        dist = Categorical(logits=action_logits)
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()

        return log_prob, entropy, state_value


# ═══════════════════════════════════════════════════════════════════════════════
# Rollout Buffer — Stores experience for PPO updates
# ═══════════════════════════════════════════════════════════════════════════════

class RolloutBuffer:
    """Stores trajectories collected during environment interaction."""

    def __init__(self):
        self.observations = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []

    def store(self, obs, action, log_prob, reward, value, done):
        self.observations.append(obs)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)

    def clear(self):
        self.observations = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []

    def __len__(self):
        return len(self.observations)


# ═══════════════════════════════════════════════════════════════════════════════
# PPO Agent — The Brain
# ═══════════════════════════════════════════════════════════════════════════════

class PPOAgent:
    """
    PPO agent with entropy bonus and action diversity penalty.

    This is Adam's brain. It starts with no knowledge and learns
    through trial and error, driven only by sensory feedback.
    """

    def __init__(self, obs_dim: int, action_dim: int):
        cfg = PPO_CONFIG

        self.device = torch.device("cpu")  # CPU-only for 55MB constraint
        self.actor_critic = ActorCritic(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=cfg["hidden_dim"],
        ).to(self.device)

        self.optimizer = optim.Adam(
            self.actor_critic.parameters(),
            lr=cfg["learning_rate"],
            eps=1e-5,
        )

        self.buffer = RolloutBuffer()

        # PPO hyperparameters
        self.gamma = cfg["gamma"]
        self.gae_lambda = cfg["gae_lambda"]
        self.clip_epsilon = cfg["clip_epsilon"]
        self.entropy_coef = cfg["entropy_coef"]
        self.value_loss_coef = cfg["value_loss_coef"]
        self.max_grad_norm = cfg["max_grad_norm"]
        self.ppo_epochs = cfg["ppo_epochs"]
        self.mini_batch_size = cfg["mini_batch_size"]

        # Action diversity tracking
        self.action_history = []
        self.diversity_penalty = cfg["diversity_penalty"]
        self.diversity_window = cfg["diversity_window"]
        self.action_history_max = cfg["action_history_max"]

        # Training statistics
        self.total_updates = 0

    def select_action(self, observation: np.ndarray) -> tuple:
        """
        Select an action based on current observation.

        Returns: (action_index, log_prob, value)
        """
        obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action, log_prob, entropy, value = self.actor_critic.get_action(obs_tensor)

        return action, log_prob.item(), value.item()

    def compute_diversity_penalty(self, action_index: int) -> float:
        """
        Compute a reward penalty if Adam repeats the same action
        too many times in a row. This is Fix 2 from Training Run 3.
        """
        self.action_history.append(action_index)
        if len(self.action_history) > self.action_history_max:
            self.action_history.pop(0)

        # Check if the last N actions are all the same
        if len(self.action_history) >= self.diversity_window:
            recent = self.action_history[-self.diversity_window:]
            if len(set(recent)) == 1:
                return self.diversity_penalty

        return 0.0

    def compute_rewards_to_go(self, rewards, dones, last_value: float) -> list:
        """Compute discounted returns (rewards-to-go)."""
        returns = []
        R = last_value

        for i in reversed(range(len(rewards))):
            if dones[i]:
                R = 0
            R = rewards[i] + self.gamma * R
            returns.insert(0, R)

        return returns

    def compute_gae(self, rewards, values, dones, last_value: float) -> tuple:
        """
        Compute Generalized Advantage Estimation (GAE).
        Returns advantages and returns.
        """
        advantages = []
        gae = 0

        # Add the last value estimate for bootstrapping
        values_with_last = values + [last_value]

        for i in reversed(range(len(rewards))):
            delta = rewards[i] + self.gamma * values_with_last[i + 1] * (1 - dones[i]) - values_with_last[i]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[i]) * gae
            advantages.insert(0, gae)

        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = advantages + torch.FloatTensor(values).to(self.device)

        return advantages, returns

    def update(self, last_value: float = 0.0) -> dict:
        """
        Perform a PPO update using collected experience.

        Includes:
        - Clipped surrogate objective (policy loss)
        - Value function loss
        - Entropy bonus (Fix 1 from Training Run 3)

        Returns training statistics.
        """
        if len(self.buffer) == 0:
            return {}

        # Compute advantages using GAE
        advantages, returns = self.compute_gae(
            self.buffer.rewards,
            self.buffer.values,
            self.buffer.dones,
            last_value,
        )

        # Convert buffer to tensors
        obs_tensor = torch.FloatTensor(np.array(self.buffer.observations)).to(self.device)
        actions_tensor = torch.LongTensor(self.buffer.actions).to(self.device)
        old_log_probs = torch.FloatTensor(self.buffer.log_probs).to(self.device)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO update for multiple epochs
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        total_loss = 0
        num_updates = 0

        dataset_size = len(self.buffer)
        indices = np.arange(dataset_size)

        for _ in range(self.ppo_epochs):
            np.random.shuffle(indices)

            for start in range(0, dataset_size, self.mini_batch_size):
                end = min(start + self.mini_batch_size, dataset_size)
                batch_idx = indices[start:end]

                # Get batch data
                batch_obs = obs_tensor[batch_idx]
                batch_actions = actions_tensor[batch_idx]
                batch_old_log_probs = old_log_probs[batch_idx]
                batch_advantages = advantages[batch_idx]
                batch_returns = returns[batch_idx]

                # Evaluate current policy on batch
                new_log_probs, entropy, state_values = self.actor_critic.evaluate(
                    batch_obs, batch_actions
                )

                state_values = state_values.squeeze()

                # ── Policy Loss (Clipped Surrogate) ─────────────────────────
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # ── Value Loss ──────────────────────────────────────────────
                value_loss = nn.MSELoss()(state_values, batch_returns)

                # ── Entropy Bonus (Fix 1 — prevents mono-behavior collapse) ─
                entropy_loss = -self.entropy_coef * entropy.mean()

                # ── Total Loss ──────────────────────────────────────────────
                loss = policy_loss + self.value_loss_coef * value_loss + entropy_loss

                # Gradient step
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                total_loss += loss.item()
                num_updates += 1

        # Clear buffer after update
        self.buffer.clear()
        self.total_updates += 1

        return {
            "policy_loss": total_policy_loss / max(num_updates, 1),
            "value_loss": total_value_loss / max(num_updates, 1),
            "entropy": total_entropy / max(num_updates, 1),
            "total_loss": total_loss / max(num_updates, 1),
            "updates": self.total_updates,
        }

    def save(self, path: str):
        """Save model weights."""
        torch.save({
            "model_state": self.actor_critic.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "total_updates": self.total_updates,
        }, path)
        print(f"[Brain] Model saved to {path}")

    def load(self, path: str):
        """Load model weights."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.actor_critic.load_state_dict(checkpoint["model_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.total_updates = checkpoint.get("total_updates", 0)
        print(f"[Brain] Model loaded from {path} (updates: {self.total_updates})")


# ═══════════════════════════════════════════════════════════════════════════════
# Observation Builder — Converts Adam's state to neural network input
# ═══════════════════════════════════════════════════════════════════════════════

def build_observation(adam, world) -> np.ndarray:
    """
    Convert Adam's internal state and world context into a numeric
    observation vector for the neural network.

    This is Adam's raw sensory experience — no words, no concepts,
    just raw numbers representing bodily sensations and environmental cues.
    """
    # Normalize Adam's internal state to [0, 1]
    obs = [
        adam.health / 100.0,       # 0: health (0=dead, 1=full)
        adam.hunger / 100.0,       # 1: hunger (0=full, 1=starving)
        adam.energy / 100.0,       # 2: energy (0=exhausted, 1=full)
        adam.stress / 100.0,       # 3: stress (0=calm, 1=terrified)
    ]

    # World context — one-hot encoded for categorical, normalized for numeric
    time_of_day_map = {
        "dawn": 0, "morning": 1, "midday": 2, "afternoon": 3,
        "dusk": 4, "night": 5, "deep_night": 6,
    }
    weather_map = {
        "clear": 0, "cloudy": 1, "raining": 2, "cold_wind": 3, "hot": 4,
    }

    # Time of day as normalized value
    time_idx = time_of_day_map.get(world.time_of_day, 0)
    obs.append(time_idx / 6.0)  # 4: time of day

    # Weather as normalized value
    weather_idx = weather_map.get(world.weather, 0)
    obs.append(weather_idx / 4.0)  # 5: weather

    # Temperature normalized (rough range: 0-35 C)
    obs.append(min(max(world.temperature / 35.0, 0), 1))  # 6: temperature

    # Event category detection (binary flags from world event text)
    event_lower = world.last_event.lower() if world.last_event else ""

    # Is there food nearby?
    food_keywords = ["berry", "fruit", "sweet smell", "soft thing", "hangs"]
    obs.append(float(any(k in event_lower for k in food_keywords)))  # 7: food_present

    # Is there water nearby?
    water_keywords = ["water", "river", "wet", "still water"]
    obs.append(float(any(k in event_lower for k in water_keywords)))  # 8: water_present

    # Is there danger?
    danger_keywords = ["sharp", "bite", "large and dark", "moves in the tall", "loud crack", "boom"]
    obs.append(float(any(k in event_lower for k in danger_keywords)))  # 9: danger_present

    # Is there shelter?
    shelter_keywords = ["dark opening", "dry place", "rock", "protected"]
    obs.append(float(any(k in event_lower for k in shelter_keywords)))  # 10: shelter_present

    # Is it dark?
    dark_keywords = ["no light", "cannot see", "dark"]
    obs.append(float(any(k in event_lower for k in dark_keywords)))  # 11: is_dark

    # Is it cold?
    cold_keywords = ["cold", "freezing", "frost"]
    obs.append(float(any(k in event_lower for k in cold_keywords)))  # 12: is_cold

    # Is Adam tired?
    obs.append(1.0 if adam.energy < 20 else 0.0)  # 13: exhausted

    # Is Adam starving?
    obs.append(1.0 if adam.hunger > 70 else 0.0)  # 14: starving

    return np.array(obs, dtype=np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
# Reward Function — What Adam feels
# ═══════════════════════════════════════════════════════════════════════════════

def compute_reward(adam, outcome: dict, action_index: int) -> float:
    """
    Compute reward based on Adam's sensory experience.

    This is the core of what Adam "feels" — not what we tell him to do,
    but the raw sensory consequences of his actions.

    Base reward structure:
    - Small positive reward for staying alive each tick
    - Positive reward for health/stress improvements
    - Negative reward for health/stress damage
    - No explicit reward for any specific action — emergent behavior only

    Plus the diversity penalty (Fix 2) to prevent mono-behavior collapse.
    """
    reward = 0.0

    # ── Survival bonus ─────────────────────────────────────────────────────
    # Adam gets a small positive signal just for being alive
    reward += 0.1

    # ── Health changes ─────────────────────────────────────────────────────
    health_delta = outcome.get("health_delta", 0)
    if health_delta > 0:
        reward += 0.3  # feeling better is rewarding
    elif health_delta < 0:
        reward -= 0.5  # pain is punishing

    # ── Hunger relief ──────────────────────────────────────────────────────
    hunger_delta = outcome.get("hunger_delta", 0)
    if hunger_delta < 0:
        reward += 0.3  # eating/drinking feels good
    elif hunger_delta > 0:
        reward -= 0.2  # getting hungrier feels bad

    # ── Energy recovery ────────────────────────────────────────────────────
    energy_delta = outcome.get("energy_delta", 0)
    if energy_delta > 0:
        reward += 0.2  # recovering energy feels good
    elif energy_delta < 0:
        reward -= 0.1  # spending energy is mildly costly

    # ── Critical state penalties ───────────────────────────────────────────
    if adam.hunger >= 90:
        reward -= 0.3  # starving is painful
    if adam.energy <= 10:
        reward -= 0.2  # exhaustion is painful
    if adam.health <= 30:
        reward -= 0.3  # near death is terrifying
    if adam.stress >= 60:
        reward -= 0.2  # high stress is uncomfortable

    # ── Death penalty ──────────────────────────────────────────────────────
    if not adam.is_alive:
        reward -= 1.0  # dying is the worst outcome

    return reward
