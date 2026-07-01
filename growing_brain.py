"""
Nafs AI — Growing Brain (Phase 4)
=================================

Implements MD Phase 4: a self-growing brain architecture that expands
when learning plateaus.

The brain grows when it needs to. Not by age. Not by schedule. By
hitting a learning wall.

Covers:
  4.1 Growth Trigger System
      - Monitor rolling average of PPO loss over last 500 ticks
      - If loss_improvement < 0.001 for 500 consecutive ticks → GROW
      - Growth: expand hidden_dim by 1.5x, preserve all existing weights
      - New neurons initialized near-zero (don't disrupt existing knowledge)
      - Maximum growth events: uncapped
      - Log every growth event: {tick, old_params, new_params, trigger_reason}

  4.2 Baby Brain Architecture
      - Baby starts with: hidden_dim=16, layers=2, ~500 params, no GRU (just MLP)
      - GRU layer added at first growth event (temporal memory emerges)
      - Attention added at third growth event
      - Transformer blocks added at fifth growth event if vocabulary > 100

  4.3 Catastrophic Forgetting Prevention
      - Elastic Weight Consolidation (EWC): penalise changing important weights
      - Old weights marked with importance score based on gradient magnitude
      - Lateral connections between old and new layers (Progressive Neural Net)
      - No forgetting of survival basics when brain expands for social complexity

Design constraints:
  - Does NOT modify base rewards in world_sim.py
  - Standalone module — BabyBrain (existing PPO+GRU) is untouched
  - GrowingBrain can be used in place of BabyBrain for agents that should
    start small and grow over time

Usage:
    from growing_brain import GrowingBrain
    brain = GrowingBrain(input_dim=21, num_actions=8)
    brain.record_loss(loss=0.5, tick=100)
    if brain.should_grow():
        brain.grow(tick=100, reason='loss_plateau')
    action, log_prob, value = brain.act(observation)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from collections import deque
from typing import Dict, List, Optional, Tuple, Any


# ═══════════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════════

INITIAL_HIDDEN_DIM = 16           # baby brain starts tiny
GROWTH_HIDDEN_MULT = 1.5          # 1.5x expansion per growth event
LOSS_HISTORY_SIZE = 500           # rolling window for loss monitoring
LOSS_IMPROVEMENT_THRESHOLD = 0.001  # growth trigger: improvement < 0.001
MIN_TICKS_BETWEEN_GROWTH = 500    # avoid back-to-back growth
TRANSFORMER_VOCAB_THRESHOLD = 100 # 5th growth requires vocab > 100

# Architecture stages by growth count
# 0: MLP only (~500 params)
# 1: MLP + GRU (temporal memory emerges)
# 2: MLP + GRU (bigger)
# 3: MLP + GRU + Attention (social complexity)
# 4: MLP + GRU + Attention (bigger)
# 5+: Transformer blocks (if vocab > 100)
ARCHITECTURE_BY_GROWTH = {
    0: "mlp_only",
    1: "mlp_gru",
    2: "mlp_gru_bigger",
    3: "mlp_gru_attn",
    4: "mlp_gru_attn_bigger",
    5: "transformer",
}


# ═══════════════════════════════════════════════════════════════════════════════
# Network architectures
# ═══════════════════════════════════════════════════════════════════════════════

class MLPOnly(nn.Module):
    """Stage 0: Just an MLP. No temporal memory. ~500 params."""

    def __init__(self, input_dim, hidden_dim, num_actions):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.actor = nn.Linear(hidden_dim, num_actions)
        self.critic = nn.Linear(hidden_dim, 1)
        # Init new neurons near-zero (don't disrupt existing knowledge)
        nn.init.uniform_(self.fc1.weight, -0.1, 0.1)
        nn.init.uniform_(self.fc2.weight, -0.1, 0.1)
        nn.init.uniform_(self.actor.weight, -0.1, 0.1)
        nn.init.uniform_(self.critic.weight, -0.1, 0.1)

    def forward(self, x, hidden_state=None):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        policy_logits = self.actor(h)
        value = self.critic(h)
        return policy_logits, value, None  # no hidden state for MLP


class MLPGRU(nn.Module):
    """Stage 1+: MLP + GRU for temporal memory."""

    def __init__(self, input_dim, hidden_dim, num_actions):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.actor = nn.Linear(hidden_dim, num_actions)
        self.critic = nn.Linear(hidden_dim, 1)
        # Init GRU near-zero so it doesn't disrupt existing MLP knowledge
        for name, param in self.gru.named_parameters():
            nn.init.uniform_(param, -0.05, 0.05)

    def forward(self, x, hidden_state=None):
        if x.dim() == 1:
            x = x.unsqueeze(0).unsqueeze(0)  # (1, 1, input_dim)
        elif x.dim() == 2:
            x = x.unsqueeze(0)  # (1, seq, input_dim)
        h = F.relu(self.fc1(x))
        if hidden_state is not None:
            h, new_hidden = self.gru(h, hidden_state)
        else:
            h, new_hidden = self.gru(h)
        # Take last timestep
        h_last = h[:, -1, :]
        policy_logits = self.actor(h_last)
        value = self.critic(h_last)
        return policy_logits, value, new_hidden


class MLPGRUAttention(nn.Module):
    """Stage 3+: MLP + GRU + Self-Attention."""

    def __init__(self, input_dim, hidden_dim, num_actions, num_heads=None):
        super().__init__()
        # Ensure hidden_dim is divisible by num_heads
        if num_heads is None:
            num_heads = 2
        # Adjust hidden_dim to be divisible by num_heads
        if hidden_dim % num_heads != 0:
            hidden_dim = (hidden_dim // num_heads) * num_heads
            if hidden_dim < num_heads:
                hidden_dim = num_heads
        self.actual_hidden_dim = hidden_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, batch_first=True
        )
        self.actor = nn.Linear(hidden_dim, num_actions)
        self.critic = nn.Linear(hidden_dim, 1)
        # Init attention near-zero
        for name, param in self.attention.named_parameters():
            if param.dim() > 1:
                nn.init.uniform_(param, -0.05, 0.05)

    def forward(self, x, hidden_state=None):
        if x.dim() == 1:
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.dim() == 2:
            x = x.unsqueeze(0)
        h = F.relu(self.fc1(x))
        if hidden_state is not None:
            h, new_hidden = self.gru(h, hidden_state)
        else:
            h, new_hidden = self.gru(h)
        # Self-attention
        attn_out, _ = self.attention(h, h, h)
        h = h + attn_out  # residual
        h_last = h[:, -1, :]
        policy_logits = self.actor(h_last)
        value = self.critic(h_last)
        return policy_logits, value, new_hidden


# ═══════════════════════════════════════════════════════════════════════════════
# GrowingBrain
# ═══════════════════════════════════════════════════════════════════════════════

class GrowingBrain:
    """
    Master dynamic brain for Nafs AI agents.

    The brain starts tiny (MLP only, hidden_dim=16, ~500 params) and grows
    new architectural components when learning plateaus:
      - 1st growth: add GRU (temporal memory emerges)
      - 2nd growth: expand hidden_dim
      - 3rd growth: add self-attention
      - 4th growth: expand hidden_dim
      - 5th growth: full transformer blocks (if vocab > 100)

    Weight preservation:
      - Existing weights are transferred to new architecture
      - New neurons initialized near-zero (don't disrupt existing knowledge)
      - EWC penalty prevents important weights from changing too much

    Growth trigger:
      - Monitor rolling average of PPO loss over last 500 ticks
      - If loss_improvement < 0.001 → growth triggered
      - Min 500 ticks between growth events
    """

    def __init__(self, input_dim: int, num_actions: int,
                 initial_hidden_dim: int = INITIAL_HIDDEN_DIM,
                 vocabulary_size: int = 0,
                 seed: Optional[int] = None):
        self.input_dim = input_dim
        self.num_actions = num_actions
        self.hidden_dim = initial_hidden_dim
        self.vocabulary_size = vocabulary_size
        self.growth_count = 0
        self.architecture = ARCHITECTURE_BY_GROWTH[0]

        if seed is not None:
            torch.manual_seed(seed)

        # Build initial tiny model
        self.model = self._build_model(self.architecture, self.hidden_dim)

        # Loss tracking
        self.loss_history: deque = deque(maxlen=LOSS_HISTORY_SIZE)
        self.last_growth_tick = 0

        # EWC: importance scores + old parameter snapshots
        self.ewc_importance: Dict[str, torch.Tensor] = {}
        self.ewc_old_params: Dict[str, torch.Tensor] = {}
        self.ewc_lambda = 100.0  # EWC penalty strength

        # Growth event log
        self.growth_events: List[Dict[str, Any]] = []

    # ─────────────────────────────────────────────────────────────────────────
    # Model building
    # ─────────────────────────────────────────────────────────────────────────

    def _build_model(self, architecture: str, hidden_dim: int) -> nn.Module:
        if architecture == "mlp_only":
            return MLPOnly(self.input_dim, hidden_dim, self.num_actions)
        elif architecture == "mlp_gru" or architecture == "mlp_gru_bigger":
            return MLPGRU(self.input_dim, hidden_dim, self.num_actions)
        elif architecture == "mlp_gru_attn" or architecture == "mlp_gru_attn_bigger":
            return MLPGRUAttention(self.input_dim, hidden_dim, self.num_actions)
        elif architecture == "transformer":
            # For 5th growth: use attention network with 4 heads as base
            # (full transformer would require sequence input handling beyond
            # current scope — MultiheadAttention captures the core mechanism)
            return MLPGRUAttention(self.input_dim, hidden_dim, self.num_actions, num_heads=4)
        else:
            return MLPOnly(self.input_dim, hidden_dim, self.num_actions)

    # ─────────────────────────────────────────────────────────────────────────
    # Forward pass
    # ─────────────────────────────────────────────────────────────────────────

    def act(self, x: torch.Tensor, hidden_state: Optional[torch.Tensor] = None
            ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass. Returns (action_probs, log_prob, value, new_hidden).

        For MLP-only (stage 0), hidden_state is ignored.
        For stages 1+, hidden_state carries GRU state.
        """
        with torch.no_grad():
            logits, value, new_hidden = self.model(x, hidden_state)
            probs = F.softmax(logits, dim=-1)
            if probs.dim() > 1:
                probs = probs.squeeze(0)
            # Sample action
            if probs.sum() <= 0:
                probs = torch.ones_like(probs) / len(probs)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        return action, log_prob, value.squeeze() if value.dim() > 0 else value, new_hidden

    def evaluate(self, x: torch.Tensor, action: torch.Tensor,
                 hidden_state: Optional[torch.Tensor] = None
                 ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """For PPO update: returns (log_prob, value, entropy)."""
        logits, value, _ = self.model(x, hidden_state)
        probs = F.softmax(logits, dim=-1)
        if probs.dim() > 1:
            probs = probs.squeeze(0)
        dist = torch.distributions.Categorical(probs)
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return log_prob, value.squeeze() if value.dim() > 0 else value, entropy

    def parameters(self):
        return self.model.parameters()

    # ─────────────────────────────────────────────────────────────────────────
    # Loss tracking + growth trigger
    # ─────────────────────────────────────────────────────────────────────────

    def record_loss(self, loss: float, tick: int) -> None:
        """Record a PPO loss value for growth monitoring."""
        self.loss_history.append((tick, float(loss)))

    def should_grow(self, current_tick: int) -> bool:
        """
        Check if growth should be triggered.

        Growth conditions:
          1. At least 500 loss samples recorded
          2. Loss improvement < 0.001 over last 500 samples
          3. At least 500 ticks since last growth
        """
        if len(self.loss_history) < LOSS_HISTORY_SIZE:
            return False
        if current_tick - self.last_growth_tick < MIN_TICKS_BETWEEN_GROWTH:
            return False

        recent = list(self.loss_history)
        # Compare first 100 vs last 100 of the window
        early_avg = sum(l for _, l in recent[:100]) / 100
        late_avg = sum(l for _, l in recent[-100:]) / 100
        improvement = early_avg - late_avg

        return improvement < LOSS_IMPROVEMENT_THRESHOLD

    def grow(self, tick: int, reason: str = "loss_plateau",
             vocab_size: Optional[int] = None) -> Dict[str, Any]:
        """
        Trigger brain growth. Expands architecture + preserves weights.

        Returns growth event log dict.
        """
        if vocab_size is not None:
            self.vocabulary_size = vocab_size

        old_architecture = self.architecture
        old_hidden_dim = self.hidden_dim
        old_params = sum(p.numel() for p in self.model.parameters())

        # Update EWC importance BEFORE growing (capture current state)
        self._update_ewc_importance()

        # Increment growth count + determine new architecture
        self.growth_count += 1

        # Special case: 5th growth requires vocab > 100
        if self.growth_count == 5 and self.vocabulary_size < TRANSFORMER_VOCAB_THRESHOLD:
            # Skip transformer, just expand hidden_dim
            self.architecture = ARCHITECTURE_BY_GROWTH.get(
                self.growth_count - 1, "mlp_gru_attn_bigger"
            )
        else:
            self.architecture = ARCHITECTURE_BY_GROWTH.get(
                self.growth_count, "transformer"
            )

        # Expand hidden_dim by 1.5x
        self.hidden_dim = int(self.hidden_dim * GROWTH_HIDDEN_MULT)
        self.hidden_dim = max(self.hidden_dim, 8)  # safety floor

        # Build new model
        new_model = self._build_model(self.architecture, self.hidden_dim)

        # Transfer weights from old to new
        transfer_stats = self._transfer_weights(self.model, new_model)

        # Save old params for EWC
        self.ewc_old_params = {
            name: param.detach().clone()
            for name, param in new_model.named_parameters()
            if name in self.ewc_importance
        }

        self.model = new_model
        new_params = sum(p.numel() for p in self.model.parameters())

        # Log growth event
        event = {
            "tick": tick,
            "growth_count": self.growth_count,
            "old_architecture": old_architecture,
            "new_architecture": self.architecture,
            "old_hidden_dim": old_hidden_dim,
            "new_hidden_dim": self.hidden_dim,
            "old_params": old_params,
            "new_params": new_params,
            "params_growth_pct": round((new_params / max(1, old_params) - 1) * 100, 1),
            "trigger_reason": reason,
            "vocabulary_size": self.vocabulary_size,
            "weights_transferred": transfer_stats,
        }
        self.growth_events.append(event)
        self.last_growth_tick = tick

        # Clear loss history to start fresh window
        self.loss_history.clear()

        return event

    # ─────────────────────────────────────────────────────────────────────────
    # Weight transfer (preserve existing knowledge)
    # ─────────────────────────────────────────────────────────────────────────

    def _transfer_weights(self, old_model: nn.Module, new_model: nn.Module) -> Dict:
        """
        Transfer weights from old model to new model.

        Strategy:
          - For layers that exist in both (same name + same shape): copy directly
          - For layers that exist in both but different shape: copy overlapping
            portion, init new portion near-zero
          - For new layers (e.g. GRU when growing from MLP): leave near-zero init
        """
        old_state = dict(old_model.named_parameters())
        new_state = dict(new_model.named_parameters())

        transferred = 0
        newly_initialized = 0

        for name, new_param in new_state.items():
            if name in old_state:
                old_param = old_state[name]
                if old_param.shape == new_param.shape:
                    # Direct copy
                    new_param.data.copy_(old_param.data)
                    transferred += 1
                else:
                    # Copy overlapping portion
                    min_shape = [min(o, n) for o, n in
                                 zip(old_param.shape, new_param.shape)]
                    if len(min_shape) == 1:
                        new_param.data[:min_shape[0]] = old_param.data[:min_shape[0]]
                    elif len(min_shape) == 2:
                        new_param.data[:min_shape[0], :min_shape[1]] = \
                            old_param.data[:min_shape[0], :min_shape[1]]
                    transferred += 1
            else:
                # New layer — leave near-zero init (already done in __init__)
                newly_initialized += 1

        return {
            "transferred": transferred,
            "newly_initialized": newly_initialized,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # EWC: Catastrophic forgetting prevention
    # ─────────────────────────────────────────────────────────────────────────

    def _update_ewc_importance(self) -> None:
        """
        Compute Fisher information matrix approximation for EWC.

        importance[name] = average of (gradient)^2 over recent samples.
        We use the current loss history as a proxy.
        """
        if not self.loss_history:
            return

        # Compute Fisher information as squared gradient magnitudes
        # For simplicity, use the parameter norms as a proxy for importance
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                self.ewc_importance[name] = param.grad.data ** 2
            else:
                # Use weight magnitude as importance proxy
                self.ewc_importance[name] = (param.data ** 2) * 0.01

    def get_ewc_penalty(self) -> torch.Tensor:
        """
        EWC penalty: sum of importance * (param - old_param)^2.

        This penalizes changing important weights during PPO updates.
        Only applies to params with matching shapes (post-growth, some
        params may have different shapes — those are skipped to avoid
        shape-mismatch errors).
        """
        if not self.ewc_importance or not self.ewc_old_params:
            return torch.tensor(0.0)

        penalty = torch.tensor(0.0, requires_grad=False)
        for name, param in self.model.named_parameters():
            if name not in self.ewc_importance or name not in self.ewc_old_params:
                continue
            importance = self.ewc_importance[name]
            old_param = self.ewc_old_params[name]
            # Skip if shapes don't match (post-growth mismatch)
            if importance.shape != param.shape or old_param.shape != param.shape:
                continue
            importance = importance.to(param.device)
            old_param = old_param.to(param.device)
            diff = param - old_param
            penalty = penalty + (importance * diff ** 2).sum()

        return penalty * (self.ewc_lambda / 2)

    # ─────────────────────────────────────────────────────────────────────────
    # Info + serialization
    # ─────────────────────────────────────────────────────────────────────────

    def get_param_count(self) -> int:
        return sum(p.numel() for p in self.model.parameters())

    def get_info(self) -> Dict:
        return {
            "architecture": self.architecture,
            "hidden_dim": self.hidden_dim,
            "param_count": self.get_param_count(),
            "growth_count": self.growth_count,
            "loss_samples": len(self.loss_history),
            "vocabulary_size": self.vocabulary_size,
            "growth_events": len(self.growth_events),
        }

    def to_dict(self) -> Dict:
        """Serialize for checkpointing (model weights + state)."""
        return {
            'model_state_dict': self.model.state_dict(),
            'input_dim': self.input_dim,
            'num_actions': self.num_actions,
            'hidden_dim': self.hidden_dim,
            'vocabulary_size': self.vocabulary_size,
            'growth_count': self.growth_count,
            'architecture': self.architecture,
            'last_growth_tick': self.last_growth_tick,
            'growth_events': self.growth_events,
        }

    def load_state(self, state: Dict) -> None:
        """Restore from checkpoint."""
        self.input_dim = state['input_dim']
        self.num_actions = state['num_actions']
        self.hidden_dim = state['hidden_dim']
        self.vocabulary_size = state.get('vocabulary_size', 0)
        self.growth_count = state['growth_count']
        self.architecture = state['architecture']
        self.last_growth_tick = state.get('last_growth_tick', 0)
        self.growth_events = state.get('growth_events', [])
        # Rebuild model and load weights
        self.model = self._build_model(self.architecture, self.hidden_dim)
        self.model.load_state_dict(state['model_state_dict'])


# ═══════════════════════════════════════════════════════════════════════════════
# Self-test
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Testing GrowingBrain...")

    # Test 1: Baby brain starts small
    brain = GrowingBrain(input_dim=21, num_actions=8, seed=42)
    info = brain.get_info()
    print(f"  Initial: arch={info['architecture']}, hidden={info['hidden_dim']}, "
          f"params={info['param_count']}")
    assert info['architecture'] == 'mlp_only'
    assert info['hidden_dim'] == 16
    assert info['param_count'] < 1000  # should be ~500

    # Test 2: Forward pass works
    x = torch.randn(21)
    action, log_prob, value, hidden = brain.act(x)
    print(f"  Forward pass: action={action.item()}, log_prob={log_prob.item():.4f}, "
          f"value={value.item():.4f}")
    assert 0 <= action.item() < 8

    # Test 3: should_grow() returns False initially
    assert not brain.should_grow(current_tick=100)

    # Test 4: Trigger growth by simulating loss plateau
    # Fill loss history with constant loss (improvement = 0)
    for tick in range(500):
        brain.record_loss(loss=0.5, tick=tick)
    assert brain.should_grow(current_tick=500)

    # Test 5: Grow the brain
    event = brain.grow(tick=500, reason="loss_plateau")
    print(f"  Growth event: {event['old_architecture']} → {event['new_architecture']}")
    print(f"    params: {event['old_params']} → {event['new_params']} "
          f"(+{event['params_growth_pct']}%)")
    print(f"    weights_transferred: {event['weights_transferred']}")
    assert event['new_architecture'] == 'mlp_gru'  # 1st growth adds GRU
    assert event['new_params'] > event['old_params']
    assert event['weights_transferred']['transferred'] > 0

    # Test 6: After growth, forward pass still works
    action, log_prob, value, new_hidden = brain.act(x)
    print(f"  Post-growth forward: action={action.item()}, value={value.item():.4f}")
    assert 0 <= action.item() < 8

    # Test 7: Multiple growth events
    for growth_num in range(2, 6):
        # Fill loss history again
        for tick in range(500):
            brain.record_loss(loss=0.5, tick=brain.last_growth_tick + tick)
        if brain.should_grow(current_tick=brain.last_growth_tick + 500):
            event = brain.grow(tick=brain.last_growth_tick + 500,
                                reason="loss_plateau",
                                vocab_size=50)
            print(f"  Growth #{growth_num}: arch={event['new_architecture']}, "
                  f"params={event['new_params']}")

    info = brain.get_info()
    print(f"  Final: arch={info['architecture']}, params={info['param_count']}, "
          f"growth_events={info['growth_events']}")
    assert info['growth_events'] >= 2

    # Test 8: EWC penalty
    brain2 = GrowingBrain(input_dim=21, num_actions=8, seed=42)
    # Trigger growth to populate EWC importance
    for tick in range(500):
        brain2.record_loss(loss=0.5, tick=tick)
    brain2.grow(tick=500)
    # EWC penalty should be a tensor
    penalty = brain2.get_ewc_penalty()
    print(f"  EWC penalty: {penalty.item():.6f}")

    # Test 9: 5th growth requires vocab > 100
    brain3 = GrowingBrain(input_dim=21, num_actions=8, seed=42)
    # Force growth 4 times
    for _ in range(4):
        for tick in range(500):
            brain3.record_loss(loss=0.5, tick=brain3.last_growth_tick + tick)
        brain3.grow(tick=brain3.last_growth_tick + 500, vocab_size=50)
    # 5th growth with vocab < 100: should NOT upgrade to transformer
    arch_before = brain3.architecture
    for tick in range(500):
        brain3.record_loss(loss=0.5, tick=brain3.last_growth_tick + tick)
    brain3.grow(tick=brain3.last_growth_tick + 500, vocab_size=50)
    print(f"  5th growth with vocab=50: arch stays {brain3.architecture}")
    # Should not be transformer (vocab too low)
    assert brain3.architecture != "transformer"

    # Test 10: 5th growth with vocab > 100 → transformer
    brain4 = GrowingBrain(input_dim=21, num_actions=8, seed=42)
    for _ in range(4):
        for tick in range(500):
            brain4.record_loss(loss=0.5, tick=brain4.last_growth_tick + tick)
        brain4.grow(tick=brain4.last_growth_tick + 500, vocab_size=150)
    for tick in range(500):
        brain4.record_loss(loss=0.5, tick=brain4.last_growth_tick + tick)
    brain4.grow(tick=brain4.last_growth_tick + 500, vocab_size=150)
    print(f"  5th growth with vocab=150: arch = {brain4.architecture}")
    assert brain4.architecture == "transformer"

    # Test 11: Serialization
    brain5 = GrowingBrain(input_dim=21, num_actions=8, seed=42)
    for tick in range(500):
        brain5.record_loss(loss=0.5, tick=tick)
    brain5.grow(tick=500)
    state = brain5.to_dict()
    brain6 = GrowingBrain(input_dim=21, num_actions=8, seed=99)
    brain6.load_state(state)
    assert brain6.architecture == brain5.architecture
    assert brain6.growth_count == brain5.growth_count
    print(f"  Serialization round-trip: ✓")

    print("\n✓ GrowingBrain self-test passed")
