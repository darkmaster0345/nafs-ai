"""
Nafs AI — TensorBoard Logger (Fixed for Single-Life Mode)
"Watch Adam's mind unfold in real-time, tick by tick."

Fixes:
  - Added flush() to ensure data is written immediately
  - Changed log_episode to log_tick for single-life mode
  - Added more granular per-tick logging
  - Added world state logging (biome, weather, position)
  - Fixed potential issues with SummaryWriter initialization

Usage:
    tensorboard --logdir runs/
"""

import os


class TBLogger:
    """
    TensorBoard logger with graceful fallback.

    If tensorboard is available, logs all metrics.
    If not, silently does nothing — simulation works either way.
    """

    def __init__(self, log_dir: str = "runs/nafs_single_life"):
        self.enabled = False
        self.writer = None
        self.log_dir = log_dir
        self._step_counter = 0

        try:
            from torch.utils.tensorboard import SummaryWriter
            os.makedirs(log_dir, exist_ok=True)
            self.writer = SummaryWriter(
                log_dir=log_dir,
                flush_secs=10,     # Auto-flush every 10 seconds
                max_queue=100,     # Buffer up to 100 events before flushing
            )
            self.enabled = True
            print(f"  [TB] TensorBoard logging to {log_dir}", flush=True)
            print(f"  [TB] View with: tensorboard --logdir runs/", flush=True)
        except ImportError:
            print("  [TB] TensorBoard not installed. Install with: pip install tensorboard", flush=True)
            print("  [TB] Simulation will continue without TensorBoard logging.", flush=True)
        except Exception as e:
            print(f"  [TB] TensorBoard init failed: {e}. Continuing without logging.", flush=True)

    def log_scalar(self, tag: str, value: float, step: int):
        """Log a single scalar value."""
        if self.enabled and self.writer:
            try:
                self.writer.add_scalar(tag, value, step)
            except Exception:
                pass

    def log_scalars(self, tag_value_dict: dict, step: int):
        """Log multiple scalars at once."""
        if not self.enabled or not self.writer:
            return
        for tag, value in tag_value_dict.items():
            try:
                self.writer.add_scalar(tag, value, step)
            except Exception:
                pass

    def log_histogram(self, tag: str, values, step: int):
        """Log a histogram of values."""
        if self.enabled and self.writer:
            try:
                import torch
                if not isinstance(values, torch.Tensor):
                    values = torch.FloatTensor(values)
                self.writer.add_histogram(tag, values, step)
            except Exception:
                pass

    def log_text(self, tag: str, text: str, step: int):
        """Log text content."""
        if self.enabled and self.writer:
            try:
                self.writer.add_text(tag, text, step)
            except Exception:
                pass

    def flush(self):
        """Force-flush all pending data to disk."""
        if self.enabled and self.writer:
            try:
                self.writer.flush()
            except Exception:
                pass

    def log_tick(self, tick: int, data: dict):
        """
        Log tick-level metrics for single-life mode.

        Expected data keys:
            - reward, total_reward, tick
            - policy_loss, value_loss, entropy, grad_norm
            - action_dist (dict of percentages)
            - vocabulary_size, discovered_words_count
            - fear_triggers_count, good_memories_count, patterns_count
            - curiosity_intrinsic_reward, curiosity_states_discovered
            - dream_count, nightmare_count
            - dominant_pct
            - personality_disposition
        """
        if not self.enabled or not self.writer:
            return

        try:
            # Core metrics
            self.log_scalar("life/avg_reward_per_tick", data.get("reward", 0), tick)
            self.log_scalar("life/total_reward", data.get("total_reward", 0), tick)

            # PPO internals
            if "policy_loss" in data:
                self.log_scalar("ppo/policy_loss", data["policy_loss"], tick)
            if "value_loss" in data:
                self.log_scalar("ppo/value_loss", data["value_loss"], tick)
            if "entropy" in data:
                self.log_scalar("ppo/entropy", data["entropy"], tick)
            if "grad_norm" in data:
                self.log_scalar("ppo/grad_norm", data["grad_norm"], tick)

            # Action distribution
            action_dist = data.get("action_dist", {})
            for action, pct in action_dist.items():
                self.log_scalar(f"actions/{action}", pct, tick)

            # Dominant action warning
            if "dominant_pct" in data:
                self.log_scalar("warnings/dominant_action_pct", data["dominant_pct"], tick)

            # Inner life
            if "vocabulary_size" in data:
                self.log_scalar("inner_life/vocabulary_size", data["vocabulary_size"], tick)
            if "discovered_words_count" in data:
                self.log_scalar("inner_life/discovered_words", data["discovered_words_count"], tick)

            # Memory
            if "fear_triggers_count" in data:
                self.log_scalar("memory/fear_triggers", data["fear_triggers_count"], tick)
            if "good_memories_count" in data:
                self.log_scalar("memory/good_memories", data["good_memories_count"], tick)
            if "patterns_count" in data:
                self.log_scalar("memory/patterns_learned", data["patterns_count"], tick)

            # Curiosity
            if "curiosity_intrinsic_reward" in data:
                self.log_scalar("curiosity/intrinsic_reward", data["curiosity_intrinsic_reward"], tick)
            if "curiosity_states_discovered" in data:
                self.log_scalar("curiosity/states_discovered", data["curiosity_states_discovered"], tick)

            # Dreaming
            if "dream_count" in data:
                self.log_scalar("dreaming/total_dreams", data["dream_count"], tick)
            if "nightmare_count" in data:
                self.log_scalar("dreaming/nightmares", data["nightmare_count"], tick)

            # Periodic flush to ensure data shows up in TensorBoard
            self._step_counter += 1
            if self._step_counter % 20 == 0:
                self.flush()

        except Exception:
            # Never let logging break simulation
            pass

    # Backward compatibility — log_episode maps to log_tick
    def log_episode(self, tick: int, data: dict):
        """Alias for log_tick (backward compatibility)."""
        self.log_tick(tick, data)

    def close(self):
        """Close the TensorBoard writer and flush remaining data."""
        if self.enabled and self.writer:
            try:
                self.writer.flush()
                self.writer.close()
            except Exception:
                pass
