"""
Nafs AI — TensorBoard Logger
"Watch Adam's mind unfold in real-time."

Provides structured TensorBoard logging for all training metrics.
If TensorBoard is not installed, falls back to no-op silently
so training still works without it.

Usage:
    tensorboard --logdir runs/

Metrics logged:
    - Training: episode reward, survival length, loss components
    - Actions: action distribution histogram
    - Inner Life: thought count, vocabulary size, emotion distribution
    - Memory: fear triggers, good memories, patterns learned
    - Curiosity: intrinsic reward, states discovered
    - Dreaming: dream count, nightmare ratio
    - PPO: entropy, value loss, policy loss, grad norm
"""

import os


class TBLogger:
    """
    TensorBoard logger with graceful fallback.

    If tensorboard is available, logs all metrics.
    If not, silently does nothing — training works either way.
    """

    def __init__(self, log_dir: str = "runs/nafs_run"):
        self.enabled = False
        self.writer = None
        self.log_dir = log_dir

        try:
            from torch.utils.tensorboard import SummaryWriter
            os.makedirs(log_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir=log_dir)
            self.enabled = True
            print(f"  [TB] TensorBoard logging to {log_dir}", flush=True)
            print(f"  [TB] View with: tensorboard --logdir runs/", flush=True)
        except ImportError:
            print("  [TB] TensorBoard not installed. Install with: pip install tensorboard", flush=True)
            print("  [TB] Training will continue without TensorBoard logging.", flush=True)
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

    def log_episode(self, episode: int, data: dict):
        """
        Log all episode-level metrics.

        Expected data keys:
            - reward, survival_length, best_survival
            - policy_loss, value_loss, entropy, grad_norm
            - action_counts (dict), action_dist (dict of percentages)
            - vocabulary_size, discovered_words_count
            - fear_triggers_count, good_memories_count, patterns_count
            - curiosity_intrinsic_reward, curiosity_states_discovered
            - dream_count, nightmare_count, peaceful_count
            - dominant_action, dominant_pct
            - personality_disposition
            - thought_sample (str), reflection_sample (str)
        """
        if not self.enabled or not self.writer:
            return

        try:
            # Core training metrics
            self.log_scalar("train/episode_reward", data.get("reward", 0), episode)
            self.log_scalar("train/survival_length", data.get("survival_length", 0), episode)
            self.log_scalar("train/best_survival", data.get("best_survival", 0), episode)

            # PPO internals
            if "policy_loss" in data:
                self.log_scalar("ppo/policy_loss", data["policy_loss"], episode)
            if "value_loss" in data:
                self.log_scalar("ppo/value_loss", data["value_loss"], episode)
            if "entropy" in data:
                self.log_scalar("ppo/entropy", data["entropy"], episode)
            if "grad_norm" in data:
                self.log_scalar("ppo/grad_norm", data["grad_norm"], episode)

            # Action distribution
            action_dist = data.get("action_dist", {})
            for action, pct in action_dist.items():
                self.log_scalar(f"actions/{action}", pct, episode)

            # Dominant action tracking (early warning for mono-behavior)
            if "dominant_pct" in data:
                self.log_scalar("warnings/dominant_action_pct", data["dominant_pct"], episode)

            # Inner life metrics
            if "vocabulary_size" in data:
                self.log_scalar("inner_life/vocabulary_size", data["vocabulary_size"], episode)
            if "discovered_words_count" in data:
                self.log_scalar("inner_life/discovered_words", data["discovered_words_count"], episode)

            # Memory metrics
            if "fear_triggers_count" in data:
                self.log_scalar("memory/fear_triggers", data["fear_triggers_count"], episode)
            if "good_memories_count" in data:
                self.log_scalar("memory/good_memories", data["good_memories_count"], episode)
            if "patterns_count" in data:
                self.log_scalar("memory/patterns_learned", data["patterns_count"], episode)

            # Curiosity metrics
            if "curiosity_intrinsic_reward" in data:
                self.log_scalar("curiosity/intrinsic_reward", data["curiosity_intrinsic_reward"], episode)
            if "curiosity_states_discovered" in data:
                self.log_scalar("curiosity/states_discovered", data["curiosity_states_discovered"], episode)

            # Dreaming metrics
            if "dream_count" in data:
                self.log_scalar("dreaming/total_dreams", data["dream_count"], episode)
            if "nightmare_count" in data:
                self.log_scalar("dreaming/nightmares", data["nightmare_count"], episode)

            # Text samples
            if "thought_sample" in data:
                self.log_text("samples/thought", data["thought_sample"], episode)
            if "reflection_sample" in data:
                self.log_text("samples/reflection", data["reflection_sample"], episode)

        except Exception as e:
            # Never let logging break training
            pass

    def close(self):
        """Close the TensorBoard writer."""
        if self.enabled and self.writer:
            try:
                self.writer.close()
            except Exception:
                pass
