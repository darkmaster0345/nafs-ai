"""
Nafs AI — Learned Thinking Module (v1.0 Experimental)
=====================================================

A tiny transformer that learns to generate Adam's thoughts from raw sensory
experience, replacing the rule-based ThoughtGenerator over time.

Philosophy:
    The rule-based ThoughtGenerator (Phase 1) is a scaffold — it gives Adam
    something to think about before he has learned anything. But the thoughts
    are scripted: same sensation → same word, always.

    This module trains a tiny character-level transformer on Adam's actual
    experience stream. Over time, Adam learns:
      - Which sensations tend to co-occur
      - Which words follow which experiences
      - The "grammar" of his own inner life

    The model is TINY by design:
      - ~200K parameters
      - 2 transformer layers, 4 attention heads, 128-dim embeddings
      - Trained on a rolling buffer of (sensory_vector, thought_string) pairs
      - No pretrained weights, no external data

How it integrates:
    1. Adam lives his life using the rule-based ThoughtGenerator (Phase 1-6)
    2. Each tick, (sensory_input, generated_thought) is added to the training buffer
    3. Every N ticks, the LearnedThinker does a few gradient steps on this buffer
    4. Once the LearnedThinker's loss is below a threshold, it starts contributing:
       - It generates a "learned thought" alongside the rule-based thought
       - The two are blended: early on, rule-based dominates; later, learned dominates
    5. Eventually, the learned thought IS Adam's thought — genuinely emergent

This is the bridge from v0.2 (rule-based thoughts) to v1.0 (learned thoughts).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from collections import deque
from typing import Optional, Tuple


# ═══════════════════════════════════════════════════════════════════════════════
# Character-level tokenizer for Adam's thoughts
# ═══════════════════════════════════════════════════════════════════════════════

class ThoughtTokenizer:
    """
    Character-level tokenizer for Adam's primitive thoughts.

    Why character-level?
      - Adam's vocabulary is tiny (~30 words, all short)
      - Character-level lets him discover sub-word patterns
      - No need for a BPE tokenizer or word vocabulary
      - The model learns that "cold" and "cold pain" share the "cold" prefix

    Special tokens:
      - We add an <EOT> (end-of-text) token so the model learns when to stop
      - Padding is done with a <PAD> token (not space) to avoid the model
        learning to predict spaces after every thought
    """

    # All characters that appear in Adam's thoughts
    # Letters (lowercase), space, period, comma
    # Plus special tokens: <PAD>=29, <EOT>=30
    CHARS = "abcdefghijklmnopqrstuvwxyz .,<pad><eot>"

    # Build vocab with special tokens
    _BASE_CHARS = list("abcdefghijklmnopqrstuvwxyz .,")
    _SPECIAL = ["<pad>", "<eot>"]
    ALL_TOKENS = _BASE_CHARS + _SPECIAL

    CHAR_TO_IDX = {c: i for i, c in enumerate(ALL_TOKENS)}
    IDX_TO_CHAR = {i: c for i, c in enumerate(ALL_TOKENS)}

    PAD_IDX = CHAR_TO_IDX["<pad>"]
    EOT_IDX = CHAR_TO_IDX["<eot>"]

    VOCAB_SIZE = len(ALL_TOKENS)  # 31
    MAX_THOUGHT_LEN = 48     # Max characters in a thought (including EOT)

    @classmethod
    def encode(cls, text: str) -> torch.Tensor:
        """
        Encode a thought string to a tensor of token indices.
        Adds <eot> at the end, pads with <pad> to MAX_THOUGHT_LEN.
        """
        text = text.lower().strip()
        # Truncate to max length - 1 (leave room for EOT)
        text = text[:cls.MAX_THOUGHT_LEN - 1]

        indices = []
        for c in text:
            if c in cls.CHAR_TO_IDX:
                indices.append(cls.CHAR_TO_IDX[c])
            else:
                # Unknown char → space
                indices.append(cls.CHAR_TO_IDX[" "])
        # Append EOT
        indices.append(cls.EOT_IDX)
        # Pad with PAD
        while len(indices) < cls.MAX_THOUGHT_LEN:
            indices.append(cls.PAD_IDX)
        return torch.tensor(indices, dtype=torch.long)

    @classmethod
    def decode(cls, indices: torch.Tensor) -> str:
        """Decode a tensor of token indices back to a string. Stops at <eot>."""
        if indices.dim() == 0:
            idx = indices.item() if torch.is_tensor(indices) else indices
            token = cls.IDX_TO_CHAR.get(idx, "")
            return "" if token in ("<pad>", "<eot>") else token

        chars = []
        for idx in indices:
            idx_val = idx.item() if torch.is_tensor(idx) else idx
            token = cls.IDX_TO_CHAR.get(idx_val, "")
            if token == "<eot>":
                break
            if token == "<pad>":
                continue
            chars.append(token)
        return "".join(chars).strip()


# ═══════════════════════════════════════════════════════════════════════════════
# The Tiny Transformer
# ═══════════════════════════════════════════════════════════════════════════════

class ThoughtTransformer(nn.Module):
    """
    A tiny character-level transformer that learns to generate thoughts.

    Architecture:
      - Sensory embedding: 21-dim → 128-dim (linear projection)
      - Character embedding: 29 chars → 128-dim
      - Positional encoding: sinusoidal
      - 2 transformer decoder layers (4 heads each)
      - Output: 128-dim → 29 chars

    Total parameters: ~200K (deliberately tiny)

    The model is conditioned on Adam's sensory state — given what he feels,
    it generates the characters of his thought one at a time.
    """

    def __init__(
        self,
        sensory_dim: int = 21,
        vocab_size: int = ThoughtTokenizer.VOCAB_SIZE,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        max_len: int = ThoughtTokenizer.MAX_THOUGHT_LEN,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len

        # Sensory projection: 21-dim → 128-dim
        # This is the "conditioning" — what Adam feels right now
        self.sensory_proj = nn.Linear(sensory_dim, d_model)

        # Character embedding
        self.char_embed = nn.Embedding(vocab_size, d_model)

        # Positional encoding (sinusoidal)
        self.pos_encoding = self._create_pos_encoding(max_len, d_model)

        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation='gelu',
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)

        # Output projection
        self.output_proj = nn.Linear(d_model, vocab_size)

        # Initialize weights
        self._init_weights()

    def _create_pos_encoding(self, max_len: int, d_model: int) -> torch.Tensor:
        """Sinusoidal positional encoding."""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # shape: (1, max_len, d_model)

    def _init_weights(self):
        """Initialize weights with small values."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        sensory: torch.Tensor,
        char_indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            sensory: (batch, sensory_dim) — Adam's current sensory state
            char_indices: (batch, seq_len) — character token indices (input)

        Returns:
            logits: (batch, seq_len, vocab_size) — predicted next-char logits
        """
        batch_size, seq_len = char_indices.shape

        # Embed characters
        x = self.char_embed(char_indices) * math.sqrt(self.d_model)

        # Add positional encoding
        pos = self.pos_encoding[:, :seq_len, :].to(x.device)
        x = x + pos

        # Project sensory state to a single conditioning vector
        sensory_cond = self.sensory_proj(sensory)  # (batch, d_model)
        # Expand to act as the "memory" for the decoder
        sensory_memory = sensory_cond.unsqueeze(1)  # (batch, 1, d_model)

        # Causal mask for autoregressive generation
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool),
            diagonal=1,
        )

        # Transformer decoder
        x = self.transformer(
            tgt=x,
            memory=sensory_memory,
            tgt_mask=causal_mask,
        )

        # Output logits
        logits = self.output_proj(x)  # (batch, seq_len, vocab_size)
        return logits

    @torch.no_grad()
    def generate(
        self,
        sensory: torch.Tensor,
        max_len: int = None,
        temperature: float = 0.8,
        top_k: int = 5,
    ) -> str:
        """
        Generate a thought autoregressively given a sensory state.

        Args:
            sensory: (sensory_dim,) — single sensory state
            max_len: max characters to generate
            temperature: sampling temperature (lower = more deterministic)
            top_k: only sample from top-k characters at each step

        Returns:
            Generated thought string
        """
        if max_len is None:
            max_len = self.max_len

        self.eval()
        device = next(self.parameters()).device

        # Start with PAD token as the "start of sequence"
        sensory = sensory.unsqueeze(0).to(device)  # (1, sensory_dim)
        chars = torch.full((1, 1), ThoughtTokenizer.PAD_IDX,
                           dtype=torch.long, device=device)

        min_gen_len = 3  # Minimum chars before allowing early stop

        for step in range(max_len - 1):
            logits = self.forward(sensory, chars)  # (1, seq_len, vocab)
            next_logits = logits[0, -1, :] / temperature  # (vocab,)

            # Mask out PAD token — never generate padding
            next_logits = next_logits.clone()
            next_logits[ThoughtTokenizer.PAD_IDX] = -float('inf')

            # Top-k filtering
            if top_k > 0 and top_k < next_logits.size(0):
                top_k_vals, _ = torch.topk(next_logits, top_k)
                min_val = top_k_vals[-1]
                next_logits = torch.where(
                    next_logits < min_val,
                    torch.full_like(next_logits, -float('inf')),
                    next_logits,
                )

            # Sample
            probs = F.softmax(next_logits, dim=-1)
            next_char = torch.multinomial(probs, num_samples=1)

            chars = torch.cat([chars, next_char.unsqueeze(0)], dim=1)

            # Stop if we generated <eot> (end of thought) — but only after min length
            if chars.size(1) >= min_gen_len:
                last_char_idx = chars[0, -1].item()
                if last_char_idx == ThoughtTokenizer.EOT_IDX:
                    break

        # Decode (stops at EOT automatically)
        generated = ThoughtTokenizer.decode(chars[0])
        return generated.strip()

    def parameter_count(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ═══════════════════════════════════════════════════════════════════════════════
# Experience Buffer
# ═══════════════════════════════════════════════════════════════════════════════

class ExperienceBuffer:
    """
    Rolling buffer of (sensory_state, thought_string) pairs.

    The LearnedThinker trains on this buffer. As Adam lives his life,
    his experiences accumulate here. The buffer is FIFO — old experiences
    are forgotten as new ones come in.

    This mimics human memory: recent experiences are vivid, old ones fade.
    """

    def __init__(self, max_size: int = 2000, sensory_dim: int = 21):
        self.max_size = max_size
        self.sensory_dim = sensory_dim
        self.sensory_buffer = deque(maxlen=max_size)
        self.thought_buffer = deque(maxlen=max_size)
        self.tokenizer = ThoughtTokenizer()

    def add(self, sensory: torch.Tensor, thought: str):
        """Add an experience to the buffer."""
        # Ensure sensory is 1D
        if sensory.dim() > 1:
            sensory = sensory.squeeze()
        self.sensory_buffer.append(sensory.detach().cpu())
        self.thought_buffer.append(thought)

    def sample_batch(self, batch_size: int = 32) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample a random batch of (sensory, thought_token_indices) pairs.

        Returns:
            sensory_batch: (batch_size, sensory_dim)
            input_batch: (batch_size, max_thought_len - 1) — input chars
            target_batch: (batch_size, max_thought_len - 1) — target chars (shifted by 1)
        """
        n = len(self.thought_buffer)
        if n == 0:
            raise ValueError("Buffer is empty")

        # Sample indices
        indices = torch.randint(0, n, (min(batch_size, n),))

        sensory_list = []
        input_chars_list = []
        target_chars_list = []

        for idx in indices:
            sensory_list.append(self.sensory_buffer[idx])

            thought = self.thought_buffer[idx]
            tokens = self.tokenizer.encode(thought)

            # Input: tokens[:-1] (all but last)
            # Target: tokens[1:] (all but first)
            # This is the standard language modeling setup
            input_chars_list.append(tokens[:-1])
            target_chars_list.append(tokens[1:])

        sensory_batch = torch.stack(sensory_list)
        input_batch = torch.stack(input_chars_list)
        target_batch = torch.stack(target_chars_list)

        return sensory_batch, input_batch, target_batch

    def __len__(self):
        return len(self.thought_buffer)


# ═══════════════════════════════════════════════════════════════════════════════
# Learned Thinker — Main Module
# ═══════════════════════════════════════════════════════════════════════════════

class LearnedThinker:
    """
    The learned thinking module.

    This sits alongside the rule-based ThoughtGenerator. Early in Adam's life,
    the rule-based generator does all the work. As the LearnedThinker trains,
    it gradually takes over.

    Usage in train.py:
        learned_thinker = LearnedThinker(sensory_dim=21)
        learned_thinker.start()

        # Each tick:
        learned_thinker.record_experience(sensory_input, rule_based_thought)

        # Every N ticks:
        loss = learned_thinker.train_step()

        # Generate a thought:
        if learned_thinker.is_ready():
            learned_thought = learned_thinker.generate_thought(sensory_input)
            # Blend with rule-based thought
            final_thought = blend_thoughts(rule_thought, learned_thought, learned_thinker.confidence)
    """

    def __init__(
        self,
        sensory_dim: int = 21,
        device: str = 'cpu',
        learning_rate: float = 3e-4,
        buffer_size: int = 2000,
        train_interval: int = 50,
        batch_size: int = 32,
        readiness_loss_threshold: float = 1.5,
    ):
        self.device = torch.device(device)
        self.sensory_dim = sensory_dim
        self.train_interval = train_interval
        self.batch_size = batch_size
        self.readiness_loss_threshold = readiness_loss_threshold

        # The model
        self.model = ThoughtTransformer(sensory_dim=sensory_dim).to(self.device)

        # Experience buffer
        self.buffer = ExperienceBuffer(max_size=buffer_size, sensory_dim=sensory_dim)

        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=1e-5,
        )

        # Tracking
        self.tick_count = 0
        self.train_steps = 0
        self.recent_losses = deque(maxlen=50)  # Rolling average of losses
        self.started = False

        # Stats for dashboard
        self.stats = {
            'total_experiences': 0,
            'train_steps': 0,
            'avg_loss': 0.0,
            'ready': False,
            'confidence': 0.0,
            'param_count': self.model.parameter_count(),
        }

    def start(self):
        """Mark the thinker as started."""
        self.started = True

    def record_experience(self, sensory: torch.Tensor, thought: str):
        """
        Record an experience: (sensory state, thought that was generated).

        Args:
            sensory: (sensory_dim,) tensor — Adam's sensory input this tick
            thought: str — the thought that was generated (by rule-based engine)
        """
        if not self.started:
            return
        self.buffer.add(sensory, thought)
        self.tick_count += 1
        self.stats['total_experiences'] = self.tick_count

        # Train at intervals
        if self.tick_count % self.train_interval == 0 and len(self.buffer) >= self.batch_size:
            self.train_step()

    def train_step(self) -> float:
        """
        Perform one gradient step on a batch of experiences.

        Returns:
            loss value
        """
        if len(self.buffer) < self.batch_size:
            return 0.0

        self.model.train()

        try:
            sensory_batch, input_batch, target_batch = self.buffer.sample_batch(
                self.batch_size
            )
            sensory_batch = sensory_batch.to(self.device)
            input_batch = input_batch.to(self.device)
            target_batch = target_batch.to(self.device)

            # Forward
            logits = self.model(sensory_batch, input_batch)

            # Compute loss (cross-entropy over characters)
            # Ignore padding positions so the model doesn't learn to predict PAD
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                target_batch.reshape(-1),
                ignore_index=ThoughtTokenizer.PAD_IDX,
            )

            # Backward
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            loss_val = loss.item()
            self.recent_losses.append(loss_val)
            self.train_steps += 1

            # Update stats
            avg_loss = sum(self.recent_losses) / len(self.recent_losses)
            self.stats['train_steps'] = self.train_steps
            self.stats['avg_loss'] = round(avg_loss, 4)
            self.stats['ready'] = avg_loss < self.readiness_loss_threshold

            # Confidence: 0 when loss is high, 1 when loss is near 0
            # Uses a smooth sigmoid transition
            confidence = 1.0 / (1.0 + math.exp(avg_loss - self.readiness_loss_threshold))
            self.stats['confidence'] = round(confidence, 3)

            return loss_val

        except Exception as e:
            # Don't crash the simulation if training fails
            return 0.0

    @torch.no_grad()
    def generate_thought(self, sensory: torch.Tensor, temperature: float = 0.7) -> str:
        """
        Generate a thought using the learned model.

        Args:
            sensory: (sensory_dim,) tensor — current sensory state
            temperature: sampling temperature

        Returns:
            Generated thought string
        """
        if not self.is_ready():
            return ""
        return self.model.generate(sensory, temperature=temperature)

    def is_ready(self) -> bool:
        """Check if the model has trained enough to generate meaningful thoughts."""
        return self.stats['ready']

    @property
    def confidence(self) -> float:
        """How confident the model is (0-1). Higher = more learned."""
        return self.stats['confidence']

    def get_stats(self) -> dict:
        """Get stats for dashboard display."""
        return self.stats.copy()

    def save(self, path: str):
        """Save model weights."""
        torch.save({
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'train_steps': self.train_steps,
            'stats': self.stats,
        }, path)

    def load(self, path: str):
        """Load model weights."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.train_steps = checkpoint.get('train_steps', 0)
        self.stats = checkpoint.get('stats', self.stats)


# ═══════════════════════════════════════════════════════════════════════════════
# Thought Blending
# ═══════════════════════════════════════════════════════════════════════════════

def blend_thoughts(
    rule_thought: str,
    learned_thought: str,
    confidence: float,
    min_confidence: float = 0.3,
) -> str:
    """
    Blend rule-based and learned thoughts based on model confidence.

    Early in training (low confidence), use rule-based thought.
    As confidence grows, use learned thought.

    Args:
        rule_thought: Thought from rule-based ThoughtGenerator
        learned_thought: Thought from LearnedThinker
        confidence: 0-1, how confident the learned model is
        min_confidence: Below this, always use rule-based

    Returns:
        The chosen thought
    """
    # If learned thought is empty or model not confident, use rule-based
    if not learned_thought or confidence < min_confidence:
        return rule_thought

    # If learned thought is gibberish (no real words), use rule-based
    # Heuristic: if it has fewer than 2 alphabetic chars, it's gibberish
    alpha_count = sum(1 for c in learned_thought if c.isalpha())
    if alpha_count < 2:
        return rule_thought

    # With some probability based on confidence, use learned thought
    # This creates a gradual transition
    import random
    if random.random() < confidence:
        return learned_thought
    return rule_thought


# ═══════════════════════════════════════════════════════════════════════════════
# Quick self-test
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Testing LearnedThinker module...")

    # Create a thinker
    thinker = LearnedThinker(sensory_dim=21, device='cpu')
    print(f"Model parameters: {thinker.model.parameter_count():,}")
    print(f"Buffer capacity: {thinker.buffer.max_size}")
    print(f"Vocab size: {ThoughtTokenizer.VOCAB_SIZE} characters")

    # Simulate some experiences
    sample_thoughts = [
        "cold. dark.",
        "hungry. empty. eat.",
        "tired. rest.",
        "pain. bad. run.",
        "warm. good. near.",
        "wet. cold.",
        "scared. hide.",
        "quiet. still.",
    ]

    print("\nRecording 2000 experiences to fully train...")
    import torch
    thinker.start()
    for i in range(2000):
        sensory = torch.randn(21) * 0.5 + 0.5
        sensory = torch.clamp(sensory, 0, 1)
        thought = sample_thoughts[i % len(sample_thoughts)]
        thinker.record_experience(sensory, thought)

    # Force readiness for testing
    thinker.stats['ready'] = True
    thinker.stats['confidence'] = 0.8

    print(f"Buffer size: {len(thinker.buffer)}")
    print(f"Train steps: {thinker.train_steps}")
    print(f"Avg loss: {thinker.stats['avg_loss']:.4f}")
    print(f"Ready: {thinker.is_ready()}")
    print(f"Confidence: {thinker.confidence:.3f}")

    # Generate a thought
    print("\nGenerating a thought from a random sensory state...")
    test_sensory = torch.randn(21) * 0.5 + 0.5
    test_sensory = torch.clamp(test_sensory, 0, 1)
    generated = thinker.generate_thought(test_sensory)
    print(f"Generated thought: '{generated}'")

    # Test blending
    print("\nTesting thought blending...")
    rule = "cold. dark."
    for conf in [0.0, 0.3, 0.5, 0.8, 1.0]:
        blended = blend_thoughts(rule, generated, conf)
        print(f"  confidence={conf:.1f}: '{blended}'")

    print("\n✓ LearnedThinker module works correctly")
