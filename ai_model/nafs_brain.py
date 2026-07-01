"""
Nafs AI — Standalone Brain Module
=================================
Self-contained definitions of:
  - BabyBrain            (PPO + GRU actor-critic, 348,169 params)
  - ThoughtTransformer   (character-level transformer, 539,935 params)
  - ThoughtTokenizer     (char-level tokenizer, vocab=31)

This file has NO dependencies on the rest of the nafs-ai project.
Only requires: torch >= 2.0

Use it to load Adam's or Eve's brain from a .pt checkpoint and run inference
on a raw 21-dim sensory vector.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── BabyBrain (PPO Actor-Critic + GRU) ────────────────────────────────────────
class BabyBrain(nn.Module):
    """
    PPO Actor-Critic network with GRU temporal memory.
    Input : 21-dim sensory vector
    Output: 8 action logits + 1 state value
    """
    def __init__(self, input_dim: int = 21, hidden_dim: int = 256, num_actions: int = 8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.actor_linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.actor_output  = nn.Linear(hidden_dim, num_actions)
        self.critic_linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.critic_output  = nn.Linear(hidden_dim, 1)

    def forward(self, sensory_input, hidden_state):
        sensory_input = sensory_input.unsqueeze(1)  # (B,1,input_dim)
        gru_out, next_hidden = self.gru(sensory_input, hidden_state)
        gru_out = gru_out.squeeze(1)
        action_logits = self.actor_output(F.relu(self.actor_linear1(gru_out)))
        state_value   = self.critic_output(F.relu(self.critic_linear1(gru_out)))
        return action_logits, state_value, next_hidden

    def init_hidden(self, batch_size: int = 1):
        return torch.zeros(1, batch_size, self.hidden_dim)


# ── Character-level tokenizer ─────────────────────────────────────────────────
class ThoughtTokenizer:
    _BASE_CHARS = list("abcdefghijklmnopqrstuvwxyz .,")
    _SPECIAL    = ["<pad>", "<eot>"]
    ALL_TOKENS  = _BASE_CHARS + _SPECIAL
    CHAR_TO_IDX = {c: i for i, c in enumerate(ALL_TOKENS)}
    IDX_TO_CHAR = {i: c for i, c in enumerate(ALL_TOKENS)}
    PAD_IDX     = CHAR_TO_IDX["<pad>"]
    EOT_IDX     = CHAR_TO_IDX["<eot>"]
    VOCAB_SIZE  = len(ALL_TOKENS)              # 31
    MAX_THOUGHT_LEN = 48

    @classmethod
    def encode(cls, text: str) -> torch.Tensor:
        text = text.lower().strip()[: cls.MAX_THOUGHT_LEN - 1]
        idx = [cls.CHAR_TO_IDX.get(c, cls.CHAR_TO_IDX[" "]) for c in text]
        idx.append(cls.EOT_IDX)
        while len(idx) < cls.MAX_THOUGHT_LEN:
            idx.append(cls.PAD_IDX)
        return torch.tensor(idx, dtype=torch.long)

    @classmethod
    def decode(cls, indices) -> str:
        chars = []
        for i in indices:
            v = i.item() if torch.is_tensor(i) else i
            t = cls.IDX_TO_CHAR.get(v, "")
            if t == "<eot>":  break
            if t == "<pad>":  continue
            chars.append(t)
        return "".join(chars).strip()


# ── Tiny character-level transformer for thoughts ─────────────────────────────
class ThoughtTransformer(nn.Module):
    """
    Conditions on the 21-dim sensory vector and generates a thought
    character-by-character (max 48 chars).
    """
    def __init__(self, sensory_dim: int = 21,
                 vocab_size: int = ThoughtTokenizer.VOCAB_SIZE,
                 d_model: int = 128, n_heads: int = 4, n_layers: int = 2,
                 max_len: int = ThoughtTokenizer.MAX_THOUGHT_LEN,
                 dropout: float = 0.1):
        super().__init__()
        self.d_model  = d_model
        self.max_len  = max_len
        self.sensory_proj = nn.Linear(sensory_dim, d_model)
        self.char_embed   = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = self._create_pos_encoding(max_len, d_model)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 4, dropout=dropout,
            batch_first=True, activation='gelu',
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)
        self.output_proj = nn.Linear(d_model, vocab_size)
        self._init_weights()

    def _create_pos_encoding(self, max_len, d_model):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, sensory, char_indices):
        # sensory:     (B, sensory_dim)
        # char_indices:(B, seq_len)
        sensory_emb = self.sensory_proj(sensory).unsqueeze(1)  # (B,1,d_model)
        char_emb    = self.char_embed(char_indices) * math.sqrt(self.d_model)
        seq_len     = char_indices.size(1)
        char_emb    = char_emb + self.pos_encoding[:, :seq_len, :].to(char_emb.device)
        memory      = sensory_emb.expand(-1, char_emb.size(1), -1)  # broadcast over seq
        # Decoder: target = char_emb, memory = sensory condition
        out = self.transformer(char_emb, memory)
        return self.output_proj(out)


# ── Convenience loader ────────────────────────────────────────────────────────
ACTION_NAMES = ["EXPLORE","EAT","DRINK","SLEEP","HIDE","MOVE","FLEE","IDLE"]


def load_agent(checkpoint_path: str, device: str = "cpu"):
    """
    Load a Nafs AI agent checkpoint.

    Returns:
        brain:    BabyBrain instance (with weights loaded)
        thinker:  ThoughtTransformer instance (with weights loaded)
        meta:     dict of metadata stored in the checkpoint
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    brain   = BabyBrain()
    thinker = ThoughtTransformer()
    if "model_state_dict" in ckpt:
        brain.load_state_dict(ckpt["model_state_dict"])
    if "thinker_state" in ckpt:
        thinker.load_state_dict(ckpt["thinker_state"])
    elif "learned_state" in ckpt:
        thinker.load_state_dict(ckpt["learned_state"])
    brain.eval(); thinker.eval()
    meta = {k: v for k, v in ckpt.items()
            if k not in ("model_state_dict","thinker_state","learned_state",
                         "optimizer_state_dict","optimizer_state")}
    return brain, thinker, meta


@torch.no_grad()
def think(brain, thinker, sensory_vector, hidden=None, max_new_tokens=48):
    """
    Given a 21-dim sensory vector, return:
        action_name : str
        action_prob : float
        value       : float
        thought     : str
        new_hidden  : next GRU hidden state
    """
    if hidden is None:
        hidden = brain.init_hidden(1)
    s = torch.tensor(sensory_vector, dtype=torch.float32).unsqueeze(0)
    logits, value, hidden = brain(s, hidden)
    probs = F.softmax(logits, dim=-1)
    a_idx = int(probs.argmax(dim=-1).item())
    action_name = ACTION_NAMES[a_idx]
    action_prob = float(probs[0, a_idx].item())

    # Generate thought autoregressively
    generated = []
    cur = ThoughtTokenizer.encode("").unsqueeze(0)  # (1, 48) all PAD
    cur[:, 0] = ThoughtTokenizer.EOT_IDX  # start token (decoder will predict next)
    # Simpler approach: feed sensory as memory, generate one char at a time
    # The model expects (B, seq_len) input; we start with just EOT then grow.
    seq = torch.full((1, 1), ThoughtTokenizer.EOT_IDX, dtype=torch.long)
    for _ in range(max_new_tokens):
        logits_c = thinker(s, seq)  # (1, seq_len, vocab)
        next_idx = int(logits_c[0, -1].argmax().item())
        if next_idx == ThoughtTokenizer.EOT_IDX:
            break
        generated.append(ThoughtTokenizer.IDX_TO_CHAR.get(next_idx, ""))
        seq = torch.cat([seq, torch.tensor([[next_idx]])], dim=1)
        if seq.size(1) >= ThoughtTokenizer.MAX_THOUGHT_LEN:
            break
    thought = "".join(generated).strip() or "(silence)"
    return action_name, action_prob, float(value.item()), thought, hidden


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python nafs_brain.py <checkpoint.pt>")
        sys.exit(1)
    brain, thinker, meta = load_agent(sys.argv[1])
    print("Loaded:", sys.argv[1])
    print("Meta:", meta)
    print(f"BabyBrain params   : {sum(p.numel() for p in brain.parameters()):,}")
    print(f"Thinker   params   : {sum(p.numel() for p in thinker.parameters()):,}")

    # Random sensory test
    sensory = torch.randn(21).tolist()
    action, prob, value, thought, _ = think(brain, thinker, sensory)
    print(f"\nTest inference on random sensory vector:")
    print(f"  Action : {action} (p={prob:.3f})")
    print(f"  Value  : {value:.3f}")
    print(f"  Thought: \"{thought}\"")
