#!/usr/bin/env python3
"""
Nafs AI — Standalone Runner for Adam
========================================
Runs Adam's brain WITHOUT the rest of the project.
Just needs: nafs_brain.py + adam.pt in the same folder, and torch.

Usage:
    python run_adam.py                  # random sensory test
    python run_adam.py 0.5 0.2 0.0 ...  # pass a 21-dim sensory vector
    python run_adam.py --interactive    # interactive REPL, type sensory vecs
"""
import os, sys, math, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nafs_brain import load_agent, think, ACTION_NAMES

CKPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "adam.pt")

def print_inference(sensory, brain, thinker, hidden=None):
    action, prob, value, thought, hidden = think(brain, thinker, sensory, hidden)
    print(f"  Action  : {action}  (p={prob:.3f})")
    print(f"  Value   : {value:+.3f}")
    print(f"  Thought : \"{thought}\"")
    return hidden

def main():
    brain, thinker, meta = load_agent(CKPT)
    print("=" * 60)
    print(f"  Nafs AI — {meta.get('agent','Adam')} (standalone)")
    print("=" * 60)
    print(f"  Checkpoint : {CKPT}")
    print(f"  Source     : {meta.get('source_checkpoint', 'random init (untrained)')}")
    if 'tick' in meta:
        print(f"  Trained at : tick {meta['tick']}")
    print(f"  BabyBrain params    : {sum(p.numel() for p in brain.parameters()):,}")
    print(f"  Thinker   params    : {sum(p.numel() for p in thinker.parameters()):,}")
    print(f"  Input dim  : {meta.get('input_dim', 21)}")
    print(f"  Actions    : {', '.join(meta.get('action_names', ACTION_NAMES))}")
    print(f"  Starting vocab: {', '.join(meta.get('starting_vocab', []))}")
    print("=" * 60)

    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        print("\nInteractive mode. Enter 21 comma-separated numbers (or 'q' to quit).")
        hidden = None
        while True:
            try:
                line = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if line.lower() in ("q","quit","exit"):
                break
            try:
                vals = [float(x) for x in line.replace(",", " ").split()]
                if len(vals) != 21:
                    print(f"  Need 21 values, got {len(vals)}. Try again.")
                    continue
                hidden = print_inference(vals, brain, thinker, hidden)
            except ValueError as e:
                print(f"  Parse error: {e}")
        return

    # Default: random sensory vector (or use CLI args)
    if len(sys.argv) > 1:
        try:
            sensory = [float(x) for x in sys.argv[1:]]
            if len(sensory) != 21:
                print(f"  Warning: expected 21 values, got {len(sensory)}; padding/truncating.")
                sensory = (sensory + [0.0]*21)[:21]
        except ValueError:
            sensory = [random.random() for _ in range(21)]
    else:
        print("\nNo sensory vector provided — using random vector.")
        sensory = [random.random() for _ in range(21)]

    print(f"\nSensory input (21-dim):")
    for i, v in enumerate(sensory):
        print(f"  [{i:2d}] {v:+.3f}")
    print(f"\nInference:")
    print_inference(sensory, brain, thinker)

if __name__ == "__main__":
    main()
