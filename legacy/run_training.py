#!/usr/bin/env python3
"""
Nafs AI — Training Runner
Runs training and saves output to log file. Handles errors gracefully.
"""
import sys
import os
import time
import signal

# Redirect stdout/stderr to log file
log_path = "/home/z/my-project/download/nafs_training_output.log"
log_file = open(log_path, "w", buffering=1)  # line buffered
sys.stdout = log_file
sys.stderr = log_file

print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Training runner started", flush=True)

try:
    # Import and run training
    sys.path.insert(0, "/home/z/my-project/nafs-ai")
    from train import train
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Import successful, starting training...", flush=True)
    train()
except Exception as e:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] FATAL ERROR: {e}", flush=True)
    import traceback
    traceback.print_exc()
finally:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Training runner finished", flush=True)
    log_file.close()
