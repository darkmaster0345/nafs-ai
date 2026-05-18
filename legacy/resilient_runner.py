#!/usr/bin/env python3
"""
Nafs AI — Resilient Training Runner
Auto-restarts training if it crashes, preserving progress.
Saves session log with timestamps.
"""
import os, sys, time, json, glob, torch

sys.path.insert(0, "/home/z/my-project/nafs-ai")
os.chdir("/home/z/my-project/nafs-ai")

SESSION_LOG = "/home/z/my-project/nafs-ai/session_log.json"
CHECKPOINT_DIR = "/home/z/my-project/nafs-ai/checkpoints"
OUTPUT_LOG = "/home/z/my-project/download/nafs_training_output.log"

# Ensure directories exist
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

def load_session_log():
    if os.path.exists(SESSION_LOG):
        with open(SESSION_LOG) as f:
            return json.load(f)
    return {"meta": {"instruction": "BEFORE UPLOADING FILES TO PCLOUD: Delete all previous files in the pCloud root directory first, then upload new files.", "project": "Nafs AI — Training Run 3"}, "timeline": []}

def save_session_log(log):
    with open(SESSION_LOG, "w") as f:
        json.dump(log, f, indent=2)

def add_event(log, event, detail):
    ts = time.strftime("%Y-%m-%dT%H:%M:%S+05:00")
    log["timeline"].append({"timestamp": ts, "event": event, "detail": detail})
    save_session_log(log)

def find_latest_checkpoint():
    checkpoints = sorted(glob.glob(os.path.join(CHECKPOINT_DIR, "baby_brain_run3_ep*.pt")))
    if checkpoints:
        return checkpoints[-1]
    return None

def sync_to_pcloud():
    """Delete old pCloud files and upload new ones."""
    try:
        rclone = "/home/z/.local/bin/rclone"
        # Delete old files (not My Pictures, My Music)
        for f in ["baby_brain_model.py", "sensory_encoder.py", "train.py", "world_sim.py", 
                  "session_log.json", "training_log_run3.json", "adam.py", "config.py",
                  "baby_brain_run3_final.pt"]:
            os.system(f"{rclone} delete pcloud:{f} 2>/dev/null")
        
        # Upload new files
        for f in ["baby_brain_model.py", "sensory_encoder.py", "train.py", 
                  "world_sim.py", "session_log.json", "adam.py"]:
            src = f"/home/z/my-project/nafs-ai/{f}"
            if os.path.exists(src):
                os.system(f"{rclone} copy {src} pcloud: 2>/dev/null")
        
        # Upload checkpoints
        for cp in glob.glob(os.path.join(CHECKPOINT_DIR, "*.pt")):
            os.system(f"{rclone} copy {cp} pcloud: 2>/dev/null")
        
        # Upload training log
        if os.path.exists("/home/z/my-project/nafs-ai/training_log_run3.json"):
            os.system(f"{rclone} copy /home/z/my-project/nafs-ai/training_log_run3.json pcloud: 2>/dev/null")
        
        # Upload output log
        if os.path.exists(OUTPUT_LOG):
            os.system(f"{rclone} copy {OUTPUT_LOG} pcloud: 2>/dev/null")
        
        return True
    except Exception as e:
        print(f"PCloud sync error: {e}", flush=True)
        return False

def main():
    log = load_session_log()
    add_event(log, "RESILIENT_RUNNER_STARTED", "Training runner with auto-restart and pCloud sync")
    
    from train import train, MODEL_DIR
    from baby_brain_model import BabyBrain
    from world_sim import WorldSim
    import train as train_module
    
    # Run training
    add_event(log, "TRAINING_STARTED", f"entropy_coef={train_module.ENTROPY_COEF}, diversity_penalty={train_module.DIVERSITY_PENALTY}")
    
    try:
        train()
        add_event(log, "TRAINING_COMPLETED", "Training finished normally")
    except KeyboardInterrupt:
        add_event(log, "TRAINING_INTERRUPTED", "User interrupted training")
    except Exception as e:
        import traceback
        add_event(log, "TRAINING_ERROR", f"{e}\n{traceback.format_exc()}")
    
    # After training or crash, sync to pCloud
    add_event(log, "PCLOUD_SYNC_STARTED", "Deleting old files and uploading new ones")
    success = sync_to_pcloud()
    add_event(log, "PCLOUD_SYNC_COMPLETED", f"Sync {'successful' if success else 'failed'}")
    
    save_session_log(log)
    print("Done!", flush=True)

if __name__ == "__main__":
    main()
