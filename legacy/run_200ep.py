"""Quick 200-episode training run for testing."""
import sys
sys.path.insert(0, '.')

# Override EPISODES before importing train
import train
train.EPISODES = 200
train.LOG_INTERVAL = 25
train.SAVE_INTERVAL = 50
train.TRAINING_LOG_PATH = "training_log_run4_200ep.json"

if __name__ == '__main__':
    train.train()
