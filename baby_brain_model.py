
import torch
import torch.nn as nn
import torch.nn.functional as F

class BabyBrain(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_actions):
        super(BabyBrain, self).__init__()
        self.hidden_dim = hidden_dim

        # GRU layer
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)

        # Actor head: outputs probabilities over actions
        self.actor_linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.actor_output = nn.Linear(hidden_dim, num_actions)

        # Critic head: outputs state value scalar
        self.critic_linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.critic_output = nn.Linear(hidden_dim, 1)

    def forward(self, sensory_input, hidden_state):
        # sensory_input: (batch_size, input_dim)
        # hidden_state: (1, batch_size, hidden_dim) - GRU expects (num_layers * num_directions, batch, hidden_size)

        # Add a sequence dimension for GRU (batch_size, 1, input_dim)
        sensory_input = sensory_input.unsqueeze(1)

        # Pass through GRU
        gru_out, next_hidden_state = self.gru(sensory_input, hidden_state)
        # gru_out: (batch_size, 1, hidden_dim)
        # next_hidden_state: (1, batch_size, hidden_dim)

        # Remove sequence dimension for actor and critic heads
        gru_out = gru_out.squeeze(1) # (batch_size, hidden_dim)

        # Actor head — return raw logits for stable PPO with Categorical distribution
        # (softmax is applied inside Categorical, not here, for numerical stability)
        actor_hidden = F.relu(self.actor_linear1(gru_out))
        action_logits = self.actor_output(actor_hidden)

        # Critic head
        critic_hidden = F.relu(self.critic_linear1(gru_out))
        state_value = self.critic_output(critic_hidden)

        return action_logits, state_value, next_hidden_state

    def init_hidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_dim)

# Helper function to count parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == '__main__':
    input_dim = 12  # 12-dimensional sensory vector
    hidden_dim = 256 # 64 hidden units for GRU
    num_actions = 8 # EXPLORE, EAT, DRINK, SLEEP, HIDE, MOVE, FLEE, IDLE

    model = BabyBrain(input_dim, hidden_dim, num_actions)
    print(f"Model Architecture:\n{model}")

    total_params = count_parameters(model)
    print(f"Total trainable parameters: {total_params}")

    # Test a forward pass
    batch_size = 1
    sensory_input = torch.randn(batch_size, input_dim)
    hidden_state = model.init_hidden(batch_size)

    action_logits, state_value, next_hidden_state = model(sensory_input, hidden_state)

    print(f"\nSensory Input Shape: {sensory_input.shape}")
    print(f"Initial Hidden State Shape: {hidden_state.shape}")
    print(f"Action Logits Shape: {action_logits.shape}")
    print(f"State Value Shape: {state_value.shape}")
    print(f"Next Hidden State Shape: {next_hidden_state.shape}")

    assert action_logits.shape == (batch_size, num_actions)
    assert state_value.shape == (batch_size, 1)
    assert next_hidden_state.shape == (1, batch_size, hidden_dim)

    # Check parameter count for typical range (50K-500K)
    assert 50000 <= total_params <= 500000, f"Parameter count {total_params} not in range [50K, 500K]"
    print("Parameter count is within the specified range.")
