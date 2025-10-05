import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np

class ActorCriticNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(ActorCriticNetwork, self).__init__()
        self.shared_layer = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU()
        )
        # Actor head: Outputs action probabilities
        self.actor_head = nn.Linear(128, action_size)
        
        # Critic head: Outputs a single value for the state
        self.critic_head = nn.Linear(128, 1)

    def forward(self, state):
        x = self.shared_layer(state)
        action_logits = self.actor_head(x)
        state_value = self.critic_head(x)
        # We use softmax on the logits later to get probabilities
        return action_logits, state_value

class ActorCriticAgent:
    def __init__(self, state_size, action_size, learning_rate=0.002, gamma=0.99):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network = ActorCriticNetwork(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        self.gamma = gamma

        self.state_size = state_size
        self.action_size = action_size
        
    def choose_action(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action_logits, _ = self.network(state_tensor)
        probabilities = torch.softmax(action_logits, dim=-1)
        
        dist = Categorical(probabilities)
        action = dist.sample()
        
        return action.item()

    def learn(self, state, action, reward, next_state):
        """Update the networks at every single step."""
        # Skip learning if action is None (no action was taken)
        if action is None:
            return
            
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
        reward_tensor = torch.FloatTensor([reward]).unsqueeze(0).to(self.device)
        action_tensor = torch.LongTensor([action]).unsqueeze(0).to(self.device)

        # Get values from the network
        action_logits, state_value = self.network(state_tensor)
        _, next_state_value = self.network(next_state_tensor)
        
        # --- Calculate losses ---
        # 1. Critic Loss (how wrong was the state-value estimate?)
        advantage = reward_tensor + self.gamma * next_state_value - state_value
        critic_loss = advantage.pow(2).mean() # Mean Squared Error

        # 2. Actor Loss (how to update the policy?)
        probabilities = torch.softmax(action_logits, dim=-1)
        dist = Categorical(probabilities)
        log_prob = dist.log_prob(action_tensor.squeeze())
        # Use advantage as the baseline for the update
        actor_loss = (-log_prob * advantage.detach()).mean()

        # Combine losses and update
        total_loss = actor_loss + 0.5 * critic_loss # 0.5 to balance the critic loss
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()