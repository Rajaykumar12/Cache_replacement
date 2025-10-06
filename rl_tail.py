import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

class ValueNetwork(nn.Module):
    """A network that takes item features and outputs a single value."""
    def __init__(self, state_size):
        super(ValueNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1) # Single output node for the value
        )
    def forward(self, state):
        return self.network(state)

class ValueDQNAgent:
    """An RL agent that learns a state-value function for cache items."""
    def __init__(self, state_size: int, learning_rate=0.001, gamma=0.95):
        self.state_size = state_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.q_network = ValueNetwork(state_size).to(self.device)
        self.target_network = ValueNetwork(state_size).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.loss_function = nn.MSELoss()
        self.memory = deque(maxlen=50000)
        self.gamma = gamma

    def get_value(self, state):
        """Predicts the value of a single item's state."""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            value = self.q_network(state_tensor)
        return value.item()

    def remember(self, state, reward, next_state):
        """Stores an experience tuple in the replay buffer."""
        self.memory.append((state, reward, next_state))

    def learn(self, batch_size: int):
        """Trains the ValueNetwork using a batch of experiences."""
        if len(self.memory) < batch_size:
            return
        
        minibatch = random.sample(self.memory, batch_size)
        states, rewards, next_states = zip(*minibatch)

        states = torch.FloatTensor(np.array(states)).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)

        current_q_values = self.q_network(states)
        
        with torch.no_grad():
            next_q_values = self.target_network(next_states)
        
        # The Bellman equation for a state-value function
        target_q_values = rewards + (self.gamma * next_q_values)

        loss = self.loss_function(current_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
    def update_target_network(self):
        """Updates the target network with the weights from the main network."""
        self.target_network.load_state_dict(self.q_network.state_dict())