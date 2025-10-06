import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np

class PolicyNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(PolicyNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, action_size),
            nn.Softmax(dim=-1) # Outputs a probability distribution
        )

    def forward(self, state):
        return self.network(state)

class PolicyGradientAgent:
    def __init__(self, state_size, action_size, learning_rate=0.001, gamma=0.99):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_network = PolicyNetwork(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=learning_rate)
        self.gamma = gamma
        
        # Add these missing attributes
        self.state_size = state_size
        self.action_size = action_size
        
        # Buffers to store experiences for an episode
        self.rewards = []
        self.log_probs = []

    def choose_action(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        # Get action probabilities from the policy network
        probabilities = self.policy_network(state_tensor)
        
        # Sample an action from the distribution
        dist = Categorical(probabilities)
        action = dist.sample()
        
        # Store the log probability of the chosen action (needed for learning)
        self.log_probs.append(dist.log_prob(action))
        
        return action.item()

    def remember(self, state, action, reward, next_state):
        """Store reward for policy gradient learning (compatibility with RLCache)"""
        # Policy gradient doesn't use experience replay like DQN
        # We just store the reward for the current episode
        self.rewards.append(reward)

    def learn(self):
        """Update the policy network using the rewards from a completed episode."""
        if len(self.rewards) == 0:
            return
            
        discounted_returns = []
        current_return = 0
        
        # Calculate the discounted return for each step, working backwards
        for reward in reversed(self.rewards):
            current_return = reward + self.gamma * current_return
            discounted_returns.insert(0, current_return)
            
        returns_tensor = torch.FloatTensor(discounted_returns).to(self.device)
        
        # Normalize returns for stability
        if len(returns_tensor) > 1:
            returns_tensor = (returns_tensor - returns_tensor.mean()) / (returns_tensor.std() + 1e-9)
        
        policy_loss = []
        for log_prob, R in zip(self.log_probs, returns_tensor):
            policy_loss.append(-log_prob * R)
        
        if len(policy_loss) > 0:
            self.optimizer.zero_grad()
            loss = torch.cat(policy_loss).sum()
            loss.backward()
            self.optimizer.step()
        
        # Clear the episode buffers
        self.rewards = []
        self.log_probs = []