import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque, OrderedDict
import pandas as pd

class ValueNetwork(nn.Module):
    def __init__(self, state_size, hidden_dims=[64, 64]):
        super(ValueNetwork, self).__init__()
        layers = []
        input_dim = state_size
        for h_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, h_dim))
            layers.append(nn.ReLU())
            input_dim = h_dim
        layers.append(nn.Linear(input_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, state):
        return self.network(state)

class ValueDQNAgent:
    def __init__(self, state_size: int, learning_rate=0.001, gamma=0.95, feature_mask=None, hidden_dims=[64, 64]):
        self.state_size = state_size
        self.feature_mask = feature_mask # List of booleans [recency, freq, rank]
        
        # Adjust state size based on mask
        if self.feature_mask:
            self.input_size = sum(self.feature_mask)
        else:
            self.input_size = state_size
            
        self.device = torch.device("cpu")
        
        self.q_network = ValueNetwork(self.input_size, hidden_dims).to(self.device)
        self.target_network = ValueNetwork(self.input_size, hidden_dims).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.loss_function = nn.MSELoss()
        self.memory = deque(maxlen=50000)
        self.gamma = gamma

    def get_values(self, states):
        if self.feature_mask:
            states = states[:, self.feature_mask]
            
        with torch.no_grad():
            states_tensor = torch.FloatTensor(states).to(self.device)
            values = self.q_network(states_tensor)
        return values.cpu().numpy().flatten()

    def get_numpy_weights(self):
        """Extracts weights and biases for fast numpy-based inference."""
        params = list(self.q_network.parameters())
        return [p.detach().cpu().numpy() for p in params]

    def remember(self, state, reward, next_state):
        self.memory.append((state, reward, next_state))

    def learn(self, batch_size: int):
        if len(self.memory) < batch_size: return
        
        minibatch = random.sample(self.memory, batch_size)
        states, rewards, next_states = zip(*minibatch)

        states = torch.FloatTensor(np.array(states)).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)

        current_q_values = self.q_network(states)
        with torch.no_grad():
            next_q_values = self.target_network(next_states)
        
        target_q_values = rewards + (self.gamma * next_q_values)
        loss = self.loss_function(current_q_values, target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def save(self, path):
        torch.save(self.q_network.state_dict(), path)

    def load(self, path):
        self.q_network.load_state_dict(torch.load(path, map_location=self.device))
        self.q_network.eval()

class TrainingCache:
    def __init__(self, capacity: int, agent: ValueDQNAgent):
        self.capacity, self.agent = capacity, agent
        self.cache, self.freq, self.ts = OrderedDict(), {}, 0
        self.hits, self.misses = 0, 0
        
    def _get_state(self, item_id):
        recency = self.ts - self.cache.get(item_id, self.ts)
        frequency = self.freq.get(item_id, 1)
        try: rank = list(self.cache.keys()).index(item_id) + 1
        except ValueError: rank = self.capacity
        return np.array([recency, frequency, rank / self.capacity])

    def process_request(self, item_id):
        self.ts += 1
        curr_state = self._get_state(item_id)
        self.freq[item_id] = self.freq.get(item_id, 0) + 1
        
        if item_id in self.cache:
            reward, self.hits = 1.0, self.hits + 1
            self.cache.move_to_end(item_id)
        else:
            reward, self.misses = -1.0, self.misses + 1
            if len(self.cache) >= self.capacity: self.cache.popitem(last=False)
            
        self.cache[item_id] = self.ts
        next_state = self._get_state(item_id)
        self.agent.remember(curr_state, reward, next_state)

# Minimal Training Block to generate the model
if __name__ == "__main__":
    print("--- Training RL Agent ---")
    requests = pd.read_csv("data/training_data.csv")
    agent = ValueDQNAgent(state_size=3)
    env = TrainingCache(capacity=30, agent=agent)

    for i, row in requests.iterrows():
        env.process_request(int(row['item_id']))
        if (i + 1) % 64 == 0: agent.learn(64)
        if (i + 1) % 100 == 0: agent.update_target_network()
        
    agent.save("models/rl_eviction_model.pth")
    print(f"Training Complete. Final Hit Rate: {(env.hits / (env.hits + env.misses))*100:.2f}%")