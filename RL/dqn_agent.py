import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        # MODIFICATION: Made the network wider for more capacity
        self.layer1 = nn.Linear(state_size, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, action_size)

    def forward(self, state):
        x = torch.relu(self.layer1(state))
        x = torch.relu(self.layer2(x))
        return self.layer3(x)

class DQNAgent:
    def __init__(self, state_size: int, action_size: int):
        self.state_size = state_size
        self.action_size = action_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.q_network = QNetwork(state_size, action_size).to(self.device)
        self.target_network = QNetwork(state_size, action_size).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=0.001)
        self.loss_function = nn.MSELoss()

        # MODIFICATION: Increased memory and slowed decay for more robust learning
        self.memory = deque(maxlen=20000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9999 

    def remember(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))

    def choose_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        return np.argmax(q_values.cpu().data.numpy())

    def learn(self, batch_size: int):
        if len(self.memory) < batch_size:
            return
            
        minibatch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states = zip(*minibatch)

        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)

        current_q_values = self.q_network(states).gather(1, actions)
        
        # --- MODIFICATION: Double DQN Implementation ---
        # 1. Get the best action for the next state from the MAIN network
        next_actions = self.q_network(next_states).argmax(dim=1, keepdim=True)
        # 2. Get the Q-value for that action from the TARGET network
        with torch.no_grad():
            next_q_values = self.target_network(next_states).gather(1, next_actions)
        # -----------------------------------------------

        target_q_values = rewards + (self.gamma * next_q_values)

        loss = self.loss_function(current_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())