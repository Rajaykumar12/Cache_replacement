import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import pandas as pd
import time
from collections import OrderedDict
import matplotlib.pyplot as plt



class ValueNetwork(nn.Module):
    """A network that takes item features and outputs a single value score."""
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
        
        # Add tracking for training metrics
        self.training_losses = []
        self.training_rewards = []
        self.episode_rewards = []

    def get_value(self, state):
        """Predicts the value of a single item's state."""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            value = self.q_network(state_tensor)
        return value.item()

    def get_values(self, states):
        """Predicts the values for a batch of item states."""
        with torch.no_grad():
            states_tensor = torch.FloatTensor(states).to(self.device)
            values = self.q_network(states_tensor)
        return values.cpu().numpy().flatten()

    def remember(self, state, reward, next_state):
        """Stores an experience tuple in the replay buffer."""
        self.memory.append((state, reward, next_state))

    def learn(self, batch_size: int):
        """Trains the ValueNetwork using a batch of experiences."""
        if len(self.memory) < batch_size:
            return 0
        
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
        
        # Track training loss
        loss_value = loss.item()
        self.training_losses.append(loss_value)
        return loss_value

    def update_target_network(self):
        """Updates the target network with the weights from the main network."""
        self.target_network.load_state_dict(self.q_network.state_dict())

    def save(self, path):
        """Saves the trained weights of the main Q-network."""
        torch.save(self.q_network.state_dict(), path)
        print(f"Model saved to {path}")

    def load(self, path):
        """Loads pre-trained weights into the main Q-network."""
        self.q_network.load_state_dict(torch.load(path, map_location=self.device))
        self.q_network.eval() # Set the network to evaluation mode
        print(f"Model loaded from {path}")

# ===================================================================
# =================== TRAINING ENVIRONMENT ==========================
# ===================================================================

class TrainingCache:
    """A minimal cache environment for the agent to learn in."""
    def __init__(self, capacity: int, agent: ValueDQNAgent):
        self.capacity, self.agent, self.NUM_FEATURES = capacity, agent, agent.state_size
        self.cache, self.item_history_frequency, self.current_timestamp = OrderedDict(), {}, 0
        
        # Add performance tracking
        self.hits = 0
        self.misses = 0
        self.episode_rewards = []
        self.hit_rates = []
        
    def _get_item_state(self, item_id):
        recency = self.current_timestamp - self.cache.get(item_id, self.current_timestamp)
        frequency = self.item_history_frequency.get(item_id, 1)
        try: rank = list(self.cache.keys()).index(item_id) + 1
        except ValueError: rank = self.capacity
        recency_rank = rank / self.capacity
        return np.array([recency, frequency, recency_rank])
    def process_request(self, item_id: int):
        self.current_timestamp += 1
        current_state = self._get_item_state(item_id)
        self.item_history_frequency[item_id] = self.item_history_frequency.get(item_id, 0) + 1
        if item_id in self.cache:
            reward = 1.0
            self.hits += 1
            self.cache.move_to_end(item_id)
        else:
            reward = -1.0
            self.misses += 1
            if len(self.cache) >= self.capacity: 
                self.cache.popitem(last=False)
        self.cache[item_id] = self.current_timestamp
        next_state = self._get_item_state(item_id)
        self.agent.remember(current_state, reward, next_state)
        
        # Track rewards and performance
        self.agent.training_rewards.append(reward)
        
        return reward
    
    def get_hit_rate(self):
        total = self.hits + self.misses
        return (self.hits / total) * 100 if total > 0 else 0

def generate_training_graphs(agent, training_cache, save_prefix="rl_training"):
    """Generate comprehensive training visualization graphs."""
    
    print("\nGenerating training performance graphs...")
    
    # Create a figure with multiple subplots
    fig = plt.figure(figsize=(20, 12))
    
    # 1. Training Loss Over Time
    ax1 = plt.subplot(2, 3, 1)
    if agent.training_losses:
        loss_smoothed = pd.Series(agent.training_losses).rolling(window=100, min_periods=1).mean()
        plt.plot(agent.training_losses, alpha=0.3, color='red', label='Raw Loss')
        plt.plot(loss_smoothed, color='darkred', linewidth=2, label='Smoothed Loss (100-pt avg)')
        plt.xlabel('Training Step')
        plt.ylabel('Loss')
        plt.title('Training Loss Over Time', fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    # 2. Cumulative Reward Over Time
    ax2 = plt.subplot(2, 3, 2)
    if agent.training_rewards:
        cumulative_rewards = np.cumsum(agent.training_rewards)
        reward_smoothed = pd.Series(agent.training_rewards).rolling(window=1000, min_periods=1).mean()
        
        plt.plot(cumulative_rewards, color='blue', linewidth=2, label='Cumulative Reward')
        plt.xlabel('Training Step')
        plt.ylabel('Cumulative Reward')
        plt.title('Cumulative Training Reward', fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    # 3. Average Reward (Moving Window)
    ax3 = plt.subplot(2, 3, 3)
    if agent.training_rewards:
        window_size = 1000
        avg_rewards = pd.Series(agent.training_rewards).rolling(window=window_size, min_periods=1).mean()
        plt.plot(avg_rewards, color='green', linewidth=2)
        plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        plt.xlabel('Training Step')
        plt.ylabel(f'Average Reward ({window_size}-step window)')
        plt.title('Average Reward Trend', fontweight='bold')
        plt.grid(True, alpha=0.3)
    
    # 4. Reward Distribution
    ax4 = plt.subplot(2, 3, 4)
    if agent.training_rewards:
        plt.hist(agent.training_rewards, bins=50, alpha=0.7, color='purple', edgecolor='black')
        plt.xlabel('Reward Value')
        plt.ylabel('Frequency')
        plt.title('Training Reward Distribution', fontweight='bold')
        plt.axvline(x=np.mean(agent.training_rewards), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(agent.training_rewards):.3f}')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    # 5. Learning Progress (Loss vs Reward)
    ax5 = plt.subplot(2, 3, 5)
    if agent.training_losses and agent.training_rewards:
        # Sample data points for cleaner visualization
        sample_size = min(5000, len(agent.training_losses))
        indices = np.linspace(0, len(agent.training_losses)-1, sample_size, dtype=int)
        
        loss_sample = [agent.training_losses[i] for i in indices]
        reward_sample = [agent.training_rewards[i] for i in indices]
        
        plt.scatter(loss_sample, reward_sample, alpha=0.5, s=2)
        plt.xlabel('Training Loss')
        plt.ylabel('Immediate Reward')
        plt.title('Loss vs Reward Relationship', fontweight='bold')
        plt.grid(True, alpha=0.3)
    
    # 6. Training Statistics Table
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('tight')
    ax6.axis('off')
    
    # Calculate training statistics
    if agent.training_rewards and agent.training_losses:
        stats_data = [
            ['Total Training Steps', f"{len(agent.training_rewards):,}"],
            ['Final Hit Rate', f"{training_cache.get_hit_rate():.2f}%"],
            ['Total Hits', f"{training_cache.hits:,}"],
            ['Total Misses', f"{training_cache.misses:,}"],
            ['Average Reward', f"{np.mean(agent.training_rewards):.4f}"],
            ['Final Loss', f"{agent.training_losses[-1]:.4f}" if agent.training_losses else "N/A"],
            ['Min Loss', f"{min(agent.training_losses):.4f}" if agent.training_losses else "N/A"],
            ['Max Reward', f"{max(agent.training_rewards):.4f}"],
            ['Min Reward', f"{min(agent.training_rewards):.4f}"],
            ['Reward Std Dev', f"{np.std(agent.training_rewards):.4f}"]
        ]
        
        table = ax6.table(cellText=stats_data,
                         colLabels=['Metric', 'Value'],
                         cellLoc='center',
                         loc='center')
        
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1.2, 2)
        
        # Style the table
        for i in range(len(stats_data) + 1):
            for j in range(2):
                cell = table[(i, j)]
                if i == 0:  # Header row
                    cell.set_facecolor('#34495e')
                    cell.set_text_props(weight='bold', color='white')
                else:
                    cell.set_facecolor('#ecf0f1' if i % 2 == 0 else 'white')
                cell.set_edgecolor('black')
                cell.set_linewidth(1)
    
    ax6.set_title('Training Statistics Summary', fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(f'{save_prefix}_comprehensive.png', dpi=300, bbox_inches='tight')
    print(f"Comprehensive training analysis saved as '{save_prefix}_comprehensive.png'")
    
    # Create a separate detailed loss analysis
    plt.figure(figsize=(15, 5))
    
    if agent.training_losses:
        # Plot 1: Loss over time with different smoothing windows
        plt.subplot(1, 3, 1)
        plt.plot(agent.training_losses, alpha=0.2, color='gray', label='Raw Loss')
        
        for window in [50, 200, 500]:
            smoothed = pd.Series(agent.training_losses).rolling(window=window, min_periods=1).mean()
            plt.plot(smoothed, linewidth=2, label=f'{window}-step avg')
        
        plt.xlabel('Training Step')
        plt.ylabel('Loss')
        plt.title('Loss Convergence Analysis', fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.yscale('log')  # Log scale for better visualization
        
        # Plot 2: Loss distribution over training phases
        plt.subplot(1, 3, 2)
        total_steps = len(agent.training_losses)
        phase_size = total_steps // 4
        
        phases = ['Early (0-25%)', 'Mid-Early (25-50%)', 'Mid-Late (50-75%)', 'Late (75-100%)']
        phase_losses = [
            agent.training_losses[:phase_size],
            agent.training_losses[phase_size:2*phase_size],
            agent.training_losses[2*phase_size:3*phase_size],
            agent.training_losses[3*phase_size:]
        ]
        
        plt.boxplot(phase_losses, labels=phases)
        plt.ylabel('Loss')
        plt.title('Loss Distribution by Training Phase', fontweight='bold')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # Plot 3: Learning rate analysis (loss improvement over time)
        plt.subplot(1, 3, 3)
        window = 1000
        loss_improvement = []
        for i in range(window, len(agent.training_losses)):
            recent_avg = np.mean(agent.training_losses[i-window:i])
            older_avg = np.mean(agent.training_losses[i-2*window:i-window]) if i >= 2*window else recent_avg
            improvement = older_avg - recent_avg  # Positive means improvement
            loss_improvement.append(improvement)
        
        plt.plot(range(window, len(agent.training_losses)), loss_improvement, color='purple', linewidth=2)
        plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        plt.xlabel('Training Step')
        plt.ylabel(f'Loss Improvement ({window}-step window)')
        plt.title('Learning Progress Rate', fontweight='bold')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_prefix}_loss_analysis.png', dpi=300, bbox_inches='tight')
    print(f"Detailed loss analysis saved as '{save_prefix}_loss_analysis.png'")

# ===================================================================
# =================== MAIN TRAINING SCRIPT ==========================
# ===================================================================

if __name__ == "__main__":
    # --- Configuration ---
    REQUEST_LOG_FILE = "data/training_data.csv"
    MODEL_SAVE_PATH = "models/rl_eviction_model.pth"
    NUM_FEATURES = 3
    CACHE_CAPACITY = 30
    BATCH_SIZE = 64
    TARGET_UPDATE_FREQUENCY = 100

    print("--- Starting RL Agent Training ---")
    requests_df = pd.read_csv(REQUEST_LOG_FILE)
    
    rl_agent = ValueDQNAgent(state_size=NUM_FEATURES)
    training_cache = TrainingCache(capacity=CACHE_CAPACITY, agent=rl_agent)

    start_time = time.time()
    total_requests = len(requests_df)
    
    # Track training progress
    training_losses = []
    hit_rates_over_time = []
    
    for i, row in requests_df.iterrows():
        item_id = int(row['item_id'])
        reward = training_cache.process_request(item_id)
        
        if (i + 1) % BATCH_SIZE == 0: 
            loss = rl_agent.learn(BATCH_SIZE)
            if loss > 0:
                training_losses.append(loss)
        
        if (i + 1) % TARGET_UPDATE_FREQUENCY == 0: 
            rl_agent.update_target_network()
            
        # Track hit rate periodically
        if (i + 1) % 5000 == 0:
            current_hit_rate = training_cache.get_hit_rate()
            hit_rates_over_time.append(current_hit_rate)
            
        if (i + 1) % 50000 == 0:
            current_hit_rate = training_cache.get_hit_rate()
            avg_loss = np.mean(training_losses[-100:]) if training_losses else 0
            print(f"  ...processed {i + 1}/{total_requests} requests. Hit Rate: {current_hit_rate:.2f}%, Avg Loss: {avg_loss:.4f}")
            
    training_time = time.time() - start_time
    final_hit_rate = training_cache.get_hit_rate()
    
    print(f"\nTraining finished in {training_time:.2f} seconds!")
    print(f"Final Training Hit Rate: {final_hit_rate:.2f}%")
    print(f"Total Training Steps: {len(rl_agent.training_rewards):,}")
    print(f"Total Hits: {training_cache.hits:,}")
    print(f"Total Misses: {training_cache.misses:,}")
    
    # Generate comprehensive training graphs
    generate_training_graphs(rl_agent, training_cache)
    
    # Create a simple performance over time graph
    if hit_rates_over_time:
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        steps = range(5000, len(requests_df) + 1, 5000)[:len(hit_rates_over_time)]
        plt.plot(steps, hit_rates_over_time, 'b-o', linewidth=2, markersize=4)
        plt.xlabel('Training Steps')
        plt.ylabel('Hit Rate (%)')
        plt.title('Hit Rate Progress During Training', fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        if training_losses:
            # Show loss trend
            smoothed_loss = pd.Series(training_losses).rolling(window=50, min_periods=1).mean()
            plt.plot(smoothed_loss, 'r-', linewidth=2)
            plt.xlabel('Training Batch')
            plt.ylabel('Average Loss')
            plt.title('Training Loss Trend', fontweight='bold')
            plt.grid(True, alpha=0.3)
            plt.yscale('log')
        
        plt.tight_layout()
        plt.savefig('rl_training_progress.png', dpi=300, bbox_inches='tight')
        print("Training progress saved as 'rl_training_progress.png'")
        plt.show()
    
    # Save the trained model's weights
    rl_agent.save(MODEL_SAVE_PATH)
    
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    print(f"📊 Final Hit Rate: {final_hit_rate:.2f}%")
    print(f"⏱️  Training Time: {training_time:.2f} seconds")
    print(f"🎯 Total Requests Processed: {total_requests:,}")
    print(f"🔥 Cache Hits: {training_cache.hits:,}")
    print(f"❌ Cache Misses: {training_cache.misses:,}")
    if rl_agent.training_rewards:
        print(f"💰 Average Reward: {np.mean(rl_agent.training_rewards):.4f}")
    if training_losses:
        print(f"📉 Final Loss: {training_losses[-1]:.4f}")
    print("="*60)