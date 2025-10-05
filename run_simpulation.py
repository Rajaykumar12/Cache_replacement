import pandas as pd
import numpy as np
import joblib
from collections import OrderedDict, deque, defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle

# Import your existing RL components
from dqn_agent import DQNAgent
from policy_agent import PolicyGradientAgent
from actor_critic_agent import ActorCriticAgent
from cache_environment import RLCache

# --- Configuration ---
CACHE_CAPACITY = 20
REQUEST_LOG_FILE = "data/training_data.csv"
XGB_FILE = "models/xgb_model.pkl"
LGB_FILE = "models/lgb_model.pkl"

# --- Reinforcement Learning Configuration ---
NUM_FEATURES = 3  # Recency, Frequency, and Recency Rank
BATCH_SIZE = 64
TARGET_UPDATE_FREQUENCY = 100
PG_EPISODE_LENGTH = 500

# ===================================================================
# =================== TRADITIONAL CACHE IMPLEMENTATIONS ===========
# ===================================================================

class LRUCache:
    def __init__(self, capacity: int):
        self.cache, self.capacity, self.hits, self.misses = OrderedDict(), capacity, 0, 0
    
    def process_request(self, item_id):
        if item_id in self.cache:
            self.hits += 1
            self.cache.move_to_end(item_id)
        else:
            self.misses += 1
            if len(self.cache) >= self.capacity:
                self.cache.popitem(last=False)
            self.cache[item_id] = True
    
    def get_hit_rate(self):
        total = self.hits + self.misses
        return (self.hits / total) * 100 if total > 0 else 0

class FIFOCache:
    def __init__(self, capacity: int):
        self.capacity, self.cache, self.queue, self.hits, self.misses = capacity, set(), deque(), 0, 0
    
    def process_request(self, item_id):
        if item_id in self.cache:
            self.hits += 1
        else:
            self.misses += 1
            if len(self.cache) >= self.capacity:
                self.cache.remove(self.queue.popleft())
            self.cache.add(item_id)
            self.queue.append(item_id)
    
    def get_hit_rate(self):
        total = self.hits + self.misses
        return (self.hits / total) * 100 if total > 0 else 0

class MLCache:
    def __init__(self, capacity: int, model):
        self.capacity, self.model = capacity, model
        self.cache, self.item_history_last_access, self.item_history_frequency = {}, {}, {}
        self.current_timestamp, self.hits, self.misses = 0, 0, 0
    
    def _evict(self):
        items_in_cache = list(self.cache.keys())
        features = [[self.current_timestamp - self.item_history_last_access.get(item, self.current_timestamp), 
                     self.item_history_frequency.get(item, 1)] for item in items_in_cache]
        predictions = self.model.predict(pd.DataFrame(features, columns=['time_since_last_access', 'frequency_count']))
        del self.cache[items_in_cache[np.argmax(predictions)]]
    
    def process_request(self, item_id):
        self.current_timestamp += 1
        self.item_history_last_access[item_id] = self.current_timestamp
        self.item_history_frequency[item_id] = self.item_history_frequency.get(item_id, 0) + 1
        
        if item_id in self.cache:
            self.hits += 1
        else:
            self.misses += 1
            if len(self.cache) >= self.capacity:
                self._evict()
            self.cache[item_id] = True
    
    def get_hit_rate(self):
        total = self.hits + self.misses
        return (self.hits / total) * 100 if total > 0 else 0

class OptimalCache:
    def __init__(self, capacity: int, future_uses: dict):
        self.capacity, self.future_uses = capacity, future_uses
        self.cache, self.hits, self.misses, self.use_pointers = set(), 0, 0, defaultdict(int)
    
    def _evict(self, current_index: int):
        farthest_item, max_distance = -1, -1
        for item in self.cache:
            future_accesses, pointer = self.future_uses[item], self.use_pointers[item]
            while pointer < len(future_accesses) and future_accesses[pointer] <= current_index:
                pointer += 1
            self.use_pointers[item] = pointer
            if pointer == len(future_accesses):
                farthest_item = item
                break 
            distance = future_accesses[pointer] - current_index
            if distance > max_distance:
                max_distance, farthest_item = distance, item
        self.cache.remove(farthest_item)
    
    def process_request(self, item_id, current_index: int):
        if item_id in self.cache:
            self.hits += 1
        else:
            self.misses += 1
            if len(self.cache) >= self.capacity:
                self._evict(current_index)
            self.cache.add(item_id)
    
    def get_hit_rate(self):
        total = self.hits + self.misses
        return (self.hits / total) * 100 if total > 0 else 0

# ===================================================================
# =================== RL CACHE WRAPPERS ============================
# ===================================================================

class PolicyGradientCache:
    def __init__(self, capacity, agent):
        self.rl_cache = RLCache(capacity, agent)
        
    def process_request(self, item_id):
        self.rl_cache.process_request(item_id)
        # Store reward for policy gradient learning
        reward = 1 if item_id in self.rl_cache.cache else -1
        self.rl_cache.agent.rewards.append(reward)
    
    def get_hit_rate(self):
        return self.rl_cache.get_hit_rate()
    
    @property
    def hits(self):
        return self.rl_cache.hits
    
    @property
    def misses(self):
        return self.rl_cache.misses

class ActorCriticCache:
    def __init__(self, capacity, agent):
        self.rl_cache = RLCache(capacity, agent)
        self.prev_state = None
        self.prev_action = None
    
    def process_request(self, item_id):
        current_state = self.rl_cache._get_state()
        
        # If we have a previous state AND action, learn from the transition
        if self.prev_state is not None and self.prev_action is not None:
            reward = 1 if item_id in self.rl_cache.cache else -1
            self.rl_cache.agent.learn(self.prev_state, self.prev_action, reward, current_state)
        
        # Process the request
        if item_id in self.rl_cache.cache:
            self.rl_cache.hits += 1
            self.rl_cache.cache[item_id] = self.rl_cache.current_timestamp
            self.rl_cache.cache.move_to_end(item_id)
            self.prev_action = None  # No action taken (cache hit)
        else:
            self.rl_cache.misses += 1
            if len(self.rl_cache.cache) >= self.rl_cache.capacity:
                # Need to evict - this is where we take an action
                action_index = self.rl_cache.agent.choose_action(current_state)
                items_in_cache = list(self.rl_cache.cache.keys())
                if action_index < len(items_in_cache):
                    del self.rl_cache.cache[items_in_cache[action_index]]
                else:
                    self.rl_cache.cache.popitem(last=False)
                self.prev_action = action_index  # Store the action we took
            else:
                self.prev_action = None  # No eviction needed, no action taken
            
            self.rl_cache.cache[item_id] = self.rl_cache.current_timestamp
        
        # Update tracking variables
        self.rl_cache.current_timestamp += 1
        self.rl_cache.item_history_frequency[item_id] = self.rl_cache.item_history_frequency.get(item_id, 0) + 1
        self.prev_state = current_state
    
    def get_hit_rate(self):
        return self.rl_cache.get_hit_rate()
    
    @property
    def hits(self):
        return self.rl_cache.hits
    
    @property
    def misses(self):
        return self.rl_cache.misses

# ===================================================================
# =================== PLOTTING FUNCTIONS ============================
# ===================================================================

def plot_cache_performance(results, save_path="cache_performance_comparison.png"):
    """Plot bar chart comparing cache hit rates"""
    plt.figure(figsize=(12, 8))
    
    # Separate results by category
    rl_methods = {k: v for k, v in results.items() if 'RL' in k}
    ml_methods = {k: v for k, v in results.items() if 'ML' in k}
    traditional_methods = {k: v for k, v in results.items() if k in ['LRU', 'FIFO']}
    optimal = {k: v for k, v in results.items() if k == 'Optimal'}
    
    # Color scheme
    colors = {'RL': '#FF6B6B', 'ML': '#4ECDC4', 'Traditional': '#45B7D1', 'Optimal': '#96CEB4'}
    
    methods, hit_rates, method_colors = [], [], []
    
    for category, methods_dict in [('RL', rl_methods), ('ML', ml_methods), 
                                   ('Traditional', traditional_methods), ('Optimal', optimal)]:
        for method, rate in methods_dict.items():
            methods.append(method)
            hit_rates.append(rate)
            method_colors.append(colors[category])
    
    bars = plt.bar(methods, hit_rates, color=method_colors, alpha=0.8, edgecolor='black', linewidth=1)
    
    plt.title('Cache Replacement Algorithm Performance Comparison', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Cache Replacement Algorithm', fontsize=12, fontweight='bold')
    plt.ylabel('Hit Rate (%)', fontsize=12, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    
    for bar, rate in zip(bars, hit_rates):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                f'{rate:.2f}%', ha='center', va='bottom', fontweight='bold')
    
    legend_elements = [Rectangle((0,0),1,1, facecolor=colors[cat], alpha=0.8, label=cat) 
                      for cat in ['RL', 'ML', 'Traditional', 'Optimal']]
    plt.legend(handles=legend_elements, loc='upper left')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_cache_statistics(caches, save_path="cache_statistics.png"):
    """Plot detailed statistics"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    cache_names = list(caches.keys())
    hits = [cache.hits for cache in caches.values()]
    misses = [cache.misses for cache in caches.values()]
    hit_rates = [cache.get_hit_rate() for cache in caches.values()]
    
    # 1. Hits vs Misses
    x_pos = np.arange(len(cache_names))
    ax1.bar(x_pos, hits, label='Hits', color='#2ECC71', alpha=0.8)
    ax1.bar(x_pos, misses, bottom=hits, label='Misses', color='#E74C3C', alpha=0.8)
    ax1.set_title('Cache Hits vs Misses', fontweight='bold')
    ax1.set_xlabel('Cache Algorithm')
    ax1.set_ylabel('Number of Requests')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(cache_names, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # 2. Top performers pie chart
    sorted_indices = np.argsort(hit_rates)[-4:]
    top_names = [cache_names[i] for i in sorted_indices]
    top_rates = [hit_rates[i] for i in sorted_indices]
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    ax2.pie(top_rates, labels=top_names, autopct='%1.2f%%', colors=colors, startangle=90)
    ax2.set_title('Top 4 Cache Algorithms by Hit Rate', fontweight='bold')
    
    # 3. Hit rate comparison
    y_pos = np.arange(len(cache_names))
    bars = ax3.barh(y_pos, hit_rates, color='#3498DB', alpha=0.8)
    ax3.set_title('Hit Rate Comparison', fontweight='bold')
    ax3.set_xlabel('Hit Rate (%)')
    ax3.set_ylabel('Cache Algorithm')
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(cache_names)
    ax3.grid(axis='x', alpha=0.3)
    
    for i, (bar, rate) in enumerate(zip(bars, hit_rates)):
        ax3.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2, 
                f'{rate:.2f}%', va='center', fontweight='bold')
    
    # 4. Performance relative to optimal
    optimal_rate = next(rate for name, rate in zip(cache_names, hit_rates) if name == 'Optimal')
    relative_performance = [(rate / optimal_rate) * 100 for rate in hit_rates]
    
    bars = ax4.bar(cache_names, relative_performance, color='#9B59B6', alpha=0.8)
    ax4.set_title('Performance Relative to Optimal (%)', fontweight='bold')
    ax4.set_xlabel('Cache Algorithm')
    ax4.set_ylabel('Relative Performance (%)')
    ax4.set_xticklabels(cache_names, rotation=45, ha='right')
    ax4.grid(axis='y', alpha=0.3)
    ax4.axhline(y=100, color='red', linestyle='--', label='Optimal (100%)')
    ax4.legend()
    
    for bar, perf in zip(bars, relative_performance):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{perf:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

# ===================================================================
# =================== MAIN SIMULATION LOGIC =========================
# ===================================================================

if __name__ == "__main__":
    print("--- Comprehensive Cache Performance Simulation ---")

    # Load resources
    try:
        requests_df = pd.read_csv(REQUEST_LOG_FILE)
        xgb_model = joblib.load(XGB_FILE)
        lgb_model = joblib.load(LGB_FILE)
    except FileNotFoundError as e:
        print(f"Error loading file: {e}")
        exit()

    print("Pre-computing future uses for Optimal algorithm...")
    future_uses = defaultdict(list)
    [future_uses[item_id].append(index) for index, item_id in enumerate(requests_df['item_id'])]
    
    # Initialize RL Agents
    state_size = CACHE_CAPACITY * NUM_FEATURES
    action_size = CACHE_CAPACITY
    dqn_agent = DQNAgent(state_size=state_size, action_size=action_size)
    pg_agent = PolicyGradientAgent(state_size=state_size, action_size=action_size)
    ac_agent = ActorCriticAgent(state_size=state_size, action_size=action_size)
    
    # Initialize all caches
    caches = {
        "FIFO": FIFOCache(capacity=CACHE_CAPACITY),
        "LRU": LRUCache(capacity=CACHE_CAPACITY),
        "ML (XGB)": MLCache(capacity=CACHE_CAPACITY, model=xgb_model),
        "ML (LGB)": MLCache(capacity=CACHE_CAPACITY, model=lgb_model),
        "RL (DQN)": RLCache(capacity=CACHE_CAPACITY, agent=dqn_agent),
        "RL (PG)": PolicyGradientCache(capacity=CACHE_CAPACITY, agent=pg_agent),
        "RL (A2C)": ActorCriticCache(capacity=CACHE_CAPACITY, agent=ac_agent),
        "Optimal": OptimalCache(capacity=CACHE_CAPACITY, future_uses=future_uses)
    }
    
    total_requests = len(requests_df)
    print(f"Starting simulation with {total_requests} requests and cache capacity of {CACHE_CAPACITY}...")
    
    for i, row in requests_df.iterrows():
        item_id = int(row['item_id'])
        
        # Process request in all caches
        for name, cache in caches.items():
            if name == "Optimal":
                cache.process_request(item_id, current_index=i)
            else:
                cache.process_request(item_id)
        
        # RL Training
        if len(dqn_agent.memory) > BATCH_SIZE:
            dqn_agent.learn(BATCH_SIZE)
        if (i + 1) % TARGET_UPDATE_FREQUENCY == 0:
            dqn_agent.update_target_network()
        if (i + 1) % PG_EPISODE_LENGTH == 0:
            pg_agent.learn()
            
        if (i + 1) % 50000 == 0:
            print(f"  ...processed {i + 1}/{total_requests} requests")

    print("\nSimulation finished!")
    print("=" * 60)
    print("FINAL RESULTS:")
    print("=" * 60)
    
    results = {name: cache.get_hit_rate() for name, cache in caches.items()}
    
    for name, hit_rate in results.items():
        print(f"{name:20}: {hit_rate:6.2f}%")
    
    print("\nGenerating plots...")
    plot_cache_performance(results)
    plot_cache_statistics(caches)
    
    print("\nPlots saved:")
    print("- cache_performance_comparison.png")
    print("- cache_statistics.png")