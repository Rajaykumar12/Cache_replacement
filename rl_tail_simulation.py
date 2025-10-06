import pandas as pd
import numpy as np
import joblib
from collections import OrderedDict
import matplotlib.pyplot as plt
import time

# Import the new RL agent from the other file
from rl_tail import ValueDQNAgent

# --- Configuration ---
CACHE_CAPACITY = 30
MODEL_FILE = "models/xgb_model.pkl"  # Supervised ML model for comparison
REQUEST_LOG_FILE = "data/training_data.csv"
HYBRID_TAIL_SAMPLE_SIZE = 16 # K, the number of cold items to evaluate

# --- Reinforcement Learning Configuration ---
NUM_FEATURES = 3 # Recency, Frequency, Recency Rank
BATCH_SIZE = 64
TARGET_UPDATE_FREQUENCY = 100

# ===================================================================
# =================== CACHE IMPLEMENTATIONS =========================
# ===================================================================

class LRUCache:
    def __init__(self, capacity: int):
        self.cache, self.capacity, self.hits, self.misses = OrderedDict(), capacity, 0, 0
    def process_request(self, item_id):
        if item_id in self.cache: self.hits += 1; self.cache.move_to_end(item_id)
        else:
            self.misses += 1
            if len(self.cache) >= self.capacity: self.cache.popitem(last=False)
            self.cache[item_id] = True
    def get_hit_rate(self): total = self.hits + self.misses; return (self.hits / total) * 100 if total > 0 else 0

class HybridCache: # Uses the pre-trained supervised ML model
    def __init__(self, capacity: int, model, tail_sample_size: int):
        self.capacity, self.model, self.k = capacity, model, tail_sample_size
        self.cache, self.item_history_last_access, self.item_history_frequency, self.current_timestamp, self.hits, self.misses = OrderedDict(), {}, {}, 0, 0, 0
    def _evict(self):
        candidate_items = list(self.cache.keys())[:self.k]
        features = [[self.current_timestamp - self.item_history_last_access.get(item, self.current_timestamp), self.item_history_frequency.get(item, 1)] for item in candidate_items]
        predictions = self.model.predict(pd.DataFrame(features, columns=['time_since_last_access', 'frequency_count']))
        del self.cache[candidate_items[np.argmax(predictions)]]
    def process_request(self, item_id):
        self.current_timestamp += 1; self.item_history_last_access[item_id] = self.current_timestamp; self.item_history_frequency[item_id] = self.item_history_frequency.get(item_id, 0) + 1
        if item_id in self.cache: self.hits += 1; self.cache.move_to_end(item_id)
        else:
            self.misses += 1
            if len(self.cache) >= self.capacity: self._evict()
            self.cache[item_id] = True
    def get_hit_rate(self): total = self.hits + self.misses; return (self.hits / total) * 100 if total > 0 else 0

class RLHybridCache: # Uses the new RL agent that learns online
    def __init__(self, capacity: int, agent: ValueDQNAgent, tail_sample_size: int):
        self.capacity, self.agent, self.k = capacity, agent, tail_sample_size
        self.cache, self.item_history_frequency, self.current_timestamp, self.hits, self.misses = OrderedDict(), {}, 0, 0, 0
    def _get_item_state(self, item_id):
        recency = self.current_timestamp - self.cache.get(item_id, self.current_timestamp)
        frequency = self.item_history_frequency.get(item_id, 1)
        try: rank = list(self.cache.keys()).index(item_id) + 1
        except ValueError: rank = self.capacity
        recency_rank = rank / self.capacity
        return np.array([recency, frequency, recency_rank])
    def _evict(self):
        candidate_items = list(self.cache.keys())[:self.k]
        candidate_states = [self._get_item_state(item) for item in candidate_items]
        values = [self.agent.get_value(state) for state in candidate_states]
        del self.cache[candidate_items[np.argmin(values)]]
    def process_request(self, item_id: int):
        self.current_timestamp += 1
        current_state = self._get_item_state(item_id)
        self.item_history_frequency[item_id] = self.item_history_frequency.get(item_id, 0) + 1
        if item_id in self.cache:
            self.hits += 1; reward = 1.0; self.cache.move_to_end(item_id)
        else:
            self.misses += 1; reward = -1.0
            if len(self.cache) >= self.capacity: self._evict()
        self.cache[item_id] = self.current_timestamp
        next_state = self._get_item_state(item_id)
        self.agent.remember(current_state, reward, next_state)
    def get_hit_rate(self): total = self.hits + self.misses; return (self.hits / total) * 100 if total > 0 else 0

# ===================================================================
# =================== MAIN SIMULATION LOGIC =========================
# ===================================================================

if __name__ == "__main__":
    print("--- Cache Performance Simulation: RL Hybrid vs ML Hybrid ---")
    try:
        requests_df = pd.read_csv(REQUEST_LOG_FILE)
        ml_model = joblib.load(MODEL_FILE)
    except FileNotFoundError as e:
        print(f"Error loading a required file: {e}"); exit()
    
    rl_agent = ValueDQNAgent(state_size=NUM_FEATURES)
    
    caches = {
        "LRU": LRUCache(capacity=CACHE_CAPACITY),
        "ML Hybrid Tail Sampling": HybridCache(capacity=CACHE_CAPACITY, model=ml_model, tail_sample_size=HYBRID_TAIL_SAMPLE_SIZE),
        "RL Hybrid Tail Sampling": RLHybridCache(capacity=CACHE_CAPACITY, agent=rl_agent, tail_sample_size=HYBRID_TAIL_SAMPLE_SIZE)
    }
    
    total_requests = len(requests_df)
    history, time_steps = {name: [] for name in caches.keys()}, []
    print(f"Starting simulation with {total_requests} requests...")
    start_time = time.time()
    
    for i, row in requests_df.iterrows():
        item_id = int(row['item_id'])
        for cache in caches.values(): cache.process_request(item_id)
        
        if (i + 1) % BATCH_SIZE == 0: rl_agent.learn(BATCH_SIZE)
        if (i + 1) % TARGET_UPDATE_FREQUENCY == 0: rl_agent.update_target_network()
            
        if (i + 1) % 50000 == 0:
            print(f"  ...processed {i + 1}/{total_requests} requests")
            history.update({name: history[name] + [cache.get_hit_rate()] for name, cache in caches.items()})
            time_steps.append(i + 1)
            
    print(f"\nSimulation finished in {time.time() - start_time:.2f} seconds!")
    print("--- FINAL CACHE PERFORMANCE RESULTS ---")
    results = {name: cache.get_hit_rate() for name, cache in caches.items()}
    for name, hit_rate in results.items(): print(f"{name}: \t{hit_rate:.2f}%")

    # print("\nGenerating performance graphs...")
    # names, hit_rates = list(results.keys()), list(results.values())
    # plt.figure(figsize=(10, 6)); bars = plt.bar(names, hit_rates, color=['skyblue', 'lightgreen', 'salmon'])
    # plt.ylabel('Cache Hit Rate (%)'); plt.title('Performance Comparison of Cache Algorithms'); plt.ylim(0, max(hit_rates) * 1.15)
    # for bar in bars: yval = bar.get_height(); plt.text(bar.get_x() + bar.get_width()/2.0, yval + 1, f'{yval:.2f}%', ha='center', va='bottom')
    # plt.tight_layout(); plt.show()

    # plt.figure(figsize=(12, 7))
    # for name, hist_data in history.items():
    #     if hist_data: plt.plot(time_steps, hist_data, label=name, marker='o', linestyle='--')
    # plt.xlabel('Number of Requests Processed'); plt.ylabel('Cache Hit Rate (%)'); plt.title('Learning Curves of Different Cache Algorithms')
    # plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()