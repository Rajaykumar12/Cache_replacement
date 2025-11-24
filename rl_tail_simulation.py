import pandas as pd
import numpy as np
import joblib
from collections import OrderedDict
import time
from itertools import islice

from rl_tail import ValueDQNAgent

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


class RLHybridCache: # Uses the pre-trained RL agent
    def __init__(self, capacity: int, agent: ValueDQNAgent, tail_sample_size: int):
        self.capacity, self.agent, self.k = capacity, agent, tail_sample_size
        self.cache, self.item_history_frequency, self.current_timestamp, self.hits, self.misses = OrderedDict(), {}, 0, 0, 0
    def _get_item_state(self, item_id, rank):
        recency = self.current_timestamp - self.cache.get(item_id, self.current_timestamp)
        frequency = self.item_history_frequency.get(item_id, 1)
        # Optimization: Rank is passed directly, avoiding O(N) search
        recency_rank = rank / self.capacity
        return np.array([recency, frequency, recency_rank])
    def _evict(self):
        # Optimization: Use islice to get candidates without creating full list
        candidate_items = list(islice(self.cache.keys(), self.k))
        
        # Optimization: Vectorized state creation and batch inference
        # Ranks are implicitly 1 to k for the first k items
        candidate_states = [self._get_item_state(item, rank+1) for rank, item in enumerate(candidate_items)]
        
        # Use batch inference
        values = self.agent.get_values(np.array(candidate_states))
        del self.cache[candidate_items[np.argmin(values)]] # Evict item with the lowest value
    def process_request(self, item_id: int):
        self.current_timestamp += 1
        self.item_history_frequency[item_id] = self.item_history_frequency.get(item_id, 0) + 1
        if item_id in self.cache:
            self.hits += 1; self.cache.move_to_end(item_id)
        else:
            self.misses += 1
            if len(self.cache) >= self.capacity: self._evict()
        self.cache[item_id] = self.current_timestamp
        # NO LEARNING HAPPENS IN THIS SCRIPT
    def get_hit_rate(self): total = self.hits + self.misses; return (self.hits / total) * 100 if total > 0 else 0

# ===================================================================
# =================== MAIN SIMULATION SCRIPT ========================
# ===================================================================

if __name__ == "__main__":
    CACHE_CAPACITY = 30
    RL_MODEL_FILE = "models/rl_eviction_model.pth"
    REQUEST_LOG_FILE = "data/training_data.csv"
    HYBRID_TAIL_SAMPLE_SIZE = 16
    NUM_FEATURES = 3
    
    try:
        requests_df = pd.read_csv(REQUEST_LOG_FILE)
    except FileNotFoundError as e:
        print(f"Error loading a required file: {e}"); exit()
    
    # Initialize and LOAD the pre-trained RL agent
    rl_agent = ValueDQNAgent(state_size=NUM_FEATURES)
    rl_agent.load(RL_MODEL_FILE)
    
    caches = {
        "LRU": LRUCache(capacity=CACHE_CAPACITY),
        "RL Hybrid (Pre-Trained)": RLHybridCache(capacity=CACHE_CAPACITY, agent=rl_agent, tail_sample_size=HYBRID_TAIL_SAMPLE_SIZE)
    }
    
    total_requests = len(requests_df)
    start_time = time.time()
    
    for i, row in requests_df.iterrows():
        item_id = int(row['item_id'])
        for cache in caches.values(): cache.process_request(item_id)
        
        if (i + 1) % 100000 == 0:
            print(f"  ...processed {i + 1}/{total_requests} requests")
            
    print(f"\nSimulation finished in {time.time() - start_time:.2f} seconds!")
    # print("--- FINAL CACHE PERFORMANCE RESULTS ---")
    results = {name: cache.get_hit_rate() for name, cache in caches.items()}
    for name, hit_rate in results.items(): print(f"{name}: \t{hit_rate:.2f}%")

