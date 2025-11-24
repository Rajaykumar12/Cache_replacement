import pandas as pd
import numpy as np
import time
from collections import OrderedDict
from itertools import islice
from rl_tail import ValueDQNAgent

# =================== CACHE CLASSES =========================

class LRUCache:
    def __init__(self, capacity: int):
        self.cache, self.capacity = OrderedDict(), capacity
        self.hits, self.misses = 0, 0
        
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


class RLHybridCache:
    def __init__(self, capacity: int, agent: ValueDQNAgent, k: int):
        self.capacity, self.agent, self.k = capacity, agent, k
        self.cache, self.freq, self.ts = OrderedDict(), {}, 0
        self.hits, self.misses = 0, 0

    def _get_item_state(self, item_id, rank):
        recency = self.ts - self.cache.get(item_id, self.ts)
        frequency = self.freq.get(item_id, 1)
        return np.array([recency, frequency, rank / self.capacity])

    def _evict(self):
        # 1. Select bottom k candidates (LRU Logic)
        candidates = list(islice(self.cache.keys(), self.k))
        
        # 2. Vectorized State Construction
        states = [self._get_item_state(item, r+1) for r, item in enumerate(candidates)]
        
        # 3. Batch Inference (Select item with lowest predicted value)
        values = self.agent.get_values(np.array(states))
        del self.cache[candidates[np.argmin(values)]]

    def process_request(self, item_id: int):
        self.ts += 1
        self.freq[item_id] = self.freq.get(item_id, 0) + 1
        
        if item_id in self.cache:
            self.hits += 1
            self.cache.move_to_end(item_id)
        else:
            self.misses += 1
            if len(self.cache) >= self.capacity:
                self._evict()
        self.cache[item_id] = self.ts

    def get_hit_rate(self):
        total = self.hits + self.misses
        return (self.hits / total) * 100 if total > 0 else 0

# =================== MAIN SIMULATION ========================

if __name__ == "__main__":
    # Settings
    CACHE_CAPACITY = 30
    TAIL_SIZE = 16
    DATA_FILE = "data/training_data.csv"
    MODEL_FILE = "models/rl_eviction_model.pth"

    # Setup
    print("--- Loading Data and Model ---")
    try:
        requests = pd.read_csv(DATA_FILE)
        rl_agent = ValueDQNAgent(state_size=3)
        rl_agent.load(MODEL_FILE)
    except FileNotFoundError as e:
        print(f"Error: {e}"); exit()

    caches = {
        "LRU": LRUCache(CACHE_CAPACITY),
        "RL-Hybrid": RLHybridCache(CACHE_CAPACITY, rl_agent, TAIL_SIZE)
    }

    model_runtimes = {name: 0.0 for name in caches}
    
    print(f"--- Processing {len(requests)} Requests ---")
    
    for _, row in requests.iterrows():
        item = int(row['item_id'])
        
        for name, cache in caches.items():
            start_t = time.perf_counter()
            cache.process_request(item)
            end_t = time.perf_counter()  
            
            model_runtimes[name] += (end_t - start_t)

    # Results
    print("\n" + "="*50)
    print("FINAL RESULTS")
    print("="*50)
    print(f"{'Algorithm':<15} | {'Hit Rate':<10} | {'Time (s)':<10}")
    print("-" * 43)
    
    for name, cache in caches.items():
        hit_rate = cache.get_hit_rate()
        runtime = model_runtimes[name]
        print(f"{name:<15} | {hit_rate:>6.2f}%    | {runtime:>8.4f}s")
    print("="*50)