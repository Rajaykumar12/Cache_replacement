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
        
        # Pre-load weights for fast inference (avoiding PyTorch overhead)
        self.weights = agent.get_numpy_weights()
        self.w1, self.b1 = self.weights[0].T, self.weights[1]
        self.w2, self.b2 = self.weights[2].T, self.weights[3]
        self.w3, self.b3 = self.weights[4].T, self.weights[5]
        
        # Pre-allocate state buffer to avoid repeated malloc/GC
        self.state_buffer = np.zeros((k, 3), dtype=np.float32)

    def _get_item_state(self, item_id, rank):
        recency = self.ts - self.cache.get(item_id, self.ts)
        frequency = self.freq.get(item_id, 1)
        return np.array([recency, frequency, rank / self.capacity])

    def _predict_fast(self, states):
        """Pure NumPy forward pass for speed."""
        # Layer 1
        x = np.dot(states, self.w1) + self.b1
        x = np.maximum(x, 0) # ReLU
        # Layer 2
        x = np.dot(x, self.w2) + self.b2
        x = np.maximum(x, 0) # ReLU
        # Output Layer
        values = np.dot(x, self.w3) + self.b3
        return values.flatten()

    def _evict(self):
        # 1. Select bottom k candidates (LRU Logic)
        candidates = list(islice(self.cache.keys(), self.k))
        
        # 2. Vectorized State Construction (Optimized)
        # Construct matrix directly: (k, 3)
        # Columns: [recency, frequency, rank_norm]
        # Rank is 1..k for these candidates
        
        current_ts = self.ts
        cap = self.capacity
        
        # Use pre-allocated buffer (Zero Allocation)
        num_candidates = len(candidates)
        states = self.state_buffer[:num_candidates] # View of the buffer
        
        # Fill matrix - avoiding list comprehension overhead
        for i, item in enumerate(candidates):
            states[i, 0] = current_ts - self.cache[item] # Recency
            states[i, 1] = self.freq[item]               # Frequency
            states[i, 2] = (i + 1) / cap                 # Rank (normalized)
        
        # 3. Fast Inference
        values = self._predict_fast(states)
        
        # 4. Evict
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
    DATA_FILE = "data/zipf_100k.csv"
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
    
    # Pre-load requests to list for faster iteration (numpy scalars can be slow to hash)
    request_items = requests['item_id'].tolist()
    
    print(f"--- Processing {len(requests)} Requests ---")
    
    # Measure total time for each cache
    for name, cache in caches.items():
        print(f"Running {name}...")
        start_t = time.perf_counter()
        
        # Inner loop optimization: avoid function calls and attribute lookups where possible
        process = cache.process_request
        for i, item in enumerate(request_items):
            process(item)
            
        end_t = time.perf_counter()
        model_runtimes[name] = end_t - start_t
        print(f"  > {name} finished in {model_runtimes[name]:.4f}s")

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
    
    # Performance Comparison
    lru_time = model_runtimes["LRU"]
    rl_time = model_runtimes["RL-Hybrid"]
    if lru_time > 0:
        ratio = rl_time / lru_time
        print(f"\n🚀 LRU is {ratio:.2f}x faster than RL-Hybrid")
    else:
        print("\n🚀 LRU runtime was 0.00s (too fast to compare)")