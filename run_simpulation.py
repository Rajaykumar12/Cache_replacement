import pandas as pd
import numpy as np
import joblib
from collections import OrderedDict, deque, defaultdict

CACHE_CAPACITY = 10
REQUEST_LOG_FILE = "data/training_data.csv"
XGB_FILE = "models/xgb_model.pkl"
LGB_FILE = "models/lgb_model.pkl"


class LRUCache:
    """A standard LRU Cache implementation."""
    def __init__(self, capacity: int):
        self.cache = OrderedDict()
        self.capacity = capacity
        self.hits = 0
        self.misses = 0

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
        self.capacity = capacity
        self.cache = set()
        self.queue = deque()
        self.hits = 0
        self.misses = 0

    def process_request(self, item_id):
        if item_id in self.cache:
            self.hits += 1
        else:
            self.misses += 1
            if len(self.cache) >= self.capacity:
                evicted_item = self.queue.popleft()
                self.cache.remove(evicted_item)
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
        if not self.cache: return
        items_in_cache = list(self.cache.keys())
        features = [[self.current_timestamp - self.item_history_last_access.get(item, self.current_timestamp), 
                     self.item_history_frequency.get(item, 1)] for item in items_in_cache]
        features_df = pd.DataFrame(features, columns=['time_since_last_access', 'frequency_count'])
        predictions = self.model.predict(features_df)
        item_to_evict = items_in_cache[np.argmax(predictions)]
        del self.cache[item_to_evict]

    def process_request(self, item_id):
        self.current_timestamp += 1
        if item_id in self.cache: self.hits += 1
        else:
            self.misses += 1
            if len(self.cache) >= self.capacity: self._evict()
            self.cache[item_id] = True
        self.item_history_last_access[item_id] = self.current_timestamp
        self.item_history_frequency[item_id] = self.item_history_frequency.get(item_id, 0) + 1

    def get_hit_rate(self):
        total = self.hits + self.misses
        return (self.hits / total) * 100 if total > 0 else 0

class OptimalCache:
    def __init__(self, capacity: int, future_uses: dict):
        self.capacity = capacity
        self.future_uses = future_uses 
        self.cache = set()
        self.hits = 0
        self.misses = 0
        self.use_pointers = defaultdict(int)

    def _evict(self, current_index: int):
        farthest_item = -1
        max_distance = -1
        
        for item in self.cache:
            # Find the next time this item will be used
            future_accesses = self.future_uses[item]
            pointer = self.use_pointers[item]
            
            # Find the first future access that is after the current_index
            next_use_index = -1
            # Advance pointer to the current position
            while pointer < len(future_accesses) and future_accesses[pointer] <= current_index:
                pointer += 1
            self.use_pointers[item] = pointer # Update pointer
            
            if pointer < len(future_accesses):
                next_use_index = future_accesses[pointer]
            
            # If the item is never used again, it's the best to evict
            if next_use_index == -1:
                farthest_item = item
                break 
            
            distance = next_use_index - current_index
            if distance > max_distance:
                max_distance = distance
                farthest_item = item
                
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

# --- 4. Main Simulation Logic ---
if __name__ == "__main__":
    print("--- Cache Performance Simulation ---")

    # Load resources
    xgb_model = joblib.load(XGB_FILE)
    lgb_model = joblib.load(LGB_FILE)
    requests_df = pd.read_csv(REQUEST_LOG_FILE)
    
    # --- NEW: Pre-computation for Optimal Cache ---
    print("Pre-computing future uses for Optimal algorithm...")
    future_uses = defaultdict(list)
    for index, item_id in enumerate(requests_df['item_id']):
        future_uses[item_id].append(index)
    
    # Initialize all caches
    lru_cache = LRUCache(capacity=CACHE_CAPACITY)
    fifo_cache = FIFOCache(capacity=CACHE_CAPACITY)
    xgb_cache = MLCache(capacity=CACHE_CAPACITY, model=xgb_model)
    lgb_cache = MLCache(capacity=CACHE_CAPACITY, model=lgb_model)
    optimal_cache = OptimalCache(capacity=CACHE_CAPACITY, future_uses=future_uses) # <-- ADDED
    
    total_requests = len(requests_df)
    print(f"Starting simulation with {total_requests} requests and cache capacity of {CACHE_CAPACITY}...")
    
    for i, row in requests_df.iterrows():
        item_id = row['item_id']
        
        lru_cache.process_request(item_id)
        fifo_cache.process_request(item_id)
        xgb_cache.process_request(item_id)
        lgb_cache.process_request(item_id)
        optimal_cache.process_request(item_id, current_index=i) 
        
        if (i + 1) % 10000 == 0:
            print(f"  ...processed {i + 1}/{total_requests} requests")

    print("\nSimulation finished!")
    print("---" * 12)
    
    print(f"FIFO Cache Hit Rate: \t\t{fifo_cache.get_hit_rate():.2f}%")
    print(f"LRU Cache Hit Rate: \t\t{lru_cache.get_hit_rate():.2f}%")
    print(f"ML-Based Cache Hit Rate: \t{xgb_cache.get_hit_rate():.2f}%")
    print(f"ML-Based Cache Hit Rate: \t{lgb_cache.get_hit_rate():.2f}%")
    print(f"Optimal Cache Hit Rate: \t{optimal_cache.get_hit_rate():.2f}%")