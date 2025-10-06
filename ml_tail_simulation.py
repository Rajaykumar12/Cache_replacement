import pandas as pd
import numpy as np
import joblib
from collections import OrderedDict
import matplotlib.pyplot as plt
import time

# --- Configuration ---
CACHE_CAPACITY = 30
MODEL_FILE = "models/lgb_model.pkl"  # Using XGBoost as the model for this simulation
REQUEST_LOG_FILE = "data/training_data.csv"

# Configuration for the new HybridCache
HYBRID_TAIL_SAMPLE_SIZE = 16 # K, 

# ===================================================================
# =================== CACHE IMPLEMENTATIONS =========================
# ===================================================================

class LRUCache:
    """The standard LRU cache implementation. Serves as the baseline."""
    def __init__(self, capacity: int):
        self.cache = OrderedDict()
        self.capacity = capacity
        self.hits = 0
        self.misses = 0

    def process_request(self, item_id):
        if item_id in self.cache:
            self.hits += 1
            self.cache.move_to_end(item_id) # Mark as most recently used
        else:
            self.misses += 1
            if len(self.cache) >= self.capacity:
                self.cache.popitem(last=False) # Evict the least recently used
            self.cache[item_id] = True

    def get_hit_rate(self):
        total = self.hits + self.misses
        return (self.hits / total) * 100 if total > 0 else 0

class MLCache:
    """The original ML-based approach.
    On a miss, it evaluates ALL N items in the cache to find a victim.
    """
    def __init__(self, capacity: int, model):
        self.capacity = capacity
        self.model = model
        self.cache = {}
        self.item_history_last_access = {}
        self.item_history_frequency = {}
        self.current_timestamp = 0
        self.hits = 0
        self.misses = 0

    def _evict(self):
        if not self.cache: return
        
        # This is an O(N) operation, where N is the cache capacity. Can be slow.
        items_in_cache = list(self.cache.keys())
        
        features = [[self.current_timestamp - self.item_history_last_access.get(item, self.current_timestamp), 
                     self.item_history_frequency.get(item, 1)] for item in items_in_cache]
        
        features_df = pd.DataFrame(features, columns=['time_since_last_access', 'frequency_count'])
        predictions = self.model.predict(features_df)
        
        # Evict item with the highest predicted 'time_to_next_request'
        item_to_evict = items_in_cache[np.argmax(predictions)]
        del self.cache[item_to_evict]

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

class HybridCache:
    """The new proposed solution (LRU Tail Sampling).
    On a miss, it only evaluates K items from the LRU tail.
    """
    def __init__(self, capacity: int, model, tail_sample_size: int):
        self.capacity = capacity
        self.model = model
        self.k = tail_sample_size
        self.cache = OrderedDict()
        self.item_history_last_access = {}
        self.item_history_frequency = {}
        self.current_timestamp = 0
        self.hits = 0
        self.misses = 0

    def _evict(self):
        if not self.cache: return

        # This is an O(K) operation, which is O(1) for a small, fixed K. Very fast.
        candidate_items = list(self.cache.keys())[:self.k]
        
        features = []
        for item in candidate_items:
            recency = self.current_timestamp - self.item_history_last_access.get(item, self.current_timestamp)
            frequency = self.item_history_frequency.get(item, 1)
            features.append([recency, frequency])
        
        features_df = pd.DataFrame(features, columns=['time_since_last_access', 'frequency_count'])
        predictions = self.model.predict(features_df)
        
        # Find the candidate with the WORST score (highest predicted time)
        victim_index = np.argmax(predictions)
        item_to_evict = candidate_items[victim_index]
        
        del self.cache[item_to_evict]

    def process_request(self, item_id):
        self.current_timestamp += 1
        self.item_history_last_access[item_id] = self.current_timestamp
        self.item_history_frequency[item_id] = self.item_history_frequency.get(item_id, 0) + 1

        if item_id in self.cache:
            self.hits += 1
            self.cache.move_to_end(item_id) # On a hit, it becomes MRU
        else:
            self.misses += 1
            if len(self.cache) >= self.capacity:
                self._evict()
            self.cache[item_id] = True # Add new item to the MRU end

    def get_hit_rate(self):
        total = self.hits + self.misses
        return (self.hits / total) * 100 if total > 0 else 0

# ===================================================================
# =================== MAIN SIMULATION LOGIC =========================
# ===================================================================

if __name__ == "__main__":
    print("--- Cache Performance Simulation ---")

    # Load resources
    try:
        requests_df = pd.read_csv(REQUEST_LOG_FILE)
        ml_model = joblib.load(MODEL_FILE)
    except FileNotFoundError as e:
        print(f"Error loading a required file: {e}"); exit()
    
    # Initialize all caches
    caches = {
        "LRU": LRUCache(capacity=CACHE_CAPACITY),
        "ML Cache (Original)": MLCache(capacity=CACHE_CAPACITY, model=ml_model),
        "Ml Cache Tail Sampling": HybridCache(capacity=CACHE_CAPACITY, model=ml_model, tail_sample_size=HYBRID_TAIL_SAMPLE_SIZE)
    }
    
    total_requests = len(requests_df)
    print(f"Starting simulation with {total_requests} requests and cache capacity of {CACHE_CAPACITY}...")
    start_time = time.time()
    
    for i, row in requests_df.iterrows():
        item_id = int(row['item_id'])
        
        for cache in caches.values():
            cache.process_request(item_id)
            
        if (i + 1) % 50000 == 0:
            print(f"  ...processed {i + 1}/{total_requests} requests")

    end_time = time.time()
    print(f"\nSimulation finished in {end_time - start_time:.2f} seconds!")
    print("---" * 15)
    print("--- FINAL CACHE PERFORMANCE RESULTS ---")
    print("---" * 15)
    
    results = {name: cache.get_hit_rate() for name, cache in caches.items()}
    for name, hit_rate in results.items():
        print(f"{name}: \t{hit_rate:.2f}%")
        
    print("---" * 15)

    # # ===================================================================
    # # =================== VISUALIZATION CODE ============================
    # # ===================================================================

    # print("\nGenerating performance graph...")

    # names = list(results.keys())
    # hit_rates = list(results.values())
    
    # plt.figure(figsize=(10, 6))
    # bars = plt.bar(names, hit_rates, color=['skyblue', 'salmon', 'lightgreen'])
    
    # plt.ylabel('Cache Hit Rate (%)')
    # plt.title('Performance Comparison of Cache Algorithms')
    # plt.ylim(0, max(hit_rates) * 1.15)
    
    # for bar in bars:
    #     yval = bar.get_height()
    #     plt.text(bar.get_x() + bar.get_width()/2.0, yval + 1, f'{yval:.2f}%', ha='center', va='bottom')
        
    # plt.tight_layout()
    # plt.show()