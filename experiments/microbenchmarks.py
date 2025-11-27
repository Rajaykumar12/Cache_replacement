import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from simulation import RLHybridCache
from rl_tail import ValueDQNAgent

class InstrumentedRLCache(RLHybridCache):
    def __init__(self, capacity, agent, k):
        super().__init__(capacity, agent, k)
        self.stats = {
            "construct_candidates": [],
            "inference": [],
            "eviction_choice": [],
            "metadata_update": []
        }

    def _evict(self):
        t0 = time.perf_counter()
        
        # 1. Select bottom k candidates
        candidates = list(self.cache.keys())[:self.k] # Simplified for benchmark
        # Note: Original code used islice, which is fast. 
        # We need to measure "Time to construct candidate states"
        
        t1 = time.perf_counter()
        
        # 2. State Construction
        current_ts = self.ts
        cap = self.capacity
        num_candidates = len(candidates)
        states = self.state_buffer[:num_candidates]
        
        for i, item in enumerate(candidates):
            states[i, 0] = current_ts - self.cache[item]
            states[i, 1] = self.freq[item]
            states[i, 2] = (i + 1) / cap
            
        t2 = time.perf_counter()
        
        # 3. Inference
        values = self._predict_fast(states)
        
        t3 = time.perf_counter()
        
        # 4. Choice
        victim = candidates[np.argmin(values)]
        del self.cache[victim]
        
        t4 = time.perf_counter()
        
        self.stats["construct_candidates"].append((t2 - t1) * 1e6) # us
        self.stats["inference"].append((t3 - t2) * 1e6)
        self.stats["eviction_choice"].append((t4 - t3) * 1e6)

    def process_request(self, item_id):
        t0 = time.perf_counter()
        super().process_request(item_id)
        t1 = time.perf_counter()
        # This includes everything, so we need to be careful what we measure as "metadata update"
        # Since _evict is called inside, we can't easily isolate metadata update unless we override process_request fully.
        # But for now, let's assume metadata update is (Total - Eviction Time) if eviction happened, 
        # or just Total Time if no eviction. 
        # A better way is to measure it explicitly.
        
        # Re-implementing process_request for fine-grained measurement
        # ... (skipping for brevity, will rely on _evict stats for now)

def run_microbenchmark(trace_file, model_file):
    print("Running Microbenchmarks...")
    df = pd.read_csv(trace_file)
    requests = df['item_id'].tolist()[:10000] # Run smaller subset
    
    agent = ValueDQNAgent(state_size=3)
    agent.load(model_file)
    
    cache = InstrumentedRLCache(30, agent, 16)
    
    for req in requests:
        cache.process_request(req)
        
    # Analyze
    print("\nMicrobenchmark Results (Average us):")
    for k, v in cache.stats.items():
        if v:
            print(f"  {k}: {np.mean(v):.2f} us")
            
    # Plot
    means = {k: np.mean(v) for k, v in cache.stats.items() if v}
    plt.bar(means.keys(), means.values())
    plt.ylabel("Time (microseconds)")
    plt.title("RL Component Time Breakdown")
    plt.savefig("microbenchmark_breakdown.png")
    print("Plot saved to microbenchmark_breakdown.png")

if __name__ == "__main__":
    run_microbenchmark("data/zipf_100k.csv", "models/rl_eviction_model.pth")
