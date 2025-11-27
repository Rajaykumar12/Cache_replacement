import pandas as pd
import numpy as np
import time
from simulation import InstrumentedRLHybridCache, LRUCache
from rl_tail import ValueDQNAgent

def run_microbenchmarks(model_file):
    print("--- 2. Microbenchmark Breakdown of RL ---")
    
    trace_file = "data_gen/zipf_alpha_1.0.csv"
    cache_size = 30
    k = 16
    
    # Load Data
    df = pd.read_csv(trace_file)
    requests = df['item_id'].tolist()
    
    # Load Agent
    agent = ValueDQNAgent(state_size=3)
    try:
        agent.load(model_file)
    except:
        print("Warning: Using random weights for microbenchmark")
        
    # Initialize Instrumented Cache
    rl_cache = InstrumentedRLHybridCache(cache_size, agent, k)
    lru_cache = LRUCache(cache_size)
    
    # Run RL
    print("Running RL-Hybrid Microbenchmark...")
    start_t = time.perf_counter()
    for req in requests:
        rl_cache.process_request(req)
    rl_total_time = time.perf_counter() - start_t
    
    # Run LRU for comparison
    print("Running LRU Microbenchmark...")
    start_t = time.perf_counter()
    for req in requests:
        lru_cache.process_request(req)
    lru_total_time = time.perf_counter() - start_t
    
    # Process Stats
    stats = rl_cache.stats
    total_evictions = stats['total_evictions']
    
    # Normalize per eviction (microseconds)
    if total_evictions > 0:
        per_eviction = {
            'State Construction': (stats['state_construction'] / total_evictions) * 1e6,
            'Inference': (stats['inference'] / total_evictions) * 1e6,
            'Eviction Choice': (stats['eviction_choice'] / total_evictions) * 1e6,
            # Metadata update is total per request, not per eviction, but we can show total
        }
    else:
        per_eviction = {}
        
    print("\n--- Results ---")
    print(f"Total RL Time: {rl_total_time:.4f}s")
    print(f"Total LRU Time: {lru_total_time:.4f}s")
    print(f"Total Evictions: {total_evictions}")
    
    print("\nPer Eviction Breakdown (us):")
    for k, v in per_eviction.items():
        print(f"  {k}: {v:.2f} us")
        
    # Save to CSV
    df_res = pd.DataFrame([per_eviction])
    df_res['Algorithm'] = 'RL-Hybrid'
    df_res['LRU Total Time (s)'] = lru_total_time
    df_res['RL Total Time (s)'] = rl_total_time
    df_res.to_csv("results_microbenchmark.csv", index=False)

if __name__ == "__main__":
    run_microbenchmarks("models/rl_eviction_model.pth")
