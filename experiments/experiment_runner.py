import pandas as pd
import numpy as np
import time
import argparse
import os
from collections import OrderedDict
from experiments.baselines import LFUCache, ARCCache
from simulation import LRUCache, RLHybridCache
from rl_tail import ValueDQNAgent
import torch

def run_experiment(algorithm, cache_size, trace_file, model_file=None, k=16):
    print(f"Running {algorithm} on {trace_file} (Size: {cache_size})...")
    
    # Load Data
    try:
        df = pd.read_csv(trace_file)
        requests = df['item_id'].tolist()
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

    # Initialize Cache
    if algorithm == 'LRU':
        cache = LRUCache(cache_size)
    elif algorithm == 'LFU':
        cache = LFUCache(cache_size)
    elif algorithm == 'ARC':
        cache = ARCCache(cache_size)
    elif algorithm == 'RL-Hybrid':
        if not model_file:
            print("Error: Model file required for RL-Hybrid")
            return None
        agent = ValueDQNAgent(state_size=3)
        agent.load(model_file)
        cache = RLHybridCache(cache_size, agent, k)
    else:
        print(f"Unknown algorithm: {algorithm}")
        return None

    # Run Simulation
    start_time = time.perf_counter()
    latencies = []
    
    # Optimization: Local variable lookup
    process = cache.process_request
    
    # Memory usage tracking (simple approximation)
    # import psutil
    # process_info = psutil.Process(os.getpid())
    # initial_memory = process_info.memory_info().rss / 1024 / 1024
    
    for req in requests:
        t0 = time.perf_counter()
        process(req)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1e6) # microseconds

    end_time = time.perf_counter()
    total_time = end_time - start_time
    
    # Metrics
    hit_rate = cache.get_hit_rate()
    avg_latency = np.mean(latencies)
    p50 = np.percentile(latencies, 50)
    p90 = np.percentile(latencies, 90)
    p99 = np.percentile(latencies, 99)
    
    # final_memory = process_info.memory_info().rss / 1024 / 1024
    # memory_usage = final_memory - initial_memory
    memory_usage = 0 # Placeholder until psutil is confirmed/installed

    results = {
        "Algorithm": algorithm,
        "Cache Size": cache_size,
        "Hit Rate (%)": hit_rate,
        "Runtime (s)": total_time,
        "Avg Latency (us)": avg_latency,
        "P50 Latency (us)": p50,
        "P90 Latency (us)": p90,
        "P99 Latency (us)": p99,
        "Memory (MB)": memory_usage
    }
    
    print(f"  > Hit Rate: {hit_rate:.2f}%, Time: {total_time:.4f}s")
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithms", nargs="+", default=["LRU", "LFU", "ARC", "RL-Hybrid"])
    parser.add_argument("--sizes", nargs="+", type=int, default=[10, 20, 30, 50, 100, 200])
    parser.add_argument("--trace", type=str, default="data/zipf_100k.csv")
    parser.add_argument("--model", type=str, default="models/rl_eviction_model.pth")
    parser.add_argument("--output", type=str, default="results.csv")
    args = parser.parse_args()

    all_results = []
    
    for size in args.sizes:
        for algo in args.algorithms:
            res = run_experiment(algo, size, args.trace, args.model)
            if res:
                all_results.append(res)

    # Save Results
    df_res = pd.DataFrame(all_results)
    df_res.to_csv(args.output, index=False)
    print(f"\nResults saved to {args.output}")
    print(df_res)
