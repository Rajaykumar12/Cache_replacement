import pandas as pd
import numpy as np
import time
from baselines import LFUCache, ARCCache
from simulation import LRUCache, RLHybridCache
from rl_tail import ValueDQNAgent, ValueNetwork
import torch

def run_ablation_experiment(variant_name, hidden_dims, trace_file, cache_size, model_file, k=16):
    """Run ablation experiment with specific architecture"""
    print(f"Running {variant_name}...")
    
    # Load Data
    try:
        df = pd.read_csv(trace_file)
        requests = df['item_id'].tolist()
    except Exception as e:
        print(f"Error loading data {trace_file}: {e}")
        return None

    # Initialize agent with specific architecture
    agent = ValueDQNAgent(state_size=3, hidden_dims=hidden_dims)
    
    # Try to load model only if dimensions match (64, 64)
    if hidden_dims == [64, 64] and variant_name == "Base (64-64-1)":
        try:
            agent.load(model_file)
            print(f"  Loaded pretrained model")
        except Exception as e:
            print(f"  Warning: Could not load model: {e}. Using random weights.")
    else:
        print(f"  Using random weights for this architecture")
    
    cache = RLHybridCache(cache_size, agent, k)

    # Run Simulation
    start_time = time.perf_counter()
    latencies = []
    
    for req in requests:
        t0 = time.perf_counter()
        cache.process_request(req)
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
    
    # Parameter count
    param_count = sum(p.numel() for p in agent.q_network.parameters())
    
    results = {
        "Variant": variant_name,
        "Hit Rate (%)": hit_rate,
        "Runtime (s)": total_time,
        "Avg Latency (us)": avg_latency,
        "P50 Latency (us)": p50,
        "P90 Latency (us)": p90,
        "P99 Latency (us)": p99,
        "Parameters": param_count,
        "Architecture": str(hidden_dims)
    }
    
    return results

def run_ablation_studies(model_file="models/rl_eviction_model.pth"):
    print("="*60)
    print("6. ABLATION STUDIES")
    print("="*60)
    
    results = []
    trace = "data_gen/zipf_alpha_1.0.csv"
    size = 30
    
    # Base RL with pretrained model
    res = run_ablation_experiment("Base (64-64-1)", [64, 64], trace, size, model_file)
    if res: results.append(res)
        
    # Smaller Networks
    res = run_ablation_experiment("MLP 32-32-1", [32, 32], trace, size, model_file)
    if res: results.append(res)
    
    res = run_ablation_experiment("MLP 16-16-1", [16, 16], trace, size, model_file)
    if res: results.append(res)
    
    res = run_ablation_experiment("MLP 8-8-1", [8, 8], trace, size, model_file)
    if res: results.append(res)
    
    # Different sized networks
    res = run_ablation_experiment("MLP 128-128-1", [128, 128], trace, size, model_file)
    if res: results.append(res)
    
    res = run_ablation_experiment("MLP 64-32-1", [64, 32], trace, size, model_file)
    if res: results.append(res)

    df = pd.DataFrame(results)
    df.to_csv("results_ablation.csv", index=False)
    
    print("\\n" + "="*60)
    print("ABLATION RESULTS")
    print("="*60)
    print(df.to_string(index=False))
    print("\\nResults saved to results_ablation.csv")

if __name__ == "__main__":
    run_ablation_studies()
