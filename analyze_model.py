import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rl_tail import ValueDQNAgent
from simulation import RLHybridCache
from itertools import islice

def analyze_model(model_file, trace_file):
    print("--- 5. RL Model Internal Behavior Analysis ---")
    
    # Load Agent
    agent = ValueDQNAgent(state_size=3)
    try:
        agent.load(model_file)
    except:
        print("Error loading model")
        return

    # 5.1 Weight Inspection
    print("\n[5.1] Weight Inspection")
    weights = agent.get_numpy_weights()
    w1 = weights[0] # Shape: (3, 64) or (64, 3) depending on implementation
    
    print(f"W1 shape: {w1.shape}")
    
    # Simple importance: sum of absolute weights connected to each input feature
    # If w1 is (in_features, out_features), sum across axis 1
    # If w1 is (out_features, in_features), sum across axis 0
    if w1.shape[0] == 3:
        feature_importance = np.sum(np.abs(w1), axis=1)
    else:
        feature_importance = np.sum(np.abs(w1), axis=0)
    
    features = ['Recency', 'Frequency', 'Rank']
    
    print("Feature Importance (Sum Abs Weights Layer 1):")
    for f, imp in zip(features, feature_importance):
        print(f"  {f}: {imp:.4f}")
        
    weights_df = pd.DataFrame({
        'Feature': features, 
        'Importance': list(feature_importance)
    })
    weights_df.to_csv("results_weights.csv", index=False)
    
    # 5.2 Q-Value Analysis & 5.4 Feature Correlation
    print("\n[5.2 & 5.4] Q-Value Analysis & Correlation")
    
    # Run a short simulation to collect states and Q-values
    df = pd.read_csv(trace_file)
    requests = df['item_id'].tolist()[:10000] # Analyze first 10k
    
    cache = RLHybridCache(30, agent, 16)
    
    collected_data = []
    
    for req in requests:
        cache.process_request(req)
        
        # Sample Q-values periodically
        if len(cache.cache) >= 30 and np.random.rand() < 0.1:
            # Get current candidates and their Q-values
            q_vals = cache.get_q_values()
            if len(q_vals) > 0:
                collected_data.extend(q_vals)
                
    q_values = np.array(collected_data)
    
    print(f"Collected {len(q_values)} Q-values")
    print(f"Mean Q: {np.mean(q_values):.4f}")
    print(f"Std Q: {np.std(q_values):.4f}")
    print(f"Min Q: {np.min(q_values):.4f}")
    print(f"Max Q: {np.max(q_values):.4f}")
    
    # Save for histogram
    pd.DataFrame({'Q-Values': q_values}).to_csv("results_qvalues.csv", index=False)

if __name__ == "__main__":
    analyze_model("models/rl_eviction_model.pth", "data_gen/zipf_alpha_1.0.csv")
