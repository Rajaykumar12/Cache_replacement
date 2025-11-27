import numpy as np
import pandas as pd
import os

def generate_zipf(alpha, n, num_items):
    # Zipf distribution
    # p(x) ~ 1/x^alpha
    # We use numpy's random generator or a custom one if needed.
    # np.random.zipf requires alpha > 1. For alpha <= 1, we need an approximation.
    
    if alpha > 1:
        s = np.random.zipf(alpha, n)
        # Map to range [0, num_items-1]
        # Zipf gives values >= 1. We can modulo or clip.
        # Standard way: just use the values, but they can be unbounded.
        # Better: use bounded zipf or just modulo.
        s = (s - 1) % num_items
    else:
        # Approximate for alpha <= 1 (e.g. 0.5, 0.8, 1.0)
        # PMF: p(k) = C / k^alpha
        ranks = np.arange(1, num_items + 1)
        weights = 1 / np.power(ranks, alpha)
        weights /= weights.sum()
        s = np.random.choice(ranks, size=n, p=weights) - 1
        
    return s

def generate_uniform(n, num_items):
    return np.random.randint(0, num_items, n)

def generate_gaussian(n, num_items):
    # Gaussian centered at num_items/2
    s = np.random.normal(num_items/2, num_items/6, n)
    s = np.clip(s, 0, num_items-1).astype(int)
    return s

def generate_bursty(n, num_items):
    # Bursty: Mix of Uniform and short bursts of specific items
    s = []
    while len(s) < n:
        if np.random.rand() < 0.1: # 10% chance of burst
            burst_item = np.random.randint(0, num_items)
            burst_len = np.random.randint(10, 100)
            s.extend([burst_item] * burst_len)
        else:
            s.append(np.random.randint(0, num_items))
    return np.array(s[:n])

def generate_periodic(n, num_items):
    # Periodic: Accesses shift every period
    s = []
    period = 5000
    shift = 0
    while len(s) < n:
        # Shift popular items
        base = np.arange(shift, shift + 100) % num_items
        # 80% traffic to these 100 items
        if np.random.rand() < 0.8:
            item = np.random.choice(base)
        else:
            item = np.random.randint(0, num_items)
        s.append(item)
        
        if len(s) % period == 0:
            shift = (shift + 100) % num_items
            
    return np.array(s[:n])

def generate_adversarial(n, num_items, cache_size=30):
    # Adversarial for LRU: Cyclic scan of size cache_size + 1
    # This causes 100% miss rate for LRU
    pattern = np.arange(0, cache_size + 1)
    repeats = n // len(pattern) + 1
    s = np.tile(pattern, repeats)
    return s[:n]

def save_trace(data, filename):
    df = pd.DataFrame({'item_id': data})
    df.to_csv(filename, index=False)
    print(f"Saved {filename} ({len(data)} requests)")

if __name__ == "__main__":
    os.makedirs("data_gen", exist_ok=True)
    
    N = 100000
    ITEMS = 10000
    
    # Zipf variations
    for alpha in [0.5, 0.8, 1.0, 1.2, 1.5]:
        print(f"Generating Zipf alpha={alpha}...")
        data = generate_zipf(alpha, N, ITEMS)
        save_trace(data, f"data_gen/zipf_alpha_{alpha}.csv")
        
    # Uniform
    print("Generating Uniform...")
    data = generate_uniform(N, ITEMS)
    save_trace(data, "data_gen/uniform.csv")
    
    # Gaussian
    print("Generating Gaussian...")
    data = generate_gaussian(N, ITEMS)
    save_trace(data, "data_gen/gaussian.csv")
    
    # Bursty
    print("Generating Bursty...")
    data = generate_bursty(N, ITEMS)
    save_trace(data, "data_gen/bursty.csv")

    # Periodic
    print("Generating Periodic...")
    data = generate_periodic(N, ITEMS)
    save_trace(data, "data_gen/periodic.csv")

    # Adversarial (LRU Killer)
    print("Generating Adversarial...")
    data = generate_adversarial(N, ITEMS, cache_size=30) # Default cache size for adversarial
    save_trace(data, "data_gen/adversarial.csv")
