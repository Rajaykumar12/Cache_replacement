import pandas as pd
import numpy as np
import random
from collections import defaultdict

def generate_web_traffic_data(num_rows=100000, num_items=5000, output_file='data/web_traffic_100k.csv'):
    print(f"Generating {num_rows} requests with Web Traffic characteristics...")
    print("- Simulating temporal locality with shifting 'Hot Sets'")
    print("- Simulating long-tail popularity")

    # --- Traffic Generation Logic ---
    # Web traffic often has a "working set" that changes over time (e.g., breaking news).
    # We simulate this by having a "Hot Set" of items that receives most traffic.
    # Every N requests, we drift the Hot Set.
    
    hot_set_size = int(num_items * 0.05) # 5% of items are hot
    hot_traffic_ratio = 0.70             # 70% of traffic goes to hot items
    shift_interval = 2000                # Shift hot set every 2000 requests
    
    all_items = np.arange(1, num_items + 1)
    
    # Initial Hot Set
    hot_set = np.random.choice(all_items, hot_set_size, replace=False)
    
    item_ids = []
    
    print("Generating request stream...")
    for i in range(num_rows):
        # Drift the hot set periodically
        if i % shift_interval == 0 and i > 0:
            # Replace 10% of the hot set with new random items
            num_replace = int(hot_set_size * 0.1)
            new_items = np.random.choice(all_items, num_replace, replace=False)
            # Randomly replace indices
            replace_indices = np.random.choice(len(hot_set), num_replace, replace=False)
            hot_set[replace_indices] = new_items
            
        # Decide if request is for a Hot item or Cold item
        if random.random() < hot_traffic_ratio:
            # Pick from Hot Set (Zipfian distribution within hot set for realism)
            # Using a simple power law approximation by squaring random numbers
            # or just random choice for simplicity + locality
            item = np.random.choice(hot_set)
        else:
            # Pick from Cold Set (Long Tail)
            item = np.random.choice(all_items)
            
        item_ids.append(item)
    
    # --- Feature Calculation (Same as before) ---
    print("Calculating features (recency, frequency, next_arrival)...")
    
    data = []
    last_access = {}
    frequency = defaultdict(int)
    item_indices = defaultdict(list)
    
    # Pre-scan for indices
    for t, item_id in enumerate(item_ids):
        item_indices[item_id].append(t)
        
    for t, item_id in enumerate(item_ids):
        # Timestamp
        timestamp = t
        
        # Time since last access
        if item_id in last_access:
            time_since_last = t - last_access[item_id]
        else:
            time_since_last = 0
            
        # Frequency
        frequency[item_id] += 1
        freq_count = frequency[item_id]
        
        # Time to next request
        occurrences = item_indices[item_id]
        current_idx = freq_count - 1
        
        if current_idx + 1 < len(occurrences):
            next_timestamp = occurrences[current_idx + 1]
            time_to_next = next_timestamp - t
        else:
            time_to_next = 0
            
        last_access[item_id] = t
        
        data.append({
            'timestamp': timestamp,
            'item_id': item_id,
            'time_since_last_access': time_since_last,
            'frequency_count': freq_count,
            'time_to_next_request': time_to_next
        })
        
        if (t + 1) % 20000 == 0:
            print(f"Processed {t + 1}/{num_rows} rows")

    df = pd.DataFrame(data)
    cols = ['timestamp', 'item_id', 'time_since_last_access', 'frequency_count', 'time_to_next_request']
    df = df[cols]
    
    print(f"Saving to {output_file}...")
    df.to_csv(output_file, index=False)
    print("Done!")

if __name__ == "__main__":
    generate_web_traffic_data()
