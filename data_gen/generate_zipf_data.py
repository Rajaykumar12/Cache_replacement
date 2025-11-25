import pandas as pd
import numpy as np
from collections import defaultdict

def generate_zipf_data(num_rows=100000, num_items=1000, alpha=1.2, output_file='data/zipf_100k.csv'):
    print(f"Generating {num_rows} requests with Zipfian distribution (alpha={alpha})...")
    
    # Generate item IDs using Zipfian distribution
    # numpy.random.zipf returns values starting from 1
    item_ids = np.random.zipf(alpha, num_rows)
    
    # Modulo to keep item IDs within a reasonable range if needed, 
    # but Zipf naturally has a long tail. Let's just use them as is or map them.
    # To ensure we have a fixed vocabulary size if desired, we could map them, 
    # but standard Zipf is fine.
    
    data = []
    last_access = {}
    frequency = defaultdict(int)
    
    # First pass: Generate basic features
    # We need to look ahead for 'time_to_next_request', so we'll store indices first
    item_indices = defaultdict(list)
    for t, item_id in enumerate(item_ids):
        item_indices[item_id].append(t)
        
    print("Calculating features...")
    
    for t, item_id in enumerate(item_ids):
        # Timestamp (simple increment)
        timestamp = t
        
        # Time since last access
        if item_id in last_access:
            time_since_last = t - last_access[item_id]
        else:
            time_since_last = 0 # Or some default value for first access
            
        # Frequency count (including this one)
        frequency[item_id] += 1
        freq_count = frequency[item_id]
        
        # Time to next request
        # Find the current index in the list of indices for this item
        # This search can be slow if not optimized. 
        # Optimization: We can pop from the list since we iterate sequentially.
        # But popping from front is O(N). 
        # Better: Use a pointer for each item.
        
        # Actually, let's just pre-calculate next occurrence
        # We can do this by iterating backwards or using the indices list
        
        # Let's use the indices list efficiently
        # current_occurrence_idx is frequency[item_id] - 1
        # next_occurrence_idx is frequency[item_id]
        
        occurrences = item_indices[item_id]
        current_idx_in_occurrences = freq_count - 1
        
        if current_idx_in_occurrences + 1 < len(occurrences):
            next_timestamp = occurrences[current_idx_in_occurrences + 1]
            time_to_next = next_timestamp - t
        else:
            time_to_next = 0 # No future request
            
        last_access[item_id] = t
        
        data.append({
            'timestamp': timestamp,
            'item_id': item_id,
            'time_since_last_access': time_since_last,
            'frequency_count': freq_count,
            'time_to_next_request': time_to_next
        })
        
        if (t + 1) % 10000 == 0:
            print(f"Processed {t + 1}/{num_rows} rows")

    df = pd.DataFrame(data)
    
    # Reorder columns to match training_data.csv
    cols = ['timestamp', 'item_id', 'time_since_last_access', 'frequency_count', 'time_to_next_request']
    df = df[cols]
    
    print(f"Saving to {output_file}...")
    df.to_csv(output_file, index=False)
    print("Done!")

if __name__ == "__main__":
    generate_zipf_data()
