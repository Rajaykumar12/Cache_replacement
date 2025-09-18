import pandas as pd
import numpy as np

# --- Parameters you can change ---
NUM_REQUESTS = 20000  # Total number of requests to generate
NUM_ITEMS = 1000      # Total number of unique items in the system
ZIPF_PARAM_A = 1.1    # How skewed the popularity is. >1.0 is typical. Higher means more skewed.
FILENAME = "requests.csv"

# --- Generation Logic ---
# Generate the item IDs based on a Zipfian distribution
# An item with rank 'i' is chosen with probability proportional to 1/i^a
ranks = np.arange(1, NUM_ITEMS + 1)
probabilities = 1 / (ranks**ZIPF_PARAM_A)
probabilities /= np.sum(probabilities) # Normalize to sum to 1

# Generate the sequence of requests
# 'p' is the probability distribution for the choices
requested_items = np.random.choice(ranks, size=NUM_REQUESTS, p=probabilities)

# Create the timestamps (a simple incrementing integer is fine for a prototype)
timestamps = np.arange(NUM_REQUESTS)

# Create a Pandas DataFrame and save to CSV
df = pd.DataFrame({
    'timestamp': timestamps,
    'item_id': requested_items
})

df.to_csv(FILENAME, index=False)

print(f"Successfully generated '{FILENAME}' with {NUM_REQUESTS} requests.")
print("Sample of the data:")
print(df.head())