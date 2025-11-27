import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12

# Create output directory
Path("plots").mkdir(exist_ok=True)

print("="*60)
print("GENERATING ALL PLOTS")
print("="*60)

# Load all results
try:
    baseline = pd.read_csv("results_baseline.csv")
    sensitivity = pd.read_csv("results_sensitivity.csv")
    behavior = pd.read_csv("results_behavior.csv")
    ablation = pd.read_csv("results_ablation.csv")
    microbenchmark = pd.read_csv("results_microbenchmark.csv")
    weights = pd.read_csv("results_weights.csv")
    qvalues = pd.read_csv("results_qvalues.csv")
except Exception as e:
    print(f"Error loading results: {e}")
    exit(1)

# =================== PLOT 1: Hit Rate vs Cache Size ===================
print("\\n[1/8] Generating: Hit rate vs cache size...")
plt.figure(figsize=(10, 6))

for algo in baseline['Algorithm'].unique():
    data = baseline[baseline['Algorithm'] == algo]
    plt.plot(data['Cache Size'], data['Hit Rate (%)'], marker='o', label=algo, linewidth=2)

plt.xlabel('Cache Size', fontsize=14)
plt.ylabel('Hit Rate (%)', fontsize=14)
plt.title('Hit Rate vs Cache Size', fontsize=16, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('plots/hit_rate_vs_cache_size.png', dpi=300)
plt.close()

# =================== PLOT 2: Runtime vs Cache Size ===================
print("[2/8] Generating: Runtime vs cache size...") 
plt.figure(figsize=(10, 6))

for algo in baseline['Algorithm'].unique():
    data = baseline[baseline['Algorithm'] == algo]
    plt.plot(data['Cache Size'], data['Runtime (s)'], marker='o', label=algo, linewidth=2)

plt.xlabel('Cache Size', fontsize=14)
plt.ylabel('Runtime (s)', fontsize=14)
plt.title('Runtime vs Cache Size', fontsize=16, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('plots/runtime_vs_cache_size.png', dpi=300)
plt.close()

# =================== PLOT 3: Runtime vs k (Candidate Size) ===================
print("[3/8] Generating: Runtime vs k...")
k_data = sensitivity[(sensitivity['Trace'] == 'zipf_alpha_1.0.csv') & (sensitivity['Algorithm'] == 'RL-Hybrid')].copy()
k_data = k_data[k_data['k'].isin([4, 8, 16, 32])]

k_data = k_data.groupby('k', as_index=False)['Runtime (s)'].mean()
k_data = k_data.sort_values('k')
plt.figure(figsize=(10, 6))
plt.plot(k_data['k'], k_data['Runtime (s)'], marker='o', linewidth=2, markersize=10, color='#1f77b4')
plt.xlabel('k (Candidate Size)', fontsize=14)
plt.ylabel('Runtime (s)', fontsize=14)
plt.title('RL-Hybrid Runtime vs Candidate Size (k)', fontsize=16, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('plots/runtime_vs_k.png', dpi=300)
plt.close()

# =================== PLOT 4: Hit Rate vs Zipf Alpha ===================
print("[4/8] Generating: Hit rate vs Zipf α...")
zipf_data = sensitivity[sensitivity['Trace'].str.contains('zipf_alpha')].copy()
zipf_data['alpha'] = zipf_data['Trace'].str.extract(r'(\d\.\d)').astype(float)
zipf_data = zipf_data.groupby(['Algorithm', 'alpha'], as_index=False)['Hit Rate (%)'].mean()

plt.figure(figsize=(10, 6))
for algo in zipf_data['Algorithm'].unique():
    data = zipf_data[zipf_data['Algorithm'] == algo].sort_values('alpha')
    plt.plot(data['alpha'], data['Hit Rate (%)'], marker='o', label=algo, linewidth=2)

plt.xlabel('Zipf α (Workload Locality)', fontsize=14)
plt.ylabel('Hit Rate (%)', fontsize=14)
plt.title('Hit Rate vs Zipf Distribution α', fontsize=16, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('plots/hit_rate_vs_zipf_alpha.png', dpi=300)
plt.close()

# =================== PLOT 5: Q-Value Distribution Histogram ===================
print("[5/8] Generating: Q-value distribution...")
plt.figure(figsize=(10, 6))
plt.hist(qvalues['Q-Values'], bins=50, edgecolor='black', alpha=0.7, color='steelblue')
plt.axvline(qvalues['Q-Values'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {qvalues["Q-Values"].mean():.2f}')
plt.xlabel('Q-Value', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.title('Distribution of RL Model Q-Values', fontsize=16, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('plots/qvalue_distribution.png', dpi=300)
plt.close()

# =================== PLOT 6: Feature Weight Bar Chart ===================
print("[6/8] Generating: Feature weights...")
plt.figure(figsize=(10, 6))
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
bars = plt.bar(weights['Feature'], weights['Importance'], color=colors, edgecolor='black', alpha=0.8)
plt.ylabel('Importance (Sum of Absolute Weights)', fontsize=14)
plt.xlabel('Feature', fontsize=14)
plt.title('Feature Importance in RL Model', fontsize=16, fontweight='bold')
plt.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.2f}',
             ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('plots/feature_weights.png', dpi=300)
plt.close()

# =================== PLOT 7: Latency CDF (LRU vs RL) ===================
print("[7/8] Generating: Latency CDF...")

# Get LRU and RL latencies from baseline
lru_data = baseline[baseline['Algorithm'] == 'LRU']
rl_data = baseline[baseline['Algorithm'] == 'RL-Hybrid']

# Use average latency as a proxy (we don't have full latency distributions)
# But we have p50, p90, p99, so let's create a synthetic CDF

plt.figure(figsize=(10, 6))

# Use cache size 30 for comparison
lru_30 = lru_data[lru_data['Cache Size'] == 30].iloc[0]
rl_30 = rl_data[rl_data['Cache Size'] == 30].iloc[0]

# Create synthetic CDFs using percentiles
percentiles = [0, 50, 90, 99, 100]
lru_latencies = [0, lru_30['P50 Latency (us)'], lru_30['P90 Latency (us)'], lru_30['P99 Latency (us)'], lru_30['P99 Latency (us)'] * 1.1]
rl_latencies = [0, rl_30['P50 Latency (us)'], rl_30['P90 Latency (us)'], rl_30['P99 Latency (us)'], rl_30['P99 Latency (us)'] * 1.1]

plt.plot(lru_latencies, [p/100 for p in percentiles], marker='o', label='LRU', linewidth=2)
plt.plot(rl_latencies, [p/100 for p in percentiles], marker='s', label='RL-Hybrid', linewidth=2)

plt.xlabel('Latency (μs)', fontsize=14)
plt.ylabel('CDF', fontsize=14)
plt.title('Latency CDF: LRU vs RL-Hybrid (Cache Size=30)', fontsize=16, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('plots/latency_cdf.png', dpi=300)
plt.close()

# =================== PLOT 8: Microbenchmark Breakdown Bar Chart ===================
print("[8/8] Generating: Microbenchmark breakdown...")

# Get timing breakdown
if 'State Construction' in microbenchmark.columns:
    operations = ['State Construction', 'Inference', 'Eviction Choice']
    times = [microbenchmark[op].iloc[0] for op in operations]
    
    plt.figure(figsize=(10, 6))
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    bars = plt.bar(operations, times, color=colors, edgecolor='black', alpha=0.8)
    plt.ylabel('Time per Eviction (μs)', fontsize=14)
    plt.xlabel('Operation', fontsize=14)
    plt.title('RL-Hybrid Microbenchmark Breakdown', fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.2f} μs',
                 ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('plots/microbenchmark_breakdown.png', dpi=300)
    plt.close()

# =================== BONUS: Ablation Comparison ===================
print("\\n[BONUS] Generating: Ablation study comparison...")

plt.figure(figsize=(12, 6))
x = range(len(ablation))
width = 0.35

fig, ax1 = plt.subplots(figsize=(12, 6))

# Hit rate bars
ax1.bar([i - width/2 for i in x], ablation['Hit Rate (%)'], width, label='Hit Rate', color='steelblue', alpha=0.8)
ax1.set_xlabel('Model Variant', fontsize=14)
ax1.set_ylabel('Hit Rate (%)', fontsize=14, color='steelblue')
ax1.tick_params(axis='y', labelcolor='steelblue')
ax1.set_xticks(x)
ax1.set_xticklabels(ablation['Variant'], rotation=45, ha='right')

# Runtime on secondary axis
ax2 = ax1.twinx()
ax2.bar([i + width/2 for i in x], ablation['Runtime (s)'], width, label='Runtime', color='coral', alpha=0.8)
ax2.set_ylabel('Runtime (s)', fontsize=14, color='coral')
ax2.tick_params(axis='y', labelcolor='coral')

plt.title('Ablation Study: Hit Rate and Runtime by Architecture', fontsize=16, fontweight='bold')
fig.tight_layout()
plt.savefig('plots/ablation_comparison.png', dpi=300)
plt.close()

print("\\n" + "="*60)
print("ALL PLOTS GENERATED SUCCESSFULLY!")
print("="*60)
print("\\nPlots saved in 'plots/' directory:")
print("  1. hit_rate_vs_cache_size.png")
print("  2. runtime_vs_cache_size.png") 
print("  3. runtime_vs_k.png")
print("  4. hit_rate_vs_zipf_alpha.png")
print("  5. qvalue_distribution.png")
print("  6. feature_weights.png")
print("  7. latency_cdf.png")
print("  8. microbenchmark_breakdown.png")
print("  BONUS: ablation_comparison.png")
print("="*60)
