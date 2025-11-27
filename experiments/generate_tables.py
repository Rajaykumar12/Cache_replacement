import pandas as pd
import numpy as np
from pathlib import Path

print("="*80)
print("GENERATING FINAL SUMMARY TABLES FOR RESEARCH PAPER")
print("="*80)

# Load all results
baseline = pd.read_csv("results_baseline.csv")
sensitivity = pd.read_csv("results_sensitivity.csv")
behavior = pd.read_csv("results_behavior.csv")
ablation = pd.read_csv("results_ablation.csv")
microbenchmark = pd.read_csv("results_microbenchmark.csv")
weights = pd.read_csv("results_weights.csv")
qvalues = pd.read_csv("results_qvalues.csv")

# Create tables directory
Path("tables").mkdir(exist_ok=True)

# =================== TABLE 1: Performance Summary ===================
print("\\n[1/5] Generating Performance Summary Table...")

# Use cache size 30 as representative
perf_summary = baseline[baseline['Cache Size'] == 30].copy()
perf_summary = perf_summary[['Algorithm', 'Hit Rate (%)', 'Runtime (s)', 'P99 Latency (us)', 'Memory (MB)']]
perf_summary['Cache Size'] = 30
perf_summary = perf_summary[['Algorithm', 'Cache Size', 'Hit Rate (%)', 'Runtime (s)', 'P99 Latency (us)', 'Memory (MB)']]

# Round for readability
perf_summary['Hit Rate (%)'] = perf_summary['Hit Rate (%)'].round(2)
perf_summary['Runtime (s)'] = perf_summary['Runtime (s)'].round(4)
perf_summary['P99 Latency (us)'] = perf_summary['P99 Latency (us)'].round(2)

perf_summary.to_csv("tables/1_performance_summary.csv", index=False)
print("\\nPerformance Summary Table (Cache Size = 30):")
print(perf_summary.to_string(index=False))

# =================== TABLE 2: Sensitivity Analysis Summary ===================
print("\\n\\n[2/5] Generating Sensitivity Analysis Table...")

# Cache size variation (using baseline data)
cache_size_summary = baseline[baseline['Algorithm'].isin(['LRU', 'RL-Hybrid'])].copy()
cache_size_summary['RL Overhead (%)'] = 0.0

# Calculate RL overhead vs LRU for each cache size
for size in cache_size_summary['Cache Size'].unique():
    lru_time = cache_size_summary[(cache_size_summary['Cache Size'] == size) & 
                                  (cache_size_summary['Algorithm'] == 'LRU')]['Runtime (s)'].values
    rl_time = cache_size_summary[(cache_size_summary['Cache Size'] == size) & 
                                 (cache_size_summary['Algorithm'] == 'RL-Hybrid')]['Runtime (s)'].values
    
    if len(lru_time) > 0 and len(rl_time) > 0:
        overhead = ((rl_time[0] - lru_time[0]) / lru_time[0]) * 100
        cache_size_summary.loc[(cache_size_summary['Cache Size'] == size) & 
                              (cache_size_summary['Algorithm'] == 'RL-Hybrid'), 'RL Overhead (%)'] = overhead

sens_table = cache_size_summary[cache_size_summary['Algorithm'] == 'RL-Hybrid'][
    ['Cache Size', 'Hit Rate (%)', 'Runtime (s)', 'Avg Latency (us)', 'RL Overhead (%)']
].copy()

# Add k variant info for cache size 30
k_variants = sensitivity[sensitivity['Trace'] == 'zipf_alpha_1.0.csv'].copy()
k_variants = k_variants[k_variants['k'].isin([4, 8, 16, 32])]

sens_table = sens_table.round(2)
sens_table.to_csv("tables/2_sensitivity_cache_size.csv", index=False)

print("\\nSensitivity Analysis - Cache Size Variation:")
print(sens_table.to_string(index=False))

# K variation table
k_table = k_variants[['k', 'Hit Rate (%)', 'Runtime (s)', 'Avg Latency (us)']].round(2)
k_table.to_csv("tables/2_sensitivity_k_variation.csv", index=False)
print("\\nSensitivity Analysis - k Variation (Cache Size = 30):")
print(k_table.to_string(index=False))

# =================== TABLE 3: Workload Behavior Summary ===================
print("\\n\\n[3/5] Generating Workload Behavior Table...")

workload_summary = sensitivity[sensitivity['Algorithm'].isin(['LRU', 'RL-Hybrid'])].copy()

# Pivot to compare LRU vs RL side by side
workload_pivot = workload_summary.pivot_table(
    index='Trace',
    columns='Algorithm',
    values=['Hit Rate (%)', 'Runtime (s)'],
    aggfunc='first'
)

workload_pivot.columns = [f'{col[1]} {col[0]}' for col in workload_pivot.columns]
workload_pivot = workload_pivot.reset_index()

# Calculate slowdown
workload_pivot['RL Slowdown (x)'] = (workload_pivot['RL-Hybrid Runtime (s)'] / 
                                     workload_pivot['LRU Runtime (s)']).round(2)

# Add behavior notes
def get_behavior_note(trace):
    if 'zipf' in trace:
        return "Skewed distribution"
    elif 'uniform' in trace:
        return "Random access"
    elif 'gaussian' in trace:
        return "Localized access"
    elif 'bursty' in trace:
        return "Burst patterns"
    elif 'periodic' in trace:
        return "Phase shifts"
    elif 'adversarial' in trace:
        return "LRU killer"
    return "Other"

workload_pivot['Workload Type'] = workload_pivot['Trace'].apply(get_behavior_note)

# Reorder columns
workload_table = workload_pivot[[
    'Trace', 'Workload Type',
    'LRU Hit Rate (%)', 'RL-Hybrid Hit Rate (%)',
    'LRU Runtime (s)', 'RL-Hybrid Runtime (s)',
    'RL Slowdown (x)'
]].round(2)

workload_table.to_csv("tables/3_workload_behavior.csv", index=False)
print("\\nWorkload Behavior Comparison:")
print(workload_table.to_string(index=False))

# =================== TABLE 4: Ablation Study Summary ===================
print("\\n\\n[4/5] Generating Ablation Study Table...")

ablation_table = ablation[['Variant', 'Hit Rate (%)', 'Runtime (s)', 'Parameters', 'Architecture']].copy()
ablation_table = ablation_table.round(2)

# Add notes
def get_ablation_note(variant):
    if 'Base' in variant:
        return "Pretrained model"
    elif '128' in variant:
        return "Larger network"
    elif '8-' in variant:
        return "Smallest viable"
    else:
        return "Random weights"

ablation_table['Notes'] = ablation_table['Variant'].apply(get_ablation_note)

ablation_table.to_csv("tables/4_ablation_study.csv", index=False)
print("\\nAblation Study Results:")
print(ablation_table.to_string(index=False))

# =================== TABLE 5: Model Analysis Summary ===================
print("\\n\\n[5/5] Generating Model Analysis Table...")

model_analysis = {
    'Metric': [
        'Feature Importance - Recency',
        'Feature Importance - Frequency',
        'Feature Importance - Rank',
        'Q-Value Mean',
        'Q-Value Std',
        'Q-Value Min',
        'Q-Value Max',
        'Q-Value Range',
        'Total Parameters (Base Model)',
        'State Construction Time (μs)',
        'Inference Time (μs)',
        'Eviction Choice Time (μs)',
        'LRU vs RL Slowdown (Cache=30)'
    ],
    'Value': [
        weights.loc[weights['Feature'] == 'Recency', 'Importance'].values[0],
        weights.loc[weights['Feature'] == 'Frequency', 'Importance'].values[0],
        weights.loc[weights['Feature'] == 'Rank', 'Importance'].values[0],
        qvalues['Q-Values'].mean(),
        qvalues['Q-Values'].std(),
        qvalues['Q-Values'].min(),
        qvalues['Q-Values'].max(),
        qvalues['Q-Values'].max() - qvalues['Q-Values'].min(),
        4481,  # From ablation table
        microbenchmark['State Construction'].iloc[0] if 'State Construction' in microbenchmark.columns else 0,
        microbenchmark['Inference'].iloc[0] if 'Inference' in microbenchmark.columns else 0,
        microbenchmark['Eviction Choice'].iloc[0] if 'Eviction Choice' in microbenchmark.columns else 0,
        baseline[(baseline['Cache Size'] == 30) & (baseline['Algorithm'] == 'RL-Hybrid')]['Runtime (s)'].values[0] /
        baseline[(baseline['Cache Size'] == 30) & (baseline['Algorithm'] == 'LRU')]['Runtime (s)'].values[0]
    ]
}

model_table = pd.DataFrame(model_analysis)
model_table['Value'] = model_table['Value'].round(4)
model_table.to_csv("tables/5_model_analysis.csv", index=False)
print("\\nModel Analysis Summary:")
print(model_table.to_string(index=False))

# =================== PAPER VALUES EXTRACTION ===================
print("\\n\\n" + "="*80)
print("KEY VALUES FOR RESEARCH PAPER")
print("="*80)

paper_values = {}

# Get key comparisons (Cache Size = 30)
lru_30 = baseline[(baseline['Algorithm'] == 'LRU') & (baseline['Cache Size'] == 30)].iloc[0]
rl_30 = baseline[(baseline['Algorithm'] == 'RL-Hybrid') & (baseline['Cache Size'] == 30)].iloc[0]

print("\\n📊 BASELINE PERFORMANCE (Cache Size = 30):")
print(f"   LRU Hit Rate: {lru_30['Hit Rate (%)']:.2f}%")
print(f"   RL Hit Rate: {rl_30['Hit Rate (%)']:.2f}%")
print(f"   Hit Rate Improvement: {(rl_30['Hit Rate (%)'] - lru_30['Hit Rate (%)']):.2f} percentage points")
print(f"   RL Slowdown: {(rl_30['Runtime (s)'] / lru_30['Runtime (s)']):.2f}x")

print("\\n⚡ LATENCY PERCENTILES (Cache Size = 30):")
print(f"   LRU P50: {lru_30['P50 Latency (us)']:.2f} μs")
print(f"   RL P50: {rl_30['P50 Latency (us)']:.2f} μs")
print(f"   LRU P99: {lru_30['P99 Latency (us)']:.2f} μs")
print(f"   RL P99: {rl_30['P99 Latency (us)']:.2f} μs")

print("\\n🔬 MICROBENCHMARK BREAKDOWN:")
print(f"   State Construction: {microbenchmark['State Construction'].iloc[0]:.2f} μs per eviction")
print(f"   Inference Time: {microbenchmark['Inference'].iloc[0]:.2f} μs per eviction")
print(f"   Eviction Choice: {microbenchmark['Eviction Choice'].iloc[0]:.2f} μs per eviction")
print(f"   Total Per Eviction: {(microbenchmark['State Construction'].iloc[0] + microbenchmark['Inference'].iloc[0] + microbenchmark['Eviction Choice'].iloc[0]):.2f} μs")

print("\\n🧠 MODEL CHARACTERISTICS:")
print(f"   Total Parameters: 4,481")
print(f"   Feature Importance (Recency): {weights.loc[weights['Feature'] == 'Recency', 'Importance'].values[0]:.2f}")
print(f"   Feature Importance (Frequency): {weights.loc[weights['Feature'] == 'Frequency', 'Importance'].values[0]:.2f}")
print(f"   Feature Importance (Rank): {weights.loc[weights['Feature'] == 'Rank', 'Importance'].values[0]:.2f}")
print(f"   Q-Value Mean: {qvalues['Q-Values'].mean():.4f}")
print(f"   Q-Value Variance: {qvalues['Q-Values'].var():.4f}")

# Best/Worst performance
print("\\n📈 SCALABILITY:")
rl_baseline = baseline[baseline['Algorithm'] == 'RL-Hybrid'].copy()
best_idx = rl_baseline['Hit Rate (%)'].idxmax()
best_cache = rl_baseline.loc[best_idx]
print(f"   Best RL Performance: {best_cache['Hit Rate (%)']:.2f}% at cache size {int(best_cache['Cache Size'])}")

# K sensitivity
k_data = sensitivity[sensitivity['k'].isin([4, 8, 16, 32])].copy()
best_k_idx = k_data['Hit Rate (%)'].idxmax()
k_best = k_data.loc[best_k_idx]
print(f"   Optimal k value: k={int(k_best['k'])} with {k_best['Hit Rate (%)']:.2f}% hit rate")

print("\\n" + "="*80)
print("ALL TABLES GENERATED SUCCESSFULLY!")
print("="*80)
print("\\nTables saved in 'tables/' directory:")
print("  1. 1_performance_summary.csv")
print("  2. 2_sensitivity_cache_size.csv")
print("  3. 2_sensitivity_k_variation.csv")
print("  4. 3_workload_behavior.csv")
print("  5. 4_ablation_study.csv")
print("  6. 5_model_analysis.csv")
print("="*80)
