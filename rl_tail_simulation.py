import pandas as pd
import numpy as np
import joblib
from collections import OrderedDict
import matplotlib.pyplot as plt
import time
import torch # Still needed for FloatTensor in agent

# Import the agent definition from the training script
from rl_tail import ValueDQNAgent

# ===================================================================
# =================== CACHE IMPLEMENTATIONS =========================
# ===================================================================

class LRUCache:
    def __init__(self, capacity: int):
        self.cache, self.capacity, self.hits, self.misses = OrderedDict(), capacity, 0, 0
    def process_request(self, item_id):
        if item_id in self.cache: self.hits += 1; self.cache.move_to_end(item_id)
        else:
            self.misses += 1
            if len(self.cache) >= self.capacity: self.cache.popitem(last=False)
            self.cache[item_id] = True
    def get_hit_rate(self): total = self.hits + self.misses; return (self.hits / total) * 100 if total > 0 else 0

class MLHybridCache: # Uses the pre-trained supervised ML model
    def __init__(self, capacity: int, model, tail_sample_size: int):
        self.capacity, self.model, self.k = capacity, model, tail_sample_size
        self.cache, self.item_history_last_access, self.item_history_frequency = OrderedDict(), {}, {}
        self.current_timestamp, self.hits, self.misses = 0, 0, 0
    def _evict(self):
        candidate_items = list(self.cache.keys())[:self.k]
        features = [[self.current_timestamp - self.item_history_last_access.get(item, self.current_timestamp), 
                     self.item_history_frequency.get(item, 1)] for item in candidate_items]
        predictions = self.model.predict(pd.DataFrame(features, columns=['time_since_last_access', 'frequency_count']))
        del self.cache[candidate_items[np.argmax(predictions)]]
    def process_request(self, item_id):
        self.current_timestamp += 1; self.item_history_last_access[item_id] = self.current_timestamp; self.item_history_frequency[item_id] = self.item_history_frequency.get(item_id, 0) + 1
        if item_id in self.cache: self.hits += 1; self.cache.move_to_end(item_id)
        else:
            self.misses += 1
            if len(self.cache) >= self.capacity: self._evict()
            self.cache[item_id] = True
    def get_hit_rate(self): total = self.hits + self.misses; return (self.hits / total) * 100 if total > 0 else 0

class RLHybridCache: # Uses the pre-trained RL agent
    def __init__(self, capacity: int, agent: ValueDQNAgent, tail_sample_size: int):
        self.capacity, self.agent, self.k = capacity, agent, tail_sample_size
        self.cache, self.item_history_frequency, self.current_timestamp, self.hits, self.misses = OrderedDict(), {}, 0, 0, 0
    def _get_item_state(self, item_id):
        recency = self.current_timestamp - self.cache.get(item_id, self.current_timestamp)
        frequency = self.item_history_frequency.get(item_id, 1)
        try: rank = list(self.cache.keys()).index(item_id) + 1
        except ValueError: rank = self.capacity
        recency_rank = rank / self.capacity
        return np.array([recency, frequency, recency_rank])
    def _evict(self):
        candidate_items = list(self.cache.keys())[:self.k]
        candidate_states = [self._get_item_state(item) for item in candidate_items]
        values = [self.agent.get_value(state) for state in candidate_states]
        del self.cache[candidate_items[np.argmin(values)]] # Evict item with the lowest value
    def process_request(self, item_id: int):
        self.current_timestamp += 1
        self.item_history_frequency[item_id] = self.item_history_frequency.get(item_id, 0) + 1
        if item_id in self.cache:
            self.hits += 1; self.cache.move_to_end(item_id)
        else:
            self.misses += 1
            if len(self.cache) >= self.capacity: self._evict()
        self.cache[item_id] = self.current_timestamp
        # NO LEARNING HAPPENS IN THIS SCRIPT
    def get_hit_rate(self): total = self.hits + self.misses; return (self.hits / total) * 100 if total > 0 else 0

# ===================================================================
# =================== MAIN SIMULATION SCRIPT ========================
# ===================================================================

if __name__ == "__main__":
    # --- Configuration ---
    CACHE_CAPACITY = 30
    SUPERVISED_MODEL_FILE = "models/xgb_model.pkl"
    RL_MODEL_FILE = "models/rl_eviction_model.pth"
    REQUEST_LOG_FILE = "data/training_data.csv"
    HYBRID_TAIL_SAMPLE_SIZE = 16
    NUM_FEATURES = 3
    
    # print("--- Focused Cache Performance Simulation ---")
    try:
        requests_df = pd.read_csv(REQUEST_LOG_FILE)
        ml_model = joblib.load(SUPERVISED_MODEL_FILE)
    except FileNotFoundError as e:
        print(f"Error loading a required file: {e}"); exit()
    
    # Initialize and LOAD the pre-trained RL agent
    rl_agent = ValueDQNAgent(state_size=NUM_FEATURES)
    rl_agent.load(RL_MODEL_FILE)
    
    caches = {
        "LRU": LRUCache(capacity=CACHE_CAPACITY),
        "ML Hybrid (Supervised)": MLHybridCache(capacity=CACHE_CAPACITY, model=ml_model, tail_sample_size=HYBRID_TAIL_SAMPLE_SIZE),
        "RL Hybrid (Pre-Trained)": RLHybridCache(capacity=CACHE_CAPACITY, agent=rl_agent, tail_sample_size=HYBRID_TAIL_SAMPLE_SIZE)
    }
    
    total_requests = len(requests_df)
    # print(f"Starting simulation with {total_requests} requests...")
    start_time = time.time()
    
    for i, row in requests_df.iterrows():
        item_id = int(row['item_id'])
        for cache in caches.values(): cache.process_request(item_id)
        
        if (i + 1) % 100000 == 0:
            print(f"  ...processed {i + 1}/{total_requests} requests")
            
    print(f"\nSimulation finished in {time.time() - start_time:.2f} seconds!")
    # print("--- FINAL CACHE PERFORMANCE RESULTS ---")
    results = {name: cache.get_hit_rate() for name, cache in caches.items()}
    for name, hit_rate in results.items(): print(f"{name}: \t{hit_rate:.2f}%")

    print("\nGenerating comprehensive performance graphs...")
    
    # Create a figure with multiple subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Cache Performance Analysis', fontsize=16, fontweight='bold')
    
    names, hit_rates = list(results.keys()), list(results.values())
    colors = ['#3498db', '#2ecc71', '#e74c3c']  # Blue, Green, Red
    
    # 1. Bar Chart - Hit Rates Comparison
    bars = ax1.bar(names, hit_rates, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    ax1.set_ylabel('Cache Hit Rate (%)', fontweight='bold')
    ax1.set_title('Cache Hit Rate Comparison', fontweight='bold')
    ax1.set_ylim(0, max(hit_rates) * 1.2)
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, hit_rate in zip(bars, hit_rates):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + max(hit_rates)*0.01,
                f'{hit_rate:.2f}%', ha='center', va='bottom', fontweight='bold')
    
    # Rotate x-axis labels for better readability
    ax1.tick_params(axis='x', rotation=15)
    
    # 2. Horizontal Bar Chart for easier reading
    y_pos = np.arange(len(names))
    bars_h = ax2.barh(y_pos, hit_rates, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(names)
    ax2.set_xlabel('Cache Hit Rate (%)', fontweight='bold')
    ax2.set_title('Cache Performance (Horizontal View)', fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, (bar, hit_rate) in enumerate(zip(bars_h, hit_rates)):
        ax2.text(hit_rate + max(hit_rates)*0.01, i, f'{hit_rate:.2f}%', 
                va='center', ha='left', fontweight='bold')
    
    # 3. Performance Improvement Chart
    baseline_performance = hit_rates[0]  # LRU as baseline
    improvements = [(rate - baseline_performance) for rate in hit_rates]
    improvement_colors = ['gray' if imp <= 0 else '#27ae60' if imp > 0 else '#e74c3c' for imp in improvements]
    
    bars_imp = ax3.bar(names, improvements, color=improvement_colors, alpha=0.8, edgecolor='black', linewidth=1)
    ax3.set_ylabel('Improvement over LRU (%)', fontweight='bold')
    ax3.set_title('Performance Improvement vs LRU Baseline', fontweight='bold')
    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax3.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, improvement in zip(bars_imp, improvements):
        height = bar.get_height()
        y_pos = height + (max(improvements) - min(improvements))*0.02 if height >= 0 else height - (max(improvements) - min(improvements))*0.02
        ax3.text(bar.get_x() + bar.get_width()/2., y_pos,
                f'{improvement:+.2f}%', ha='center', va='bottom' if height >= 0 else 'top', fontweight='bold')
    
    ax3.tick_params(axis='x', rotation=15)
    
    # 4. Performance Statistics Table
    ax4.axis('tight')
    ax4.axis('off')
    
    # Calculate additional statistics
    total_requests_per_cache = [cache.hits + cache.misses for cache in caches.values()]
    hit_counts = [cache.hits for cache in caches.values()]
    miss_counts = [cache.misses for cache in caches.values()]
    
    table_data = []
    for i, (name, hit_rate) in enumerate(results.items()):
        table_data.append([
            name,
            f"{hit_rate:.2f}%",
            f"{hit_counts[i]:,}",
            f"{miss_counts[i]:,}",
            f"{total_requests_per_cache[i]:,}",
            f"{improvements[i]:+.2f}%"
        ])
    
    table = ax4.table(cellText=table_data,
                     colLabels=['Algorithm', 'Hit Rate', 'Hits', 'Misses', 'Total', 'vs LRU'],
                     cellLoc='center',
                     loc='center')
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2)
    
    # Style the table
    for i in range(len(table_data) + 1):
        for j in range(6):
            cell = table[(i, j)]
            if i == 0:  # Header row
                cell.set_facecolor('#34495e')
                cell.set_text_props(weight='bold', color='white')
            else:
                if j == 1:  # Hit rate column
                    if float(table_data[i-1][1].replace('%', '')) > baseline_performance:
                        cell.set_facecolor('#d5f4e6')  # Light green for better performance
                    else:
                        cell.set_facecolor('#fdf2e9')  # Light orange for baseline
                cell.set_edgecolor('black')
                cell.set_linewidth(1)
    
    ax4.set_title('Detailed Performance Statistics', fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    # Save the comprehensive graph
    plt.savefig('cache_performance_analysis.png', dpi=300, bbox_inches='tight')
    print("Comprehensive analysis saved as 'cache_performance_analysis.png'")
    
    # Create a separate figure for algorithm comparison
    plt.figure(figsize=(12, 8))
    
    # Create a more detailed comparison chart
    x = np.arange(len(names))
    width = 0.35
    
    fig2, (ax5, ax6) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Performance vs Complexity Chart
    complexity_scores = [1, 3, 4]  # Relative complexity scores (1=simple, 5=complex)
    scatter = ax5.scatter(complexity_scores, hit_rates, s=[200, 300, 400], 
                         c=colors, alpha=0.7, edgecolors='black', linewidth=2)
    
    for i, name in enumerate(names):
        ax5.annotate(name, (complexity_scores[i], hit_rates[i]), 
                    xytext=(5, 5), textcoords='offset points', 
                    fontweight='bold', fontsize=10)
    
    ax5.set_xlabel('Algorithm Complexity (Relative)', fontweight='bold')
    ax5.set_ylabel('Cache Hit Rate (%)', fontweight='bold')
    ax5.set_title('Performance vs Complexity Trade-off', fontweight='bold')
    ax5.grid(True, alpha=0.3)
    ax5.set_xlim(0.5, 4.5)
    
    # Miss Rate Comparison (complementary view)
    miss_rates = [100 - rate for rate in hit_rates]
    bars_miss = ax6.bar(names, miss_rates, color=['#e74c3c', '#f39c12', '#3498db'], 
                       alpha=0.8, edgecolor='black', linewidth=1)
    ax6.set_ylabel('Cache Miss Rate (%)', fontweight='bold')
    ax6.set_title('Cache Miss Rate Comparison', fontweight='bold')
    ax6.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, miss_rate in zip(bars_miss, miss_rates):
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height + max(miss_rates)*0.01,
                f'{miss_rate:.2f}%', ha='center', va='bottom', fontweight='bold')
    
    ax6.tick_params(axis='x', rotation=15)
    
    plt.tight_layout()
    plt.savefig('cache_detailed_comparison.png', dpi=300, bbox_inches='tight')
    print("Detailed comparison saved as 'cache_detailed_comparison.png'")
    
    # Print summary statistics
    print("\n" + "="*60)
    print("PERFORMANCE SUMMARY")
    print("="*60)
    best_algorithm = max(results, key=results.get)
    worst_algorithm = min(results, key=results.get)
    
    print(f"🏆 Best Performing Algorithm: {best_algorithm} ({results[best_algorithm]:.2f}%)")
    print(f"📉 Baseline Algorithm: {worst_algorithm} ({results[worst_algorithm]:.2f}%)")
    print(f"📊 Performance Improvement: {results[best_algorithm] - results[worst_algorithm]:.2f} percentage points")
    print(f"📈 Relative Improvement: {((results[best_algorithm] / results[worst_algorithm]) - 1) * 100:.2f}%")
    
    print("\n" + "="*60)
    
    plt.show()