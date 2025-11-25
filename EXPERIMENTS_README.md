# RL Cache Experimentation - Results Summary

## Overview

This directory contains comprehensive experimental results for the RL-based cache eviction algorithm research paper.

**Status**: ✅ **All Experiments Complete**

---

## 📊 Quick Access

### Key Results (Cache Size = 30)

| Metric | LRU | RL-Hybrid | Improvement |
|--------|-----|-----------|-------------|
| **Hit Rate** | 25.20% | 35.11% | **+9.91 pp** |
| **Runtime** | 0.031s | 1.173s | 38x slower |
| **P99 Latency** | 0.73 μs | 20.21 μs | 27.7x slower |

### Microbenchmark Breakdown (per eviction)
- State Construction: **5.03 μs**
- Model Inference: **10.80 μs** (63% of overhead)
- Eviction Choice: **1.32 μs**

---

## 📁 Directory Structure

```
├── data_gen/              # Generated workload traces
│   ├── zipf_alpha_*.csv  # Zipf distributions (α = 0.5, 0.8, 1.0, 1.2, 1.5)
│   ├── uniform.csv       # Uniform random
│   ├── gaussian.csv      # Gaussian distribution
│   ├── bursty.csv        # Bursty workload
│   ├── periodic.csv      # Phase-shifting workload
│   └── adversarial.csv   # LRU killer
│
├── results_*.csv          # Raw experimental results
│   ├── results_baseline.csv       # Main comparison (LRU, LFU, ARC, RL)
│   ├── results_sensitivity.csv    # Cache size & k variation
│   ├── results_behavior.csv       # Workload behavior tests
│   ├── results_ablation.csv       # Architecture ablation
│   ├── results_microbenchmark.csv # Timing breakdown
│   ├── results_weights.csv        # Feature importance
│   └── results_qvalues.csv        # Q-value distribution
│
├── tables/                # Formatted tables for paper
│   ├── 1_performance_summary.csv
│   ├── 2_sensitivity_*.csv
│   ├── 3_workload_behavior.csv
│   ├── 4_ablation_study.csv
│   └── 5_model_analysis.csv
│
└── plots/                 # Visualizations (PNG, 300 DPI)
    ├── hit_rate_vs_cache_size.png
    ├── runtime_vs_cache_size.png
    ├── runtime_vs_k.png
    ├── hit_rate_vs_zipf_alpha.png
    ├── qvalue_distribution.png
    ├── feature_weights.png
    ├── latency_cdf.png
    ├── microbenchmark_breakdown.png
    └── ablation_comparison.png
```

---

## 🔬 Experiments Completed

### ✅ 1. Baseline Performance
- Algorithms: LRU, LFU, ARC, RL-Hybrid
- Cache sizes: 10, 20, 30, 50, 100, 200
- File: `results_baseline.csv`
- Plots: `hit_rate_vs_cache_size.png`, `runtime_vs_cache_size.png`

### ✅ 2. Microbenchmark Analysis
- Per-eviction timing breakdown
- File: `results_microbenchmark.csv`
- Plot: `microbenchmark_breakdown.png`

### ✅ 3. Sensitivity Analysis
- **3.1**: Cache size variation (10-200)
- **3.2**: Candidate size k (4, 8, 16, 32)
- **3.3**: Workload distributions (8 types)
- Files: `results_sensitivity.csv`, `tables/2_sensitivity_*.csv`
- Plots: `runtime_vs_k.png`, `hit_rate_vs_zipf_alpha.png`

### ✅ 4. Workload Behavior
- Periodic, adversarial, bursty patterns
- File: `results_behavior.csv`, `tables/3_workload_behavior.csv`

### ✅ 5. Model Internal Analysis
- Feature importance (Recency: 19.48, Frequency: 18.14, Rank: 20.81)
- Q-value statistics (μ=-2.01, σ=1.69)
- Files: `results_weights.csv`, `results_qvalues.csv`
- Plots: `feature_weights.png`, `qvalue_distribution.png`

### ✅ 6. Ablation Studies
- 6 architectural variants (64-64-1, 32-32-1, 16-16-1, 8-8-1, 128-128-1, 64-32-1)
- File: `results_ablation.csv`, `tables/4_ablation_study.csv`
- Plot: `ablation_comparison.png`

---

## 🎯 Key Findings

### Strengths
1. **+9.91pp hit rate improvement** over LRU on moderate Zipf workloads (α=1.0)
2. **Best performance** at large cache sizes (47.54% @ size 200)
3. **Efficient small model**: 16-16-1 variant achieves 36.12% with only 353 parameters
4. **Rank is most important feature** (20.81 weight sum), indicating LRU-like bias with refinements

### Limitations
1. **38x runtime overhead** - requires hardware acceleration for deployment
2. **No advantage on uniform or highly skewed distributions**
3. **Inference dominates overhead** (10.80 μs = 63% of eviction time)

### Optimal Configuration
- **Cache Size**: Larger the better (performance scales well)
- **k value**: k=16 offers best quality/speed trade-off
- **Architecture**: 16-16-1 MLP (faster, smaller, similar accuracy)

---

## 📈 Values for Paper Sections

### Abstract
- Hit rate: **+9.91pp improvement** (25.20% → 35.11%)
- Overhead: **38x slowdown** vs LRU

### Results - Baseline
- See `tables/1_performance_summary.csv`
- LFU achieves best overall (39.36%), RL second place at 35.11%

### Results - Scalability
- Linear scaling with cache size (28.68% @ size 10 → 47.54% @ size 200)
- Overhead remains constant (~3700%)

### Results - Workload Analysis
- See `tables/3_workload_behavior.csv`
- Best on: Zipf α=1.0-1.2 (+8-10pp)
- Worst on: Uniform, Gaussian (no improvement)

### Discussion - Model Behavior
- Feature importance: Rank \u003e Recency \u003e Frequency
- Q-value range: 14.37 (indicates good differentiation)
- See `tables/5_model_analysis.csv`

---

## 🚀 Scripts

Run these to regenerate results:

```bash
# 1. Generate workload traces
python workload_generator.py

# 2. Run all experiments
python run_experiments.py --model models/rl_eviction_model.pth

# 3. Run microbenchmarks
python run_microbenchmarks.py

# 4. Analyze model internals
python analyze_model.py

# 5. Run ablation studies
python run_ablation.py

# 6. Generate all plots
python generate_plots.py

# 7. Generate all tables
python generate_tables.py
```

---

## 📋 Paper Integration Checklist

- [x] Baseline comparison table
- [x] Cache size sensitivity plot
- [x] k parameter sensitivity plot
- [x] Workload behavior table
- [x] Microbenchmark breakdown
- [ x] Feature importance bar chart
- [x] Q-value distribution histogram
- [x] Latency CDF comparison
- [x] Ablation comparison table
- [x] All key metrics extracted

---

## 💡 Future Work (for paper discussion)

1. **Hardware Acceleration**: FPGA/ASIC implementation to reduce 38x overhead
2. **Hybrid Approach**: RL only for specific workload patterns (α=0.8-1.2)
3. **Quantization**: 8-bit weights could reduce inference time 2-4x
4. **Online Learning**: Real-time adaptation to workload shifts

---

## 📝 Notes

- Total requests per experiment: 100,000
- RL model parameters: 4,481 (base 64-64-1)
- All plots generated at 300 DPI for publication quality
- Statistical significance testing skipped (single run per config)

**For full experiment details, see:** [walkthrough.md](/.gemini/antigravity/brain/98fda506-a013-437a-95f0-b3ce76637595/walkthrough.md)
