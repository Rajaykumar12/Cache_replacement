# Cache Replacement Algorithm Comparison

A comprehensive simulation and comparison of different cache replacement algorithms including traditional methods (LRU, FIFO), machine learning-based approaches using XGBoost and LightGBM, and reinforcement learning using Actor-Critic methods.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Cache Algorithms](#cache-algorithms)
- [Machine Learning Models](#machine-learning-models)
- [Reinforcement Learning](#reinforcement-learning)
- [Data Generation](#data-generation)
- [Results](#results)
- [Configuration](#configuration)
- [Contributing](#contributing)

## Overview

This project implements and compares multiple cache replacement algorithms to determine which performs best under different workload patterns. The project includes traditional algorithms, machine learning-based approaches, and cutting-edge reinforcement learning methods that adapt to changing access patterns.

### Key Highlights

- **6 Cache Algorithms**: FIFO, LRU, ML-based (XGBoost), ML-based (LightGBM), Actor-Critic RL, and Optimal
- **Real-world Simulation**: Uses Zipfian distribution to simulate realistic access patterns
- **Reinforcement Learning**: Actor-Critic agent that learns optimal replacement policies
- **Performance Metrics**: Comprehensive hit rate analysis
- **Extensible Design**: Easy to add new algorithms or modify existing ones

## Features

- **Multiple Cache Algorithms**: Compare traditional, ML-based, and RL approaches
- **Realistic Data Generation**: Zipfian distribution mimics real-world access patterns
- **Machine Learning Integration**: XGBoost and LightGBM models for intelligent caching
- **Reinforcement Learning**: Actor-Critic agent for adaptive policy learning
- **Performance Analysis**: Detailed hit rate comparisons
- **Configurable Parameters**: Easily adjust cache size, dataset size, and distribution parameters
- **Optimal Baseline**: Implementation of the theoretical optimal algorithm for comparison
- **GPU Support**: CUDA acceleration for reinforcement learning training

## Project Structure

```
Cache_replacement/
├── README.md
├── actor_critic_agent.py   # Actor-Critic RL implementation
├── data/
│   ├── app.py              # Data generation script
│   └── training_data.csv   # Generated request dataset
├── models/
│   ├── lgb_model.py        # LightGBM model training
│   ├── lgb_model.pkl       # Trained LightGBM model
│   ├── xgb_model.py        # XGBoost model training
│   └── xgb_model.pkl       # Trained XGBoost model
└── run_simpulation.py      # Main simulation script
```

## Installation

### Prerequisites

- Python 3.7+
- pip package manager

### Dependencies

```bash
pip install pandas numpy scikit-learn lightgbm xgboost joblib torch
```

For GPU support (optional):
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Clone the Repository

```bash
git clone https://github.com/Rajaykumar12/Cache_replacement.git
cd Cache_replacement
```

## Quick Start

### 1. Generate Training Data

```bash
cd data
python app.py
```

This creates `training_data.csv` with a realistic request pattern.

### 2. Train Machine Learning Models

```bash
# Train LightGBM model
cd models
python lgb_model.py

# Train XGBoost model
python xgb_model.py
```

### 3. Run the Simulation

```bash
python run_simpulation.py
```

### Expected Output

```
--- Cache Performance Simulation ---
Starting simulation with 20000 requests and cache capacity of 10...
  ...processed 10000/20000 requests
  ...processed 20000/20000 requests

Simulation finished!
------------------------------------
FIFO Cache Hit Rate:         15.23%
LRU Cache Hit Rate:          18.45%
ML-Based Cache Hit Rate:     22.67%
ML-Based Cache Hit Rate:     23.12%
Actor-Critic RL Hit Rate:    24.89%
Optimal Cache Hit Rate:      28.90%
```

## Cache Algorithms

### 1. FIFO (First In, First Out)
- **Strategy**: Evicts the oldest item in cache
- **Use Case**: Simple, predictable behavior
- **Complexity**: O(1) operations

### 2. LRU (Least Recently Used)
- **Strategy**: Evicts the least recently accessed item
- **Use Case**: Good for temporal locality patterns
- **Complexity**: O(1) operations with OrderedDict

### 3. ML-Based Cache (XGBoost)
- **Strategy**: Predicts time to next access for each item
- **Features**: Time since last access, frequency count
- **Use Case**: Complex access patterns with dependencies

### 4. ML-Based Cache (LightGBM)
- **Strategy**: Similar to XGBoost but with different algorithm
- **Features**: Same feature set as XGBoost
- **Use Case**: Often faster training and inference

### 5. Actor-Critic Reinforcement Learning
- **Strategy**: Learns optimal replacement policy through trial and error
- **Architecture**: Neural network with shared layers, separate actor and critic heads
- **Features**: Adapts to changing patterns, real-time learning
- **Use Case**: Dynamic environments with evolving access patterns

### 6. Optimal Algorithm
- **Strategy**: Theoretical optimal (Belady's algorithm)
- **Use Case**: Upper bound baseline for comparison
- **Note**: Requires future knowledge (not practical in real systems)

## Machine Learning Models

### Feature Engineering

The ML models use two key features:

1. **Time Since Last Access**: How long ago was this item accessed?
2. **Frequency Count**: How often has this item been accessed?

### Target Variable

**Time to Next Request**: How long until this item will be accessed again?

### Model Training

```python
# Example model configuration
lgb.LGBMRegressor(
    objective='regression_l1',
    n_estimators=1000,
    learning_rate=0.05,
    num_leaves=31,
    random_state=42
)
```

## Reinforcement Learning

### Actor-Critic Architecture

The Actor-Critic agent uses a neural network with:

- **Shared Layer**: 128-unit fully connected layer with ReLU activation
- **Actor Head**: Outputs action probabilities for cache replacement decisions
- **Critic Head**: Estimates state values for policy evaluation

### Key Components

```python
# Initialize Actor-Critic agent
from actor_critic_agent import ActorCriticAgent

agent = ActorCriticAgent(
    state_size=10,      # Cache state representation size
    action_size=4,      # Number of possible replacement actions
    learning_rate=0.002,
    gamma=0.99          # Discount factor
)
```

### Learning Process

- **Real-time Learning**: Updates policy after every cache operation
- **Advantage Function**: A(s,a) = R + γV(s') - V(s) guides policy updates
- **Loss Functions**: 
  - Critic Loss: Mean Squared Error between predicted and target values
  - Actor Loss: Policy gradient with advantage as baseline

### Training Features

- Single-step updates for immediate adaptation
- Automatic GPU utilization if available
- Robust handling of edge cases
- Categorical action distribution with softmax

## Data Generation

The project uses a **Zipfian distribution** to generate realistic access patterns:

- **Parameter**: `ZIPF_PARAM_A = 1.1` (higher = more skewed)
- **Items**: 1000 unique items
- **Requests**: 20,000 total requests
- **Pattern**: Some items are accessed much more frequently than others

### Customization

Edit `data/app.py` to modify:

```python
NUM_REQUESTS = 20000    # Total requests
NUM_ITEMS = 1000        # Unique items
ZIPF_PARAM_A = 1.1      # Distribution skewness
```

## Results

Typical performance hierarchy:

1. **Optimal Algorithm** (~29%) - Theoretical upper bound
2. **Actor-Critic RL** (~25%) - Best adaptive algorithm
3. **ML-Based (LightGBM)** (~23%) - Best traditional ML approach
4. **ML-Based (XGBoost)** (~23%) - Close second in ML category
5. **LRU** (~18%) - Good traditional algorithm
6. **FIFO** (~15%) - Baseline traditional algorithm

*Results may vary based on data distribution, cache size, and RL training progress*

### Performance Analysis

- **Actor-Critic RL** excels in dynamic environments where access patterns change over time
- **ML-based approaches** perform well when historical patterns are indicative of future behavior
- **Traditional algorithms** provide consistent baseline performance

## Configuration

### Cache Settings

Edit `run_simpulation.py`:

```python
CACHE_CAPACITY = 10              # Cache size
REQUEST_LOG_FILE = "data/training_data.csv"
XGB_FILE = "models/xgb_model.pkl"
LGB_FILE = "models/lgb_model.pkl"
```

### Actor-Critic Parameters

Modify RL agent settings:

```python
# In actor_critic_agent.py
agent = ActorCriticAgent(
    state_size=10,
    action_size=4,
    learning_rate=0.002,    # Learning rate
    gamma=0.99              # Discount factor
)
```

### Model Parameters

Modify model training in `models/lgb_model.py` or `models/xgb_model.py`:

```python
# LightGBM example
lgb.LGBMRegressor(
    objective='regression_l1',
    n_estimators=1000,
    learning_rate=0.05,
    num_leaves=31
)
```

## Adding New Algorithms

To add a new cache algorithm:

1. **Create a new class** following the interface:
   ```python
   class NewCache:
       def __init__(self, capacity: int):
           self.capacity = capacity
           self.hits = 0
           self.misses = 0
       
       def process_request(self, item_id):
           # Your algorithm logic here
           pass
       
       def get_hit_rate(self):
           total = self.hits + self.misses
           return (self.hits / total) * 100 if total > 0 else 0
   ```

2. **Add to simulation** in `run_simpulation.py`:
   ```python
   new_cache = NewCache(capacity=CACHE_CAPACITY)
   
   # In the simulation loop:
   new_cache.process_request(item_id)
   
   # In results:
   print(f"New Cache Hit Rate: {new_cache.get_hit_rate():.2f}%")
   ```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Areas for Contribution

- Implement additional RL algorithms (DQN, PPO, etc.)
- Add more sophisticated state representations
- Create visualization tools for cache performance
- Implement multi-level cache hierarchies
- Add real-world trace file support

## License

This project is open source and available under the [MIT License](LICENSE).

## Author

**Rajay Kumar** - [Rajaykumar12](https://github.com/Rajaykumar12)