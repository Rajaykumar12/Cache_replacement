import numpy as np
from collections import OrderedDict
from dqn_agent import DQNAgent 

# --- Reinforcement Learning Cache Environment ---
class RLCache:
    """The cache environment that the DQN agent interacts with."""
    def __init__(self, capacity: int, agent: DQNAgent):
        self.capacity = capacity
        self.agent = agent
        
        # MODIFICATION: Use an OrderedDict to automatically track recency order
        self.cache = OrderedDict()
        
        # History for feature calculation
        self.item_history_frequency = {}
        
        self.current_timestamp = 0
        self.hits = 0
        self.misses = 0

    def _get_state(self):
        """Converts the current cache content into a fixed-size state vector."""
        state = np.zeros(self.agent.state_size)
        # MODIFICATION: We now have 3 features per item
        num_features = 3 
        
        # Items are naturally ordered by recency in the OrderedDict
        items_in_cache = list(self.cache.keys())
        
        for i, item_id in enumerate(items_in_cache):
            if i < self.capacity:
                # Feature 1: Time since last access (recency)
                state[i * num_features] = self.current_timestamp - self.cache[item_id] 
                
                # Feature 2: Frequency count
                state[i * num_features + 1] = self.item_history_frequency.get(item_id, 1)
                
                # MODIFICATION: Feature 3 - Normalized Recency Rank
                # The first item (LRU) gets rank 1.0, last item (MRU) gets a low rank
                state[i * num_features + 2] = (i + 1) / self.capacity

        return state

    def process_request(self, item_id: int):
        """Processes a single item request and triggers the RL loop if needed."""
        self.current_timestamp += 1
        reward = 0
        
        # Update frequency history
        self.item_history_frequency[item_id] = self.item_history_frequency.get(item_id, 0) + 1

        if item_id in self.cache:
            # --- CACHE HIT ---
            self.hits += 1
            reward = 1 # Positive reward
            # MODIFICATION: Update timestamp and move item to the end (Most Recently Used)
            self.cache[item_id] = self.current_timestamp
            self.cache.move_to_end(item_id)
        else:
            # --- CACHE MISS ---
            self.misses += 1
            reward = -1 # Negative reward
            
            if len(self.cache) >= self.capacity:
                # GET STATE AND CHOOSE ACTION
                current_state = self._get_state()
                action_index = self.agent.choose_action(current_state)
                
                # EVICT THE CHOSEN ITEM
                items_in_cache = list(self.cache.keys())
                if action_index < len(items_in_cache):
                    evicted_item = items_in_cache[action_index]
                    del self.cache[evicted_item]
                else:
                    self.cache.popitem(last=False)
                
                # LEARN FROM EXPERIENCE - only for agents that have remember method
                next_state = self._get_state()
                if hasattr(self.agent, 'remember'):
                    self.agent.remember(current_state, action_index, reward, next_state)
            
            # ADD THE NEW ITEM
            self.cache[item_id] = self.current_timestamp
    
    def get_hit_rate(self):
        total = self.hits + self.misses
        return (self.hits / total) * 100 if total > 0 else 0