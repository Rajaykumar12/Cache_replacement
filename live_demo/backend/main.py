import sys
import os
import asyncio
import json
import time
import numpy as np
import pandas as pd
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

# Add parent directory to path to import existing modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from simulation import LRUCache, RLHybridCache
from rl_tail import ValueDQNAgent
from experiments.baselines import LFUCache, ARCCache
from experiments.workload_generator import (
    generate_zipf, generate_uniform, generate_gaussian, 
    generate_bursty, generate_periodic, generate_adversarial
)

from fastapi.staticfiles import StaticFiles

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load RL Model globally for now (or load on demand)
MODEL_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../models/rl_eviction_model.pth'))
rl_agent = ValueDQNAgent(state_size=3)
try:
    rl_agent.load(MODEL_FILE)
    print("RL Model loaded successfully.")
except Exception as e:
    print(f"Warning: Could not load RL model: {e}")

def get_data_generator(dist_name, n, items):
    if dist_name == 'zipf': return generate_zipf(1.2, n, items) # Default alpha 1.2
    if dist_name == 'uniform': return generate_uniform(n, items)
    if dist_name == 'gaussian': return generate_gaussian(n, items)
    if dist_name == 'bursty': return generate_bursty(n, items)
    if dist_name == 'periodic': return generate_periodic(n, items)
    if dist_name == 'adversarial': return generate_adversarial(n, items)
    return generate_zipf(1.0, n, items)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        # Wait for initialization message
        data = await websocket.receive_text()
        config = json.loads(data)
        
        model_name = config.get("model", "LRU")
        distribution = config.get("distribution", "zipf")
        cache_size = int(config.get("cacheSize", 30))
        num_requests = int(config.get("numRequests", 1000))
        speed_delay = float(config.get("speedDelay", 0.1)) # seconds
        
        print(f"Starting simulation: {model_name}, {distribution}, Size: {cache_size}")

        # Initialize All Caches
        caches = {
            "LRU": LRUCache(cache_size),
            "LFU": LFUCache(cache_size),
            "ARC": ARCCache(cache_size),
            "RL": RLHybridCache(cache_size, rl_agent, k=16)
        }

        # Generate Data
        items = 1000 # Pool of items
        requests = get_data_generator(distribution, num_requests, items)
        
        # Generate Data
        items = 1000 # Pool of items
        requests_data = get_data_generator(distribution, num_requests, items)
        # Convert to list for shared access (read-only is fine)
        requests_list = requests_data.tolist() if isinstance(requests_data, np.ndarray) else list(requests_data)

        async def run_model_simulation(name, cache, reqs, speed_factor):
            try:
                for i, item_id in enumerate(reqs):
                    start_time = time.perf_counter()
                    cache.process_request(item_id)
                    end_time = time.perf_counter()
                    
                    inference_time = (end_time - start_time) # seconds
                    
                    # Simulate processing delay
                    # speed_factor is a multiplier. 
                    # If speed_delay (from config) is 0.05, we treat it as a base delay.
                    # But user wants "time taken to execute it".
                    # So we sleep for inference_time * factor.
                    # To make it visible, we might need a large factor, e.g., 100x or 1000x real time, 
                    # OR just add the user's base delay + inference_time.
                    
                    # Let's use a hybrid: Base Delay + (Inference Time * Multiplier)
                    # This ensures even fast models have a "tick" but slow ones are slower.
                    delay = speed_delay + (inference_time * 100) 
                    
                    metrics = {
                        "model": name,
                        "step": i + 1,
                        "itemId": int(item_id),
                        "hits": cache.hits,
                        "misses": cache.misses,
                        "hitRate": cache.get_hit_rate(),
                        "inferenceTime": inference_time * 1000, # ms
                        "cacheSize": len(cache.cache) if hasattr(cache, 'cache') else 0
                    }
                    
                    await websocket.send_json(metrics)
                    await asyncio.sleep(delay)
                    
                # Send completion message for this model
                await websocket.send_json({"model": name, "status": "done"})
                
            except Exception as e:
                print(f"Error in {name} simulation: {e}")

        # Create tasks
        tasks = []
        caches = {
            "LRU": LRUCache(cache_size),
            "LFU": LFUCache(cache_size),
            "ARC": ARCCache(cache_size),
            "RL": RLHybridCache(cache_size, rl_agent, k=16)
        }
        
        for name, cache in caches.items():
            tasks.append(asyncio.create_task(run_model_simulation(name, cache, requests_list, speed_delay)))
            
        # Wait for all to complete
        await asyncio.gather(*tasks)
                
            # Check for control messages (pause/stop) - simplified for now
            # In a real app, we'd need a separate control channel or non-blocking receive
            
    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"Error: {e}")
        await websocket.close()

# Serve frontend (Must be last to avoid overriding API routes)
app.mount("/", StaticFiles(directory=os.path.abspath(os.path.join(os.path.dirname(__file__), "../frontend")), html=True), name="static")
