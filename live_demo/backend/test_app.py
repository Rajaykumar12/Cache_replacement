import asyncio
import websockets
import json
import requests
import sys
import time

async def test_websocket():
    uri = "ws://localhost:8000/ws"
    async with websockets.connect(uri) as websocket:
        # Send config (model param is ignored now)
        config = {
            "distribution": "zipf",
            "cacheSize": 10,
            "numRequests": 50,
            "speedDelay": 0.01
        }
        await websocket.send(json.dumps(config))
        
        # Receive messages
        count = 0
        try:
            while True:
                message = await asyncio.wait_for(websocket.recv(), timeout=2.0)
                data = json.loads(message)
                
                # Verify structure
                if "status" in data and data["status"] == "done":
                    continue
                    
                if "model" not in data:
                    print("Error: 'model' key missing in payload")
                    return False
                
                if data["model"] not in ["LRU", "LFU", "ARC", "RL"]:
                    print(f"Error: Unknown model in payload. Got: {data['model']}")
                    return False
                
                count += 1
                if count >= 50:
                    break
        except asyncio.TimeoutError:
            print("Timeout waiting for message")
            return False
            
        print(f"Successfully received {count} messages via WebSocket.")
        return True

def test_http():
    try:
        response = requests.get("http://localhost:8000/")
        if response.status_code == 200 and "<title>Live Cache Simulation</title>" in response.text:
            print("HTTP Index page served successfully.")
            return True
        else:
            print(f"HTTP Check Failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"HTTP Connection Failed: {e}")
        return False

async def main():
    print("Testing HTTP...")
    if not test_http():
        sys.exit(1)
        
    print("Testing WebSocket...")
    if not await test_websocket():
        sys.exit(1)
        
    print("All tests passed!")

if __name__ == "__main__":
    asyncio.run(main())
