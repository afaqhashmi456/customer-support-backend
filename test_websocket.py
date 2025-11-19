"""
Test WebSocket connection directly to diagnose issues.
"""
import asyncio
import websockets
import json
import sys

async def test_websocket():
    # Get token from command line or use a test token
    token = sys.argv[1] if len(sys.argv) > 1 else "test_token"
    
    uri = f"ws://localhost:8000/ws/chat?token={token}"
    print(f"Connecting to: {uri}")
    
    try:
        async with websockets.connect(uri) as websocket:
            print("✅ Connected successfully!")
            
            # Send a test message
            test_message = {"message": "Hello, test message"}
            print(f"Sending: {test_message}")
            await websocket.send(json.dumps(test_message))
            
            # Wait for response
            print("Waiting for response...")
            response = await websocket.recv()
            print(f"Received: {response}")
            
    except websockets.exceptions.InvalidStatusCode as e:
        print(f"❌ Connection failed with status code: {e.status_code}")
        print(f"Response headers: {e.headers}")
    except Exception as e:
        print(f"❌ Error: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_websocket())

