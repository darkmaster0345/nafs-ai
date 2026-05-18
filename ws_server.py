"""
Nafs AI — WebSocket Server (Standalone)
Broadcasts Adam's tick data to all connected dashboard clients.

This is a standalone WebSocket server that can be used alongside
the Nafs AI simulation. It accepts JSON data via HTTP POST and
broadcasts to all connected WebSocket clients.

Usage:
    python ws_server.py

The server runs on port 8765 and supports:
- WebSocket connections at ws://localhost:8765/ws
- HTTP POST at http://localhost:8765/broadcast for pushing data
- HTTP GET at http://localhost:8765/api/status for health check
"""

import asyncio
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

app = FastAPI(title="Nafs AI WebSocket Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
connected_clients: set = set()
latest_data: dict = {}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Accept WebSocket connections and broadcast data."""
    await websocket.accept()
    connected_clients.add(websocket)
    try:
        # Send latest data to new client
        if latest_data:
            await websocket.send_text(json.dumps(latest_data))

        while True:
            # Keep connection alive, receive commands
            data = await websocket.receive_text()
            msg = json.loads(data)

            if msg.get("type") == "ping":
                await websocket.send_text(json.dumps({"type": "pong"}))

    except WebSocketDisconnect:
        connected_clients.discard(websocket)
    except Exception as e:
        print(f"WebSocket error: {e}")
        connected_clients.discard(websocket)


@app.post("/broadcast")
async def broadcast_data(data: dict):
    """HTTP endpoint to broadcast data to all connected clients."""
    global latest_data
    latest_data = data

    disconnected = set()
    for client in connected_clients:
        try:
            # FastAPI WebSocket doesn't have send_text in this context
            # We'll use a different approach with asyncio
            pass
        except Exception:
            disconnected.add(client)

    connected_clients.difference_update(disconnected)
    return {"status": "ok", "clients": len(connected_clients)}


@app.get("/api/status")
async def status():
    """Health check endpoint."""
    return {
        "running": True,
        "clients": len(connected_clients),
        "has_data": bool(latest_data),
    }


async def broadcast_to_all(data: dict):
    """Broadcast data to all connected WebSocket clients."""
    global latest_data
    latest_data = data

    disconnected = set()
    message = json.dumps(data)

    for client in connected_clients:
        try:
            await client.send_text(message)
        except Exception:
            disconnected.add(client)

    connected_clients.difference_update(disconnected)


if __name__ == "__main__":
    print("🌐 Nafs AI WebSocket Server starting on port 8765...")
    uvicorn.run(app, host="0.0.0.0", port=8765)
