"""
Nafs AI — WebSocket Server (Standalone, v0.3 — Stability-Hardened)
Broadcasts Adam's (and Eve's) tick data to all connected dashboard clients.

This is a standalone WebSocket server that can be used alongside
the Nafs AI simulation. It accepts JSON data via HTTP POST and
broadcasts to all connected WebSocket clients.

Usage:
    python ws_server.py

The server runs on port 8765 and supports:
- WebSocket connections at ws://localhost:8765/ws
- HTTP POST at http://localhost:8765/broadcast for pushing data
- HTTP GET at http://localhost:8765/api/status for health check

v0.3 Stability Fixes:
  - /broadcast now ACTUALLY broadcasts to all WS clients (was a no-op `pass` before)
  - Per-client send queue with backpressure (slow clients get dropped, not block everyone)
  - Background broadcaster task drains queue at fixed rate
  - Ping/pong heartbeat every 20s to detect dead connections
  - Connection cleanup on WebSocketDisconnect + generic Exception
  - Bounded client set to prevent unbounded growth
  - Structured logging with timestamps
  - Graceful shutdown on SIGINT/SIGTERM
"""

import asyncio
import json
import sys
import os
import signal
import time
from collections import deque

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

# ═══════════════════════════════════════════════════════════════════════════════
# Connection state — per-client send queue with backpressure
# ═══════════════════════════════════════════════════════════════════════════════

MAX_CLIENTS = 50                  # Hard cap to prevent unbounded growth
CLIENT_QUEUE_MAX = 200            # Per-client backlog; slow clients drop oldest
BROADCASTER_DRAIN_INTERVAL = 0.005  # 200 Hz drain rate (faster than tick rate)

connected_clients: dict = {}      # websocket -> { "queue": deque, "last_seen": float }
latest_data: dict = {}            # Last received data (for late-joiner replay)
_broadcaster_task = None
_startup_time = time.time()


def _log(msg: str):
    """Structured logging with uptime."""
    uptime = time.time() - _startup_time
    print(f"[ws-server {uptime:7.1f}s] {msg}", flush=True)


# ═══════════════════════════════════════════════════════════════════════════════
# WebSocket endpoint
# ═══════════════════════════════════════════════════════════════════════════════

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Accept WebSocket connections and stream data to them."""
    global connected_clients

    # Reject if at capacity
    if len(connected_clients) >= MAX_CLIENTS:
        await websocket.accept()
        await websocket.send_text(json.dumps({
            "type": "error",
            "message": f"Server at capacity ({MAX_CLIENTS} clients)",
        }))
        await websocket.close(code=1013, reason="Try again later")
        return

    await websocket.accept()
    client_id = id(websocket)
    connected_clients[client_id] = {
        "ws": websocket,
        "queue": deque(maxlen=CLIENT_QUEUE_MAX),
        "last_seen": time.time(),
        "messages_sent": 0,
        "messages_dropped": 0,
    }
    _log(f"Client connected (id={client_id}, total={len(connected_clients)})")

    # Replay latest data to late joiner
    if latest_data:
        try:
            await websocket.send_text(json.dumps(latest_data))
        except Exception:
            pass

    try:
        while True:
            # Wait for commands from client (with timeout for heartbeat)
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=20.0)
                msg = json.loads(data)

                if msg.get("type") == "ping":
                    connected_clients[client_id]["last_seen"] = time.time()
                    await websocket.send_text(json.dumps({
                        "type": "pong",
                        "server_time": time.time(),
                    }))

            except asyncio.TimeoutError:
                # No message in 20s — send heartbeat ping
                try:
                    await websocket.send_text(json.dumps({
                        "type": "heartbeat",
                        "server_time": time.time(),
                    }))
                except Exception:
                    # Client is dead — break out
                    break

    except WebSocketDisconnect:
        _log(f"Client disconnected cleanly (id={client_id})")
    except Exception as e:
        _log(f"Client error (id={client_id}): {type(e).__name__}: {e}")
    finally:
        connected_clients.pop(client_id, None)


# ═══════════════════════════════════════════════════════════════════════════════
# HTTP broadcast endpoint — receives data from WSBridge, enqueues to all clients
# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/broadcast")
async def broadcast_data(data: dict):
    """HTTP endpoint: enqueue data to every connected client's send queue."""
    global latest_data
    latest_data = data

    enqueued = 0
    dropped = 0
    stale_clients = []

    for client_id, state in list(connected_clients.items()):
        # Skip if queue is full (will drop oldest due to deque maxlen)
        before = len(state["queue"])
        state["queue"].append(data)
        after = len(state["queue"])
        if after > before:
            enqueued += 1
        else:
            dropped += 1
            state["messages_dropped"] += 1

        # Mark stale client if last_seen is >5 min ago (will be cleaned by broadcaster)
        if time.time() - state["last_seen"] > 300:
            stale_clients.append(client_id)

    # Clean up stale clients
    for cid in stale_clients:
        ws = connected_clients.get(cid, {}).get("ws")
        if ws:
            try:
                await ws.close(code=1000, reason="stale")
            except Exception:
                pass
        connected_clients.pop(cid, None)
        _log(f"Cleaned stale client (id={cid})")

    return {
        "status": "ok",
        "clients": len(connected_clients),
        "enqueued": enqueued,
        "dropped": dropped,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Background broadcaster task — drains queues and sends to clients
# ═══════════════════════════════════════════════════════════════════════════════

async def _broadcaster_loop():
    """Continuously drain client queues and send messages."""
    _log("Broadcaster task started")
    while True:
        try:
            dead_clients = []
            for client_id, state in list(connected_clients.items()):
                ws = state["ws"]
                q = state["queue"]

                # Drain up to 10 messages per client per cycle (batching)
                messages_to_send = []
                while q and len(messages_to_send) < 10:
                    messages_to_send.append(q.popleft())

                if not messages_to_send:
                    continue

                # Concatenate as JSON array (1 send call instead of N — much faster)
                try:
                    payload = json.dumps(messages_to_send)
                    await ws.send_text(payload)
                    state["messages_sent"] += len(messages_to_send)
                except Exception as e:
                    _log(f"Send failed (id={client_id}): {type(e).__name__}: {e}")
                    dead_clients.append(client_id)

            # Remove dead clients
            for cid in dead_clients:
                ws = connected_clients.get(cid, {}).get("ws")
                if ws:
                    try:
                        await ws.close()
                    except Exception:
                        pass
                connected_clients.pop(cid, None)
                _log(f"Removed dead client (id={cid})")

        except Exception as e:
            _log(f"Broadcaster error: {type(e).__name__}: {e}")

        await asyncio.sleep(BROADCASTER_DRAIN_INTERVAL)


@app.on_event("startup")
async def _startup():
    """Start the background broadcaster task on app startup."""
    global _broadcaster_task
    _broadcaster_task = asyncio.create_task(_broadcaster_loop())
    _log(f"Server started — broadcaster draining at {1/BROADCASTER_DRAIN_INTERVAL:.0f} Hz")


@app.on_event("shutdown")
async def _shutdown():
    """Clean up on shutdown."""
    if _broadcaster_task:
        _broadcaster_task.cancel()
    _log("Server shutting down")


# ═══════════════════════════════════════════════════════════════════════════════
# Health & status
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/api/status")
async def status():
    """Health check endpoint."""
    total_sent = sum(s["messages_sent"] for s in connected_clients.values())
    total_dropped = sum(s["messages_dropped"] for s in connected_clients.values())
    return {
        "running": True,
        "uptime_sec": round(time.time() - _startup_time, 1),
        "clients": len(connected_clients),
        "max_clients": MAX_CLIENTS,
        "has_data": bool(latest_data),
        "messages_sent_total": total_sent,
        "messages_dropped_total": total_dropped,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════

def _handle_signal(signum, frame):
    """Graceful shutdown on signal."""
    _log(f"Received signal {signum} — shutting down")
    sys.exit(0)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)
    _log("Nafs AI WebSocket Server starting on port 8765...")
    uvicorn.run(app, host="0.0.0.0", port=8765, log_level="warning")
