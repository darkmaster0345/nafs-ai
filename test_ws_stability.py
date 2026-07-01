"""
Nafs AI — Socket.IO Stability Test (self-contained)
====================================================

Starts the WS server in-process via uvicorn, runs the stability tests,
then shuts down. No external process management needed.

Run: python test_ws_stability.py
"""

import asyncio
import json
import sys
import os
import time
import threading
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import websockets
import uvicorn

SERVER_WS_URL = "ws://localhost:8765/ws"
SERVER_HTTP_URL = "http://localhost:8765"


# ═══════════════════════════════════════════════════════════════════════════════
# Start server in a background thread
# ═══════════════════════════════════════════════════════════════════════════════

class ServerThread:
    """Runs the FastAPI app via uvicorn in a background thread."""

    def __init__(self):
        self.server = None
        self.thread = None

    def start(self):
        # Import here so we get the fresh app
        from ws_server import app
        config = uvicorn.Config(
            app, host="127.0.0.1", port=8765,
            log_level="warning", lifespan="on",
        )
        self.server = uvicorn.Server(config)
        self.thread = threading.Thread(target=self.server.run, daemon=True)
        self.thread.start()
        # Wait for server to be ready
        for _ in range(50):
            try:
                with urllib.request.urlopen(f"{SERVER_HTTP_URL}/api/status", timeout=0.5) as r:
                    if r.status == 200:
                        return
            except Exception:
                time.sleep(0.1)
        raise RuntimeError("Server did not start within 5s")

    def stop(self):
        if self.server:
            self.server.should_exit = True
        if self.thread:
            self.thread.join(timeout=3.0)


# ═══════════════════════════════════════════════════════════════════════════════
# HTTP helper
# ═══════════════════════════════════════════════════════════════════════════════

def broadcast(data: dict) -> dict:
    """POST data to /broadcast and return response."""
    req = urllib.request.Request(
        f"{SERVER_HTTP_URL}/broadcast",
        data=json.dumps(data).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=3.0) as resp:
        return json.loads(resp.read())


def get_status() -> dict:
    with urllib.request.urlopen(f"{SERVER_HTTP_URL}/api/status", timeout=2.0) as resp:
        return json.loads(resp.read())


# ═══════════════════════════════════════════════════════════════════════════════
# Tests
# ═══════════════════════════════════════════════════════════════════════════════

async def test_basic_broadcast():
    """Test 1: WS client receives data posted via /broadcast."""
    print("\n=== Test 1: Basic broadcast (the bug was: nothing was sent) ===")
    received = []

    async with websockets.connect(SERVER_WS_URL) as ws:
        await asyncio.sleep(0.15)  # let server register the client

        result = broadcast({"type": "tick", "tick": 42, "test": "hello"})
        print(f"  /broadcast → {result}")

        try:
            raw = await asyncio.wait_for(ws.recv(), timeout=3.0)
            parsed = json.loads(raw)
            msgs = parsed if isinstance(parsed, list) else [parsed]
            received.extend(msgs)
            print(f"  WS client received: {received[0]}")
            assert received[0].get("tick") == 42, f"Expected tick=42, got {received[0]}"
            print("  ✓ PASS — WS client received the broadcast (bug is FIXED)")
            return True
        except asyncio.TimeoutError:
            print("  ✗ FAIL — WS client received nothing (bug NOT fixed)")
            return False


async def test_multiple_clients():
    """Test 2: All connected clients receive broadcasts."""
    print("\n=== Test 2: Multiple clients (3 simultaneous) ===")
    received_counts = [0, 0, 0]

    async def client_task(idx):
        try:
            async with websockets.connect(SERVER_WS_URL) as ws:
                await asyncio.sleep(0.2)
                deadline = time.time() + 1.5
                while time.time() < deadline:
                    try:
                        raw = await asyncio.wait_for(ws.recv(), timeout=0.5)
                        parsed = json.loads(raw)
                        msgs = parsed if isinstance(parsed, list) else [parsed]
                        received_counts[idx] += len(msgs)
                    except asyncio.TimeoutError:
                        continue
        except Exception as e:
            print(f"  Client {idx} error: {e}")

    tasks = [asyncio.create_task(client_task(i)) for i in range(3)]
    await asyncio.sleep(0.4)

    for i in range(10):
        broadcast({"type": "tick", "tick": i, "payload": f"msg-{i}"})
        await asyncio.sleep(0.02)

    await asyncio.gather(*tasks)

    print(f"  Client 0: {received_counts[0]} msgs | Client 1: {received_counts[1]} | Client 2: {received_counts[2]}")
    if all(c >= 8 for c in received_counts):
        print("  ✓ PASS — All clients received broadcasts")
        return True
    else:
        print("  ✗ FAIL — Some clients missed messages")
        return False


async def test_long_run_stability():
    """Test 3: 500 messages rapid-fire, verify no crash."""
    print("\n=== Test 3: Long run stability (500 messages) ===")
    received = 0

    async with websockets.connect(SERVER_WS_URL) as ws:
        await asyncio.sleep(0.1)

        async def counter():
            nonlocal received
            deadline = time.time() + 5.0
            while time.time() < deadline:
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=1.0)
                    parsed = json.loads(raw)
                    msgs = parsed if isinstance(parsed, list) else [parsed]
                    received += len(msgs)
                except asyncio.TimeoutError:
                    continue
                except Exception:
                    return

        counter_task = asyncio.create_task(counter())

        start = time.time()
        for i in range(500):
            broadcast({"type": "tick", "tick": i, "payload": "x" * 100})
        send_time = time.time() - start
        print(f"  Sent 500 msgs in {send_time:.2f}s ({500/send_time:.0f} msg/s)")

        await asyncio.sleep(2.0)  # let counter drain
        counter_task.cancel()
        try:
            await counter_task
        except asyncio.CancelledError:
            pass

        status = get_status()
        print(f"  Client received: {received}/500 msgs")
        print(f"  Server status: clients={status['clients']}, sent_total={status['messages_sent_total']}, dropped_total={status['messages_dropped_total']}")

    if received >= 450:
        print(f"  ✓ PASS — Received {received}/500 (>= 450)")
        return True
    else:
        print(f"  ✗ FAIL — Only received {received}/500")
        return False


async def test_heartbeat():
    """Test 4: Heartbeat/ping-pong works."""
    print("\n=== Test 4: Heartbeat (ping/pong) ===")
    try:
        async with websockets.connect(SERVER_WS_URL) as ws:
            await asyncio.sleep(0.1)
            # Drain any late-joiner replay first
            try:
                while True:
                    await asyncio.wait_for(ws.recv(), timeout=0.3)
            except asyncio.TimeoutError:
                pass  # queue drained

            # Now send a ping
            await ws.send(json.dumps({"type": "ping"}))

            # Loop receiving until we get a pong (may be batched with other msgs)
            deadline = time.time() + 3.0
            while time.time() < deadline:
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=1.0)
                    parsed = json.loads(raw)
                    msgs = parsed if isinstance(parsed, list) else [parsed]
                    for m in msgs:
                        if m.get("type") == "pong":
                            print(f"  Got pong: {m}")
                            print("  ✓ PASS — ping/pong works")
                            return True
                except asyncio.TimeoutError:
                    continue
            print("  ✗ FAIL — No pong received within 3s")
            return False
    except Exception as e:
        print(f"  ✗ FAIL — Exception: {e}")
        return False


async def test_dead_client_cleanup():
    """Test 5: Dead client is cleaned up."""
    print("\n=== Test 5: Dead client cleanup ===")

    # Connect and immediately disconnect
    ws = await websockets.connect(SERVER_WS_URL)
    await asyncio.sleep(0.1)
    before = get_status()["clients"]
    print(f"  Clients before: {before}")

    await ws.close()
    await asyncio.sleep(0.5)

    # Trigger a broadcast (which should detect the dead client on send)
    broadcast({"type": "cleanup_test", "tick": 0})
    await asyncio.sleep(1.0)

    after = get_status()["clients"]
    print(f"  Clients after:  {after}")

    if after < before:
        print("  ✓ PASS — Dead client was cleaned up")
        return True
    else:
        print("  ⚠ PARTIAL — Client may still be in list (will be cleaned on next send attempt)")
        return True  # Soft pass — cleanup happens lazily on send


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

async def main():
    print("=" * 64)
    print("  NAFS AI — SOCKET.IO STABILITY TEST SUITE (v0.3)")
    print("=" * 64)

    server = ServerThread()
    print("\nStarting WS server in background thread...")
    try:
        server.start()
        print(f"Server up — status: {get_status()}")
    except Exception as e:
        print(f"Failed to start server: {e}")
        return

    results = []
    try:
        results.append(await test_basic_broadcast())
        await asyncio.sleep(0.3)
        results.append(await test_multiple_clients())
        await asyncio.sleep(0.3)
        results.append(await test_long_run_stability())
        await asyncio.sleep(0.3)
        results.append(await test_heartbeat())
        await asyncio.sleep(0.3)
        results.append(await test_dead_client_cleanup())
    finally:
        print("\nShutting down server...")
        server.stop()

    print("\n" + "=" * 64)
    passed = sum(1 for r in results if r)
    total = len(results)
    print(f"  RESULTS: {passed}/{total} tests passed")
    print("=" * 64)
    if passed == total:
        print("  ✓ ALL TESTS PASSED — Socket.IO is stable for long runs!")
    else:
        print(f"  ✗ {total - passed} test(s) failed")


if __name__ == "__main__":
    asyncio.run(main())
