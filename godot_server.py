"""
Nafs AI — Godot HTTP State Server (Phase 11)
=============================================

Serves simulation state at http://localhost:5000/state for the Godot client
(NetworkController.gd) to poll. The state is built by GodotBridge inside
EngineOrchestrator and written to godot_state.json on every 10th tick.

This server is intentionally tiny: it just serves the file. The heavy
lifting (state assembly) is done by the orchestrator running in the
training loop.

Endpoints:
    GET /state          — Full simulation state (latest tick)
    GET /state?tick=N   — (Future) scrub to historical tick
    GET /health         — Liveness probe
    GET /               — Server info

Usage:
    # Terminal 1: Run training (writes godot_state.json every 10 ticks)
    python train_multi_agent.py --max-ticks 50000 --tick-delay 0

    # Terminal 2: Run this HTTP server
    python godot_server.py

    # Terminal 3 (user's local machine): Run Godot client
    godot --path godot/
    # Godot polls http://localhost:5000/state every 0.5s automatically

If running the Godot client on a remote machine, set up an SSH tunnel:
    ssh -L 5000:localhost:5000 user@remote-host
    # Then point Godot at http://localhost:5000/state on your local machine
"""

import os
import json
import time
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn

# Default paths
STATE_FILE = os.environ.get("NAFS_STATE_FILE", "godot_state.json")
PORT = int(os.environ.get("NAFS_GODOT_PORT", "5000"))
HOST = os.environ.get("NAFS_GODOT_HOST", "0.0.0.0")


class GodotStateHandler(BaseHTTPRequestHandler):
    """HTTP handler that serves the Godot state file + simple endpoints."""

    def _send_json(self, code: int, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path: str, content_type: str = "application/json"):
        if not os.path.exists(path):
            self._send_json(404, {"error": "not_found", "path": path})
            return
        try:
            with open(path, "rb") as f:
                body = f.read()
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
        except Exception as e:
            self._send_json(500, {"error": str(e)})

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.end_headers()

    def do_GET(self):
        path = self.path.split("?", 1)[0]  # strip query string
        if path == "/" or path == "":
            self._send_json(200, {
                "server": "Nafs AI Godot State Server",
                "version": "1.0",
                "endpoints": ["/state", "/health", "/summary"],
                "state_file": STATE_FILE,
                "state_file_exists": os.path.exists(STATE_FILE),
                "state_file_mtime": (
                    os.path.getmtime(STATE_FILE)
                    if os.path.exists(STATE_FILE) else None
                ),
                "current_time": time.time(),
            })
        elif path == "/state":
            self._send_file(STATE_FILE)
        elif path == "/health":
            alive = os.path.exists(STATE_FILE)
            age_s = (time.time() - os.path.getmtime(STATE_FILE)) if alive else None
            self._send_json(200, {
                "ok": True,
                "state_file_exists": alive,
                "state_file_age_seconds": age_s,
                "server_pid": os.getpid(),
            })
        elif path == "/summary":
            # Read orchestrator summary if present
            self._send_file("orchestrator_summary.json")
        elif path == "/oee":
            self._send_file("oee_status.json")
        elif path == "/events":
            self._send_file("events.jsonl", content_type="application/x-ndjson")
        else:
            self._send_json(404, {"error": "not_found", "path": path})

    def log_message(self, format, *args):
        # Suppress noisy request logs by default
        # (uncomment to debug)
        # super().log_message(format, *args)
        pass


class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    """Threaded HTTP server so multiple clients can poll simultaneously."""
    daemon_threads = True
    allow_reuse_address = True


# Module-level singleton so we can start/stop the server from train_multi_agent.py
_running_server: ThreadingHTTPServer = None
_server_thread: threading.Thread = None


def start_server_in_thread(port: int = PORT, host: str = HOST,
                            state_file: str = None) -> bool:
    """Start the HTTP server in a background thread.

    Returns True if the server started (or was already running).
    Safe to call from train_multi_agent.py.
    """
    global _running_server, _server_thread, STATE_FILE
    if _running_server is not None:
        return True  # already running
    if state_file:
        STATE_FILE = state_file
    try:
        _running_server = ThreadingHTTPServer((host, port), GodotStateHandler)
        _server_thread = threading.Thread(
            target=_running_server.serve_forever, daemon=True,
            name="NafsGodotHTTPServer",
        )
        _server_thread.start()
        print(f"  \U0001f310 Godot HTTP server: http://{host}:{port}/state "
              f"(serving {STATE_FILE})", flush=True)
        return True
    except Exception as e:
        print(f"  \u26a0\ufe0f Could not start Godot HTTP server: {e}", flush=True)
        _running_server = None
        return False


def stop_server_in_thread():
    """Stop the background HTTP server."""
    global _running_server, _server_thread
    if _running_server is not None:
        try:
            _running_server.shutdown()
            _running_server.server_close()
        except Exception:
            pass
        _running_server = None
        _server_thread = None


def main():
    # Allow running from any directory — cd to project root if state file is there
    global STATE_FILE
    project_root = os.path.dirname(os.path.abspath(__file__))
    if not os.path.isabs(STATE_FILE) and not os.path.exists(STATE_FILE):
        candidate = os.path.join(project_root, STATE_FILE)
        if os.path.exists(candidate):
            STATE_FILE = candidate

    server = ThreadingHTTPServer((HOST, PORT), GodotStateHandler)
    print(f"\n{'=' * 60}")
    print(f"  Nafs AI — Godot State Server (Phase 11)")
    print(f"{'-' * 60}")
    print(f"  Listening:  http://{HOST}:{PORT}")
    print(f"  State file: {STATE_FILE}")
    print(f"  Endpoints:")
    print(f"    GET /state    — Full simulation state (polled by Godot)")
    print(f"    GET /health   — Liveness probe")
    print(f"    GET /summary  — Orchestrator summary")
    print(f"    GET /oee      — OEE criteria status")
    print(f"    GET /events   — Event log (JSONL)")
    print(f"{'-' * 60}")
    print(f"  Godot client should poll: http://localhost:{PORT}/state")
    print(f"  (set this in godot/NetworkController.gd server_url)")
    print(f"{'=' * 60}\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  Server shutting down...")
        server.shutdown()


if __name__ == "__main__":
    main()
