"""
Nafs AI — WebSocket Bridge
Connects the simulation to the WebSocket server for dashboard streaming.

Usage in train.py:
    from ws_bridge import WSBridge

    bridge = WSBridge()
    bridge.start()

    # During simulation:
    bridge.send_tick_data({...})
    bridge.send_death_data({...})
    bridge.send_birth_data({...})

    # When done:
    bridge.stop()

The bridge runs in a separate thread and handles connection failures gracefully.
The simulation works perfectly fine without the dashboard running.
"""

import json
import threading
import time
import queue
import urllib.request
import urllib.error
from typing import Optional


class WSBridge:
    """
    Bridges the Nafs AI simulation to the WebSocket server.

    Runs in a separate thread so it doesn't block the simulation.
    Handles connection failures gracefully — simulation works without dashboard.
    """

    def __init__(self, server_url: str = "http://localhost:8765"):
        self.server_url = server_url
        self.message_queue = queue.Queue(maxsize=1000)
        self.running = False
        self._thread: Optional[threading.Thread] = None
        self._connected = False

    def start(self):
        """Start the bridge in a background thread."""
        if self.running:
            return

        self.running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        print("[WSBridge] Started — will connect when server is available")

    def stop(self):
        """Stop the bridge."""
        self.running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        print("[WSBridge] Stopped")

    def send_tick_data(self, data: dict):
        """
        Send tick data to the dashboard.

        Args:
            data: Dictionary with tick data including:
                - type: "tick"
                - tick: tick number
                - alive: bool
                - adam_stats: dict with health, hunger, energy, stress, pain
                - world_state: dict with biome, weather, temperature, etc.
                - action: str
                - reward: float
                - total_reward: float
                - thought: str
                - emotion: str
                - action_counts: dict
                - biome_map: list of lists (optional)
                - ... other fields
        """
        data["type"] = "tick"
        self._enqueue(data)

    def send_death_data(self, data: dict):
        """
        Send death event to the dashboard.

        Args:
            data: Dictionary with death data including:
                - type: "death"
                - tick: final tick number
                - total_reward: float
                - action_counts: dict
                - ... other final stats
        """
        data["type"] = "death"
        self._enqueue(data)

    def send_birth_data(self, data: dict):
        """
        Send birth event to the dashboard.

        Args:
            data: Dictionary with birth data including:
                - type: "birth"
                - biome: str
                - weather: str
                - position: [x, y]
                - time_of_day: int
                - ... other birth info
        """
        data["type"] = "birth"
        self._enqueue(data)

    def _enqueue(self, data: dict):
        """Add message to the queue, dropping old messages if full."""
        try:
            self.message_queue.put_nowait(data)
        except queue.Full:
            # Drop the oldest message and try again
            try:
                self.message_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self.message_queue.put_nowait(data)
            except queue.Full:
                pass

    def _run(self):
        """Main loop running in background thread."""
        while self.running:
            try:
                # Try to send queued messages via HTTP
                while not self.message_queue.empty():
                    try:
                        data = self.message_queue.get_nowait()
                        self._send_http(data)
                    except queue.Empty:
                        break

                time.sleep(0.01)  # Small sleep to prevent CPU spinning

            except Exception as e:
                if self._connected:
                    print(f"[WSBridge] Error: {e}")
                self._connected = False
                time.sleep(1.0)  # Wait before retrying

    def _send_http(self, data: dict):
        """Send data via HTTP POST to the WebSocket server."""
        try:
            payload = json.dumps(data).encode("utf-8")
            req = urllib.request.Request(
                f"{self.server_url}/broadcast",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=2.0) as resp:
                if resp.status == 200:
                    self._connected = True
        except (urllib.error.URLError, ConnectionRefusedError, TimeoutError):
            self._connected = False
        except Exception:
            self._connected = False

    @property
    def is_connected(self) -> bool:
        """Check if the bridge is connected to the server."""
        return self._connected
