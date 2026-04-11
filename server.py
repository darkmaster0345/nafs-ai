import threading
import time
import json
import logging
import traceback
from flask import Flask, request, jsonify
from flask_cors import CORS
from world import World
from adam import Adam
from brain import ask_brain
from config import BRAIN_CONFIG, SIM_CONFIG

app = Flask(__name__)
CORS(app)

# ── Global State ─────────────────────────────────────────────────────────────
world = World()
adam = Adam()
current_command = {"action": "IDLE", "target": "", "thought": "Waking up...", "emotion": "neutral"}
last_godot_state = {}
lock = threading.Lock()

def brain_loop():
    """Background thread to handle the slow LLM processing."""
    global current_command
    while True:
        try:
            # Prepare context for the brain
            with lock:
                event = world.get_event()
                outcome_text = ""

                # If we have a previous action, apply it to get outcome
                if adam.last_action:
                    outcome = world.apply_action(
                        adam.last_action,
                        adam.last_target if hasattr(adam, 'last_target') else "",
                        adam.__dict__
                    )
                    adam.apply_outcome(outcome)
                    outcome_text = outcome.get("outcome_text", "")

            # Call LLM (Slow)
            response = ask_brain(adam, event, outcome_text)

            # Validate and update state
            with lock:
                if adam.response_is_clean(response):
                    adam.last_action = response.get("action", "IDLE")
                    adam.last_target = response.get("target", "")
                    adam.current_emotion = response.get("emotion", "uncertain")

                    # Update command for Godot
                    current_command = {
                        "action": response.get("action", "IDLE"),
                        "target": response.get("target", ""),
                        "thought": response.get("thought", ""),
                        "emotion": response.get("emotion", "uncertain")
                    }

                    # Remember
                    adam.remember(
                        tick=adam.age_ticks, # Simplified tick
                        event=event,
                        thought=response.get("thought", ""),
                        action=response.get("action", "IDLE"),
                        emotion=response.get("emotion", "uncertain"),
                        outcome=outcome_text
                    )

                # Advance simulation internal state
                world.tick_forward()
                adam.apply_time_passage(
                    hunger_rate=SIM_CONFIG["hunger_rate"],
                    energy_drain=SIM_CONFIG["energy_drain"]
                )

        except Exception:
            logging.exception("[Server] Brain Loop Error — full traceback:")
            time.sleep(2)  # brief pause before retry

        # Poll interval from config
        time.sleep(BRAIN_CONFIG["poll_interval"])

@app.route('/update', methods=['POST'])
def update():
    """
    Endpoint for Godot to send state and receive commands.
    Expected JSON: {
        "position": {"x": 0, "y": 0},
        "events": [{"type": "collision", "name": "berry_bush"}]
    }
    """
    global last_godot_state
    data = request.json

    if not isinstance(data, dict):
        return jsonify({"error": "Invalid JSON payload"}), 400

    if "events" in data:
        if not isinstance(data["events"], list):
            return jsonify({"error": "events must be a list"}), 400

    with lock:
        # Save Godot state for context (though simulation is the source of truth)
        last_godot_state = data

        # Process events from Godot (e.g. Adam touched something)
        if "events" in data:
            for event in data["events"]:
                if event["type"] == "collision":
                    # Feed this into the world event pool if necessary
                    # For now, just log or trigger a "touch" interaction
                    print(f"[Server] Godot Event: {event['name']} collision")

        # Return the latest command from the brain
        return jsonify({
            "status": "ok",
            "command": current_command,
            "stats": {
                "health": adam.health,
                "hunger": adam.hunger,
                "energy": adam.energy,
                "stress": adam.stress
            }
        })

if __name__ == '__main__':
    # Start brain thread
    bt = threading.Thread(target=brain_loop, daemon=True)
    bt.start()

    # Start Flask server
    app.run(host='0.0.0.0', port=5000)
