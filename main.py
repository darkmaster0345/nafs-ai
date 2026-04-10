"""
Nafs AI — v0.1 CLI
"What emerges when code has no memory of the world?"

Run: python main.py
"""

import time
import sys
import os
from config import BRAIN_CONFIG, SIM_CONFIG
from world import World
from adam import Adam
from brain import ask_brain

# ── Terminal Colors ───────────────────────────────────────────────────────────
class C:
    RESET   = "\033[0m"
    GREY    = "\033[90m"
    WHITE   = "\033[97m"
    YELLOW  = "\033[93m"
    CYAN    = "\033[96m"
    RED     = "\033[91m"
    GREEN   = "\033[92m"
    MAGENTA = "\033[95m"
    BLUE    = "\033[94m"
    BOLD    = "\033[1m"
    DIM     = "\033[2m"

def clear():
    os.system("cls" if os.name == "nt" else "clear")

def print_header():
    print(f"{C.BOLD}{C.CYAN}")
    print("  ███╗   ██╗ █████╗ ███████╗███████╗     █████╗ ██╗")
    print("  ████╗  ██║██╔══██╗██╔════╝██╔════╝    ██╔══██╗██║")
    print("  ██╔██╗ ██║███████║█████╗  ███████╗    ███████║██║")
    print("  ██║╚██╗██║██╔══██║██╔══╝  ╚════██║    ██╔══██║██║")
    print("  ██║ ╚████║██║  ██║██║     ███████║    ██║  ██║██║")
    print("  ╚═╝  ╚═══╝╚═╝  ╚═╝╚═╝     ╚══════╝    ╚═╝  ╚═╝╚═╝")
    print(f"{C.RESET}")
    print(f"  {C.DIM}\"What emerges when code has no memory of the world?\"{C.RESET}")
    print(f"  {C.DIM}v0.1 — CLI Simulation{C.RESET}\n")

def print_tick(tick: int, world, adam):
    print(f"\n{C.BOLD}{'─' * 60}{C.RESET}")
    print(f"  {C.GREY}{world.status_line()}  |  Tick #{tick}{C.RESET}")
    print(f"  {C.DIM}{adam.status_line()}{C.RESET}")
    print(f"{'─' * 60}{C.RESET}")

def print_event(event: str):
    print(f"\n  {C.WHITE}🌍 World:{C.RESET}  {event}")

def print_response(response: dict, outcome: str):
    thought  = response.get("thought", "")
    dialogue = response.get("dialogue", "")
    action   = response.get("action", "IDLE")
    target   = response.get("target", "")
    emotion  = response.get("emotion", "")

    print(f"\n  {C.MAGENTA}💭 Thought:{C.RESET}  {C.DIM}{thought}{C.RESET}")

    if dialogue and dialogue.strip():
        print(f"  {C.YELLOW}💬 Says:{C.RESET}    \"{dialogue}\"")

    action_str = f"{action}"
    if target:
        action_str += f" → {target}"
    print(f"  {C.CYAN}⚡ Action:{C.RESET}  {action_str}")
    print(f"  {C.BLUE}🫀 Feels:{C.RESET}   {emotion}")

    if outcome:
        print(f"\n  {C.GREEN}↩ Outcome:{C.RESET} {C.DIM}{outcome}{C.RESET}")

def print_death(adam):
    print(f"\n{C.RED}{C.BOLD}")
    print("  ╔══════════════════════════════════╗")
    print("  ║          Adam has died.          ║")
    print(f"  ║   Age: {adam.age_ticks:>4} ticks              ║")
    print(f"  ║   Memories: {len(adam.long_term):>4}                ║")
    print("  ╚══════════════════════════════════╝")
    print(f"{C.RESET}")
    print(f"  {C.DIM}memory.json preserved. Run again to start a new life.{C.RESET}\n")

def check_api_key():
    if not BRAIN_CONFIG["api_key"]:
        print(f"\n{C.RED}[Error] GROQ_API_KEY not found.{C.RESET}")
        print(f"  Create a .env file with: GROQ_API_KEY=your_key_here")
        print(f"  Get a free key at: https://console.groq.com\n")
        sys.exit(1)


# ── Main Simulation Loop ──────────────────────────────────────────────────────

def run():
    check_api_key()
    clear()
    print_header()

    world = World()
    adam  = Adam()

    print(f"  {C.GREEN}Simulation starting...{C.RESET}")
    print(f"  {C.DIM}Adam wakes up.{C.RESET}\n")
    time.sleep(2)

    tick = 0

    try:
        while adam.is_alive:
            tick += 1

            # ── World tick ────────────────────────────────────────────────
            world.tick_forward()
            adam.apply_time_passage(
                hunger_rate  = SIM_CONFIG["hunger_rate"],
                energy_drain = SIM_CONFIG["energy_drain"],
            )

            # ── Generate world event ──────────────────────────────────────
            event = world.get_event()

            # ── Print tick header ─────────────────────────────────────────
            print_tick(tick, world, adam)
            print_event(event)

            # ── Ask Adam's brain ──────────────────────────────────────────
            print(f"\n  {C.DIM}[thinking...]{C.RESET}", end="\r")

            outcome_text = ""
            if adam.last_action:
                # Get outcome of last action first
                outcome = world.apply_action(
                    adam.last_action,
                    "",
                    adam.__dict__
                )
                adam.apply_outcome(outcome)
                outcome_text = outcome.get("outcome_text", "")

            response = ask_brain(adam, event, outcome_text)

            # ── Validate response ─────────────────────────────────────────
            if not adam.response_is_clean(response):
                print(f"  {C.RED}[World Knowledge Detected — Regenerating]{C.RESET}")
                response = ask_brain(adam, event, outcome_text)

            # ── Apply response ────────────────────────────────────────────
            adam.last_action   = response.get("action", "IDLE")
            adam.last_thought  = response.get("thought", "")
            adam.current_emotion = response.get("emotion", "uncertain")

            # ── Display ───────────────────────────────────────────────────
            print_response(response, outcome_text)

            # ── Store memory ──────────────────────────────────────────────
            adam.remember(
                tick     = tick,
                event    = event,
                thought  = response.get("thought", ""),
                action   = response.get("action", "IDLE"),
                emotion  = response.get("emotion", "uncertain"),
                outcome  = outcome_text,
            )

            # ── Check death ───────────────────────────────────────────────
            if not adam.is_alive:
                print_death(adam)
                break

            # ── Wait for next tick ────────────────────────────────────────
            time.sleep(BRAIN_CONFIG["poll_interval"])

    except KeyboardInterrupt:
        print(f"\n\n  {C.YELLOW}[Observer] Simulation paused.{C.RESET}")
        print(f"  {C.DIM}Adam's memory has been saved to memory.json{C.RESET}")
        print(f"  {C.DIM}Run again to continue.{C.RESET}\n")


if __name__ == "__main__":
    run()
