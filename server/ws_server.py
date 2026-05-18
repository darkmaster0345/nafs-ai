"""
Nafs AI — WebSocket Server
Streams Adam's life tick-by-tick to the web GUI.
"""

import asyncio
import json
import copy
import random
import sys
import os

# Add parent dir to path so we can import nafs modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from world_sim import WorldSim, BIOMES, WEATHER_TYPES
from sensory_encoder import encode_sensory_input, INPUT_DIM
from baby_brain_model import BabyBrain
from thought_engine import ThoughtEngine
from curiosity import CuriosityModule
from dreaming import DreamEngine

import torch
import torch.optim as optim

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global simulation state
simulation_running = False
simulation_task = None
connected_clients = set()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    connected_clients.add(websocket)
    try:
        while True:
            # Wait for commands from the client
            data = await websocket.receive_text()
            msg = json.loads(data)

            if msg.get("type") == "start_simulation":
                global simulation_running, simulation_task
                if not simulation_running:
                    simulation_running = True
                    tick_delay = msg.get("tick_delay", 0.05)
                    simulation_task = asyncio.create_task(
                        run_simulation(websocket, tick_delay)
                    )
            elif msg.get("type") == "stop_simulation":
                simulation_running = False
            elif msg.get("type") == "get_world_map":
                # Send the world map data
                await send_world_map(websocket)
    except WebSocketDisconnect:
        connected_clients.discard(websocket)
        simulation_running = False
    except Exception as e:
        print(f"WebSocket error: {e}")
        connected_clients.discard(websocket)
        simulation_running = False


async def send_world_map(websocket):
    """Send the world map data to the client."""
    env = WorldSim()
    env.reset()

    # Extract a portion of the map for display
    map_data = []
    for y in range(min(env.world_map.height, 64)):
        row = []
        for x in range(min(env.world_map.width, 64)):
            biome = env.world_map.get_biome(x, y)
            row.append(biome)
        map_data.append(row)

    await websocket.send_text(json.dumps({
        "type": "world_map",
        "width": env.world_map.width,
        "height": env.world_map.height,
        "map": map_data,
    }))


async def run_simulation(websocket, tick_delay=0.05):
    """Run the full Nafs AI simulation, streaming tick data."""
    global simulation_running

    DEVICE = torch.device("cpu")
    HIDDEN_DIM = 256
    NUM_ACTIONS = len(WorldSim.ACTIONS)

    # Initialize
    model = BabyBrain(INPUT_DIM, HIDDEN_DIM, NUM_ACTIONS).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=3e-4, eps=1e-5)

    env = WorldSim()
    world_state, adam_stats_dict = env.reset()

    # Send world map first
    map_data = []
    for y in range(min(env.world_map.height, 64)):
        row = []
        for x in range(min(env.world_map.width, 64)):
            biome = env.world_map.get_biome(x, y)
            row.append(biome)
        map_data.append(row)

    await websocket.send_text(json.dumps({
        "type": "world_map",
        "width": env.world_map.width,
        "height": env.world_map.height,
        "map": map_data,
    }))

    # Send birth event
    biome_name = world_state.get('biome', 'plains')
    biome_data = BIOMES.get(biome_name, {})
    await websocket.send_text(json.dumps({
        "type": "birth",
        "biome": biome_name,
        "biome_emoji": biome_data.get('emoji', ''),
        "biome_desc": biome_data.get('desc', ''),
        "weather": world_state.get('weather', 'clear'),
        "position": [world_state.get('adam_x', 0), world_state.get('adam_y', 0)],
        "time_of_day": world_state.get('time_of_day', 12),
        "world_size": [env.world_map.width, env.world_map.height],
    }))

    # Initialize modules
    thought_engine = ThoughtEngine(memory_size=10)
    curiosity = CuriosityModule(
        curiosity_bonus=0.15,
        curiosity_decay=0.98,
        min_curiosity=0.01,
    )
    dream_engine = DreamEngine()

    # Encode initial observation
    init_phase5 = thought_engine.get_phase5_signals(world_state, adam_stats_dict)
    sensory_input = encode_sensory_input(
        world_state, adam_stats_dict,
        fear_signal=init_phase5['fear_signal'],
        pleasure_signal=init_phase5['pleasure_signal'],
        pattern_confidence=init_phase5['pattern_confidence'],
    ).to(DEVICE)
    hidden_state = model.init_hidden(1).to(DEVICE)

    # Tracking
    action_history = []
    all_action_counts = {a: 0 for a in WorldSim.ACTIONS}
    total_reward = 0.0
    tick = 0
    episode_intrinsic_reward = 0.0

    # PPO buffer
    obs_list, actions_list, old_log_probs = [], [], []
    rewards_list, values_list, masks_list = [], [], []

    GAMMA = 0.99
    GAE_LAMBDA = 0.95
    CLIP_EPSILON = 0.2
    VALUE_LOSS_COEF = 0.5
    MAX_GRAD_NORM = 0.5
    ENTROPY_COEF = 0.05
    DIVERSITY_PENALTY = 0.25
    DIVERSITY_WINDOW = 5
    PPO_UPDATE_INTERVAL = 64
    REFLECTION_INTERVAL = 20
    PATTERN_CONFIDENCE_THRESHOLD = 0.5

    latest_thought = "quiet. still."
    latest_emotion = "uncertain"
    latest_reflection = None
    latest_dream = None
    recent_actions = []

    alive = True
    while alive and simulation_running:
        try:
            tick += 1
            prev_stats = copy.deepcopy(adam_stats_dict)

            # Phase 5
            phase5 = thought_engine.get_phase5_signals(world_state, adam_stats_dict)

            # Get action
            with torch.no_grad():
                obs_list.append(sensory_input.clone())
                action_logits, state_value, hidden_state = model(
                    sensory_input.unsqueeze(0), hidden_state
                )
                action_logits_sq = action_logits.squeeze(0)
                state_value_sq = state_value.squeeze()
                action_dist = torch.distributions.Categorical(logits=action_logits_sq)
                action_idx = action_dist.sample()
                action = WorldSim.ACTIONS[action_idx.item()]

            # Step world
            next_world_state, next_adam_stats_dict, reward, done = env.step(action)

            # Curiosity
            intrinsic_reward = curiosity.compute_intrinsic_reward(
                world_state, adam_stats_dict
            )
            reward += intrinsic_reward
            episode_intrinsic_reward += intrinsic_reward

            # Diversity penalty
            action_history.append(action_idx.item())
            if len(action_history) > 10:
                action_history.pop(0)
            if len(action_history) >= DIVERSITY_WINDOW:
                if len(set(action_history[-DIVERSITY_WINDOW:])) == 1:
                    reward -= DIVERSITY_PENALTY

            # Phase 6 reflection
            suggested = phase5.get('suggested_action')
            confidence = phase5.get('pattern_confidence', 0)
            if suggested and confidence >= PATTERN_CONFIDENCE_THRESHOLD:
                if action == suggested:
                    reward += 0.05
                else:
                    reward -= 0.02

            # Track actions
            all_action_counts[action] = all_action_counts.get(action, 0) + 1
            actions_list.append(action_idx.item())
            old_log_probs.append(action_dist.log_prob(action_idx).item())
            rewards_list.append(reward)
            values_list.append(state_value_sq.item())
            masks_list.append(1.0 - done)

            # Phase 1: Inner experience
            experience = thought_engine.experience(
                world_state=next_world_state,
                adam_stats=next_adam_stats_dict,
                action=action,
                prev_stats=prev_stats,
                tick=tick,
                reward=reward,
            )
            latest_thought = experience.get('thought', 'quiet. still.')
            latest_emotion = experience.get('emotion', 'uncertain')

            # Reflection
            recent_actions.append(action)
            if tick % REFLECTION_INTERVAL == 0 and tick > 0:
                reflection = thought_engine.reflect(
                    world_state, adam_stats_dict,
                    recent_actions=recent_actions
                )
                if reflection.get('has_reflection'):
                    latest_reflection = reflection

            # Dreaming
            if action == "SLEEP" and tick > 1:
                dream = dream_engine.dream(
                    thought_engine.persistent_memory,
                    thought_engine.memory
                )
                if dream.get('dream_type') != 'empty':
                    latest_dream = dream

            # Next observation
            next_phase5 = thought_engine.get_phase5_signals(next_world_state, next_adam_stats_dict)
            sensory_input = encode_sensory_input(
                next_world_state, next_adam_stats_dict,
                fear_signal=next_phase5['fear_signal'],
                pleasure_signal=next_phase5['pleasure_signal'],
                pattern_confidence=next_phase5['pattern_confidence'],
            ).to(DEVICE)

            total_reward += reward

            # PPO Update
            ppo_data = None
            if tick % PPO_UPDATE_INTERVAL == 0 and len(obs_list) > 0:
                R = 0.0
                if not done:
                    with torch.no_grad():
                        _, last_val, _ = model(sensory_input.unsqueeze(0), hidden_state)
                        R = last_val.item()

                advantages = []
                gae = 0.0
                next_values = values_list[1:] + [R]
                for i in reversed(range(len(rewards_list))):
                    delta = rewards_list[i] + GAMMA * next_values[i] * masks_list[i] - values_list[i]
                    gae = delta + GAMMA * GAE_LAMBDA * masks_list[i] * gae
                    advantages.insert(0, gae)

                advantages_t = torch.FloatTensor(advantages)
                returns_t = advantages_t + torch.FloatTensor(values_list)
                if len(advantages_t) > 1:
                    advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

                old_log_probs_t = torch.FloatTensor(old_log_probs)
                actions_t = torch.LongTensor(actions_list)

                new_log_probs, new_entropies, new_values = [], [], []
                h = model.init_hidden(1).to(DEVICE)
                for i in range(len(obs_list)):
                    a_logits, s_value, h = model(obs_list[i].unsqueeze(0), h.detach())
                    a_logits_sq = a_logits.squeeze(0)
                    s_value_sq = s_value.squeeze()
                    dist = torch.distributions.Categorical(logits=a_logits_sq)
                    new_log_probs.append(dist.log_prob(actions_t[i]))
                    new_entropies.append(dist.entropy())
                    new_values.append(s_value_sq)

                new_log_probs_t = torch.stack(new_log_probs)
                new_entropies_t = torch.stack(new_entropies)
                new_values_t = torch.stack(new_values)

                ratio = torch.exp(new_log_probs_t - old_log_probs_t)
                surr1 = ratio * advantages_t
                surr2 = torch.clamp(ratio, 1 - CLIP_EPSILON, 1 + CLIP_EPSILON) * advantages_t
                policy_loss = -torch.min(surr1, surr2).mean()
                import torch.nn.functional as F
                value_loss = F.mse_loss(new_values_t, returns_t)
                entropy_loss = -ENTROPY_COEF * new_entropies_t.mean()
                loss = policy_loss + VALUE_LOSS_COEF * value_loss + entropy_loss

                optimizer.zero_grad()
                loss.backward()
                import torch.nn as nn
                grad_norm = nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                optimizer.step()

                ppo_data = {
                    "policy_loss": round(policy_loss.item(), 4),
                    "value_loss": round(value_loss.item(), 4),
                    "entropy": round(new_entropies_t.mean().item(), 4),
                    "grad_norm": round(grad_norm.item() if isinstance(grad_norm, torch.Tensor) else float(grad_norm), 4),
                }

                obs_list, actions_list, old_log_probs = [], [], []
                rewards_list, values_list, masks_list = [], [], []

            # Send tick data to client
            tick_data = {
                "type": "tick",
                "tick": tick,
                "action": action,
                "reward": round(reward, 3),
                "total_reward": round(total_reward, 2),
                "thought": latest_thought,
                "emotion": latest_emotion,
                "biome": next_world_state.get('biome', 'plains'),
                "weather": next_world_state.get('weather', 'clear'),
                "position": [next_world_state.get('adam_x', 0), next_world_state.get('adam_y', 0)],
                "facing": next_world_state.get('facing', 'north'),
                "time_of_day": next_world_state.get('time_of_day', 12),
                "temperature": round(next_world_state.get('temperature', 20), 1),
                "visibility": round(next_world_state.get('visibility', 1.0), 2),
                "vitals": {
                    "health": round(next_adam_stats_dict.get('health', 100), 1),
                    "hunger": round(next_adam_stats_dict.get('hunger', 0), 1),
                    "energy": round(next_adam_stats_dict.get('energy', 100), 1),
                    "stress": round(next_adam_stats_dict.get('stress', 0), 1),
                    "pain": round(next_adam_stats_dict.get('pain', 0), 1),
                },
                "world_events": {
                    "food_nearby": next_world_state.get('smell_food', 0),
                    "danger_nearby": next_world_state.get('smell_danger', 0),
                    "water_nearby": next_world_state.get('proximity_entity', 0),
                },
                "action_counts": all_action_counts,
                "intrinsic_reward": round(episode_intrinsic_reward, 2),
                "curiosity_states": curiosity.get_curiosity_stats()['total_states_discovered'],
                "dream_count": dream_engine.get_dream_stats()['total_dreams'],
                "vocabulary_size": len(thought_engine.get_vocabulary()),
                "discovered_words": thought_engine.get_discovered_vocabulary() if hasattr(thought_engine, 'get_discovered_vocabulary') and thought_engine.get_discovered_vocabulary() else [],
                "personality": thought_engine.get_personality(),
            }

            if ppo_data:
                tick_data["ppo"] = ppo_data

            if latest_reflection and latest_reflection.get('has_reflection'):
                tick_data["reflection"] = latest_reflection.get('reflection', '')
                tick_data["reflection_personality"] = latest_reflection.get('personality', '')
                latest_reflection = None

            if latest_dream:
                tick_data["dream"] = {
                    "type": latest_dream.get('dream_type', ''),
                    "thoughts": latest_dream.get('thoughts', []),
                }
                latest_dream = None

            if "new_words" in experience:
                tick_data["new_words"] = [
                    {"word": w, "meaning": m} for w, m in experience["new_words"]
                ]

            await websocket.send_text(json.dumps(tick_data))

            # Check death
            if done:
                alive = False
                # Send death event
                personality = thought_engine.get_personality()
                cs = curiosity.get_curiosity_stats()
                ds = dream_engine.get_dream_stats()

                await websocket.send_text(json.dumps({
                    "type": "death",
                    "tick": tick,
                    "total_reward": round(total_reward, 2),
                    "biome": next_world_state.get('biome', 'plains'),
                    "personality": personality,
                    "curiosity_stats": cs,
                    "dream_stats": ds,
                    "action_counts": all_action_counts,
                    "vocabulary_size": len(thought_engine.get_vocabulary()),
                }))
                simulation_running = False
                break

            world_state = next_world_state
            adam_stats_dict = next_adam_stats_dict

            # Delay between ticks
            if tick_delay > 0:
                await asyncio.sleep(tick_delay)

        except Exception as e:
            import traceback
            print(f"Error at tick {tick}: {e}")
            traceback.print_exc()
            continue


@app.get("/api/status")
async def status():
    return {"running": simulation_running, "clients": len(connected_clients)}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8765)
