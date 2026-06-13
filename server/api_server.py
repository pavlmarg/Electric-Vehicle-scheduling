import sys
import os
import numpy as np
import random
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from stable_baselines3 import PPO

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from environments.citygrid import CityMap
from environments.traffic_generator import TrafficGenerator
from baselines.benchmark import GreedyHeuristicBaseline
from environments.ev_gym_env import EVFleetEnv

app = FastAPI(title="EV Fleet Simulator API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

try:
    ai_model = PPO.load("ppo_model")
    print("✓ AI model loaded successfully.")
except Exception as e:
    print(f"⚠ AI model not found: {e}")
    ai_model = None


class SimulationRequest(BaseModel):
    algorithm: str  # "baseline" or "ai"
    profile: int    # 0-9
    seed: int


STATE_CODE = {
    'IDLE': 0,
    'WITH_CUSTOMER': 1,
    'REBALANCING': 2,
    'WAITING_FOR_CHARGER': 3,
    'CHARGING': 4,
    'STRANDED': 5,
}

LEASING_COST_EUR = 40.0
FRAME_INTERVAL = 15   # capture one frame every N minutes
NUM_VEHICLES = 750


def _build_frame(fleet, stations):
    """Snapshot of all taxis + per-station queue lengths."""
    taxis = [
        {
            "id": ev.id,
            "x": round(ev.location[0], 2),
            "y": round(ev.location[1], 2),
            "s": STATE_CODE.get(ev.state, 0),
            "soc": round(ev.current_soc, 3),
        }
        for ev in fleet
    ]
    queues = [st["queue_length"] for st in stations]
    return {"taxis": taxis, "queues": queues}


def _compute_stats(fleet, customers_served, abandoned, total_wait_time):
    total_requested = customers_served + abandoned
    service_rate = (customers_served / total_requested * 100) if total_requested > 0 else 0.0
    avg_wait = (total_wait_time / customers_served) if customers_served > 0 else 0.0
    net_profit = sum(e.daily_revenue - e.daily_charging_cost - LEASING_COST_EUR for e in fleet)
    stranded = sum(1 for e in fleet if e.state == "STRANDED")
    return {
        "net_profit": round(net_profit, 2),
        "service_rate": round(service_rate, 2),
        "avg_wait_time": round(avg_wait, 2),
        "customers_served": customers_served,
        "customers_abandoned": abandoned,
        "stranded_taxis": stranded,
    }


@app.get("/health")
def health():
    return {"status": "ok", "ai_model_loaded": ai_model is not None}


@app.post("/simulate")
def run_simulation(req: SimulationRequest):
    if req.algorithm not in ("baseline", "ai"):
        raise HTTPException(status_code=400, detail="algorithm must be 'baseline' or 'ai'")
    if not 0 <= req.profile <= 9:
        raise HTTPException(status_code=400, detail="profile must be 0-9")
    if req.algorithm == "ai" and ai_model is None:
        raise HTTPException(status_code=503, detail="AI model not loaded on server")

    print(f"--> Request: algo={req.algorithm}, profile={req.profile}, seed={req.seed}")

    np.random.seed(req.seed)
    random.seed(req.seed)

    frames = []
    queues_over_time = []   
    avg_soc_over_time = []  

    # =========================================================
    # BASELINE
    # =========================================================
    if req.algorithm == "baseline":
        city = CityMap(width_km=20.0, height_km=20.0, num_stations=16, num_hubs=4)
        generator = TrafficGenerator(city, num_vehicles=NUM_VEHICLES, seed=req.seed)
        generator.client_manager.current_profile = generator.client_manager.all_profiles[req.profile]
        fleet = generator.generate_initial_fleet()
        solver = GreedyHeuristicBaseline(city)

        total_wait_time = 0.0
        customers_served = 0
        abandoned_total = 0

        for minute in range(1440):
            generator.generate_new_demands(minute)
            wait_times, abandoned = generator.process_waitlist(minute, fleet)
            total_wait_time += sum(wait_times)
            customers_served += len(wait_times)
            abandoned_total += abandoned

            for ev in fleet:
                ev.update_time(minute)

                if ev.state == "IDLE":
                    action, target_pos, dist, duration = solver.route_ev(ev)
                    if action is not None:
                        if action == "REBALANCE":
                            ev.state = "REBALANCING"
                            ev.target_pos = target_pos
                            ev.arrival_time = minute + duration
                        else:
                            ev.dispatch_to_station(target_pos, action, dist, duration, minute)
                            city.add_to_queue(action)

                elif ev.state == "REBALANCING":
                    if minute >= getattr(ev, "arrival_time", minute):
                        ev.location = ev.target_pos
                        ev.state = "IDLE"

                elif ev.state == "WAITING_FOR_CHARGER":
                    ev.total_waiting_time += 1
                    charger = city.occupy_charger(ev.target_station_idx)
                    if charger:
                        city.remove_from_queue(ev.target_station_idx)
                        ev.state = "CHARGING"
                        ev.charger_type = charger

                elif ev.state == "CHARGING":
                    power = city.charger_specs[ev.charger_type]["power"]
                    price = city.get_electricity_price(minute, ev.charger_type)
                    station_idx = ev.target_station_idx
                    ev.charge(power_kw=power, price_per_kwh=price)
                    if ev.state == "IDLE":
                        city.release_charger(station_idx, ev.charger_type)

            # Time-series data (every minute)
            queues_over_time.append(sum(st["queue_length"] for st in city.stations))
            avg_soc_over_time.append(sum(ev.current_soc for ev in fleet) / len(fleet))

            if minute % FRAME_INTERVAL == 0:
                frames.append(_build_frame(fleet, city.stations))

        stations_data = [
            {"id": i, "x": st["location"][0], "y": st["location"][1], "type": st["type"]}
            for i, st in enumerate(city.stations)
        ]
        stats = _compute_stats(fleet, customers_served, abandoned_total, total_wait_time)

    # =========================================================
    # AI (PPO)
    # =========================================================
    else:
        env = EVFleetEnv(num_vehicles=NUM_VEHICLES)
        obs, _ = env.reset(seed=req.seed)
        env.generator.client_manager.current_profile = (
            env.generator.client_manager.all_profiles[req.profile]
        )

        terminated = False
        last_captured_minute = -1

        while not terminated:
            action, _ = ai_model.predict(obs, deterministic=True)

            # Safety shield
            if env.taxis_needing_action:
                taxi = env.taxis_needing_action[0]
                if taxi.current_soc < 0.25 and action >= 16:
                    best_idx, best_score = 0, float("inf")
                    for i, st in enumerate(env.city.stations):
                        dist = env.city.calculate_manhattan_dist(taxi.location, st["location"])
                        score = dist + st["queue_length"] * 2.0
                        if score < best_score:
                            best_score = score
                            best_idx = i
                    action = best_idx

            obs, reward, terminated, truncated, info = env.step(action)

            # Capture frame when a new FRAME_INTERVAL boundary is crossed
            cur = env.current_minute
            boundary = (cur // FRAME_INTERVAL) * FRAME_INTERVAL
            if boundary != last_captured_minute and cur > 0:
                frames.append(_build_frame(env.fleet, env.city.stations))
                last_captured_minute = boundary

        # Align time-series lengths to 1440 if the env stopped early
        queues_over_time = env.queues_over_time[:1440]
        avg_soc_over_time = env.avg_soc_over_time[:1440]

        stations_data = [
            {"id": i, "x": st["location"][0], "y": st["location"][1], "type": st["type"]}
            for i, st in enumerate(env.city.stations)
        ]
        stats = _compute_stats(
            env.fleet,
            env.total_customers_served,
            env.total_abandoned,
            env.total_wait_time,
        )

    return {
        "status": "success",
        "stats": stats,
        "frames": frames,
        "stations": stations_data,
        "queues_over_time": queues_over_time,
        "avg_soc_over_time": [round(v, 4) for v in avg_soc_over_time],
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)