import os
import sys
import numpy as np
import random
from stable_baselines3 import PPO
import csv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from environments.ev_gym_env import EVFleetEnv

def evaluate_single_profile(model, env, profile_idx):
    """Αξιολογεί το AI μοντέλο σε ένα συγκεκριμένο προφίλ ζήτησης."""
    # 1. Κάνουμε reset το περιβάλλον με σταθερό seed
    obs, _ = env.reset(seed=50)
    
    # 2. ΕΠΙΒΟΛΗ ΤΟΥ ΠΡΟΦΙΛ (Αμέσως μετά το reset)
    env.generator.client_manager.current_profile = env.generator.client_manager.all_profiles[profile_idx]
    
    terminated = False
    safety_overrides_count = 0
    dead_taxis_set = set()

    # Simulation Loop
    while not terminated:
        action, _states = model.predict(obs, deterministic=True)
        
        # --- SAFETY SHIELD (ΑΣΠΙΔΑ ΠΡΟΣΤΑΣΙΑΣ) ---
        if len(env.taxis_needing_action) > 0:
            current_taxi = env.taxis_needing_action[0]
            
            if current_taxi.current_soc < 0.25 and action >= 16:
                safety_overrides_count += 1
                best_station = 0
                best_score = float('inf')
                
                for i, st in enumerate(env.city.stations):
                    dist = env.city.calculate_manhattan_dist(current_taxi.location, st['location'])
                    score = dist + (st['queue_length'] * 2.0) 
                    
                    if score < best_score:
                        best_score = score
                        best_station = i
                
                action = best_station
        # ----------------------------------------

        obs, reward, terminated, truncated, info = env.step(action)
        
        for ev in env.fleet:
            if ev.state == 'STRANDED' and ev.id not in dead_taxis_set:
                dead_taxis_set.add(ev.id)

    # --- ΥΠΟΛΟΓΙΣΜΟΙ ΑΠΟΤΕΛΕΣΜΑΤΩΝ ΓΙΑ ΤΟ ΣΥΓΚΕΚΡΙΜΕΝΟ ΠΡΟΦΙΛ ---
    LEASING_COST_EUR = 40.0 
    total_net_profit = sum(e.daily_revenue - e.daily_charging_cost - LEASING_COST_EUR for e in env.fleet)
    
    total_requested = env.total_customers_served + env.total_abandoned
    service_rate = (env.total_customers_served / total_requested * 100) if total_requested > 0 else 0.0
    
    # ΠΡΟΣΟΧΗ: Υποθέτουμε ότι έχεις αλλάξει το env.total_stars σε env.total_wait_time στο ev_gym_env.py!
    avg_wait_time = (env.total_wait_time / env.total_customers_served) if env.total_customers_served > 0 else 0.0

    return {
        "profit": total_net_profit,
        "service_rate": service_rate,
        "wait_time": avg_wait_time,
        "overrides": safety_overrides_count,
        "dead_taxis": len(dead_taxis_set),
        "queues_over_time": env.queues_over_time, 
        "avg_soc_over_time": env.avg_soc_over_time
    }

def main():
    NUM_VEHICLES = 750
    
    # Κλείδωμα seeds
    np.random.seed(50)
    random.seed(50)
    
    print("--- 1. INITIALIZING ENVIRONMENT & LOADING AI MODEL ---")
    env = EVFleetEnv(num_vehicles=NUM_VEHICLES)

    model_path = "ppo_model" 
    
    if not os.path.exists(model_path + ".zip"):
        print(f"ΣΦΑΛΜΑ: Το αρχείο {model_path}.zip δεν βρέθηκε!")
        return
        
    model = PPO.load(model_path)

    profile_names = [
        "Normal", "Commuter", "Saturday", "Sunday", "Low", 
        "High Stress", "Flattened", "Bimodal", "Event", "Early Spike"
    ]
    
    all_ai_results = []

    print("\n--- 2. STARTING AI BENCHMARK ACROSS 10 PROFILES ---")
    print(f"{'Profile':<15} | {'Profit (€)':<12} | {'Service %':<10} | {'Wait(m)':<7} | {'Overrides'}")
    print("-" * 65)

    for i in range(10):
        res = evaluate_single_profile(model, env, i)
        all_ai_results.append(res)
        
        # Εκτύπωση αποτελεσμάτων για το προφίλ
        print(f"{profile_names[i]:<15} | {res['profit']:>10.2f}€ | {res['service_rate']:>8.1f}% | {res['wait_time']:>6.1f}m | {res['overrides']:>5}")

    print("=" * 65)
    print("Benchmarking Complete!")

if __name__ == "__main__":
    main()