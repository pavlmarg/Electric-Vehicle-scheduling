import os
import sys
import numpy as np
import random
from stable_baselines3 import PPO
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from environments.ev_gym_env import EVFleetEnv
import csv

def evaluate():
    NUM_VEHICLES = 750
    
    print("--- 1. INITIALIZING CONTINUOUS SPACE MAP & FLEET FOR AI ---")
    np.random.seed(50)
    random.seed(50)
    
    env = EVFleetEnv(num_vehicles=NUM_VEHICLES)

    model_path = "ppo_fleet_model_v6" # Προσοχή: Βάλε το σωστό όνομα του τελευταίου σου μοντέλου!
    
    if not os.path.exists(model_path + ".zip"):
        print(f"ΣΦΑΛΜΑ: Το αρχείο {model_path}.zip δεν βρέθηκε! Σιγουρέψου ότι έτρεξες το train_ppo.py")
        return
        
    print(f"Loading model: {model_path}")
    model = PPO.load(model_path)

    np.random.seed(50)
    obs, _ = env.reset(seed=50)
    
    terminated = False
    
    print("\n--- 2. STARTING AI SIMULATION LOOP (48 HOURS) ---")
    
    last_printed_hour = -1
    dead_taxis_set = set()
    
    # Μετρητής για να δούμε πόσες φορές χρειάστηκε να επέμβει ο "Φύλακας"
    safety_overrides_count = 0 
    
    while not terminated:
        # Το AI προτείνει μια κίνηση
        action, _states = model.predict(obs, deterministic=True)
        
        # =================================================================
        # --- SAFETY SHIELD (ΑΣΠΙΔΑ ΠΡΟΣΤΑΣΙΑΣ) ---
        # =================================================================
        if len(env.taxis_needing_action) > 0:
            current_taxi = env.taxis_needing_action[0]
            
            # Αν το ταξί κινδυνεύει (< 20%) ΚΑΙ το AI επέλεξε κάτι άλλο εκτός από σταθμό (16 ή 17)
            if current_taxi.current_soc < 0.25 and action >= 16:
                safety_overrides_count += 1
                
                # Αγνοούμε το AI! Βρίσκουμε τον βέλτιστο σταθμό
                best_station = 0
                best_score = float('inf')
                
                for i, st in enumerate(env.city.stations):
                    dist = env.city.calculate_manhattan_dist(current_taxi.location, st['location'])
                    # Σκορ Σταθμού: Θέλουμε μικρή απόσταση, αλλά και μικρή ουρά. 
                    # Βάζουμε ποινή * 2.0 στην ουρά για να αποφύγουμε κλειστούς σταθμούς.
                    score = dist + (st['queue_length'] * 2.0) 
                    
                    if score < best_score:
                        best_score = score
                        best_station = i
                
                # Επιβάλλουμε τη σωτήρια κίνηση!
                action = best_station
        # =================================================================

        obs, reward, terminated, truncated, info = env.step(action)
        
        current_hour = env.current_minute // 60
        
        for ev in env.fleet:
            if ev.state == 'STRANDED' and ev.id not in dead_taxis_set:
                print(f"💀 [Ώρα {env.current_minute//60:02d}:{env.current_minute%60:02d}] SOS: Το Ταξί {ev.id} έμεινε από μπαταρία!")
                dead_taxis_set.add(ev.id)
        
        if current_hour > last_printed_hour and current_hour < 48:
            with_cust = sum(1 for e in env.fleet if e.state == 'WITH_CUSTOMER')
            idle = sum(1 for e in env.fleet if e.state == 'IDLE')
            charging = sum(1 for e in env.fleet if e.state == 'CHARGING')
            waiting = sum(1 for e in env.fleet if e.state == 'WAITING_FOR_CHARGER')
            rebalancing = sum(1 for e in env.fleet if e.state == 'REBALANCING')
            
            avg_stars = (env.total_stars / env.total_customers_served) if env.total_customers_served > 0 else 5.0
            waitlist_len = len(env.generator.waitlist)
            
            print(f"[Ώρα {current_hour:02d}:00] Ταξί(Ελεύθ:{idle:3d}|Πελάτης:{with_cust:3d}|Επιστρέφ:{rebalancing:3d}|Φορτίζ:{charging:2d}|Ουρά:{waiting:2d}) | Πελάτες Αναμονή: {waitlist_len:3d} | Αστέρια: {avg_stars:.1f} | Νεκρά: {len(dead_taxis_set)}")
            last_printed_hour = current_hour

    print("\n--- 3. SAVING HISTORY LOG ---")
    LEASING_COST_EUR = 40.0 
    
    with open('history_ai_model.csv', mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['EV_ID', 'Status', 'Distance_km', 'Gross_Profit_Eur', 'Charging_Cost_Eur', 'Net_Profit_Eur', 'Times_Charged', 'Wait_Time_mins', 'Final_SoC', 'Customers_Served'])
        
        for ev in env.fleet:
            net_profit = ev.daily_revenue - ev.daily_charging_cost - LEASING_COST_EUR
            writer.writerow([
                ev.id, ev.state, f"{ev.total_km_driven:.2f}", f"{ev.daily_revenue:.2f}", 
                f"{ev.daily_charging_cost:.2f}", f"{net_profit:.2f}", ev.times_charged, 
                ev.total_waiting_time, f"{ev.current_soc:.2f}", ev.customers_served
            ])

    successful_charges = sum(e.times_charged for e in env.fleet)
    total_net_profit = sum(e.daily_revenue - e.daily_charging_cost - LEASING_COST_EUR for e in env.fleet)
    final_stranded = sum(1 for e in env.fleet if e.state == 'STRANDED')
    final_avg_stars = (env.total_stars / env.total_customers_served) if env.total_customers_served > 0 else 0.0
    
    print("\n" + "="*60)
    print("--- SIMULATION COMPLETE (AI FLEET MANAGER) ---")
    print(f"Συνολικός Στόλος: {NUM_VEHICLES} Οχήματα")
    print(f"Επεμβάσεις Ασφαλείας (Safety Overrides): {safety_overrides_count}")
    print(f"Ολοκληρωμένες Φορτίσεις: {successful_charges}")
    print(f"Συνολική Ενέργεια Δικτύου: {env.total_energy_kwh:.2f} kWh")
    print(f"ΣΥΝΟΛΙΚΟ ΚΑΘΑΡΟ ΚΕΡΔΟΣ ΕΤΑΙΡΕΙΑΣ: {total_net_profit:.2f} €")
    print("-" * 60)
    print(f"Εξυπηρετήθηκαν: {env.total_customers_served} Πελάτες")
    print(f"Εγκατέλειψαν (Χαμένα Έσοδα): {env.total_abandoned} Πελάτες")
    print(f"Μέση Βαθμολογία Στόλου (Αστέρια): {final_avg_stars:.2f} / 5.00")
    print(f"Οχήματα που έμειναν από μπαταρία: {final_stranded}")
    print("="*60)

if __name__ == "__main__":
    evaluate()