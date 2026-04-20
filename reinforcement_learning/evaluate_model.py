import numpy as np
import csv
from stable_baselines3 import PPO
from environments.ev_gym_env import EVFleetEnv

def evaluate_ai_model():
    print("--- 1. LOADING ENVIRONMENT AND AI MODEL ---")
    
    # Ίδιο Seed με το baseline για να έχουμε τα ΙΔΙΑ ακριβώς αυτοκίνητα
    np.random.seed(50) 
    
    # Αρχικοποίηση του περιβάλλοντος
    env = EVFleetEnv(city_size_km=25, num_stations=12, num_vehicles=1000)
    
    try:
        # Φόρτωση του εκπαιδευμένου μοντέλου
        model = PPO.load("models/ppo_ev_fleet_model")
        print("Το μοντέλο φορτώθηκε επιτυχώς!\n")
    except FileNotFoundError:
        print("ΣΦΑΛΜΑ: Το αρχείο του μοντέλου δεν βρέθηκε. Τρέξε πρώτα το train_ppo.py!")
        return

    print("--- 2. RUNNING 24-HOUR SIMULATION (PPO AI IN CONTROL) ---")
    obs, info = env.reset(seed=42) 
    done = False
    decisions_made = 0
    
    # Μεταβλητή για να θυμάται ποια ώρα τύπωσε τελευταία φορά
    last_printed_hour = -1 
    
    while not done:
        # --- ΕΛΕΓΧΟΣ ΚΑΙ ΕΚΤΥΠΩΣΗ ΑΝΑ ΩΡΑ ---
        current_hour = env.current_minute // 60
        if current_hour > last_printed_hour and current_hour < 24:
            working = sum(1 for e in env.fleet if e.status == 'driving')
            charging = sum(1 for e in env.fleet if e.status == 'charging')
            waiting = sum(1 for e in env.fleet if e.status == 'waiting')
            stranded = sum(1 for e in env.fleet if e.status == 'stranded')
            
            print(f"[Ώρα {current_hour:02d}:00] Στο δρόμο: {working:4d} | Φορτίζουν: {charging:3d} | Αναμονή: {waiting:3d} | Crashed: {stranded:3d}")
            last_printed_hour = current_hour

        # --- ΑΠΟΦΑΣΗ AI ΚΑΙ ΕΚΤΕΛΕΣΗ ---
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        
        done = terminated or truncated
        decisions_made += 1

    print(f"\nΗ προσομοίωση ολοκληρώθηκε! Συνολικές αποφάσεις AI: {decisions_made}")
    
    print("\n--- 3. SAVING HISTORY LOG (PPO) ---")
    PROFIT_PER_KM = 0.70 
    fleet = env.fleet 
    
    with open('history_ppo.csv', mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['EV_ID', 'Status', 'Distance_km', 'Gross_Profit_Eur', 'Charging_Cost_Eur', 'Net_Profit_Eur', 'Times_Charged', 'Wait_Time_mins', 'Final_SoC'])
        
        total_net_profit = 0.0
        successful_charges = 0
        stranded = 0
        
        for ev in fleet:
            gross_profit = ev.total_distance * PROFIT_PER_KM
            net_profit = gross_profit - ev.total_cost
            
            total_net_profit += net_profit
            successful_charges += ev.times_charged
            if ev.status == 'stranded':
                stranded += 1
                
            writer.writerow([
                ev.id, ev.status, f"{ev.total_distance:.2f}", f"{gross_profit:.2f}", 
                f"{ev.total_cost:.2f}", f"{net_profit:.2f}", ev.times_charged, 
                ev.total_waiting_time, f"{ev.current_soc:.2f}"
            ])

    # --- 4. ΤΕΛΙΚΑ ΑΠΟΤΕΛΕΣΜΑΤΑ ΠΡΟΣΟΜΟΙΩΣΗΣ ---
    print("\n" + "="*55)
    print("--- AI PERFORMANCE RESULTS (PPO) ---")
    print(f"Συνολικός Στόλος: 1000 Οχήματα")
    print(f"Ολοκληρωμένες Φορτίσεις: {successful_charges}")
    print(f"Οχήματα που έμειναν από μπαταρία (Stranded): {stranded}")
    print(f"ΣΥΝΟΛΙΚΟ ΚΑΘΑΡΟ ΚΕΡΔΟΣ ΣΤΟΛΟΥ: {total_net_profit:.2f} €")
    print("="*55)

if __name__ == "__main__":
    evaluate_ai_model()