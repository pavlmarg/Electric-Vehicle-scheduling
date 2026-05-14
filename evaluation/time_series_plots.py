import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

# Βοηθάει την Python να βρει τους φακέλους σου
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from baselines.main_simulation import run_simulation_for_profile
from environments.ev_gym_env import EVFleetEnv
from reinforcement_learning.evaluate_model import evaluate_single_profile

def generate_all_micro_graphs():
    profile_names = [
        "Normal", "Commuter", "Saturday", "Sunday", "Low", 
        "High Stress", "Flattened", "Bimodal", "Event", "Early Spike"
    ]
    
    # Δημιουργία ξεχωριστού φακέλου για να μην γίνει χαμός με 20 εικόνες
    output_dir = "thesis_timeseries_graphs"
    os.makedirs(output_dir, exist_ok=True)
    
    print("--- Φόρτωση Περιβάλλοντος και Μοντέλου AI ---")
    env = EVFleetEnv(num_vehicles=750)
    model = PPO.load("ppo_fleet_model_v6") # Έλεγξε το όνομα!
    
    minutes = np.arange(1440)
    hours = minutes / 60

    print("\n--- Έναρξη Μαζικής Παραγωγής Γραφημάτων ---")
    
    for i, p_name in enumerate(profile_names):
        print(f"[{i+1}/10] Τρέχουν οι προσομοιώσεις για: {p_name} ...")
        
        # 1. Τρέχουμε Baseline
        base_results = run_simulation_for_profile(i)
        base_q = base_results['queues_over_time']
        base_soc = np.array(base_results['avg_soc_over_time']) * 100
        
        # 2. Τρέχουμε AI
        ai_results = evaluate_single_profile(model, env, i)
        ai_q = ai_results['queues_over_time']
        ai_soc = np.array(ai_results['avg_soc_over_time']) * 100
        
        # Καθαρίζουμε το όνομα για να γίνει ωραίο όνομα αρχείου (π.χ. "High Stress" -> "high_stress")
        safe_name = p_name.replace(" ", "_").lower()
        
        # --- ΓΡΑΦΗΜΑ 1: Η ΟΥΡΑ ---
        fig1, ax1 = plt.subplots(figsize=(12, 5))
        ax1.plot(hours, base_q, label='Greedy Baseline', color='#e74c3c', linewidth=2, alpha=0.9)
        ax1.plot(hours, ai_q, label='PPO AI Agent', color='#3498db', linewidth=2, alpha=0.9)
        ax1.fill_between(hours, base_q, alpha=0.2, color='#e74c3c')
        ax1.fill_between(hours, ai_q, alpha=0.2, color='#3498db')

        ax1.set_title(f'Συνολική Ουρά στους Σταθμούς Φόρτισης - Προφίλ: {p_name}', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Ώρα της Ημέρας', fontsize=12)
        ax1.set_ylabel('Αριθμός Ταξί στην Ουρά', fontsize=12)
        ax1.set_xticks(np.arange(0, 25, 2))
        ax1.grid(True, linestyle='--', alpha=0.6)
        ax1.legend()
        fig1.tight_layout()
        
        fig1.savefig(os.path.join(output_dir, f'queue_{safe_name}.png'), dpi=300)
        plt.close(fig1) # Κλείνουμε το figure για να μην γεμίσει η RAM!

        # --- ΓΡΑΦΗΜΑ 2: Η ΜΠΑΤΑΡΙΑ (SoC) ---
        fig2, ax2 = plt.subplots(figsize=(12, 5))
        ax2.plot(hours, base_soc, label='Greedy Baseline', color='#e74c3c', linewidth=2)
        ax2.plot(hours, ai_soc, label='PPO AI Agent', color='#2ecc71', linewidth=2)

        ax2.set_title(f'Μέση Στάθμη Μπαταρίας (SoC) Στόλου - Προφίλ: {p_name}', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Ώρα της Ημέρας', fontsize=12)
        ax2.set_ylabel('Μέσο SoC (%)', fontsize=12)
        ax2.set_xticks(np.arange(0, 25, 2))
        ax2.set_ylim(0, 100)
        ax2.axhline(20, color='black', linestyle=':', label='Όριο Κινδύνου (20%)')
        ax2.grid(True, linestyle='--', alpha=0.6)
        ax2.legend()
        fig2.tight_layout()
        
        fig2.savefig(os.path.join(output_dir, f'soc_{safe_name}.png'), dpi=300)
        plt.close(fig2)

        print(f"  -> Αποθηκεύτηκαν επιτυχώς!")

    print(f"\n=======================================================")
    print(f"ΤΕΛΟΣ! Δημιουργήθηκαν {len(profile_names) * 2} γραφήματα.")
    print(f"Μπορείς να τα βρεις στον φάκελο: {os.path.abspath(output_dir)}")
    print(f"=======================================================")

if __name__ == "__main__":
    generate_all_micro_graphs()