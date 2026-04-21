import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from stable_baselines3 import PPO
from environments.ev_gym_env import EVFleetEnv

def main():
    print("--- 1. Φόρτωση Περιβάλλοντος (City & Fleet) ---")
    env = EVFleetEnv(num_vehicles=400)

    # Δημιουργούμε φάκελο για τα γραφήματα της εκπαίδευσης
    log_dir = "./tensorboard_logs/"
    os.makedirs(log_dir, exist_ok=True)

    print("--- 2. Δημιουργία Νευρωνικού Δικτύου (PPO) ---")
    # Το MlpPolicy σημαίνει ότι θα χρησιμοποιήσουμε ένα κλασικό Multi-Layer Perceptron
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1, 
        learning_rate=0.0003,
        gamma=0.99,           
        tensorboard_log=log_dir
    )

    print("--- 3. Έναρξη Εκπαίδευσης (Training) ---")
    # ΑΥΞΗΣΑΜΕ ΤΑ ΒΗΜΑΤΑ ΣΕ 250.000! 
    # Με τον σωστό χρονοδιακόπτη, αυτό αντιστοιχεί σε ~40-50 πλήρεις ημέρες.
    timesteps = 250_000 
    
    # Αλλάζουμε το όνομα σε v2 για να βλέπουμε καθαρά τα νέα γραφήματα στο TensorBoard
    model.learn(total_timesteps=timesteps, tb_log_name="PPO_FleetManager_v2")

    print("--- 4. Αποθήκευση Μοντέλου ---")
    model.save("ppo_fleet_model_v2")
    print("Το AI εκπαιδεύτηκε και αποθηκεύτηκε επιτυχώς στο αρχείο 'ppo_fleet_model_v2.zip'!")

if __name__ == "__main__":
    main()