import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor 
from environments.ev_gym_env import EVFleetEnv

def make_env():
    def _init():
        # Προσοχή: Εδώ είναι 1000 οχήματα! Αν θες να το συγκρίνεις 
        # με το Baseline των 400, άλλαξέ το σε 400.
        env = EVFleetEnv(num_vehicles=1000)
        return Monitor(env) 
    return _init

def main():
    print("--- 1. Φόρτωση Περιβάλλοντος σε Παράλληλη Επεξεργασία ---")
    
    num_cpu = 4 
    env = SubprocVecEnv([make_env() for _ in range(num_cpu)])

    log_dir = "./tensorboard_logs/"
    os.makedirs(log_dir, exist_ok=True)

    print("--- 2. Φόρτωση Υπάρχοντος Νευρωνικού Δικτύου (PPO) ---")
    model_path = "ppo_fleet_model_v6"
    
    # Φορτώνουμε το παλιό μοντέλο και το συνδέουμε με το τωρινό περιβάλλον και log
    if os.path.exists(model_path + ".zip"):
        model = PPO.load(model_path, env=env, tensorboard_log=log_dir)
        print(f"Το μοντέλο {model_path} φορτώθηκε επιτυχώς!")
    else:
        print(f"ΣΦΑΛΜΑ: Το αρχείο {model_path}.zip δεν βρέθηκε!")
        return

    print("--- 3. Συνέχιση Εκπαίδευσης (Training) ---")
    timesteps = 1000000
    
    # Το reset_num_timesteps=False συνεχίζει το γράφημα στο Tensorboard ομαλά!
    model.learn(
        total_timesteps=timesteps, 
        tb_log_name="PPO_Conti", 
        reset_num_timesteps=False
    )

    print("--- 4. Αποθήκευση Νέου Μοντέλου ---")
    model.save("ppo_fleet_model_v7") 
    print("Το AI εκπαιδεύτηκε για ακόμα 500.000 βήματα και αποθηκεύτηκε ως v5!")

if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    main()