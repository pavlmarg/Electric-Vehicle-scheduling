import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
# 1. ΚΑΝΟΥΜΕ IMPORT TO MONITOR
from stable_baselines3.common.monitor import Monitor 
from environments.ev_gym_env import EVFleetEnv

def make_env():
    def _init():
        # 2. ΦΤΙΑΧΝΟΥΜΕ ΤΟ ΠΕΡΙΒΑΛΛΟΝ
        env = EVFleetEnv(num_vehicles=750)
        # 3. ΤΟ ΤΥΛΙΓΟΥΜΕ ΜΕ ΤΟ MONITOR
        return Monitor(env) 
    return _init

def main():
    print("--- 1. Φόρτωση Περιβάλλοντος σε Παράλληλη Επεξεργασία ---")
    
    num_cpu = 4 
    env = SubprocVecEnv([make_env() for _ in range(num_cpu)])

    log_dir = "./tensorboard_logs/"
    os.makedirs(log_dir, exist_ok=True)

    print("--- 2. Δημιουργία Νευρωνικού Δικτύου (PPO) ---")
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1, 
        learning_rate=0.0003,
        gamma=0.99, 
        policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
        tensorboard_log=log_dir
    )

    print("--- 3. Έναρξη Εκπαίδευσης (Training) ---")
    timesteps = 1000000
    
    # Αλλάζουμε το όνομα για να δεις το νέο γράφημα σωστά
    model.learn(total_timesteps=timesteps, tb_log_name="PPO_ContinuousCity")

    print("--- 4. Αποθήκευση Μοντέλου ---")
    model.save("ppo_fleet_model")
    print("Το AI εκπαιδεύτηκε και αποθηκεύτηκε επιτυχώς!")

if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    main()