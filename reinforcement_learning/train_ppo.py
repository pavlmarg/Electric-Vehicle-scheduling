import os
import sys
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from environments.ev_gym_env import EVFleetEnv

def make_env(rank=0):
    def _init():
        # Διατηρούμε τα 1000 οχήματα
        env = EVFleetEnv(num_vehicles=1000)
        return Monitor(env) 
    return _init

def main():
    log_dir = "./tensorboard_logs/"
    os.makedirs(log_dir, exist_ok=True)

    print("--- 1. Φόρτωση Περιβάλλοντος σε Παράλληλη Επεξεργασία ---")
    num_cpu = 4 
    env = SubprocVecEnv([make_env(i) for i in range(num_cpu)])

    print("--- 2. Δημιουργία Νευρωνικού Δικτύου (PPO) ---")
    model = PPO(
    "MlpPolicy", 
    env, 
    verbose=1, 
    learning_rate=3e-4,
    n_steps=512,          # More frequent updates early on
    batch_size=256,       # Stays clean with 4 envs × 512
    gamma=0.995,          # Slightly higher — rewards happen far in future
    ent_coef=0.005,       # Lower entropy, you want exploitation not exploration
    clip_range=0.2,       # Default, keep it
    n_epochs=10,          # How many times to reuse each batch
    policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
    tensorboard_log=log_dir
)

    print("--- 3. Έναρξη Εκπαίδευσης (500.000 βήματα) ---")
    # Χωρίς Φύλακα, το τρέξιμο θα είναι συνεχόμενο και πολύ πιο γρήγορο
    model.learn(total_timesteps=2000000, tb_log_name="PPO_EV")

    print("--- 4. Τελική Αποθήκευση ---")
    model.save("ppo_fleet_model_v5") 
    print("Η εκπαίδευση ολοκληρώθηκε! Το μοντέλο αποθηκεύτηκε ως 'ppo_fleet_model.zip'")

if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    main()