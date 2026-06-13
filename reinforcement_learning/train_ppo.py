import os
import sys
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from environments.ev_gym_env import EVFleetEnv

def make_env(rank=0):
    def _init():
        env = EVFleetEnv(num_vehicles=750)
        return Monitor(env) 
    return _init

def main():
    log_dir = "./tensorboard_logs/"
    os.makedirs(log_dir, exist_ok=True)


    num_cpu = 4 
    env = SubprocVecEnv([make_env(i) for i in range(num_cpu)])

    model = PPO(
    "MlpPolicy", 
    env, 
    verbose=1, 
    learning_rate=3e-4,
    n_steps=512,          
    batch_size=256,       
    gamma=0.995,          
    ent_coef=0.005,       
    clip_range=0.2,       
    n_epochs=10,          
    policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
    tensorboard_log=log_dir
)

    print("--- 3. Έναρξη Εκπαίδευσης (500.000 βήματα) ---")
    model.learn(total_timesteps=3000000, tb_log_name="PPO_EV")

    print("--- 4. Τελική Αποθήκευση ---")
    model.save("ppo_model") 
    print("Η εκπαίδευση ολοκληρώθηκε!")

if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    main()