import os
import supersuit as ss
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor 
from environments.ev_gym_env import EVFleetEnv

def train_fleet_manager():
    print("--- 1. INITIALIZING EV FLEET ENVIRONMENT ---")
    env = EVFleetEnv(city_size_km=25, num_stations=12, num_vehicles=1000)
    

    env = Monitor(env) 
    
    vec_env = DummyVecEnv([lambda: env])

    print("\n--- 2. BUILDING NEW PPO NEURAL NETWORK ---")
    model = PPO(
        policy="MlpPolicy",           
        env=vec_env,
        learning_rate=0.0003,         
        n_steps=2048,                 
        batch_size=64,
        gamma=0.99,                   
        ent_coef=0.01,                
        verbose=1,                    
        tensorboard_log="./ppo_tensorboard/" 
    )
    
    TIMESTEPS = 2000000
    
    model.learn(total_timesteps=TIMESTEPS, progress_bar=True)

    print("\n--- 4. SAVING THE TRAINED MODEL ---")
    os.makedirs("models", exist_ok=True)
    model.save("models/ppo_ev_fleet_model")
    print("Το νέο μοντέλο εκπαιδεύτηκε και αποθηκεύτηκε επιτυχώς!")

if __name__ == "__main__":
    train_fleet_manager()