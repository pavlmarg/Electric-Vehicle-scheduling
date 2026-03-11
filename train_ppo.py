import os
import supersuit as ss
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from environments.ev_routing_env import EVRoutingParallelEnv

def train_marl_model():
    print("--- Initializing EV Routing Environment ---")
    # 1. Create the base PettingZoo Environment
    env = EVRoutingParallelEnv(city_size_km=5, num_stations=12)

    # 2. Apply SuperSuit Wrappers
    # Converts the PettingZoo dictionary format into a vectorized format SB3 can read
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    
    # Stacks the agents so they share the exact same neural network parameters
    env = ss.concat_vec_envs_v1(env, num_vec_envs=1, num_cpus=1, base_class='stable_baselines3')

    print("--- Building PPO Actor-Critic Network ---")
    # 3. Initialize the PPO Algorithm
    model = PPO(
        MlpPolicy, 
        env, 
        verbose=1, 
        learning_rate=3e-4, 
        batch_size=256,
        tensorboard_log="./ppo_ev_tensorboard/"
    )

    print("--- Starting Training Loop ---")
    # 4. Train the Model (100,000 steps is a solid initial test)
    model.learn(total_timesteps=100000)

    # 5. Save the Brain
    os.makedirs("models", exist_ok=True)
    model.save("models/ppo_marl_ev_routing")
    print("--- Training Complete! Saved to /models/ppo_marl_ev_routing.zip ---")

if __name__ == "__main__":
    train_marl_model()