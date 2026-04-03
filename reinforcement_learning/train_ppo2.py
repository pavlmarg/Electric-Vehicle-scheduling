import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor 
from environments.ev_gym_env import EVFleetEnv

def resume_training():
    print("--- 1. INITIALIZING EV FLEET ENVIRONMENT ---")
    # Χρησιμοποιούμε ακριβώς τις ίδιες παραμέτρους με την αρχική εκπαίδευση
    env = EVFleetEnv(city_size_km=25, num_stations=12, num_vehicles=1000)
    
    # Πρέπει να το κάνουμε Monitor για να συνεχίσει να στέλνει τα δεδομένα στο TensorBoard
    env = Monitor(env) 
    vec_env = DummyVecEnv([lambda: env])

    print("\n--- 2. LOADING EXISTING PPO MODEL ---")
    # Το ακριβές όνομα του μοντέλου που έσωσε το train_ppo.py
    NEW_MODEL_PATH = "models/ppo_ev_fleet_model"
    OLD_MODEL_PATH = "models/ppo_ev_fleet_model_2_5M"

    try:
        # Φορτώνουμε το μοντέλο και το συνδέουμε με το vec_env
        model = PPO.load(OLD_MODEL_PATH, env=vec_env)
        print(f"Το μοντέλο '{OLD_MODEL_PATH}' φορτώθηκε επιτυχώς!")
    except FileNotFoundError:
        print(f"ΣΦΑΛΜΑ: Δεν βρέθηκε το αρχείο {OLD_MODEL_PATH}.zip. Έλεγξε τον φάκελο models/")
        return

    print("\n--- 3. RESUMING TRAINING ---")
    STEPS_TO_TRAIN = 500000
    
    # Το reset_num_timesteps=False είναι αυτό που ενώνει τα γραφήματα στο TensorBoard
    model.learn(
        total_timesteps=STEPS_TO_TRAIN, 
        reset_num_timesteps=False, 
        progress_bar=True
    )

    print("\n--- 4. SAVING THE NEW MODEL ---")
    os.makedirs("models", exist_ok=True)
    model.save(NEW_MODEL_PATH)
    print(f"Η εκπαίδευση ολοκληρώθηκε! Το νέο μοντέλο αποθηκεύτηκε ως '{NEW_MODEL_PATH}.zip'")

if __name__ == "__main__":
    resume_training()