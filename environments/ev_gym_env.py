import numpy as np
import gymnasium as gym
from gymnasium import spaces

from environments.citygrid import CityGrid
from environments.traffic_generator import TrafficGenerator

class EVFleetEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, city_size_km=25, num_stations=12, num_vehicles=1000):
        super().__init__()
        
        self.city_size = city_size_km
        self.num_stations = num_stations
        self.num_vehicles = num_vehicles
        
        
        self.action_space = spaces.Discrete(self.num_stations)
        
        
        obs_size = 4 + self.num_stations 
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(obs_size,), dtype=np.float32)

        # Σταθερή υποδομή πόλης
        np.random.seed(50)
        self.city = CityGrid(size_km=self.city_size, num_stations=self.num_stations)
        np.random.seed(None)
        
        self.generator = None
        self.fleet = []
        
        # Μεταβλητές διαχείρισης χρόνου
        self.current_minute = 0
        self.current_ev_idx = 0  
        self.ev_needing_routing = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        if seed is not None:
            np.random.seed(seed)
        
        
        for st in self.city.stations:
            st['current_load'] = 0.0
            st['occupied'] = {'fast': 0, 'slow': 0}
            st['queue_length'] = 0
            
        # Νέα ημέρα, νέα οχήματα
        self.generator = TrafficGenerator(self.city, num_vehicles=self.num_vehicles)
        self.fleet = self.generator.generate_initial_fleet()
        
        
        self.current_minute = 0
        self.current_ev_idx = 0
        self.ev_needing_routing = None
        
        self._advance_simulation_until_routing()
        
        return self._get_obs(), {}

    def _get_obs(self):
        """Ετοιμάζει τα δεδομένα για το AI."""
        if self.ev_needing_routing is None:
            return np.zeros(self.observation_space.shape, dtype=np.float32)
            
        ev = self.ev_needing_routing
        
        car_x = np.clip(ev.location[0] / self.city_size, 0.0, 1.0)
        car_y = np.clip(ev.location[1] / self.city_size, 0.0, 1.0)
        car_soc = ev.current_soc
        time_norm = self.current_minute / 1440.0
        
        queues = [np.clip(self.city.get_queue(i) / 30.0, 0.0, 1.0) for i in range(self.num_stations)]
        
        return np.array([car_x, car_y, car_soc, time_norm] + queues, dtype=np.float32)

    def step(self, action):
        ev = self.ev_needing_routing
        target_station = self.city.stations[action]
        
        # --- ΥΠΟΛΟΓΙΣΜΟΣ REWARD ---
        dist = np.linalg.norm(ev.location - target_station['location'])
        queue = self.city.get_queue(action)
        price_now = self.city.get_electricity_price(self.current_minute, 'fast')
        
        energy_needed = dist * ev.consumption_per_km
        if (ev.current_soc * ev.battery_capacity) < energy_needed:
            reward = -5000.0 
        else:
            reward = - (dist * 2.0) - (queue * 1.0) - (price_now * 3.0)
        
        #  ΕΦΑΡΜΟΓΗ ΑΠΟΦΑΣΗΣ
        ev.target_station_idx = action
        ev.target_location = target_station['location']
        ev.status = 'moving_to_station'
        self.city.add_to_queue(action)
        
        self.ev_needing_routing = None
        
        # ΠΡΟΧΩΡΑΜΕ ΤΟΝ ΧΡΟΝΟ 
        terminated = self._advance_simulation_until_routing()
        
        obs = self._get_obs()
        return obs, reward, terminated, False, {}

    def _advance_simulation_until_routing(self):
        """
        Τρέχει τον χρόνο λεπτό προς λεπτό, συνεχίζοντας από το όχημα που είχε μείνει.
        """
        while self.current_minute < 1440:
            
            # Ελέγχουμε τα οχήματα ΞΕΚΙΝΩΝΤΑΣ από το current_ev_idx
            while self.current_ev_idx < len(self.fleet):
                ev = self.fleet[self.current_ev_idx]
                
                if ev.status == 'driving':
                    ev.drive_randomly(self.city_size)
                    if ev.status == 'routing':
                        self.ev_needing_routing = ev
                        self.current_ev_idx += 1  
                        return False 
                        
                elif ev.status == 'moving_to_station':
                    if ev.move_to_station(): 
                        self.city.remove_from_queue(ev.target_station_idx)
                        charger_assigned = self.city.occupy_charger(ev.target_station_idx)
                        if charger_assigned:
                            ev.status = 'charging'
                            ev.charger_type = charger_assigned
                            
                elif ev.status == 'waiting':
                    ev.total_waiting_time += 1
                    charger_assigned = self.city.occupy_charger(ev.target_station_idx)
                    if charger_assigned:
                        ev.status = 'charging'
                        ev.charger_type = charger_assigned
                        
                elif ev.status == 'charging':
                    power = self.city.charger_specs[ev.charger_type]['power']
                    price = self.city.get_electricity_price(self.current_minute, ev.charger_type)
                    
                    station_to_release = ev.target_station_idx
                    ev.charge(power_kw=power, price_per_kwh=price)
                    
                    if ev.status == 'driving':
                        self.city.release_charger(station_to_release, ev.charger_type)

                # Πάμε στο επόμενο όχημα για το τρέχον λεπτό
                self.current_ev_idx += 1
                
            self.current_minute += 1
            self.current_ev_idx = 0
            
        return True 