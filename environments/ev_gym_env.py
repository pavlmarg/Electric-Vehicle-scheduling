import os
import sys
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from collections import deque
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from environments.citygrid import CityMap
from environments.traffic_generator import TrafficGenerator

class EVFleetEnv(gym.Env):
    def __init__(self, num_vehicles=750): # 750 Οχήματα
        super(EVFleetEnv, self).__init__()
        self.num_vehicles = num_vehicles
        # Φορτώνουμε τον συνεχή χάρτη με τους 12 σταθμούς
        self.city = CityMap(width_km=20.0, height_km=20.0, num_stations=12, num_hubs=3)

        # 14 Επιλογές: 0-11 (Φόρτιση στους 12 σταθμούς), 12 (Stay IDLE), 13 (Rebalance)
        self.action_space = spaces.Discrete(14)
        
        # 37 Τιμές στο Observation Space
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(37,), dtype=np.float32)

        self.total_stars = 0
        self.total_customers_served = 0
        self.total_abandoned = 0
        self.total_energy_kwh = 0.0

        self.current_minute = 0
        self.taxis_needing_action = deque()
        self.previous_net_profit = 0.0
        self.idle_cooldowns = {}
        self._np_rng = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._np_rng = np.random.default_rng(seed)

        for st in self.city.stations:
            st['queue_length'] = 0
            st['occupied'] = {'fast': 0, 'slow': 0}

        self.generator = TrafficGenerator(self.city, num_vehicles=self.num_vehicles)
        self.fleet = self.generator.generate_initial_fleet()

        self.total_stars = 0
        self.total_customers_served = 0
        self.total_abandoned = 0
        self.total_energy_kwh = 0.0

        self.current_minute = 0
        self.taxis_needing_action = deque()
        self.previous_net_profit = 0.0
        self.idle_cooldowns = {ev.id: 0 for ev in self.fleet}

        self._advance_simulation_until_decision()
        return self._get_observation(), {}

    def step(self, action):
        taxi = self.taxis_needing_action.popleft()

        if action < 12: # Πάει σε έναν από τους 12 σταθμούς
            station_idx = action
            dest_pos = self.city.stations[station_idx]['location']
            dist = self.city.calculate_manhattan_dist(taxi.location, dest_pos)
            
            # Υποθέτουμε μέση ταχύτητα 30km/h (0.5 km/min)
            travel_minutes = max(1, int(dist / 0.5))

            if self.city.try_reserve_charger(station_idx):
                taxi.dispatch_to_station(
                    dest_pos, station_idx, dist, travel_minutes, self.current_minute
                )
                self.city.add_to_queue(station_idx)
            else:
                taxi.state = 'IDLE'
                self.idle_cooldowns[taxi.id] = 5

        elif action == 12: # STAY IDLE
            taxi.state = 'IDLE'
            self.idle_cooldowns[taxi.id] = 15

        elif action == 13: # REBALANCING
            # Του δίνουμε μια τυχαία θέση στο κέντρο για να κατευθυνθεί
            dest_pos = self.generator._get_random_point('center')
            dist = self.city.calculate_manhattan_dist(taxi.location, dest_pos)
            
            taxi.state = 'REBALANCING'
            taxi.target_pos = dest_pos
            taxi.arrival_time = self.current_minute + max(1, int(dist / 0.5))

        abandoned = 0
        if not self.taxis_needing_action:
            abandoned = self._advance_simulation_until_decision()

        current_net_profit = sum(
            (e.daily_revenue - e.daily_charging_cost) for e in self.fleet
        )
        
        # --- ΝΕΟ, ΑΥΣΤΗΡΟ ΣΥΣΤΗΜΑ ΑΝΤΑΜΟΙΒΩΝ (REWARD SHAPING) ---
        
        # 1. ΠΕΛΑΤΕΣ: Βαριά καμπάνα για κάθε πελάτη που φεύγει νευριασμένος (50 ευρώ ζημιά αντί για 15)
        reward = (current_net_profit - self.previous_net_profit) - (abandoned * 50.0)
        self.previous_net_profit = current_net_profit

        # 2. ΠΡΟΛΗΨΗ: Το AI τιμωρείται ΑΥΣΤΗΡΑ αν αφήσει ταξί να πέσει κάτω από το 15% (όχι 10%)
        if taxi.current_soc < 0.1:
            reward -= 50.0

        terminated = self.current_minute >= 2880

        if terminated:
            low_battery_count = sum(1 for t in self.fleet if t.current_soc < 0.35)
            stranded_count = sum(1 for t in self.fleet if t.state == 'STRANDED')
            
            reward -= low_battery_count * 5.0 
            
            reward -= stranded_count * 5000.0

        obs = self._get_observation()
        return obs, reward, terminated, False, {}

    def _get_observation(self):
        obs = np.zeros(37, dtype=np.float32)
        obs[0] = self.current_minute / 1440.0

        if self.taxis_needing_action:
            taxi = self.taxis_needing_action[0]
            obs[1] = taxi.current_soc
            # Τοποθεσία X και Y κανονικοποιημένα (0.0 έως 1.0)
            obs[2] = taxi.location[0] / self.city.width_km
            obs[3] = taxi.location[1] / self.city.height_km
        else:
            obs[1] = -1.0
            obs[2] = -1.0
            obs[3] = -1.0

        # Αποστάσεις και Ουρές για τους 12 σταθμούς
        for i in range(12):
            if self.taxis_needing_action:
                station_pos = self.city.stations[i]['location']
                dist = self.city.calculate_manhattan_dist(taxi.location, station_pos)
                # Μέγιστη απόσταση 20km για το normalization
                obs[4 + i] = min(dist / 20.0, 1.0)
            else:
                obs[4 + i] = 1.0
                
            obs[16 + i] = min(self.city.get_queue(i) / 10.0, 1.0)

        # 3x3 Heatmap Ζήτησης
        heatmap = self._calculate_heatmap()
        obs[28:37] = heatmap.flatten()

        # Κόβουμε τυχόν τιμές εκτός ορίων, αφήνοντας τα -1.0 άθικτα στα αρχικά indices
        obs[4:] = np.clip(obs[4:], 0.0, 1.0)
        return obs

    def _calculate_heatmap(self):
        """Φτιάχνει ένα 3x3 πλέγμα ζήτησης χωρίζοντας τον χάρτη στα 3"""
        grid = np.zeros((3, 3))
        if not self.generator.waitlist:
            return grid

        for cust in self.generator.waitlist:
            x, y = cust['spawn_pos']
            
            # Βρίσκουμε σε ποιο από τα 3x3 κελιά ανήκει ο πελάτης
            lon_idx = int((x / self.city.width_km) * 2.99)
            lat_idx = int((y / self.city.height_km) * 2.99)
            
            lon_idx = np.clip(lon_idx, 0, 2)
            lat_idx = np.clip(lat_idx, 0, 2)
            grid[lat_idx, lon_idx] += 1

        max_val = grid.max()
        if max_val > 0:
            grid /= max_val
        return grid

    def _advance_simulation_until_decision(self):
        total_abandoned_this_loop = 0

        while self.current_minute < 2880:
            self.generator.generate_new_demands(self.current_minute)
            ratings, abandoned = self.generator.process_waitlist(self.current_minute)

            total_abandoned_this_loop += abandoned
            self.total_abandoned += abandoned
            self.total_stars += sum(ratings)
            self.total_customers_served += len(ratings)

            for ev in self.fleet:
                ev.update_time(self.current_minute)

                if ev.state == 'REBALANCING':
                    arrival = getattr(ev, 'arrival_time', self.current_minute)
                    if self.current_minute >= arrival:
                        # Χρησιμοποιούμε τη λέξη pos (position)
                        ev.location = ev.target_pos
                        ev.state = 'IDLE'
                        self.idle_cooldowns[ev.id] = 0

                if ev.state == 'IDLE':
                    if self.idle_cooldowns.get(ev.id, 0) > 0:
                        self.idle_cooldowns[ev.id] -= 1
                    elif ev not in self.taxis_needing_action:
                        self.taxis_needing_action.append(ev)
                        
                if ev.state == 'WAITING_FOR_CHARGER':
                    ev.total_waiting_time += 1
                    charger_assigned = self.city.occupy_charger(ev.target_station_idx)
                    if charger_assigned:
                        self.city.remove_from_queue(ev.target_station_idx)
                        ev.state = 'CHARGING'
                        ev.charger_type = charger_assigned

                if ev.state == 'CHARGING':
                    p = self.city.charger_specs[ev.charger_type]['power']
                    cost = self.city.get_electricity_price(self.current_minute, ev.charger_type)
                    station_to_release = ev.target_station_idx
                    added_kwh = ev.charge(p, cost)
                    self.total_energy_kwh += added_kwh

                    if ev.state == 'IDLE':
                        self.city.release_charger(station_to_release, ev.charger_type)

            self.current_minute += 1

            if self.taxis_needing_action:
                return total_abandoned_this_loop

        return total_abandoned_this_loop