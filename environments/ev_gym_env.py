import os
import sys
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from collections import deque
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from environments.citygrid import CityMap
from environments.traffic_generator import TrafficGenerator
import random


class EVFleetEnv(gym.Env):
    def __init__(self, num_vehicles=750):
        super(EVFleetEnv, self).__init__()
        self.num_vehicles = num_vehicles

        self.city = CityMap(width_km=20.0, height_km=20.0, num_stations=16, num_hubs=4)

        self.action_space = spaces.Discrete(18)

        # Observation space expanded: 48 features
        # [0]    = sin(time)                        <- Cyclical time encoding
        # [1]    = cos(time)                        <- Cyclical time encoding
        # [2]    = taxi SoC
        # [3]    = taxi x position
        # [4]    = taxi y position
        # [5-20] = distance to each of 16 stations
        # [21-36]= queue length at each of 16 stations
        # [37-45]= 3x3 demand heatmap
        # [46]   = low_soc_ratio (stampede predictor) <- NEW
        # [47]   = waitlist pressure                  <- NEW
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(48,), dtype=np.float32)

        self.total_stars = 0
        self.total_customers_served = 0
        self.total_abandoned = 0
        self.total_energy_kwh = 0.0
        self.safety_overrides = 0  # Track how often the safety net fires

        self.current_minute = 0
        self.taxis_needing_action = deque()
        self.taxis_needing_action_set = set()

        self.previous_served = 0
        self.previous_stranded_count = 0
        self.idle_cooldowns = {}
        self._np_rng = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._np_rng = np.random.default_rng(seed)
        
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
            
        for st in self.city.stations:
            st['queue_length'] = 0
            st['occupied'] = {'fast': 0, 'slow': 0}

        self.generator = TrafficGenerator(self.city, num_vehicles=self.num_vehicles)
        self.fleet = self.generator.generate_initial_fleet()

        self.total_stars = 0
        self.total_customers_served = 0
        self.total_abandoned = 0
        self.total_energy_kwh = 0.0
        self.safety_overrides = 0

        self.current_minute = 0
        self.taxis_needing_action = deque()
        self.taxis_needing_action_set = set()

        self.previous_served = 0
        self.previous_stranded_count = 0

        self.idle_cooldowns = {ev.id: 0 for ev in self.fleet}

        self._advance_simulation_until_decision()
        return self._get_observation(), {}

    def step(self, action):
        taxi = self.taxis_needing_action.popleft()
        self.taxis_needing_action_set.remove(taxi.id)

        if action < 16:  # Charging
            station_idx = action
            dest_pos = self.city.stations[station_idx]['location']
            dist = self.city.calculate_manhattan_dist(taxi.location, dest_pos)
            travel_minutes = max(1, int(dist / 0.5))
            taxi.dispatch_to_station(dest_pos, station_idx, dist, travel_minutes, self.current_minute)
            self.city.add_to_queue(station_idx)

        elif action == 16:  # STAY IDLE
            taxi.state = 'IDLE'
            self.idle_cooldowns[taxi.id] = 5

        elif action == 17:  # REBALANCING
            dest_pos = self.generator.client_manager._get_random_point('center')
            dist = self.city.calculate_manhattan_dist(taxi.location, dest_pos)
            taxi.state = 'REBALANCING'
            taxi.target_pos = dest_pos
            taxi.arrival_time = self.current_minute + max(1, int(dist / 0.5))

        # --- Advance Simulation ---
        abandoned_this_step = 0
        if not self.taxis_needing_action:
            abandoned_this_step = self._advance_simulation_until_decision()

        # --- Collect reward data ---
        current_stranded = sum(1 for e in self.fleet if e.state == 'STRANDED')
        newly_stranded = current_stranded - self.previous_stranded_count
        newly_served = self.total_customers_served - self.previous_served

        # --- Context variables ---
        reward = 0.0
        current_hour = self.current_minute // 60
        waitlist_len = len(self.generator.waitlist)
        is_peak = (7 <= current_hour <= 9) or (16 <= current_hour <= 19)
        is_night = current_hour < 6

        # Stampede predictor: what fraction of the fleet is dangerously low?
        low_soc_ratio = sum(1 for e in self.fleet if e.current_soc < 0.40) / len(self.fleet)
        stampede_incoming = low_soc_ratio > 0.25 and not is_night  # >25% of fleet below 40%

        # ============================================================
        # 1. CUSTOMER SERVICE (Primary objective)
        # ============================================================
        reward += newly_served * 8.0
        self.previous_served = self.total_customers_served

        reward -= abandoned_this_step * 4.0
        reward -= newly_stranded * 5.0
        self.previous_stranded_count = current_stranded

        # ============================================================
        # 2. CHARGING DECISIONS (Actions 0-15)
        # ============================================================
        if action < 16:
            station_idx = action
            queue_len = self.city.stations[station_idx]['queue_length']

            # Distance cost 
            reward -= dist * 0.05

            # Exponential queue penalty 
            reward -= (queue_len ** 1.5) * 0.8

            # PEAK HOUR: Charging is almost always wrong unless critically low
            if is_peak and waitlist_len > 0:
                if taxi.current_soc > 0.25:
                    reward -= 30.0  # Strong deterrent: customers are waiting
                else:
                    reward += 10.0  # Survival instinct override: had no choice

            # Reward proactive charging proportionally
            elif is_night and waitlist_len == 0:
                if taxi.current_soc < 0.85:
                    reward += (1.0 - taxi.current_soc) * 15.0

            # DAY / OFF-PEAK
            elif not is_peak and not is_night:
                if taxi.current_soc > 0.60 and waitlist_len > 0:
                    reward -= 10.0  # Battery is fine, go serve people

                
                if stampede_incoming and taxi.current_soc < 0.50 and queue_len < 5:
                    reward += (1.0 - taxi.current_soc) * 10.0  # Proactive charging before the rush

        # ============================================================
        # 3. AVAILABILITY & MOVEMENT
        # ============================================================
        elif action == 16:  # IDLE
            if taxi.current_soc < 0.25:
                # Dangerous to stay idle — will strand soon
                reward -= ((1.0 - taxi.current_soc) ** 2) * 30.0


            if stampede_incoming and taxi.current_soc < 0.35:
                reward -= 15.0

        elif action == 17:  # REBALANCING
            reward -= dist * 0.01

            if taxi.current_soc < 0.25:
                reward -= ((1.0 - taxi.current_soc) ** 2) * 30.0

            if is_peak and waitlist_len > 50:
                reward -= 10.0  

        # ============================================================
        # 4. END OF DAY SUMMARY
        # ============================================================
        terminated = self.current_minute >= 1440

        if terminated:
            low_battery_count = sum(1 for t in self.fleet if t.current_soc < 0.35)
            reward -= low_battery_count * 0.5

        obs = self._get_observation()
        return obs, reward, terminated, False, {}

    def _get_observation(self):
        obs = np.zeros(48, dtype=np.float32)

        # [0-1] Cyclical time encoding (fixes the duplicate obs[0]==obs[1] bug)
        # sin/cos encoding means the agent understands midnight ≈ midnight, not 0 ≈ 1440
        time_norm = (self.current_minute / 1440.0) * 2.0 * np.pi
        obs[0] = np.sin(time_norm)
        obs[1] = np.cos(time_norm)

        # [2-4] Taxi state
        if self.taxis_needing_action:
            taxi = self.taxis_needing_action[0]
            obs[2] = taxi.current_soc
            obs[3] = taxi.location[0] / self.city.width_km
            obs[4] = taxi.location[1] / self.city.height_km
        else:
            obs[2] = -1.0
            obs[3] = -1.0
            obs[4] = -1.0

        # [5-20] Distance to each station
        # [21-36] Queue at each station
        for i in range(16):
            if self.taxis_needing_action:
                station_pos = self.city.stations[i]['location']
                dist = self.city.calculate_manhattan_dist(taxi.location, station_pos)
                obs[5 + i] = min(dist / 20.0, 1.0)
            else:
                obs[5 + i] = 1.0

            obs[21 + i] = min(self.city.get_queue(i) / 20.0, 1.0)

        # [37-45] 3x3 demand heatmap
        heatmap = self._calculate_heatmap()
        obs[37:46] = heatmap.flatten()

        # [46] Stampede predictor: fraction of fleet below 40% SoC
        # High value = charging rush is imminent, act proactively now
        low_soc_ratio = sum(1 for e in self.fleet if e.current_soc < 0.40) / len(self.fleet)
        obs[46] = float(low_soc_ratio)

        # [47] Waitlist pressure: normalised demand backlog
        # Tells the agent how urgently customers need service RIGHT NOW
        waitlist_pressure = min(len(self.generator.waitlist) / 500.0, 1.0)
        obs[47] = float(waitlist_pressure)

        # Clip everything except the sin/cos time features
        obs[2:] = np.clip(obs[2:], 0.0, 1.0)
        # Restore sentinel values if no taxi (clipping would have zeroed them)
        if not self.taxis_needing_action:
            obs[2] = -1.0
            obs[3] = -1.0
            obs[4] = -1.0

        return obs

    def _calculate_heatmap(self):
        grid = np.zeros((3, 3))
        if not self.generator.waitlist:
            return grid

        for cust in self.generator.waitlist:
            x, y = cust['spawn_pos']

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

        while self.current_minute < 1440:
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
                        ev.location = ev.target_pos
                        ev.state = 'IDLE'
                        self.idle_cooldowns[ev.id] = 0

                # --------------------------------------------------------
                # SAFETY OVERRIDE: If a taxi is idle and critically low,
                # send it to the nearest uncrowded station automatically.
                # This frees the RL agent to focus on STRATEGIC decisions
                # (proactive charging, rebalancing) not emergency triage.
                # --------------------------------------------------------
                if ev.state == 'IDLE' and ev.current_soc <= 0.20:
                    best_station = self._find_best_emergency_station(ev)
                    if best_station is not None:
                        dest_pos = self.city.stations[best_station]['location']
                        dist = self.city.calculate_manhattan_dist(ev.location, dest_pos)
                        travel_minutes = max(1, int(dist / 0.5))
                        ev.dispatch_to_station(dest_pos, best_station, dist, travel_minutes, self.current_minute)
                        self.city.add_to_queue(best_station)
                        self.safety_overrides += 1
                        # Remove from action queue if it was already there
                        if ev.id in self.taxis_needing_action_set:
                            self.taxis_needing_action_set.discard(ev.id)
                            self.taxis_needing_action = deque(
                                t for t in self.taxis_needing_action if t.id != ev.id
                            )
                        continue  # Skip normal idle logic below

                if ev.state == 'IDLE':
                    if self.idle_cooldowns.get(ev.id, 0) > 0:
                        self.idle_cooldowns[ev.id] -= 1
                    elif ev.id not in self.taxis_needing_action_set:
                        self.taxis_needing_action.append(ev)
                        self.taxis_needing_action_set.add(ev.id)

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

    def _find_best_emergency_station(self, ev):
        """
        Finds the nearest station with a short queue for emergency charging.
        Prioritises stations with queue < 5 to avoid adding to stampedes.
        Falls back to nearest station if all are crowded.
        """
        best_idx = None
        best_score = float('inf')

        for station in self.city.stations:
            dist = self.city.calculate_manhattan_dist(ev.location, station['location'])
            queue = self.city.get_queue(station['id'])
            # Score: distance + heavy queue penalty
            score = dist + (queue * 2.0)
            # Strongly prefer stations with short queues
            if queue < 5:
                score -= 5.0
            if score < best_score:
                best_score = score
                best_idx = station['id']

        return best_idx