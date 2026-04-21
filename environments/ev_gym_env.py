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
    def __init__(self, num_vehicles=400):
        super(EVFleetEnv, self).__init__()
        self.num_vehicles = num_vehicles
        self.city = CityMap(radius_meters=5500, num_stations=8)

        # 10 Επιλογές: 0-7 (Φόρτιση), 8 (Stay IDLE), 9 (Rebalance)
        self.action_space = spaces.Discrete(10)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(28,), dtype=np.float32)
        self.occupied_nodes = set()

        # --- Μετρητές KPIs ---
        self.total_stars = 0
        self.total_customers_served = 0
        self.total_abandoned = 0
        self.total_energy_kwh = 0.0

        self.current_minute = 0
        # ΔΙΟΡΘΩΣΗ #3: deque αντί για list ώστε το popleft() να είναι O(1)
        self.taxis_needing_action = deque()
        self.previous_net_profit = 0.0
        self.idle_cooldowns = {}
        self._np_rng = None  # Για αναπαραγωγιμότητα

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # ΔΙΟΡΘΩΣΗ #9: Αρχικοποιούμε το numpy RNG με το seed του gymnasium
        self._np_rng = np.random.default_rng(seed)

        self.occupied_nodes.clear()

        for st in self.city.stations:
            st['queue_length'] = 0
            st['occupied'] = {'fast': 0, 'slow': 0}

        self.generator = TrafficGenerator(self.city, num_vehicles=self.num_vehicles)
        self.fleet = self.generator.generate_initial_fleet()

        for ev in self.fleet:
            self.occupied_nodes.add(ev.location)

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

    def _fast_distance_km(self, node1_id, node2_id):
        """Επιταχύνει δραματικά την προσομοίωση παρακάμπτοντας τον αλγόριθμο Dijkstra."""
        n1 = self.city.G.nodes[node1_id]
        n2 = self.city.G.nodes[node2_id]
        # Μετατροπή μοιρών σε km: ~111 km/μοίρα για latitude, cosine correction για longitude
        dlat_km = (n1['y'] - n2['y']) * 111.0
        dlon_km = (n1['x'] - n2['x']) * 111.0 * np.cos(np.radians((n1['y'] + n2['y']) / 2))
        straight_line_km = np.sqrt(dlat_km**2 + dlon_km**2)
        return straight_line_km * 1.3

    def step(self, action):
        # ΔΙΟΡΘΩΣΗ #3: popleft() είναι O(1) με deque
        taxi = self.taxis_needing_action.popleft()

        if taxi.location in self.occupied_nodes:
            self.occupied_nodes.remove(taxi.location)

        # --- ΕΚΤΕΛΕΣΗ ΑΠΟΦΑΣΗΣ ---
        if action < 8:
            station_idx = action
            dest_node = self.city.stations[station_idx]['location']
            dist = self._fast_distance_km(taxi.location, dest_node)
            travel_minutes = max(1, int(dist / 0.4))

            # ΔΙΟΡΘΩΣΗ #2: Ελέγχουμε αν ο σταθμός έχει διαθέσιμο charger πριν δεσμεύσουμε
            if self.city.try_reserve_charger(station_idx):
                taxi.dispatch_to_station(
                    dest_node, station_idx, dist, travel_minutes, self.current_minute
                )
                self.city.add_to_queue(station_idx)
            else:
                # Δεν υπάρχει ελεύθερος charger — το taxi μένει IDLE με μικρό cooldown
                taxi.state = 'IDLE'
                self._ensure_unique_node(taxi)
                self.idle_cooldowns[taxi.id] = 5

        elif action == 8:
            taxi.state = 'IDLE'
            self._ensure_unique_node(taxi)
            self.idle_cooldowns[taxi.id] = 15  # Κουμπί Snooze για 15 λεπτά

        elif action == 9:
            # ΔΙΟΡΘΩΣΗ #9: Χρήση του seeded RNG αντί για np.random
            dest_node = self._np_rng.choice(self.generator.center_nodes)
            taxi.state = 'REBALANCING'
            taxi.target_node = dest_node
            taxi.arrival_time = self.current_minute + 10

        abandoned = 0
        if not self.taxis_needing_action:
            abandoned = self._advance_simulation_until_decision()

        # --- ΥΠΟΛΟΓΙΣΜΟΣ REWARD ---
        current_net_profit = sum(
            (e.daily_revenue - e.daily_charging_cost) for e in self.fleet
        )
        reward = (current_net_profit - self.previous_net_profit) - (abandoned * 15.0)
        self.previous_net_profit = current_net_profit

        # Ποινή χαμηλής μπαταρίας για το συγκεκριμένο ταξί που ρώτησε
        if taxi.current_soc < 0.10:
            reward -= 30.0

        terminated = self.current_minute >= 1440

        # --- Ο ΑΠΟΛΟΓΙΣΜΟΣ ΤΩΝ ΜΕΣΑΝΥΧΤΩΝ ---
        if terminated:
            low_battery_count = sum(1 for t in self.fleet if t.current_soc < 0.35)
            stranded_count = sum(1 for t in self.fleet if t.state == 'STRANDED')

            reward -= low_battery_count * 2.0
            reward -= stranded_count * 500.0  # -500 για κάθε νεκρό ταξί στο τέλος της μέρας

        # ΔΙΟΡΘΩΣΗ #1: Παίρνουμε observation ΠΡΙΝ επιστρέψουμε — αν υπάρχει taxi χρησιμοποιούμε
        # την κατάστασή του, αλλιώς δίνουμε ένα νόμιμο observation βασισμένο στην τρέχουσα ώρα
        obs = self._get_observation()
        return obs, reward, terminated, False, {}

    def _get_observation(self):
        """
        ΔΙΟΡΘΩΣΗ #1: Αντί για μηδενικά όταν δεν υπάρχει taxi, επιστρέφουμε
        μια έγκυρη παρατήρηση της κατάστασης του περιβάλλοντος.
        """
        obs = np.zeros(28, dtype=np.float32)
        obs[0] = self.current_minute / 1440.0

        # Αν υπάρχει taxi που περιμένει απόφαση, συμπληρώνουμε τα taxi-specific features
        if self.taxis_needing_action:
            taxi = self.taxis_needing_action[0]
            obs[1] = taxi.current_soc
            obs[2] = float(taxi.location) / max(self.city.nodes)
        else:
            # Χωρίς taxi: βάζουμε sentinel τιμές που ο agent μπορεί να μάθει να αγνοεί
            obs[1] = -1.0  # Εκτός [0,1] → σαφές σήμα "δεν υπάρχει taxi"
            obs[2] = -1.0

        for i in range(8):
            station_node = self.city.stations[i]['location']
            if self.taxis_needing_action:
                taxi = self.taxis_needing_action[0]
                fast_dist = self._fast_distance_km(taxi.location, station_node)
                obs[3 + i] = min(fast_dist / 10.0, 1.0)
            else:
                obs[3 + i] = 1.0  # Μέγιστη απόσταση ως default
            obs[11 + i] = min(self.city.get_queue(i) / 10.0, 1.0)

        heatmap = self._calculate_heatmap()
        obs[19:28] = heatmap.flatten()

        # Clamp για να μείνουμε πάντα εντός [0,1] εκτός από τους sentinels
        # (τους sentinels τους αφήνουμε ως έχουν)
        obs[3:] = np.clip(obs[3:], 0.0, 1.0)
        return obs

    def _calculate_heatmap(self):
        grid = np.zeros((3, 3))
        if not self.generator.waitlist:
            return grid

        lats = [self.city.G.nodes[n]['y'] for n in self.city.nodes]
        lons = [self.city.G.nodes[n]['x'] for n in self.city.nodes]
        min_lat, max_lat = min(lats), max(lats)
        min_lon, max_lon = min(lons), max(lons)

        lat_range = max_lat - min_lat or 1.0
        lon_range = max_lon - min_lon or 1.0

        for cust in self.generator.waitlist:
            node_data = self.city.G.nodes[cust['spawn_node']]
            lat_idx = int((node_data['y'] - min_lat) / lat_range * 2.99)
            lon_idx = int((node_data['x'] - min_lon) / lon_range * 2.99)
            lat_idx = np.clip(lat_idx, 0, 2)
            lon_idx = np.clip(lon_idx, 0, 2)
            grid[lat_idx, lon_idx] += 1

        # ΔΙΟΡΘΩΣΗ #7: Κανονικοποίηση με βάση τον πραγματικό αριθμό πελατών
        # αντί για hardcoded magic number — δεν μπορεί να ξεπεράσει το 1.0
        max_val = grid.max()
        if max_val > 0:
            grid /= max_val
        return grid

    def _ensure_unique_node(self, taxi):
        if taxi.location in self.occupied_nodes:
            neighbors = list(self.city.G.neighbors(taxi.location))
            for n in neighbors:
                if n not in self.occupied_nodes:
                    taxi.location = n
                    break
        self.occupied_nodes.add(taxi.location)

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

                # ΔΙΟΡΘΩΣΗ #5: Handler για REBALANCING state
                if ev.state == 'REBALANCING':
                    if self.current_minute >= ev.arrival_time:
                        ev.location = ev.target_node
                        ev.state = 'IDLE'
                        self.idle_cooldowns[ev.id] = 0

                if ev.state == 'IDLE':
                    # ΔΙΟΡΘΩΣΗ #4: Το cooldown μειώνεται μόνο για IDLE taxis
                    if self.idle_cooldowns[ev.id] > 0:
                        self.idle_cooldowns[ev.id] -= 1
                    elif ev not in self.taxis_needing_action:
                        self.taxis_needing_action.append(ev)
                # ΔΙΟΡΘΩΣΗ #4: Αφαιρούμε το λάθος else που μηδένιζε το cooldown
                # για οποιοδήποτε taxi που δεν ήταν IDLE

                if ev.state == 'CHARGING':
                    p = self.city.charger_specs[ev.charger_type]['power']
                    cost = self.city.get_electricity_price(self.current_minute, ev.charger_type)
                    added_kwh = ev.charge(p, cost)
                    self.total_energy_kwh += added_kwh

                    if ev.state == 'IDLE':  # Ολοκλήρωσε τη φόρτιση
                        self.city.release_charger(ev.target_station_idx, ev.charger_type)

            self.current_minute += 1

            if self.taxis_needing_action:
                return total_abandoned_this_loop

        return total_abandoned_this_loop