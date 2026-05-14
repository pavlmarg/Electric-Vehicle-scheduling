import numpy as np
import random

class ClientManager:
    def __init__(self, city_map, seed=None):
        self.city = city_map
        self.waitlist = []
        
        # Αν περαστεί seed, το εφαρμόζουμε παγκόσμια 
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        self.center_x = self.city.width_km / 2.0
        self.center_y = self.city.height_km / 2.0

        # --- ΤΑ 10 ΠΡΟΦΙΛ ΖΗΤΗΣΗΣ ---
        self.all_profiles = [
            [4, 2, 1, 1, 2, 5, 15, 35, 45, 30, 22, 24, 25, 25, 22, 28, 40, 45, 35, 28, 20, 15, 10, 6],
            [3, 2, 1, 1, 3, 10, 30, 55, 60, 40, 20, 15, 18, 15, 15, 20, 30, 35, 30, 25, 15, 10, 5, 3],
            [15, 10, 5, 2, 1, 2, 10, 20, 25, 25, 25, 25, 30, 30, 30, 35, 45, 50, 60, 55, 45, 35, 25, 20],
            [10, 8, 5, 2, 1, 1, 2, 5, 10, 15, 25, 35, 40, 45, 45, 40, 35, 30, 25, 20, 15, 10, 8, 5],
            [2, 1, 1, 1, 1, 3, 10, 25, 30, 20, 15, 15, 18, 18, 15, 18, 25, 30, 25, 20, 15, 10, 5, 3],
            [5, 3, 2, 2, 5, 15, 25, 45, 60, 45, 35, 35, 35, 35, 35, 45, 60, 65, 55, 45, 30, 20, 15, 10],
            [10, 8, 5, 5, 8, 15, 25, 28, 30, 28, 25, 25, 25, 25, 25, 28, 30, 28, 25, 25, 20, 15, 12, 10],
            [2, 1, 1, 1, 5, 15, 30, 65, 70, 35, 15, 10, 10, 10, 10, 15, 35, 70, 65, 30, 15, 10, 5, 2],
            [4, 2, 1, 1, 2, 5, 15, 30, 40, 25, 20, 20, 25, 25, 22, 25, 30, 35, 30, 25, 20, 15, 65, 40],
            [4, 2, 1, 1, 2, 5, 15, 35, 45, 30, 22, 24, 25, 50, 65, 35, 30, 35, 30, 25, 20, 15, 10, 5]
        ]
        
        probabilities = [0.28] + [0.08] * 9
        chosen_index = np.random.choice(len(self.all_profiles), p=probabilities)
        self.current_profile = self.all_profiles[chosen_index]

    def _get_random_point(self, region='center'):
        while True:
            x = self.city._snap_to_grid(np.random.uniform(0.0, self.city.width_km))
            y = self.city._snap_to_grid(np.random.uniform(0.0, self.city.height_km))
            
            dist_from_center = np.sqrt((x - self.center_x)**2 + (y - self.center_y)**2)
            
            if region == 'center' and dist_from_center <= 5.0:
                return (x, y)
            elif region == 'periphery' and dist_from_center > 5.0:
                return (x, y)

    def generate_new_demands(self, current_time_mins):
        hour = (current_time_mins // 60) % 24
        mean_demand = self.current_profile[hour]
        demand_count = np.random.poisson(mean_demand)

        if 6 <= hour <= 11:
            trip_probs = [0.35, 0.10, 0.45, 0.10]
        elif 15 <= hour <= 20:
            trip_probs = [0.35, 0.45, 0.10, 0.10]
        else:
            trip_probs = [0.55, 0.15, 0.15, 0.15]

        trip_types = ['CC', 'CP', 'PC', 'PP']

        for _ in range(demand_count):
            trip_type = np.random.choice(trip_types, p=trip_probs)
            dist_km = 0.0
            attempts = 0
            
            while dist_km < 0.5 and attempts < 10:
                if trip_type == 'CC':
                    spawn_pos = self._get_random_point('center')
                    dest_pos = self._get_random_point('center')
                elif trip_type == 'CP':
                    spawn_pos = self._get_random_point('center')
                    dest_pos = self._get_random_point('periphery')
                elif trip_type == 'PC':
                    spawn_pos = self._get_random_point('periphery')
                    dest_pos = self._get_random_point('center')
                else: 
                    spawn_pos = self._get_random_point('periphery')
                    dest_pos = self._get_random_point('periphery')

                dist_km = self.city.calculate_manhattan_dist(spawn_pos, dest_pos)
                attempts += 1

            if dist_km < 0.5:
                dist_km = 0.5

            customer = {
                'spawn_time': current_time_mins,
                'spawn_pos': spawn_pos,
                'destination_pos': dest_pos,
                'distance_km': dist_km
            }
            self.waitlist.append(customer)

    def process_waitlist(self, current_time_mins, fleet):
        avg_speed_kmh = 35.0
        speed_km_min = avg_speed_kmh / 60.0
        
        available_taxis = [t for t in fleet if t.state in ['IDLE', 'REBALANCING'] and t.current_soc > 0.0]
        
        # ΑΛΛΑΓΗ 1: Μετονομασία της λίστας σε wait_times
        wait_times_this_minute = []
        abandoned_count = 0

        while self.waitlist and available_taxis:
            customer = self.waitlist[0]
            
            best_taxi = None
            min_dist = float('inf')
            
            for taxi in available_taxis:
                dist = self.city.calculate_manhattan_dist(taxi.location, customer['spawn_pos'])
                if dist < min_dist:
                    min_dist = dist
                    best_taxi = taxi
            
            if best_taxi:
                self.waitlist.pop(0)
                available_taxis.remove(best_taxi)
                
                pickup_dist_km = min_dist
                pickup_duration_mins = int(pickup_dist_km / speed_km_min)
                
                dispatch_wait_time = current_time_mins - customer['spawn_time']
                total_wait_time = dispatch_wait_time + pickup_duration_mins
                
                # ΑΛΛΑΓΗ 2: Κρατάμε απευθείας τον χρόνο αναμονής, τέρμα τα αστέρια
                wait_times_this_minute.append(total_wait_time)
                
                fare_eur = max(4.00, 1.80 + (customer['distance_km'] * 0.90))
                
                total_trip_dist = pickup_dist_km + customer['distance_km']
                total_duration_mins = pickup_duration_mins + int(customer['distance_km'] / speed_km_min) + 2 
                
                best_taxi.start_customer_trip(
                    destination_pos=customer['destination_pos'],
                    distance_km=total_trip_dist, 
                    duration_mins=total_duration_mins, 
                    fare_eur=fare_eur,
                    current_time=current_time_mins
                )

        original_count = len(self.waitlist)
        self.waitlist = [c for c in self.waitlist if (current_time_mins - c['spawn_time']) <= 20]
        abandoned_count = original_count - len(self.waitlist)
        
        # ΑΛΛΑΓΗ 3: Επιστρέφει τη νέα λίστα
        return wait_times_this_minute, abandoned_count