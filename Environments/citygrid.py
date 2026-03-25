import numpy as np

class CityGrid:
    def __init__(self, size_km=25, num_stations=12, num_hubs=2, global_power_limit=15000000, min_dist_km=0.5):
        """
        Initializes a Micro-City environment.
        """
        self.size = size_km
        self.num_stations = num_stations
        self.num_hubs = num_hubs
        self.global_limit = global_power_limit
        self.min_dist = min_dist_km
        
        # Power specs and pricing multipliers (in kW)
        self.charger_specs = {
            'fast': {'power': 50.0, 'price_multiplier': 1.5}, 
            'slow': {'power': 11.0, 'price_multiplier': 1.0} 
        }
        
        self.peak_price = 0.80  
        self.off_peak_price = 0.20 
        
        self.stations = []
        self.generate_stations()

    def get_electricity_price(self, time_minutes, charger_type='slow'):
        """Calculates price per kWh based on Time + Charger Type."""
        if 420 <= time_minutes < 1380:
            base_price = self.peak_price
        else:
            base_price = self.off_peak_price
            
        multiplier = self.charger_specs[charger_type]['price_multiplier']
        return base_price * multiplier

    def generate_stations(self):
        """Generates stations on the micro-map."""
        print(f"--- Generating Micro-City Map ---")
        
        center = self.size / 2
        hub_centers = [
            [center, center],           
            [center * 0.5, center * 0.5]
        ]

        # --- SUPER HUBS ---
        for i in range(self.num_hubs):
            base_loc = hub_centers[i % len(hub_centers)]
            location = self.get_valid_location(base_center=base_loc, spread=0.7)
            self.create_station(i, location, p_max=3600.0, n_fast=10, n_slow=30, s_type="SUPER-HUB")

        # --- NORMAL STATIONS ---
        for i in range(self.num_hubs, self.num_stations):
            location = self.get_valid_location(base_center=None, spread=None)
            self.create_station(i, location, p_max=1080.0, n_fast=5, n_slow=15, s_type="Normal")

    def get_valid_location(self, base_center=None, spread=None):
        """Finds location respecting min_dist using Manhattan distance."""
        max_attempts = 100
        for _ in range(max_attempts):
            if base_center is not None:
                noise = np.random.uniform(-spread, spread, size=2)
                candidate = np.array(base_center) + noise
            else:
                candidate = np.random.uniform(0, self.size, size=2)
            
            candidate = np.clip(candidate, 0, self.size)
            
            too_close = False
            for station in self.stations:
                diff = np.abs(candidate - station['location'])
                dist = diff[0] + diff[1]
                if dist < self.min_dist:
                    too_close = True
                    break
            
            if not too_close:
                return candidate
        
        return candidate

    def create_station(self, s_id, location, p_max, n_fast, n_slow, s_type):
        """Creates station with current_load and queue trackers."""
        station = {
            'id': s_id,
            'type': s_type,
            'location': location,
            'p_max': float(p_max),
            'current_load': 0.0,
            'chargers': {'fast': n_fast, 'slow': n_slow},
            'occupied': {'fast': 0, 'slow': 0},
            'queue_length': 0  
        }
        self.stations.append(station)
        print(f"[{s_type}] ID {s_id}: Pos {location.round(1)} | Fast: {n_fast}, Slow: {n_slow} | Max Power: {p_max}kW")

    # --- QUEUE & CHARGER MANAGEMENT METHODS ---
    def get_queue(self, station_id):
        return self.stations[station_id]['queue_length']

    def add_to_queue(self, station_id):
        self.stations[station_id]['queue_length'] += 1

    def remove_from_queue(self, station_id):
        if self.stations[station_id]['queue_length'] > 0:
            self.stations[station_id]['queue_length'] -= 1

    def occupy_charger(self, station_id, preferred_type='fast'):
        """
        Προσπαθεί να καταλάβει φορτιστή. Ψάχνει πρώτα τον preferred_type,
        αν δεν βρει πάει στον άλλο. Επιστρέφει τον τύπο που βρήκε (π.χ. 'fast') ή False.
        """
        st = self.stations[station_id]
        
        # Προσπάθεια για preferred (συνήθως 'fast')
        if st['occupied'][preferred_type] < st['chargers'][preferred_type]:
            st['occupied'][preferred_type] += 1
            st['current_load'] += self.charger_specs[preferred_type]['power']
            return preferred_type
            
        # Αν είναι γεμάτος ο fast, προσπάθεια για 'slow'
        fallback = 'slow' if preferred_type == 'fast' else 'fast'
        if st['occupied'][fallback] < st['chargers'][fallback]:
            st['occupied'][fallback] += 1
            st['current_load'] += self.charger_specs[fallback]['power']
            return fallback
            
        return False # Όλα γεμάτα

    def release_charger(self, station_id, charger_type):
        """Ελευθερώνει την πρίζα και μειώνει το ρεύμα (load)."""
        st = self.stations[station_id]
        if st['occupied'][charger_type] > 0:
            st['occupied'][charger_type] -= 1
            st['current_load'] -= self.charger_specs[charger_type]['power']
            # Ασφάλεια για floating point errors
            if st['current_load'] < 0: 
                st['current_load'] = 0.0

    # --- EXISTING HELPER METHODS ---
    def get_closest_stations(self, ev_location, n=3):
        distances = []
        for station in self.stations:
            diff = np.abs(ev_location - station['location'])
            dist = diff[0] + diff[1] 
            distances.append((station, dist))
        distances.sort(key=lambda x: x[1])
        return distances[:n]

    def check_availability(self, station_id, charger_type):
        st = self.stations[station_id]
        return st['occupied'][charger_type] < st['chargers'][charger_type]
    
    def get_all_station_ids(self):
        return [st['id'] for st in self.stations]
        
    def get_station_capacity(self, station_id):
        for st in self.stations:
            if st['id'] == station_id:
                return st['p_max']
        return 0.0