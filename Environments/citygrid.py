import numpy as np

class CityGrid:
    def __init__(self, size_km=5, num_stations=12, num_hubs=2, global_power_limit=150000, min_dist_km=0.5):
        """
        Initializes a Micro-City environment.
        
        Args:
            size_km : City size 
            num_stations : Total number of stations 
            num_hubs : Total number of super-hubs
            global_power_limit : Grid power threshold
            min_dist_km : Min distance between stations 
        """
        self.size = size_km
        self.num_stations = num_stations
        self.num_hubs = num_hubs
        self.global_limit = global_power_limit
        self.min_dist = min_dist_km
        
        # Power specs and pricing multipliers
        self.charger_specs = {
            'fast': {'power': 50.0, 'price_multiplier': 1.5}, 
            'slow': {'power': 11.0, 'price_multiplier': 1.0} 
        }
        
        self.peak_price = 0.30  
        self.off_peak_price = 0.12 
        
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
        # Fewer anchor points for hubs in a smaller map
        hub_centers = [
            [center, center],           
            [center * 0.5, center * 0.5]
        ]

        # --- SUPER HUBS ---
        for i in range(self.num_hubs):
            base_loc = hub_centers[i % len(hub_centers)]
            # Reduced spread for smaller city
            location = self.get_valid_location(base_center=base_loc, spread=0.7)
            self.create_station(i, location, p_max=150000, n_fast=50, n_slow=100, s_type="SUPER-HUB")

        # --- NORMAL STATIONS ---
        for i in range(self.num_hubs, self.num_stations):
            location = self.get_valid_location(base_center=None, spread=None)
 
            self.create_station(i, location, p_max=10000, n_fast=15, n_slow=30, s_type="Normal")

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
        """Creates station with current_load tracker."""
        station = {
            'id': s_id,
            'type': s_type,
            'location': location,
            'p_max': float(p_max),
            'current_load': 0.0,
            'chargers': {'fast': n_fast, 'slow': n_slow},
            'occupied': {'fast': 0, 'slow': 0} 
        }
        self.stations.append(station)
        print(f"[{s_type}] ID {s_id}: Pos {location.round(1)} | Fast: {n_fast}, Slow: {n_slow}")

    def get_closest_stations(self, ev_location, n=3):
        """Finds N closest stations via Manhattan distance."""
        distances = []
        for station in self.stations:
            diff = np.abs(ev_location - station['location'])
            dist = diff[0] + diff[1] 
            distances.append((station, dist))
        
        distances.sort(key=lambda x: x[1])
        return distances[:n]

    def check_availability(self, station_id, charger_type):
        """Checks if a plug is available."""
        st = self.stations[station_id]
        return st['occupied'][charger_type] < st['chargers'][charger_type]
    
    def get_all_station_ids(self):
        """Helper for the PuLP solver to get all IDs."""
        return [st['id'] for st in self.stations]
        
    def get_station_capacity(self, station_id):
        """Helper for the PuLP solver to enforce local limits."""
        for st in self.stations:
            if st['id'] == station_id:
                return st['p_max']
        return 0.0