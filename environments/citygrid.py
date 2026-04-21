import random

class CityMap:
    def __init__(self, width_km=20.0, height_km=20.0, num_stations=2, num_hubs=1): 
        self.width_km = width_km
        self.height_km = height_km
        self.num_stations = num_stations
        self.num_hubs = num_hubs
        
        self.charger_specs = {
            'fast': {'power': 50.0}, 
            'slow': {'power': 11.0} 
        }
        
        self.prices = {
            'fast': {'peak': 0.65, 'off_peak': 0.50},
            'slow': {'peak': 0.45, 'off_peak': 0.35}
        }
        
        self.stations = []
        self._generate_stations()

    def _generate_stations(self):
        print(f"--- Δημιουργία {self.num_stations} Σταθμών στον Συνεχή Χώρο ({self.width_km}x{self.height_km} km) ---")
        
        
        random.seed(50)
        # Super-Hubs
        for i in range(self.num_hubs):
            # Τυχαίες δεκαδικές συντεταγμένες (π.χ. x=14.23, y=5.67)
            x = random.uniform(2.0, self.width_km - 2.0)
            y = random.uniform(2.0, self.height_km - 2.0)
            self.create_station(
                s_id=i, 
                position_xy=(x, y), 
                p_max=1500.0, 
                n_fast=5, 
                n_slow=5, 
                s_type="SUPER-HUB"
            )

        # Normal Stations
        for i in range(self.num_hubs, self.num_stations):
            x = random.uniform(1.0, self.width_km - 1.0)
            y = random.uniform(1.0, self.height_km - 1.0)
            self.create_station(
                s_id=i, 
                position_xy=(x, y), 
                p_max=300.0, 
                n_fast=2, 
                n_slow=2, 
                s_type="Normal"
            )
            
        random.seed(None)

    def get_driving_distance_km(self, start_pos, station_id):
        # ΑΣΤΡΑΠΙΑΙΑ MANHATTAN ΑΠΟΣΤΑΣΗ ΣΕ ΠΡΑΓΜΑΤΙΚΑ ΧΙΛΙΟΜΕΤΡΑ
        try:
            target_pos = self.stations[station_id]['location']
            dist_x = abs(start_pos[0] - target_pos[0])
            dist_y = abs(start_pos[1] - target_pos[1])
            return dist_x + dist_y
        except IndexError:
            return 15.0 
            
    def calculate_manhattan_dist(self, pos1, pos2):
        """Βοηθητική συνάρτηση για να μετράει το traffic_generator την απόσταση πελατών"""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def get_electricity_price(self, time_minutes, charger_type='slow'):
        
        minute_of_day = time_minutes % 1440 
        
        is_peak = 540 <= minute_of_day < 1320
        if is_peak:
            return self.prices[charger_type]['peak']
        else:
            return self.prices[charger_type]['off_peak']

    def create_station(self, s_id, position_xy, p_max, n_fast, n_slow, s_type):
        station = {
            'id': s_id,
            'type': s_type,
            'location': position_xy, 
            'p_max': float(p_max),
            'current_load': 0.0,
            'chargers': {'fast': n_fast, 'slow': n_slow},
            'occupied': {'fast': 0, 'slow': 0},
            'queue_length': 0  
        }
        self.stations.append(station)
        # Κάνουμε format τα floats για να τυπώνονται όμορφα (π.χ. 12.45)
        print(f"[{s_type}] ID {s_id}: Θέση ({position_xy[0]:.2f}, {position_xy[1]:.2f}) | Fast: {n_fast}, Slow: {n_slow}")

    def try_reserve_charger(self, station_id, preferred_type='fast'):
        st = self.stations[station_id]
        if st['occupied'][preferred_type] < st['chargers'][preferred_type]:
            return True
        fallback = 'slow' if preferred_type == 'fast' else 'fast'
        if st['occupied'][fallback] < st['chargers'][fallback]:
            return True
        return False

    def get_queue(self, station_id):
        return self.stations[station_id]['queue_length']

    def add_to_queue(self, station_id):
        self.stations[station_id]['queue_length'] += 1

    def remove_from_queue(self, station_id):
        if self.stations[station_id]['queue_length'] > 0:
            self.stations[station_id]['queue_length'] -= 1

    def occupy_charger(self, station_id, preferred_type='fast'):
        st = self.stations[station_id]
        if st['occupied'][preferred_type] < st['chargers'][preferred_type]:
            st['occupied'][preferred_type] += 1
            st['current_load'] += self.charger_specs[preferred_type]['power']
            return preferred_type
            
        fallback = 'slow' if preferred_type == 'fast' else 'fast'
        if st['occupied'][fallback] < st['chargers'][fallback]:
            st['occupied'][fallback] += 1
            st['current_load'] += self.charger_specs[fallback]['power']
            return fallback
            
        return False 

    def release_charger(self, station_id, charger_type):
        st = self.stations[station_id]
        if st['occupied'][charger_type] > 0:
            st['occupied'][charger_type] -= 1
            st['current_load'] -= self.charger_specs[charger_type]['power']
            if st['current_load'] < 0: 
                st['current_load'] = 0.0

    def get_closest_stations(self, ev_location, n=3):
        distances = []
        for station in self.stations:
            dist = self.get_driving_distance_km(ev_location, station['id'])
            distances.append((station, dist))
        distances.sort(key=lambda x: x[1])
        return distances[:n]

    def check_availability(self, station_id, charger_type):
        st = self.stations[station_id]
        return st['occupied'][charger_type] < st['chargers'][charger_type]