import numpy as np
import osmnx as ox
import networkx as nx
import os
import pickle

ox.settings.timeout = 600         
ox.settings.memory = 1073741824    

class CityMap:
    def __init__(self, center_point=(40.6264, 22.9484), radius_meters=5500, num_stations=6, num_hubs=2):
        """
        Initializes a real-world city environment using OSMnx for Thessaloniki.
        """
        self.center_point = center_point
        self.radius = radius_meters
        self.num_stations = num_stations
        self.num_hubs = num_hubs
        
        # Power specs and pricing multipliers (in kW)
        self.charger_specs = {
            'fast': {'power': 50.0,}, 
            'slow': {'power': 11.0,} 
        }
        
        
        self.prices = {
            'fast': {'peak': 0.65, 'off_peak': 0.50},
            'slow': {'peak': 0.45, 'off_peak': 0.35}
        }
        
        
        # 1. Φόρτωση ή Λήψη του Χάρτη
        self.G = self._load_or_download_graph()
        self.nodes = list(self.G.nodes())
        
        # 2. Δημιουργία Σταθμών σε πραγματικούς κόμβους
        self.stations = []
        self._generate_stations_on_graph()
        
        # 3. Υπολογισμός και αποθήκευση αποστάσεων (Για ταχύτητα στο RL)
        self.distance_cache = {}
        self._precompute_station_distances()

    def _load_or_download_graph(self):
        """Κατεβάζει τον χάρτη ή τον φορτώνει από τοπικό αρχείο για να γλιτώνει χρόνο."""
        filepath = f"cache/thessaloniki_{self.radius}m.graphml"
        os.makedirs("cache", exist_ok=True)
        
        if os.path.exists(filepath):
            print(f"--- Φόρτωση χάρτη πόλης από την cache: {filepath} ---")
            return ox.load_graphml(filepath)
        else:
            print(f"--- Λήψη χάρτη πόλης (Ακτίνα: {self.radius}m) ---")
            G = ox.graph_from_point(self.center_point, dist=self.radius, network_type="drive")
            ox.save_graphml(G, filepath)
            return G

    def _generate_stations_on_graph(self):
        """Τοποθετεί τους σταθμούς σε πραγματικές διασταυρώσεις (nodes)."""
        print(f"--- Δημιουργία {self.num_stations} Σταθμών στον Χάρτη ---")
        
        # Σταθερό seed για να τοποθετούνται οι σταθμοί στα ίδια σημεία σε κάθε επεισόδιο
        np.random.seed(42) 
        station_nodes = np.random.choice(self.nodes, self.num_stations, replace=False)
        np.random.seed(None) # Επαναφορά του seed για το υπόλοιπο simulation

        # --- SUPER HUBS ---
        for i in range(self.num_hubs):
            node_id = station_nodes[i]
            self.create_station(
                s_id=i, 
                location_node=node_id, 
                p_max=1500.0, 
                n_fast=5, 
                n_slow=5, 
                s_type="SUPER-HUB"
            )

        # --- NORMAL STATIONS ---
        for i in range(self.num_hubs, self.num_stations):
            node_id = station_nodes[i]
            self.create_station(
                s_id=i, 
                location_node=node_id, 
                p_max=300.0, 
                n_fast=2, 
                n_slow=2, 
                s_type="Normal"
            )

    def _precompute_station_distances(self):
        """
        Υπολογίζει τη συντομότερη διαδρομή από ΚΑΘΕ κόμβο της πόλης προς ΚΑΘΕ σταθμό.
        Παίρνει λίγο χρόνο στην αρχή, αλλά κάνει το AI 1000 φορές πιο γρήγορο.
        """
        cache_file = f"cache/distance_cache_{self.radius}m_{self.num_stations}s.pkl"
        
        if os.path.exists(cache_file):
            print("--- Φόρτωση προ-υπολογισμένων αποστάσεων από την cache ---")
            with open(cache_file, 'rb') as f:
                self.distance_cache = pickle.load(f)
            return

        print("--- Υπολογισμός αποστάσεων (Αυτό θα πάρει 1-2 λεπτά την πρώτη φορά...) ---")
        station_nodes = [st['location'] for st in self.stations]
        
        for st_id, target_node in enumerate(station_nodes):
            self.distance_cache[st_id] = {}
            # Υπολογισμός Dijkstra από τον σταθμό προς όλο το δίκτυο
            paths = nx.shortest_path_length(self.G, target=target_node, weight='length')
            
            for source_node, distance_meters in paths.items():
                self.distance_cache[st_id][source_node] = distance_meters / 1000.0 # Μετατροπή σε km
                
        with open(cache_file, 'wb') as f:
            pickle.dump(self.distance_cache, f)
        print("--- Ο υπολογισμός αποστάσεων ολοκληρώθηκε! ---")

    def get_driving_distance_km(self, start_node, station_id):
        """Επιστρέφει ακαριαία την πραγματική απόσταση οδήγησης."""
        try:
            return self.distance_cache[station_id][start_node]
        except KeyError:
            # Σε περίπτωση που ο κόμβος είναι αποκομμένος (π.χ. σε νησίδα), επιστρέφει μεγάλη ποινή
            return 15.0 

    def get_electricity_price(self, time_minutes, charger_type='slow'):
        """
        Υπολογίζει την τιμή της kWh βάσει πραγματικών τιμολογίων (Ώρα + Τύπος).
        """
        
        is_peak = 540 <= time_minutes < 1320
        
        if is_peak:
            return self.prices[charger_type]['peak']
        else:
            return self.prices[charger_type]['off_peak']

    def create_station(self, s_id, location_node, p_max, n_fast, n_slow, s_type):
        station = {
            'id': s_id,
            'type': s_type,
            'location': location_node,
            'p_max': float(p_max),
            'current_load': 0.0,
            'chargers': {'fast': n_fast, 'slow': n_slow},
            'occupied': {'fast': 0, 'slow': 0},
            'queue_length': 0  
        }
        self.stations.append(station)
        print(f"[{s_type}] ID {s_id}: Node {location_node} | Fast: {n_fast}, Slow: {n_slow} | Max Power: {p_max}kW")

    # --- QUEUE & CHARGER MANAGEMENT METHODS ---
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

    def get_closest_stations(self, ev_location_node, n=3):
        """Επιστρέφει τους n πιο κοντινούς σταθμούς με βάση την πραγματική απόσταση οδήγησης."""
        distances = []
        for station in self.stations:
            dist = self.get_driving_distance_km(ev_location_node, station['id'])
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