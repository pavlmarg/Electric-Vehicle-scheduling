import numpy as np
from environments.ev import EVTaxi 

class TrafficGenerator:
    def __init__(self, city_map, num_vehicles=400):
        """
        Διαχειρίζεται τον στόλο, τη ζήτηση και την ουρά αναμονής πελατών.
        """
        self.city = city_map
        self.num_vehicles = num_vehicles
        self.fleet = []
        self.waitlist = [] 
        
        # --- Γεωγραφικός Διαχωρισμός Κόμβων (Κέντρο vs Περίχωρα) ---
        self.center_nodes = []
        self.periphery_nodes = []
        
        center_lat, center_lon = self.city.center_point # (40.6264, 22.9484)
        
        for node_id, data in self.city.G.nodes(data=True):
            # Προσεγγιστικός υπολογισμός απόστασης (1 μοίρα = ~111 km)
            dist_km = np.sqrt((data['y'] - center_lat)**2 + (data['x'] - center_lon)**2) * 111.0
            
            if dist_km <= 2.5: # Ακτίνα 2.5km θεωρείται Κέντρο
                self.center_nodes.append(node_id)
            else:
                self.periphery_nodes.append(node_id)
                
        print(f"--- Map Loaded: {len(self.center_nodes)} Center Nodes | {len(self.periphery_nodes)} Periphery Nodes ---")

    def generate_initial_fleet(self):
        """
        Δημιουργεί τα ταξί στους κόμβους του χάρτη στην αρχή της ημέρας.
        """
        self.fleet = []
        print(f"--- Spawning Fleet of {self.num_vehicles} EV Taxis ---")
        
        # Αρχικά μοιράζουμε τα ταξί σε όλη την πόλη τυχαία
        available_nodes = self.city.nodes

        for i in range(self.num_vehicles):
            start_node = np.random.choice(available_nodes)
            taxi = EVTaxi(taxi_id=i, start_node=start_node)
            taxi.current_soc = np.random.uniform(0.30, 1.0)
            self.fleet.append(taxi)
            
        return self.fleet

    def generate_new_demands(self, current_time_mins):
        """
        Δημιουργεί νέους πελάτες βάσει ώρας με Poisson κατανομή και γεωγραφικούς κανόνες.
        """
        hour = (current_time_mins // 60) % 24
        is_rush_hour = (7 <= hour <= 9) or (16 <= hour <= 19)
        
        # Μέσος όρος πελατών ανά λεπτό
        mean_demand = 30 if is_rush_hour else 18
        
        # Χρήση κατανομής Poisson για ρεαλιστική τυχαιότητα
        demand_count = np.random.poisson(mean_demand)

        for _ in range(demand_count):
            # 1. Πού θα εμφανιστεί ο πελάτης; (70% πιθανότητα στο Κέντρο)
            if np.random.rand() < 0.70 and self.center_nodes:
                spawn_node = np.random.choice(self.center_nodes)
                
                # 2. Πού θέλει να πάει; (Από το Κέντρο: 60% μένει Κέντρο, 40% πάει Περίχωρα)
                if np.random.rand() < 0.60:
                    dest_node = np.random.choice(self.center_nodes)
                    dist_km = np.random.uniform(1.5, 4.0) # Μικρή απόσταση
                else:
                    dest_node = np.random.choice(self.periphery_nodes)
                    dist_km = np.random.uniform(3.0, 7.0) # Μεσαία απόσταση
            else:
                # Εμφάνιση στα Περίχωρα (30% πιθανότητα)
                spawn_node = np.random.choice(self.periphery_nodes)
                
                # 2. Πού θέλει να πάει; (Από τα Περίχωρα: 80% θέλει να κατέβει Κέντρο!)
                if np.random.rand() < 0.80:
                    dest_node = np.random.choice(self.center_nodes)
                    dist_km = np.random.uniform(3.0, 8.0) # Μεσαία/Μεγάλη απόσταση
                else:
                    dest_node = np.random.choice(self.periphery_nodes)
                    dist_km = np.random.uniform(4.0, 11.0) # Μεγάλη απόσταση

            customer = {
                'spawn_time': current_time_mins,
                'spawn_node': spawn_node,
                'destination_node': dest_node,
                'distance_km': dist_km
            }
            self.waitlist.append(customer)

    def process_waitlist(self, current_time_mins):
        """
        Ταιριάζει πελάτες από την ουρά με τα IDLE ταξί και υπολογίζει τα Αστέρια Αξιολόγησης.
        """
        hour = (current_time_mins // 60) % 24
        is_rush_hour = (7 <= hour <= 9) or (16 <= hour <= 19)
        avg_speed_kmh = 18.0 if is_rush_hour else 35.0
        speed_km_min = avg_speed_kmh / 60.0

        
        available_taxis = [t for t in self.fleet if t.state in ['IDLE', 'REBALANCING']]
        np.random.shuffle(available_taxis)
        
        ratings_this_minute = []
        abandoned_count = 0

        for taxi in available_taxis:
            if not self.waitlist:
                break
            
            customer = self.waitlist.pop(0) 
            wait_time = current_time_mins - customer['spawn_time']
            
            if wait_time <= 3:
                stars = 5
            elif wait_time <= 7:
                stars = 4
            elif wait_time <= 11:
                stars = 3
            elif wait_time <= 15:
                stars = 2
            else:
                stars = 1 
                
            ratings_this_minute.append(stars)
            
            duration_mins = int(customer['distance_km'] / speed_km_min) + 2 
            fare_eur = max(4.00, 1.80 + (customer['distance_km'] * 0.90))
            
            taxi.start_customer_trip(
                destination_node=customer['destination_node'],
                distance_km=customer['distance_km'],
                duration_mins=duration_mins,
                fare_eur=fare_eur,
                current_time=current_time_mins
            )

        original_count = len(self.waitlist)
        self.waitlist = [c for c in self.waitlist if (current_time_mins - c['spawn_time']) <= 15]
        abandoned_count = original_count - len(self.waitlist)
        
        return ratings_this_minute, abandoned_count