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
        
        # Το κέντρο του χάρτη είναι ακριβώς στη μέση (10.0, 10.0 για 20x20 χάρτη)
        self.center_x = self.city.width_km / 2.0
        self.center_y = self.city.height_km / 2.0
        
        print(f"--- Traffic Generator Ready: Continuous Space {self.city.width_km}x{self.city.height_km} km ---")

    def _get_random_point(self, region='center'):
        """Βοηθητική συνάρτηση: Δίνει ένα τυχαίο (X, Y) στο κέντρο ή στα περίχωρα"""
        while True:
            x = np.random.uniform(0.0, self.city.width_km)
            y = np.random.uniform(0.0, self.city.height_km)
            
            # Υπολογίζουμε την ευθεία απόσταση από το κέντρο της πόλης
            dist_from_center = np.sqrt((x - self.center_x)**2 + (y - self.center_y)**2)
            
            # Αν θεωρήσουμε "Κέντρο" μια ακτίνα 5 χιλιομέτρων
            if region == 'center' and dist_from_center <= 5.0:
                return (x, y)
            elif region == 'periphery' and dist_from_center > 5.0:
                return (x, y)

    def generate_initial_fleet(self):
        """
        Δημιουργεί τα ταξί σε τυχαίες θέσεις (X,Y) στην αρχή της ημέρας.
        """
        self.fleet = []
        print(f"--- Spawning Fleet of {self.num_vehicles} EV Taxis ---")
        
        for i in range(self.num_vehicles):
            # Τυχαία θέση οπουδήποτε στον χάρτη
            x = np.random.uniform(0.0, self.city.width_km)
            y = np.random.uniform(0.0, self.city.height_km)
            
            taxi = EVTaxi(taxi_id=i, start_pos=(x, y))
            taxi.current_soc = np.random.uniform(0.30, 1.0)
            self.fleet.append(taxi)
            
        return self.fleet

    def generate_new_demands(self, current_time_mins):
        """
        Δημιουργεί νέους πελάτες βάσει ρεαλιστικής καμπύλης ζήτησης και κατευθυντικών ροών.
        """
        hour = (current_time_mins // 60) % 24

        # 1. Κατανομή ζήτησης ανά ώρα (Πελάτες ανά λεπτό για το 24ωρο)
        # Συνολικά βγάζει ~29.000 πελάτες τη μέρα (όσους ακριβώς έβγαζε και το παλιό σύστημα, αλλά σωστά κατανεμημένους)
        demand_profile = [
            4, 2, 1, 1, 2, 5,       # 00:00 - 05:59 (Νύχτα - Ξημερώματα)
            15, 35, 45, 30, 22, 24, # 06:00 - 11:59 (Πρωινή αιχμή & Πρωί)
            25, 25, 22, 28, 40, 45, # 12:00 - 17:59 (Μεσημέρι & Απογευματινή αιχμή)
            35, 28, 20, 15, 10, 6   # 18:00 - 23:59 (Βράδυ)
        ]
        
        mean_demand = demand_profile[hour]
        demand_count = np.random.poisson(mean_demand)

        # 2. Προσδιορισμός τάσεων ροής με βάση την ώρα
        # Πιθανότητες για: [Κέντρο->Κέντρο, Κέντρο->Περίχωρα, Περίχωρα->Κέντρο, Περίχωρα->Περίχωρα]
        if 6 <= hour <= 11:
            # Πρωί: Ο κόσμος πάει από τα Περίχωρα στο Κέντρο (για δουλειά)
            trip_probs = [0.35, 0.10, 0.45, 0.10]
        elif 15 <= hour <= 20:
            # Απόγευμα/Βράδυ: Ο κόσμος επιστρέφει σπίτι (Κέντρο προς Περίχωρα)
            trip_probs = [0.35, 0.45, 0.10, 0.10]
        else:
            # Μεσημέρι & Νύχτα: Πιο ισορροπημένη κίνηση, κυρίως μέσα στο κέντρο
            trip_probs = [0.55, 0.15, 0.15, 0.15]

        trip_types = ['CC', 'CP', 'PC', 'PP']

        for _ in range(demand_count):
            # Διαλέγουμε τύπο διαδρομής βάσει των πιθανοτήτων της τρέχουσας ώρας
            trip_type = np.random.choice(trip_types, p=trip_probs)

            dist_km = 0.0
            attempts = 0
            
            # 3. Εξασφάλιση πραγματικής απόστασης > 0.5km (μέχρι 10 προσπάθειες για να μη κολλήσει)
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
                else: # 'PP'
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

    def process_waitlist(self, current_time_mins):
        """
        Ταιριάζει πελάτες από την ουρά με τα IDLE ταξί.
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
                destination_pos=customer['destination_pos'],
                distance_km=customer['distance_km'],
                duration_mins=duration_mins,
                fare_eur=fare_eur,
                current_time=current_time_mins
            )

        original_count = len(self.waitlist)
        self.waitlist = [c for c in self.waitlist if (current_time_mins - c['spawn_time']) <= 15]
        abandoned_count = original_count - len(self.waitlist)
        
        return ratings_this_minute, abandoned_count