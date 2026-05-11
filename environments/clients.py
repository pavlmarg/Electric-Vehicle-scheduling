import numpy as np

class ClientManager:
    def __init__(self, city_map):
        self.city = city_map
        self.waitlist = []
        
        # Το κέντρο του χάρτη 
        self.center_x = self.city.width_km / 2.0
        self.center_y = self.city.height_km / 2.0

    def _get_random_point(self, region='center'):
        """Δίνει ένα τυχαίο (X, Y) στο κέντρο ή στα περίχωρα, ΚΟΥΜΠΩΜΕΝΟ ΣΤΟ ΠΛΕΓΜΑ"""
        while True:
            # Χρησιμοποιούμε το _snap_to_grid της πόλης!
            x = self.city._snap_to_grid(np.random.uniform(0.0, self.city.width_km))
            y = self.city._snap_to_grid(np.random.uniform(0.0, self.city.height_km))
            
            dist_from_center = np.sqrt((x - self.center_x)**2 + (y - self.center_y)**2)
            
            if region == 'center' and dist_from_center <= 5.0:
                return (x, y)
            elif region == 'periphery' and dist_from_center > 5.0:
                return (x, y)

    def generate_new_demands(self, current_time_mins):
        """Δημιουργεί νέους πελάτες βάσει καμπύλης ζήτησης"""
        hour = (current_time_mins // 60) % 24

        demand_profile = [
        4, 2, 1, 1, 2, 5,       # 00:00 - 05:00
        15, 35, 45, 30, 22, 24, # 06:00 - 11:00 (Πρωινή Αιχμή στις 08:00)
        25, 25, 22, 28, 40, 45, # 12:00 - 17:00 (Απογευματινή Αιχμή)
        35, 28, 20, 15, 10, 6   # 18:00 - 23:00
    ]
        
        mean_demand = demand_profile[hour]
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
        """Ταιριάζει πελάτες με τον στόλο επιλέγοντας το κοντινότερο ταξί (Spatial Matching)"""
        avg_speed_kmh = 35.0
        speed_km_min = avg_speed_kmh / 60.0
        
        # ΠΡΟΣΤΕΘΗΚΕ Ο ΚΑΝΟΝΑΣ: AND t.current_soc > 0.10
        available_taxis = [t for t in fleet if t.state in ['IDLE', 'REBALANCING'] and t.current_soc > 0.0]
        
        ratings_this_minute = []
        abandoned_count = 0

        # Όσο έχουμε πελάτες και διαθέσιμα ταξί
        while self.waitlist and available_taxis:
            # Βλέπουμε τον παλαιότερο πελάτη στην ουρά
            customer = self.waitlist[0]
            
            best_taxi = None
            min_dist = float('inf')
            
            # Εύρεση του πιο κοντινού ταξί στον πελάτη
            for taxi in available_taxis:
                dist = self.city.calculate_manhattan_dist(taxi.location, customer['spawn_pos'])
                if dist < min_dist:
                    min_dist = dist
                    best_taxi = taxi
            
            if best_taxi:
                # Αφαιρούμε τον πελάτη από την ουρά και το ταξί από τα διαθέσιμα
                self.waitlist.pop(0)
                available_taxis.remove(best_taxi)
                
                # --- ΥΠΟΛΟΓΙΣΜΟΙ ΤΑΞΙ (Χρόνος να φτάσει στον πελάτη) ---
                pickup_dist_km = min_dist
                pickup_duration_mins = int(pickup_dist_km / speed_km_min)
                
                # --- ΝΕΑ ΑΞΙΟΛΟΓΗΣΗ ΠΕΛΑΤΗ (Ρεαλιστική) ---
                # Χρόνος εφαρμογής (συνήθως 0) + Χρόνος Οδήγησης Ταξί
                dispatch_wait_time = current_time_mins - customer['spawn_time']
                total_wait_time = dispatch_wait_time + pickup_duration_mins
                
                if total_wait_time <= 6: stars = 5
                elif total_wait_time <= 10: stars = 4
                elif total_wait_time <= 15: stars = 3
                elif total_wait_time <= 20: stars = 2
                else: stars = 1 
                    
                ratings_this_minute.append(stars)
                
                # Ο πελάτης πληρώνει ΜΟΝΟ για τη δική του διαδρομή
                fare_eur = max(4.00, 1.80 + (customer['distance_km'] * 0.90))
                
                # Το ταξί διανύει (και καίει ρεύμα) για: Απόσταση παραλαβής + Διαδρομή Πελάτη
                total_trip_dist = pickup_dist_km + customer['distance_km']
                total_duration_mins = pickup_duration_mins + int(customer['distance_km'] / speed_km_min) + 2 
                
                best_taxi.start_customer_trip(
                    destination_pos=customer['destination_pos'],
                    distance_km=total_trip_dist, 
                    duration_mins=total_duration_mins, 
                    fare_eur=fare_eur,
                    current_time=current_time_mins
                )

        # Εκκαθάριση εγκαταλελειμμένων: Όσοι περίμεναν > 15 λεπτά για να τους βρει σύστημα ταξί
        original_count = len(self.waitlist)
        self.waitlist = [c for c in self.waitlist if (current_time_mins - c['spawn_time']) <= 20]
        abandoned_count = original_count - len(self.waitlist)
        
        return ratings_this_minute, abandoned_count