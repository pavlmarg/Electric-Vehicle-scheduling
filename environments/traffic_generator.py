import numpy as np
# Προσοχή: Σιγουρέψου ότι το αρχείο των οχημάτων λέγεται ev.py (ή ev_taxi.py) 
# και έχει μέσα την κλάση EVTaxi που γράψαμε πριν!
from environments.ev import EVTaxi 

class TrafficGenerator:
    def __init__(self, city_map, num_vehicles=250):
        """
        Διαχειρίζεται τον στόλο και την παραγωγή ζήτησης πελατών (AMoD Dispatching).
        """
        self.city = city_map
        self.num_vehicles = num_vehicles
        self.fleet = []

    def generate_initial_fleet(self):
        """
        Δημιουργεί τα ταξί στους κόμβους (nodes) του χάρτη στην αρχή της ημέρας.
        """
        self.fleet = []
        print(f"--- Spawning Fleet of {self.num_vehicles} EV Taxis ---")
        
        # Λίστα με όλες τις διασταυρώσεις της Θεσσαλονίκης
        available_nodes = self.city.nodes

        for i in range(self.num_vehicles):
            # Τυχαίος κόμβος εκκίνησης
            start_node = np.random.choice(available_nodes)
            
            # Δημιουργία του Nissan Leaf Ταξί
            taxi = EVTaxi(taxi_id=i, start_node=start_node)
            
            # Τυχαία αρχική μπαταρία 30% - 100% για να μην πάνε όλα μαζί για φόρτιση στο 1ο λεπτό!
            taxi.current_soc = np.random.uniform(0.30, 1.0)
            
            self.fleet.append(taxi)
            
        return self.fleet

    def dispatch_rides(self, current_time_mins):
        """
        Προσομοιώνει τη ζήτηση πελατών και αναθέτει κούρσες στα ελεύθερα ταξί.
        Καλείται ΚΑΘΕ ΛΕΠΤΟ από το περιβάλλον (ev_gym_env.py).
        """
        # 1. Υπολογισμός Κίνησης & Ζήτησης βάσει Ώρας (Rush hours: 07:00-09:00 και 16:00-19:00)
        hour = (current_time_mins // 60) % 24
        is_rush_hour = (7 <= hour <= 9) or (16 <= hour <= 19)
        
        # Πόσοι πελάτες εμφανίζονται αυτό το λεπτό στην πόλη
        base_demand = 8 
        demand = int(base_demand * 2.5) if is_rush_hour else base_demand
        
        # Μέση ταχύτητα πόλης (km/h) -> Μετατροπή σε km/min
        avg_speed_kmh = 18.0 if is_rush_hour else 35.0
        speed_km_min = avg_speed_kmh / 60.0

        # 2. Βρίσκουμε ποια ταξί είναι ελεύθερα ('IDLE')
        idle_taxis = [taxi for taxi in self.fleet if taxi.state == 'IDLE']
        
        # Αν δεν υπάρχουν ελεύθερα ταξί, οι πελάτες χάνονται
        if not idle_taxis:
            return
            
        # Ανακατεύουμε τα ελεύθερα ταξί για δίκαιη κατανομή
        np.random.shuffle(idle_taxis)

        # 3. Ανάθεση Κουρσών
        rides_assigned = 0
        for taxi in idle_taxis:
            if rides_assigned >= demand:
                break # Εξυπηρετήθηκαν όλοι οι πελάτες αυτού του λεπτού
                
            # --- Δημιουργία Στατιστικής Κούρσας ---
            # Επιλέγουμε τυχαίο προορισμό στον χάρτη
            destination_node = np.random.choice(self.city.nodes)
            
            # ΣΗΜΑΝΤΙΚΟ: Δεν τρέχουμε Dijkstra για τον πελάτη γιατί θα γονατίσει το PC.
            # Υποθέτουμε μια ρεαλιστική απόσταση 2 έως 12 km για τη Θεσσαλονίκη.
            distance_km = np.random.uniform(2.0, 12.0)
            
            # Υπολογισμός χρόνου που θα δεσμευτεί το ταξί
            duration_mins = int(distance_km / speed_km_min) + 2 # +2 λεπτά για επιβίβαση
            
            # Έσοδα κούρσας (1.80 Πτώση σημαίας + 0.90/km | Ελάχιστη 4.00 ευρώ)
            fare_eur = max(4.00, 1.80 + (distance_km * 0.90))
            
            # Αναθέτουμε την κούρσα στο ταξί!
            taxi.start_customer_trip(
                destination_node=destination_node,
                distance_km=distance_km,
                duration_mins=duration_mins,
                fare_eur=fare_eur,
                current_time=current_time_mins
            )
            
            rides_assigned += 1