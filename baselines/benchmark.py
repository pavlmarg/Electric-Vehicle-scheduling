import numpy as np

class GreedyHeuristicBaseline:
    def __init__(self, city_map):
        """
        Ευρετικός αλγόριθμος διαχείρισης στόλου με Προληπτική Αναδιάταξη (Rebalancing).
        """
        self.city = city_map
        # Το απόλυτο μαθηματικό κέντρο (χρησιμοποιείται μόνο για να δούμε αν το ταξί έχει απομακρυνθεί)
        self.center_pos = (self.city.width_km / 2.0, self.city.height_km / 2.0)

    def _get_random_center_pos(self):
        """
        Επιστρέφει μια τυχαία τοποθεσία κοντά στο κέντρο (σε ακτίνα ~3km),
        κουμπωμένη πάνω στις διασταυρώσεις της πόλης.
        """
        # Διαλέγουμε τυχαία X και Y από το 7.0 έως το 13.0
        x = np.random.uniform(7.0, 13.0)
        y = np.random.uniform(7.0, 13.0)
        
        # Αν η πόλη έχει τη λειτουργία πλέγματος (που την έχει), κουμπώνουμε το σημείο!
        if hasattr(self.city, '_snap_to_grid'):
            x = self.city._snap_to_grid(x)
            y = self.city._snap_to_grid(y)
            
        return (x, y)

    def route_ev(self, ev):
        """
        Αποφασίζει αν το ταξί πρέπει να Φορτίσει, να κάνει Rebalance στο κέντρο, ή να μείνει IDLE.
        Επιστρέφει (Action, Target_Pos, Απόσταση_km, Διάρκεια_mins)
        """
        # 1. ΠΡΟΤΕΡΑΙΟΤΗΤΑ: ΕΠΙΒΙΩΣΗ (Φόρτιση)
        if ev.current_soc <= 0.25:
            return self._find_best_station(ev)
            
        # 2. ΣΤΡΑΤΗΓΙΚΗ: REBALANCING ΣΕ ΖΩΝΗ ΚΕΝΤΡΟΥ
        dist_from_absolute_center = self.city.calculate_manhattan_dist(ev.location, self.center_pos)
        
        # Αν είναι μακριά από το κέντρο (> 5km) και έχει ρεύμα, γυρνάει πίσω!
        if dist_from_absolute_center > 5.0 and ev.current_soc > 0.40:
            
            # Βρίσκει έναν τυχαίο κόμβο-στόχο μέσα στο κέντρο για να παρκάρει
            target_rebalance_pos = self._get_random_center_pos()
            
            # Υπολογίζουμε την πραγματική απόσταση μέχρι αυτό το συγκεκριμένο σημείο
            actual_dist = self.city.calculate_manhattan_dist(ev.location, target_rebalance_pos)
            duration_mins = int(actual_dist / 0.5) + 1 
            
            # Επιστρέφουμε τη λέξη "REBALANCE" αντί για ID σταθμού
            return "REBALANCE", target_rebalance_pos, actual_dist, duration_mins
            
        # 3. ΠΑΡΑΜΕΝΕΙ ΣΤΗ ΘΕΣΗ ΤΟΥ (IDLE)
        return None, None, 0.0, 0

    def _find_best_station(self, ev):
        """Η παλιά καλή λογική της εύρεσης του καλύτερου σταθμού (Manhattan + Ουρά)"""
        best_station_idx = None
        best_station_pos = None  
        best_dist = 0.0
        best_score = float('inf')
        
        for station in self.city.stations:
            dist_km = self.city.get_driving_distance_km(ev.location, station['id'])
            queue = self.city.get_queue(station['id'])
            score = dist_km + (queue * 2.0) 
            
            if score < best_score:
                best_score = score
                best_station_idx = station['id']
                best_station_pos = station['location']  
                best_dist = dist_km
                
        duration_mins = int(best_dist / 0.5) + 1 
        return best_station_idx, best_station_pos, best_dist, duration_mins