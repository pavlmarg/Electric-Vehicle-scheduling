import numpy as np

class GreedyHeuristicBaseline:
    def __init__(self, city_map):
        """
        Ευρετικός αλγόριθμος διαχείρισης στόλου.
        Λειτουργεί πλέον στον συνεχή μαθηματικό χώρο (Continuous Space 20x20km).
        """
        self.city = city_map

    def route_ev(self, ev):
        """
        Βρίσκει τον καλύτερο σταθμό με βάση τα πραγματικά χιλιόμετρα (Manhattan)
        και την ουρά. Επιστρέφει (Station_ID, Station_Pos, Απόσταση, Διάρκεια).
        """
        best_station_idx = None
        best_station_pos = None  # <-- Τέλος το "node", πλέον είναι position (x, y)
        best_dist = 0.0
        best_score = float('inf')
        
        for station in self.city.stations:
            # Πραγματική απόσταση οδήγησης (αστραπιαία Manhattan απόσταση)
            dist_km = self.city.get_driving_distance_km(ev.location, station['id'])
            
            # Ουρά στον σταθμό
            queue = self.city.get_queue(station['id'])
            
            # Κόστος: Απόσταση + (Ουρά * 2.0 km ποινή ανά αμάξι)
            score = dist_km + (queue * 2.0) 
            
            if score < best_score:
                best_score = score
                best_station_idx = station['id']
                best_station_pos = station['location']  # Αυτό είναι πλέον tuple (x, y)
                best_dist = dist_km
                
        # Υποθέτουμε μέση ταχύτητα 30 km/h (0.5 km/min) για μετάβαση σε σταθμό
        duration_mins = int(best_dist / 0.5) + 1 
                
        return best_station_idx, best_station_pos, best_dist, duration_mins