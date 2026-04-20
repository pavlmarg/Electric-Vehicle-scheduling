import numpy as np

class GreedyHeuristicBaseline:
    def __init__(self, city_map):
        """
        Ευρετικός αλγόριθμος διαχείρισης στόλου.
        Λειτουργεί με τον πραγματικό χάρτη (OSMnx).
        """
        self.city = city_map

    def route_ev(self, ev):
        """
        Βρίσκει τον καλύτερο σταθμό με βάση τα πραγματικά χιλιόμετρα 
        και την ουρά. Επιστρέφει (Station_ID, Station_Node, Απόσταση, Διάρκεια).
        """
        best_station_idx = None
        best_station_node = None
        best_dist = 0.0
        best_score = float('inf')
        
        for station in self.city.stations:
            # Πραγματική απόσταση οδήγησης (ακαριαία από την cache)
            dist_km = self.city.get_driving_distance_km(ev.location, station['id'])
            
            # Ουρά στον σταθμό
            queue = self.city.get_queue(station['id'])
            
            # Κόστος: Απόσταση + (Ουρά * 2.0 km ποινή ανά αμάξι)
            score = dist_km + (queue * 2.0) 
            
            if score < best_score:
                best_score = score
                best_station_idx = station['id']
                best_station_node = station['location']
                best_dist = dist_km
                
        # Υποθέτουμε μέση ταχύτητα 30 km/h (0.5 km/min) για μετάβαση σε σταθμό
        duration_mins = int(best_dist / 0.5) + 1 
                
        return best_station_idx, best_station_node, best_dist, duration_mins