import numpy as np

class GreedyHeuristicBaseline:
    def __init__(self, city_grid):
        """
        Αντικαθιστά τον παλιό MILP (PuLP) solver. 
        Χρησιμοποιεί έναν ευρετικό αλγόριθμο (Heuristic) για να διαχειριστεί 
        τον συνεχή στόλο σε πραγματικό χρόνο.
        """
        self.city = city_grid

    def route_ev(self, ev):
        """
        Αποφασίζει σε ποιον σταθμό πρέπει να πάει το EV.
        Λογική (Greedy): Ελαχιστοποίηση της Απόστασης + Ποινή για μεγάλες Ουρές.
        """
        best_station_idx = None
        best_score = float('inf')
        
        for idx, station in enumerate(self.city.stations):
            # Υπολογισμός Manhattan Distance
            diff = np.abs(ev.location - station['location'])
            dist = np.sum(diff)
            
            # Πόσα αυτοκίνητα περιμένουν ήδη σε αυτόν τον σταθμό;
            queue = self.city.get_queue(idx)
            
            # Κόστος: Απόσταση + (Ουρά * Βάρος Ποινής)
            # Το 1.5 είναι ένα αυθαίρετο βάρος. Σημαίνει "προτιμώ να οδηγήσω 
            # 1.5 km παραπάνω από το να περιμένω 1 ακόμα αυτοκίνητο στην ουρά"
            score = dist + (queue * 1.5) 
            
            if score < best_score:
                best_score = score
                best_station_idx = idx
                
        return best_station_idx