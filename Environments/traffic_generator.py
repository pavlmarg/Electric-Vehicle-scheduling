import numpy as np
from environments.ev import EV 

class TrafficGenerator:
    def __init__(self, citygrid, num_vehicles=1000):
        """
        Manages the creation of the Electric Fleet for the simulation.
        
        Args:
            citygrid (CityGrid): Reference to the city map.
            num_vehicles (int): The total number of EVs in the fleet.
        """
        self.city = citygrid
        self.num_vehicles = num_vehicles
        self.fleet = []

    def generate_initial_fleet(self):
        """
        Δημιουργεί τον στόλο στην αρχή της ημέρας (t=0).
        Όλα τα οχήματα "γεννιούνται" τώρα και δεν πεθαίνουν ποτέ.
        """
        self.fleet = []
        print(f"--- Spawning Fleet of {self.num_vehicles} EVs ---")
        
        for i in range(self.num_vehicles):
            # Τυχαία αρχική τοποθεσία μέσα στα όρια της πόλης (π.χ. 0 έως 5 km)
            start_loc = np.random.uniform(0, self.city.size, size=2)
            
            # Τυχαία αρχική μπαταρία από 30% έως 100% 
            # (Αυτό είναι κρίσιμο για να μην ζητήσουν και τα 1000 ταξί ρεύμα στο 1ο λεπτό!)
            start_soc = np.random.uniform(0.30, 1.0)
            
            # Δημιουργία του EV με τα νέα, απλοποιημένα ορίσματα της κλάσης EV
            new_ev = EV(
                ev_id=i,
                location=start_loc,
                battery_capacity=50.0,
                current_soc=start_soc
            )
            
            self.fleet.append(new_ev)
            
        return self.fleet