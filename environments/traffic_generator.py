import numpy as np
from environments.ev import EVTaxi 
from environments.clients import ClientManager

class TrafficGenerator:
    def __init__(self, city_map, num_vehicles=400):
        self.city = city_map
        self.num_vehicles = num_vehicles
        self.fleet = []
        
        # Ενεργοποιούμε τον διαχειριστή πελατών
        self.client_manager = ClientManager(city_map)
        
        print(f"--- Traffic Generator Ready: Continuous Space {self.city.width_km}x{self.city.height_km} km ---")

    @property
    def waitlist(self):
        """Επιστρέφει την ουρά αναμονής από τον client_manager για να μη σπάσει ο κώδικας στα άλλα αρχεία"""
        return self.client_manager.waitlist

    def generate_initial_fleet(self):
        """Δημιουργεί τα ταξί σε τυχαίες θέσεις (X,Y) ΚΟΥΜΠΩΜΕΝΑ ΣΤΟ GRID."""
        self.fleet = []
        print(f"--- Spawning Fleet of {self.num_vehicles} EV Taxis ---")
        
        for i in range(self.num_vehicles):
            # Χρησιμοποιούμε το _snap_to_grid και εδώ!
            x = self.city._snap_to_grid(np.random.uniform(0.0, self.city.width_km))
            y = self.city._snap_to_grid(np.random.uniform(0.0, self.city.height_km))
            
            taxi = EVTaxi(taxi_id=i, start_pos=(x, y))
            taxi.current_soc = np.random.uniform(0.30, 1.0)
            self.fleet.append(taxi)
            
        return self.fleet

    def generate_new_demands(self, current_time_mins):
        """Προωθεί την εντολή στον ClientManager"""
        self.client_manager.generate_new_demands(current_time_mins)

    def process_waitlist(self, current_time_mins):
        """Προωθεί την εντολή στον ClientManager και του δίνει τον στόλο"""
        return self.client_manager.process_waitlist(current_time_mins, self.fleet)