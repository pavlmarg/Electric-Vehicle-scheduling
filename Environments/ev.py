import numpy as np

class EV:
    def __init__(self, ev_id, location, battery_capacity, current_soc, target_soc, arrival_time, departure_time):
        """
        EV Agent class for a Micro-City
        """
        self.id = ev_id
        self.location = np.array(location, dtype=float)
        self.target_location = None
        
        # Battery Specs
        self.battery_capacity = battery_capacity 
        self.current_soc = current_soc           
        self.target_soc = target_soc             
        
        # Slightly higher consumption for small-scale city realism
        self.consumption_per_km = 2
        
        # Timing (Minutes of the day)
        self.arrival_time = arrival_time
        self.departure_time = departure_time
        
        # Stats
        self.status = 'driving'
        self.total_cost = 0.0

    def drive(self, speed=0.5):
        """
        Moves EV towards target using Manhattan logic.
        Note: speed is km per minute (0.5 = 30km/h).
        """
        if self.target_location is None: return False

        # Calculate Manhattan Distance: |x1-x2| + |y1-y2|
        diff = self.target_location - self.location
        dist = np.sum(np.abs(diff))
        
        # Check arrival
        if dist <= speed:
            self._consume_energy(dist)
            self.location = np.copy(self.target_location)
            self.status = 'waiting'
            return True
        
        # Move along X first, then Y
        move_x = min(abs(diff[0]), speed) * np.sign(diff[0])
        remaining_speed = speed - abs(move_x)
        
        move_y = 0.0
        if remaining_speed > 0:
            move_y = min(abs(diff[1]), remaining_speed) * np.sign(diff[1])
            
        # Update Position
        self.location[0] += move_x
        self.location[1] += move_y
        
        # Update Battery
        self._consume_energy(abs(move_x) + abs(move_y))
        return False

    def charge(self, power_kw, price_per_kwh, duration_min=1):
        """
        Charges the EV. In a per-minute simulation, duration_min is 1.
        """
        # Status must be set to charging by the station/manager
        if self.status != 'charging': return 0.0

        added_energy = power_kw * (duration_min / 60.0)
        
        # Physical limit: cannot exceed 100% capacity
        max_energy = (1.0 - self.current_soc) * self.battery_capacity
        actual_energy = min(added_energy, max_energy)
        
        # Update State
        self.current_soc += actual_energy / self.battery_capacity
        self.total_cost += actual_energy * price_per_kwh
        
        return actual_energy

    def _consume_energy(self, dist_km):
        """Helper to reduce battery and check for 'stranded' status."""
        used_kwh = dist_km * self.consumption_per_km
        self.current_soc -= used_kwh / self.battery_capacity
        
        if self.current_soc <= 0:
            self.current_soc = 0
            self.status = 'stranded'

    def __repr__(self):
        return f"EV_{self.id} | SoC: {self.current_soc:.2%} | {self.status}"