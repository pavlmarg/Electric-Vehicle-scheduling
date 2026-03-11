import numpy as np
from ev import EV

class TrafficGenerator:
    def __init__(self, citygrid):
        """
        Manages the creation of new EVs during the simulation.
        
        Args:
            citygrid (CityGrid): Reference to the city map.
        """
        self.city = citygrid
        self.ev_counter = 0
        
    def step(self, current_minute):
        """
        Decision logic for spawning a new EV based on the time of day.
        NOW RETURNS A LIST OF EVs to support large-scale simulations.
        """
        hour = (current_minute % 1440) / 60.0
        
        # Metropolis Scaling: Generate multiple cars per minute
        if (7 <= hour <= 10) or (16 <= hour <= 19):
            # Rush Hour: 3 to 9 cars per minute
            num_cars = np.random.randint(3, 10)  
        elif 1 <= hour <= 5:
            # Night Time: 10% chance of 1 car
            num_cars = np.random.choice([0, 1], p=[0.9, 0.1]) 
        else:
            # Normal Hours: 0 to 3 cars per minute
            num_cars = np.random.randint(0, 4)   
            
        spawned_evs = []
        for _ in range(num_cars):
            spawned_evs.append(self._create_ev(current_minute))
            
        return spawned_evs

    def _create_ev(self, current_time):
        self.ev_counter += 1
        
        # Random Start Location within city bounds
        start_loc = np.random.uniform(0, self.city.size, size=2)
        
        # Random Battery State 
        start_soc = np.random.uniform(0.2, 0.3)
        
        # Random Target SoC
        target_soc = np.random.uniform(0.6, 1.0)
        
        # Standard battery capacity
        capacity = 50.0 
        
        # --- NEW PHYSICS LOGIC ---
        # Calculate exactly how many minutes it physically takes to charge this car at max speed (50kW)
        needed_kwh = (target_soc - start_soc) * capacity
        min_hours_needed = needed_kwh / 50.0 
        min_minutes_needed = int(min_hours_needed * 60) + 15 # Add a 15-minute safety buffer
        
        # Parking Duration Logic
        if np.random.random() < 0.6:
            # Short stay: Ensure they stay at least as long as physically required!
            lower_bound = max(60, min_minutes_needed)
            duration = np.random.randint(lower_bound, lower_bound + 60) 
        else:
            # Long stay: 4 to 8 hours (always plenty of time)
            duration = np.random.randint(240, 480)
            
        # Departure = Current time + drive time + stay duration
        departure = current_time + 30 + duration
        
        # Create the EV instance
        new_ev = EV(
            ev_id=self.ev_counter,
            location=start_loc,
            battery_capacity=capacity,
            current_soc=start_soc,
            target_soc=target_soc,
            arrival_time=current_time,
            departure_time=departure
        )
        
        return new_ev