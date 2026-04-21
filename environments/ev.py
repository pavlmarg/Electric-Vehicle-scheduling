class EVTaxi:
    def __init__(self, taxi_id, start_pos):
        self.id = taxi_id
        
        # Η τοποθεσία πλέον είναι ένα tuple (x, y), π.χ. (12.45, 8.91)
        self.location = start_pos
        self.target_pos = None
        self.target_station_idx = None
        self.arrival_time = 0
        
        self.battery_capacity = 40.0
        self.consumption_per_km = 0.17
        self.current_soc = 1.0
        
        self.state = 'IDLE'
        self.charger_type = None
        self.busy_until = 0
        
        self.daily_revenue = 0.0
        self.daily_charging_cost = 0.0
        self.total_km_driven = 0.0
        self.times_charged = 0
        self.total_waiting_time = 0

    def start_customer_trip(self, destination_pos, distance_km, duration_mins, fare_eur, current_time):
        if self.state != 'IDLE':
            return False
            
        self.state = 'WITH_CUSTOMER'
        self.target_pos = destination_pos
        self.busy_until = current_time + duration_mins
        
        self.daily_revenue += fare_eur
        self.total_km_driven += distance_km
        self._consume_energy(distance_km)
        return True

    def dispatch_to_station(self, station_pos, station_id, distance_km, duration_mins, current_time):
        if self.state != 'IDLE':
            return False
            
        self.state = 'MOVING_TO_STATION'
        self.target_pos = station_pos
        self.target_station_idx = station_id
        self.busy_until = current_time + duration_mins
        
        self.total_km_driven += distance_km
        self._consume_energy(distance_km)
        return True

    def update_time(self, current_time):
        if current_time >= self.busy_until:
            if self.state == 'WITH_CUSTOMER':
                self.location = self.target_pos
                self.target_pos = None
                self.state = 'IDLE'
                
            elif self.state == 'MOVING_TO_STATION':
                self.location = self.target_pos
                self.state = 'WAITING_FOR_CHARGER'

    def charge(self, power_kw, price_per_kwh, duration_min=1):
        if self.state != 'CHARGING': 
            return 0.0

        added_energy = power_kw * (duration_min / 60.0)
        max_energy_needed = (1.0 - self.current_soc) * self.battery_capacity
        actual_energy = min(added_energy, max_energy_needed)
        
        self.current_soc += actual_energy / self.battery_capacity
        self.daily_charging_cost += actual_energy * price_per_kwh
        
        if self.current_soc >= 0.95:
            self.current_soc = 1.0
            self.state = 'IDLE'
            self.target_station_idx = None
            self.target_pos = None
            self.times_charged += 1
            
        return actual_energy

    def _consume_energy(self, dist_km):
        used_kwh = dist_km * self.consumption_per_km
        self.current_soc -= used_kwh / self.battery_capacity
        
        if self.current_soc <= 0:
            self.current_soc = 0.0
            self.state = 'STRANDED'

    def __repr__(self):
        # Format the location to look neat if printed, keeping 2 decimal places
        loc_str = f"({self.location[0]:.1f}, {self.location[1]:.1f})" if isinstance(self.location, tuple) else "None"
        return f"Taxi_{self.id} | Loc: {loc_str} | SoC: {self.current_soc:.0%} | State: {self.state} | Rev: {self.daily_revenue:.2f}€"