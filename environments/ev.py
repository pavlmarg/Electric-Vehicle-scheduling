class EVTaxi:
    def __init__(self, taxi_id, start_node):
        """
        EV Agent class προσαρμοσμένη για πραγματικό χάρτη (OSMnx) και οικονομικά δεδομένα.
        """
        self.id = taxi_id
        
        # --- Θέση & Προορισμός ---
        self.location = start_node       # Το τρέχον Node ID στον χάρτη
        self.target_node = None          # Το Node ID του προορισμού (Πελάτης ή Σταθμός)
        self.target_station_idx = None   # Αν πάει σε σταθμό, ποιος είναι ο ID του σταθμού
        
        # --- Τεχνικά Χαρακτηριστικά (Nissan Leaf) ---
        self.battery_capacity = 40.0     # kWh
        self.consumption_per_km = 0.17   # kWh / km (Ρεαλιστική κατανάλωση ταξί)
        self.current_soc = 1.0           # Μπαταρία στο 100%
        
        # --- Κατάσταση (AMoD Logic) ---
        # 'IDLE', 'WITH_CUSTOMER', 'MOVING_TO_STATION', 'WAITING_FOR_CHARGER', 'CHARGING', 'STRANDED'
        self.state = 'IDLE'              
        self.charger_type = None         # 'fast' ή 'slow'
        self.busy_until = 0              # Το λεπτό της ημέρας που θα ελευθερωθεί από το τρέχον task
        
        # --- Οικονομικά & Στατιστικά ---
        self.daily_revenue = 0.0         # Έσοδα από ταξίμετρο (Ευρώ)
        self.daily_charging_cost = 0.0   # Έξοδα Ρεύματος (Ευρώ)
        self.total_km_driven = 0.0       # Συνολικά διανυθέντα χιλιόμετρα
        self.times_charged = 0           
        self.total_waiting_time = 0      # Χρόνος που έχασε στις ουρές

    # ==========================================
    # 1. ΑΝΑΛΗΨΗ ΔΙΑΔΡΟΜΗΣ (ΠΕΛΑΤΗΣ)
    # ==========================================
    def start_customer_trip(self, destination_node, distance_km, duration_mins, fare_eur, current_time):
        """Το ταξί αναλαμβάνει κούρσα. Δεσμεύεται για X λεπτά και πληρώνεται."""
        if self.state != 'IDLE':
            return False
            
        self.state = 'WITH_CUSTOMER'
        self.target_node = destination_node
        self.busy_until = current_time + duration_mins
        
        # Ενημέρωση Οικονομικών & Στατιστικών
        self.daily_revenue += fare_eur
        self.total_km_driven += distance_km
        
        # Κατανάλωση Ενέργειας
        self._consume_energy(distance_km)
        return True

    # ==========================================
    # 2. ΑΝΑΧΩΡΗΣΗ ΓΙΑ ΦΟΡΤΙΣΗ (ΑΠΟΦΑΣΗ RL AI)
    # ==========================================
    def dispatch_to_station(self, station_node, station_id, distance_km, duration_mins, current_time):
        """Το ταξί πηγαίνει στον σταθμό για φόρτιση."""
        if self.state != 'IDLE':
            return False
            
        self.state = 'MOVING_TO_STATION'
        self.target_node = station_node
        self.target_station_idx = station_id
        self.busy_until = current_time + duration_mins
        
        self.total_km_driven += distance_km
        self._consume_energy(distance_km)
        return True

    # ==========================================
    # 3. ΕΛΕΓΧΟΣ ΑΦΙΞΗΣ (Ενημέρωση κάθε λεπτό)
    # ==========================================
    def update_time(self, current_time):
        """Καλείται κάθε λεπτό από το κεντρικό simulation για να δει αν έφτασε."""
        if current_time >= self.busy_until:
            if self.state == 'WITH_CUSTOMER':
                # Έφτασε στον προορισμό του πελάτη
                self.location = self.target_node
                self.target_node = None
                self.state = 'IDLE' # Ελεύθερο για νέα δουλειά ή οδηγία από το AI!
                
            elif self.state == 'MOVING_TO_STATION':
                # Έφτασε στον σταθμό, μπαίνει στην ουρά
                self.location = self.target_node
                self.state = 'WAITING_FOR_CHARGER'

    # ==========================================
    # 4. ΔΙΑΔΙΚΑΣΙΑ ΦΟΡΤΙΣΗΣ
    # ==========================================
    def charge(self, power_kw, price_per_kwh, duration_min=1):
        """Προσθέτει ρεύμα και χρεώνει την εταιρεία (1 λεπτό τη φορά)."""
        if self.state != 'CHARGING': 
            return 0.0

        # Υπολογισμός ενέργειας που παίρνει σε 1 λεπτό (power * 1/60 ώρας)
        added_energy = power_kw * (duration_min / 60.0)
        
        # Δεν μπορεί να ξεπεράσει το 100% της μπαταρίας
        max_energy_needed = (1.0 - self.current_soc) * self.battery_capacity
        actual_energy = min(added_energy, max_energy_needed)
        
        # Ενημέρωση μπαταρίας & κόστους
        self.current_soc += actual_energy / self.battery_capacity
        self.daily_charging_cost += actual_energy * price_per_kwh
        
        # Αν γέμισε (πάνω από 95%), το ταξί βγαίνει ξανά στους δρόμους!
        if self.current_soc >= 0.95:
            self.current_soc = 1.0
            self.state = 'IDLE'
            self.target_station_idx = None
            self.target_node = None
            self.times_charged += 1
            
        return actual_energy

    # ==========================================
    # 5. ΕΣΩΤΕΡΙΚΕΣ ΛΕΙΤΟΥΡΓΙΕΣ (HELPER)
    # ==========================================
    def _consume_energy(self, dist_km):
        """Αφαιρεί μπαταρία. Αν μηδενίσει, το ταξί μένει στον δρόμο."""
        used_kwh = dist_km * self.consumption_per_km
        self.current_soc -= used_kwh / self.battery_capacity
        
        if self.current_soc <= 0:
            self.current_soc = 0.0
            self.state = 'STRANDED'

    def __repr__(self):
        return f"Taxi_{self.id} | SoC: {self.current_soc:.0%} | State: {self.state} | Rev: {self.daily_revenue}€"