import numpy as np
from ev import EV

class EVRL(EV):
    def __init__(self, ev_id, location, battery_capacity, current_soc, target_soc, arrival_time, departure_time):
        super().__init__(ev_id, location, battery_capacity, current_soc, target_soc, arrival_time, departure_time)
        