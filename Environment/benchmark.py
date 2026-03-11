import pulp
import numpy as np

class SolutionWrapper:
    """
    It mimics the CPLEX solution.get_value() method perfectly.
    """
    def get_value(self, var):
        val = var.varValue
        return val if val is not None else 0.0

class CplexSolver:
    def __init__(self, city_grid):
        """
        Now powered by PuLP (Open-Source), but we keep the 'CplexSolver' name 
        so your main_optimal.py script doesn't break. No size limits!
        """
        self.city = city_grid

    def solve_day(self, ev_list):
        print(f"\n--- Building Model for {len(ev_list)} EVs ---")
        
        # Define the problem: Minimize cost
        mdl = pulp.LpProblem("EV_Optimal_Mixed", pulp.LpMinimize)
        
        # Parameters
        T_slots = 120      # Time slots in a day (1440/120 = 12 minute intervals)
        dt = 0.2      # 60 / 12 = 5 therefore 20% of the day     
        
        # --- 1. DECISION VARIABLES ---
        p = {} 
        s = {} 
        
        for ev in ev_list:
            start_slot = int(ev.arrival_time // 10)
            end_slot = int(ev.departure_time // 10)
            valid_slots = range(start_slot, min(end_slot, T_slots))
            
            max_p = getattr(ev, 'max_charging_power', 50.0)
            
            for t in valid_slots:
                p[(ev.id, t)] = pulp.LpVariable(name=f"p_{ev.id}_{t}", lowBound=0, upBound=max_p, cat='Continuous')
            
            s[ev.id] = pulp.LpVariable(name=f"slack_{ev.id}", lowBound=0, cat='Continuous')

        # --- 2. CONSTRAINTS ---
        
        # A. Demand Satisfaction (Soft)
        for ev in ev_list:
            start_slot = int(ev.arrival_time // 10)
            end_slot = int(ev.departure_time // 10)
            valid_slots = list(range(start_slot, min(end_slot, T_slots)))
            
            needed_kwh = (ev.target_soc - ev.current_soc) * ev.battery_capacity
            needed_kwh = max(0.0, needed_kwh)
            
            if valid_slots:
                charge_vars = [p[(ev.id, t)] for t in valid_slots if (ev.id, t) in p]
                if charge_vars:
                    # Power delivered + Slack >= Needed Energy
                    mdl += pulp.lpSum(charge_vars) * dt + s[ev.id] >= needed_kwh, f"Demand_{ev.id}"

        # B. Grid Limit (Global)
        for t in range(T_slots):
            active_vars = [p[(ev.id, t)] for ev in ev_list if (ev.id, t) in p]
            if active_vars:
                mdl += pulp.lpSum(active_vars) <= self.city.global_limit, f"GridLimit_{t}"

        # C. Local Station Limits & Hardware Limits
        evs_by_station = {st_id: [] for st_id in self.city.get_all_station_ids()}
        for ev in ev_list:
            if hasattr(ev, 'assigned_station_id') and ev.assigned_station_id in evs_by_station:
                evs_by_station[ev.assigned_station_id].append(ev)

        for station_id, parked_evs in evs_by_station.items():
            if not parked_evs: 
                continue 
            
            # Get exact hardware limits from the city
            st_data = next(s for s in self.city.stations if s['id'] == station_id)
            fast_limit_kw = st_data['chargers']['fast'] * 50.0
            slow_limit_kw = st_data['chargers']['slow'] * 11.0
            station_limit = st_data['p_max']
            
            for t in range(T_slots):
                fast_vars = [p[(ev.id, t)] for ev in parked_evs if ev.charger_type == 'fast' and (ev.id, t) in p]
                slow_vars = [p[(ev.id, t)] for ev in parked_evs if ev.charger_type == 'slow' and (ev.id, t) in p]
                all_vars = fast_vars + slow_vars
                
                # Apply the physical bounds
                if all_vars:
                    mdl += pulp.lpSum(all_vars) <= station_limit, f"Local_St{station_id}_T{t}"
                if fast_vars:
                    mdl += pulp.lpSum(fast_vars) <= fast_limit_kw, f"Fast_HW_St{station_id}_T{t}"
                if slow_vars:
                    mdl += pulp.lpSum(slow_vars) <= slow_limit_kw, f"Slow_HW_St{station_id}_T{t}"

        # --- 3. OBJECTIVE FUNCTION ---
        cost_terms = []
        
        for (ev_id, t), var in p.items():
            ev = next(x for x in ev_list if x.id == ev_id)
            c_type = getattr(ev, 'charger_type', 'fast') 
            
            time_min = t * 15
            price = self.city.get_electricity_price(time_min, c_type)
            cost_terms.append(var * dt * price)
            
        # Add severe penalty for slack
        for ev in ev_list:
            if ev.id in s:
                cost_terms.append(s[ev.id] * 1000)
        
        mdl += pulp.lpSum(cost_terms), "Total_Cost"

        # --- 4. SOLVE ---
        print("--- Solving... ---")
        # msg=1 prints the math engine's output to the terminal. timeLimit saves your PC.
        mdl.solve(pulp.PULP_CBC_CMD(timeLimit=24, msg=1))
        
        status_str = pulp.LpStatus[mdl.status]
        print(f"--- PuLP Status: {status_str} ---")
        
        # If it found a solution
        if mdl.status > 0 or status_str == 'Optimal':
            return SolutionWrapper(), p
        else:
            print("-> Solver failed to find a valid solution.")
            return None, None