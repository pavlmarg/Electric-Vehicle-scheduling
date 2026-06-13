import sys
import os
import numpy as np
import random
import csv
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from environments.citygrid import CityMap  
from environments.traffic_generator import TrafficGenerator
from baselines.benchmark import GreedyHeuristicBaseline

def run_simulation_for_profile(profile_idx):
    """Εκτελεί το simulation του Baseline για ένα συγκεκριμένο προφίλ."""
    NUM_VEHICLES = 750
    
    np.random.seed(50) 
    random.seed(50)
    
    city = CityMap(width_km=20.0, height_km=20.0, num_stations=16, num_hubs=4)
    generator = TrafficGenerator(city, num_vehicles=NUM_VEHICLES, seed=50)
    generator.client_manager.current_profile = generator.client_manager.all_profiles[profile_idx]
    
    fleet = generator.generate_initial_fleet()
    baseline_solver = GreedyHeuristicBaseline(city)
    
    total_wait_time = 0.0
    total_customers_served = 0
    total_abandoned_customers = 0
    total_energy_kwh = 0.0
    LEASING_COST_EUR = 40.0

    #  Λίστες για τα γραφήματα Time-Series
    queues_over_time = []
    avg_soc_over_time = []

    # Simulation Loop (1 ημέρα = 1440 λεπτά)
    for minute in range(1440): 
        generator.generate_new_demands(minute)
        
        wait_times, abandoned = generator.process_waitlist(minute, fleet)
        
        total_wait_time += sum(wait_times)
        total_customers_served += len(wait_times)
        total_abandoned_customers += abandoned
        
        for ev in fleet:
            ev.update_time(minute)
            
            if ev.state == 'IDLE':
                action, target_pos, dist, duration = baseline_solver.route_ev(ev)
                if action is not None:
                    if action == "REBALANCE":
                        ev.state = 'REBALANCING'
                        ev.target_pos = target_pos
                        ev.arrival_time = minute + duration
                    else:
                        ev.dispatch_to_station(target_pos, action, dist, duration, minute)
                        city.add_to_queue(action)
            
            elif ev.state == 'REBALANCING':
                if minute >= getattr(ev, 'arrival_time', minute):
                    ev.location = ev.target_pos
                    ev.state = 'IDLE'
            
            elif ev.state == 'WAITING_FOR_CHARGER':
                ev.total_waiting_time += 1
                charger_assigned = city.occupy_charger(ev.target_station_idx)
                if charger_assigned:
                    city.remove_from_queue(ev.target_station_idx)
                    ev.state = 'CHARGING'
                    ev.charger_type = charger_assigned
            
            elif ev.state == 'CHARGING':
                power = city.charger_specs[ev.charger_type]['power']
                price = city.get_electricity_price(minute, ev.charger_type)
                station_to_release = ev.target_station_idx
                added_kwh = ev.charge(power_kw=power, price_per_kwh=price)
                total_energy_kwh += added_kwh
                if ev.state == 'IDLE':
                    city.release_charger(station_to_release, ev.charger_type)

        # Καταγραφή δεδομένων ΛΕΠΤΟ-ΠΡΟΣ-ΛΕΠΤΟ
        current_total_queue = sum(st['queue_length'] for st in city.stations)
        current_avg_soc = sum(ev.current_soc for ev in fleet) / len(fleet)
        
        queues_over_time.append(current_total_queue)
        avg_soc_over_time.append(current_avg_soc)

    # Τελικοί υπολογισμοί για το προφίλ
    total_net_profit = sum(e.daily_revenue - e.daily_charging_cost - LEASING_COST_EUR for e in fleet)
    total_requested = total_customers_served + total_abandoned_customers
    service_rate = (total_customers_served / total_requested * 100) if total_requested > 0 else 0
    avg_wait_time = (total_wait_time / total_customers_served) if total_customers_served > 0 else 0.0

    return {
        "profit": total_net_profit,
        "service_rate": service_rate,
        "wait_time": avg_wait_time,
        "queues_over_time": queues_over_time,   
        "avg_soc_over_time": avg_soc_over_time  
    }
    
    
def main():
    profile_names = [
        "Normal", "Commuter", "Saturday", "Sunday", "Low", 
        "High Stress", "Flattened", "Bimodal", "Event", "Early Spike"
    ]
    
    all_baseline_results = []

    print(f"{'Profile':<15} | {'Profit (€)':<12} | {'Service %':<10} | {'Wait(m)':<7}")
    print("-" * 55)

    for i in range(10):
        res = run_simulation_for_profile(i)
        all_baseline_results.append(res)
        # ΑΛΛΑΓΗ: Εκτύπωση του wait_time
        print(f"{profile_names[i]:<15} | {res['profit']:>10.2f}€ | {res['service_rate']:>8.1f}% | {res['wait_time']:>6.1f}m")

if __name__ == "__main__":
    main()