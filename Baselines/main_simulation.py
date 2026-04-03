import numpy as np
import csv # Για την εξαγωγή του ιστορικού
from environments.citygrid import CityGrid
from environments.traffic_generator import TrafficGenerator
from baselines.benchmark import GreedyHeuristicBaseline

def run_headless_simulation():
    CITY_SIZE = 25 
    NUM_STATIONS = 12
    NUM_VEHICLES = 1000
    
    print("--- 1. INITIALIZING HEADLESS MICRO-CITY & FLEET ---")
    np.random.seed(50) 
    
    city = CityGrid(size_km=CITY_SIZE, num_stations=NUM_STATIONS)
    generator = TrafficGenerator(city, num_vehicles=NUM_VEHICLES)
    baseline_solver = GreedyHeuristicBaseline(city)
    
    fleet = generator.generate_initial_fleet()
    
    successful_charges = 0
    total_energy_kwh = 0.0
    
    print("\n--- 2. STARTING 24-HOUR SIMULATION LOOP ---")
    for minute in range(1440): 
        for ev in fleet:
            
            if ev.status == 'driving':
                ev.drive_randomly(CITY_SIZE)
                
            elif ev.status == 'routing':
                best_station = baseline_solver.route_ev(ev)
                ev.target_station_idx = best_station
                ev.target_location = city.stations[best_station]['location']
                ev.status = 'moving_to_station'
                city.add_to_queue(best_station)
                
            elif ev.status == 'moving_to_station':
                if ev.move_to_station(): 
                    city.remove_from_queue(ev.target_station_idx)
                    charger_assigned = city.occupy_charger(ev.target_station_idx)
                    if charger_assigned:
                        ev.status = 'charging'
                        ev.charger_type = charger_assigned
                        
            elif ev.status == 'waiting':
                ev.total_waiting_time += 1 # Καταγραφή του χρόνου που χάνει στην ουρά
                charger_assigned = city.occupy_charger(ev.target_station_idx)
                if charger_assigned:
                    ev.status = 'charging'
                    ev.charger_type = charger_assigned
                    
            elif ev.status == 'charging':
                power = city.charger_specs[ev.charger_type]['power']
                price = city.get_electricity_price(minute, ev.charger_type)
                
                # --- Η ΔΙΟΡΘΩΣΗ ΕΔΩ ---
                # Αποθηκεύουμε το ID του σταθμού ΠΡΙΝ φορτίσει το όχημα
                station_to_release = ev.target_station_idx
                
                # Φορτίζουμε το όχημα (αν γεμίσει, το target_station_idx γίνεται None)
                added_kwh = ev.charge(power_kw=power, price_per_kwh=price)
                total_energy_kwh += added_kwh
                
                # Ελέγχουμε αν γέμισε και είναι έτοιμο να φύγει
                if ev.status == 'driving':
                    # Χρησιμοποιούμε το ID που αποθηκεύσαμε!
                    city.release_charger(station_to_release, ev.charger_type)
                    successful_charges += 1
                    
        # --- Εκτύπωση Status ---
        if minute % 60 == 0:
            working = sum(1 for e in fleet if e.status == 'driving')
            charging = sum(1 for e in fleet if e.status == 'charging')
            waiting = sum(1 for e in fleet if e.status == 'waiting')
            stranded = sum(1 for e in fleet if e.status == 'stranded')
            print(f"[Ώρα {minute//60:02d}:00] Στο δρόμο: {working:4d} | Φορτίζουν: {charging:3d} | Αναμονή: {waiting:3d} | Crashed: {stranded:3d}")

    # --- 3. ΑΠΟΘΗΚΕΥΣΗ ΙΣΤΟΡΙΚΟΥ ---
    print("\n--- 3. SAVING HISTORY LOG ---")
    # Υποθέτουμε ένα κέρδος 1.5 Ευρώ για κάθε χιλιόμετρο που οδηγεί το ταξί
    PROFIT_PER_KM = 1
    
    with open('history_baseline.csv', mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['EV_ID', 'Status', 'Distance_km', 'Gross_Profit_Eur', 'Charging_Cost_Eur', 'Net_Profit_Eur', 'Times_Charged', 'Wait_Time_mins', 'Final_SoC'])
        
        for ev in fleet:
            gross_profit = ev.total_distance * PROFIT_PER_KM
            net_profit = gross_profit - ev.total_cost
            writer.writerow([
                ev.id, ev.status, f"{ev.total_distance:.2f}", f"{gross_profit:.2f}", 
                f"{ev.total_cost:.2f}", f"{net_profit:.2f}", ev.times_charged, 
                ev.total_waiting_time, f"{ev.current_soc:.2f}"
            ])

    # --- 4. ΤΕΛΙΚΑ ΑΠΟΤΕΛΕΣΜΑΤΑ ---
    final_stranded = sum(1 for e in fleet if e.status == 'stranded')
    total_net_profit = sum((e.total_distance * PROFIT_PER_KM) - e.total_cost for e in fleet)
    
    print("\n" + "="*55)
    print("--- SIMULATION COMPLETE (GREEDY BASELINE) ---")
    print(f"Συνολικός Στόλος: {NUM_VEHICLES} Οχήματα")
    print(f"Ολοκληρωμένες Φορτίσεις: {successful_charges}")
    print(f"Συνολική Ενέργεια Δικτύου: {total_energy_kwh:.2f} kWh")
    print(f"Οχήματα που έμειναν από μπαταρία: {final_stranded}")
    print(f"ΣΥΝΟΛΙΚΟ ΚΑΘΑΡΟ ΚΕΡΔΟΣ ΣΤΟΛΟΥ: {total_net_profit:.2f} €")
    print("="*55)

if __name__ == "__main__":
    run_headless_simulation()