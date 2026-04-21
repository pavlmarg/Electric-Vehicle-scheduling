import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import csv
from environments.citygrid import CityMap  
from environments.traffic_generator import TrafficGenerator
from baselines.benchmark import GreedyHeuristicBaseline

def run_headless_simulation():
    NUM_VEHICLES = 400
    
    print("--- 1. INITIALIZING REAL-WORLD MAP & FLEET ---")
    np.random.seed(50) 
    
    # Αρχικοποίηση χάρτη Θεσσαλονίκης
    city = CityMap(radius_meters=5500, num_stations=6)
    
    # Αρχικοποίηση στόλου
    generator = TrafficGenerator(city, num_vehicles=NUM_VEHICLES)
    fleet = generator.generate_initial_fleet()
    
    # Αρχικοποίηση Baseline (Διευθυντής Στόλου)
    baseline_solver = GreedyHeuristicBaseline(city)
    
    successful_charges = 0
    total_energy_kwh = 0.0
    
    # Μεταβλητές Ποιότητας (QoS) & Νεκρών Ταξί
    total_stars = 0
    total_customers_served = 0
    total_abandoned_customers = 0
    dead_taxis_set = set()  # Εδώ αποθηκεύουμε όσα ταξί ξεφορτίζουν
    
    print("\n--- 2. STARTING 24-HOUR SIMULATION LOOP ---")
    for minute in range(1440): 
        
        # 1. ΠΑΡΑΓΩΓΗ ΖΗΤΗΣΗΣ & ΟΥΡΑ ΠΕΛΑΤΩΝ
        generator.generate_new_demands(minute)
        ratings, abandoned = generator.process_waitlist(minute)
        
        # Αποθήκευση στατιστικών πελατών
        total_stars += sum(ratings)
        total_customers_served += len(ratings)
        total_abandoned_customers += abandoned
        
        # 2. ΕΝΗΜΕΡΩΣΗ ΣΤΟΛΟΥ
        for ev in fleet:
            ev.update_time(minute)
            
            # --- LIVE ΕΙΔΟΠΟΙΗΣΗ ΓΙΑ ΝΕΚΡΑ ΤΑΞΙ ---
            if ev.state == 'STRANDED' and ev.id not in dead_taxis_set:
                print(f" [Ώρα {minute//60:02d}:{minute%60:02d}] SOS: Το Ταξί {ev.id} έμεινε από μπαταρία στους δρόμους!")
                dead_taxis_set.add(ev.id)
            
            # Αν το ταξί είναι ελεύθερο
            if ev.state == 'IDLE':
                # Αν η μπαταρία είναι κάτω από 25%, πάει για φόρτιση
                if ev.current_soc <= 0.25:
                    station_idx, station_node, dist, duration = baseline_solver.route_ev(ev)
                    
                    ev.dispatch_to_station(station_node, station_idx, dist, duration, minute)
                    city.add_to_queue(station_idx)
                    
            # Αν έφτασε στον σταθμό και περιμένει
            elif ev.state == 'WAITING_FOR_CHARGER':
                ev.total_waiting_time += 1
                
                # Προσπαθεί να πάρει πρίζα
                charger_assigned = city.occupy_charger(ev.target_station_idx)
                if charger_assigned:
                    city.remove_from_queue(ev.target_station_idx)
                    ev.state = 'CHARGING'
                    ev.charger_type = charger_assigned
                    
            # Αν φορτίζει
            elif ev.state == 'CHARGING':
                power = city.charger_specs[ev.charger_type]['power']
                price = city.get_electricity_price(minute, ev.charger_type)
                
                station_to_release = ev.target_station_idx
                
                # Φόρτιση για 1 λεπτό
                added_kwh = ev.charge(power_kw=power, price_per_kwh=price)
                total_energy_kwh += added_kwh
                
                # Αν γέμισε, το ev.charge() αλλάζει το state σε 'IDLE'
                if ev.state == 'IDLE':
                    city.release_charger(station_to_release, ev.charger_type)
                    successful_charges += 1
                    
        # --- Εκτύπωση Status ανά Ώρα ---
        if minute % 60 == 0:
            with_cust = sum(1 for e in fleet if e.state == 'WITH_CUSTOMER')
            idle = sum(1 for e in fleet if e.state == 'IDLE')
            charging = sum(1 for e in fleet if e.state == 'CHARGING')
            waiting = sum(1 for e in fleet if e.state == 'WAITING_FOR_CHARGER')
            
            avg_stars = (total_stars / total_customers_served) if total_customers_served > 0 else 5.0
            
            print(f"[Ώρα {minute//60:02d}:00] Ταξί(Ελεύθ:{idle:3d}|Πελάτης:{with_cust:3d}|Φορτίζουν:{charging:2d}|ΟυράΠρίζας:{waiting:2d}) | Πελάτες σε Αναμονή: {len(generator.waitlist):3d} | Αστέρια App: {avg_stars:.1f} | Νεκρά: {len(dead_taxis_set)}")

    # --- 3. ΑΠΟΘΗΚΕΥΣΗ ΙΣΤΟΡΙΚΟΥ ---
    print("\n--- 3. SAVING HISTORY LOG ---")
    LEASING_COST_EUR = 20.0 
    
    with open('history_baseline_map.csv', mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['EV_ID', 'Status', 'Distance_km', 'Gross_Profit_Eur', 'Charging_Cost_Eur', 'Net_Profit_Eur', 'Times_Charged', 'Wait_Time_mins', 'Final_SoC'])
        
        for ev in fleet:
            net_profit = ev.daily_revenue - ev.daily_charging_cost - LEASING_COST_EUR
            
            writer.writerow([
                ev.id, ev.state, f"{ev.total_km_driven:.2f}", f"{ev.daily_revenue:.2f}", 
                f"{ev.daily_charging_cost:.2f}", f"{net_profit:.2f}", ev.times_charged, 
                ev.total_waiting_time, f"{ev.current_soc:.2f}"
            ])

    # --- 4. ΤΕΛΙΚΑ ΑΠΟΤΕΛΕΣΜΑΤΑ ---
    total_net_profit = sum(e.daily_revenue - e.daily_charging_cost - LEASING_COST_EUR for e in fleet)
    final_avg_stars = (total_stars / total_customers_served) if total_customers_served > 0 else 0
    
    print("\n" + "="*60)
    print("--- SIMULATION COMPLETE (REAL WORLD BASELINE) ---")
    print(f"Συνολικός Στόλος: {NUM_VEHICLES} Οχήματα")
    print(f"Ολοκληρωμένες Φορτίσεις: {successful_charges}")
    print(f"Συνολική Ενέργεια Δικτύου: {total_energy_kwh:.2f} kWh")
    print(f"ΣΥΝΟΛΙΚΟ ΚΑΘΑΡΟ ΚΕΡΔΟΣ ΕΤΑΙΡΕΙΑΣ: {total_net_profit:.2f} €")
    print("-" * 60)
    print(f"Εξυπηρετήθηκαν: {total_customers_served} Πελάτες")
    print(f"Εγκατέλειψαν (Χαμένα Έσοδα): {total_abandoned_customers} Πελάτες")
    print(f"Μέση Βαθμολογία Στόλου (Αστέρια): {final_avg_stars:.2f} / 5.00")
    print(f"Οχήματα που έμειναν από μπαταρία: {len(dead_taxis_set)}")
    print("="*60)

if __name__ == "__main__":
    run_headless_simulation()