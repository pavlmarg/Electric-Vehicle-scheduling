import pygame
import sys
import numpy as np
from environments.citygrid import CityGrid
from environments.traffic_generator import TrafficGenerator
from baselines.benchmark import CplexSolver


# --- PYGAME CONFIGURATION ---
WINDOW_SIZE = 600
FPS = 30 
MINUTES_PER_FRAME = 1 

# Colors
BG_COLOR = (30, 30, 30)
GRID_COLOR = (50, 50, 50)
HUB_COLOR = (0, 200, 255)      
NORMAL_COLOR = (50, 205, 50)   
EV_DRIVING = (255, 200, 0)     
EV_PARKED = (100, 100, 100)    
EV_CRASHED = (255, 0, 0)       
TEXT_COLOR = (255, 255, 255)
TOOLTIP_BG = (20, 20, 20, 230)

def save_history_log(ev_list, solution, vars_dict, city, filename):
    print(f"\nSaving Detailed History to {filename}")
    
    with open(filename, "w", encoding="utf-8") as f:
        f.write("EV_ID,Station_ID,Arr,Dep,SoC_Start,SoC_Target,SoC_Final,Energy_kWh,Cost_Euro,Status,Charger_Type\n")
        
        dt = 0.2 
        
        for ev in ev_list:
            ev_energy = 0.0
            ev_cost = 0.0
            
            if solution: 
                for t in range(120):
                    if (ev.id, t) in vars_dict:
                        power_kw = solution.get_value(vars_dict[(ev.id, t)])
                        if power_kw > 0.001:
                            slot_energy = power_kw * dt
                            price = city.get_electricity_price(t * 15, ev.charger_type)
                            ev_energy += slot_energy
                            ev_cost += slot_energy * price
            
            final_soc = min(1.0, ev.current_soc + (ev_energy / ev.battery_capacity))
            
            status = "Survived"
            if ev.status == 'stranded':
                status = "Crashed (No Battery)"
            elif final_soc < (ev.target_soc - 0.01):
                if ev.departure_time < 1440:
                    status = "Failed (Undercharged)"
        
            c_type_str = ev.charger_type.capitalize() if ev.charger_type else "N/A"

            f.write(f"{ev.id},{ev.assigned_station_id},{ev.arrival_time},{ev.departure_time},"
                    f"{ev.current_soc:.2f},{ev.target_soc:.2f},{final_soc:.2f},{ev_energy:.2f},"
                    f"{ev_cost:.2f},{status},{c_type_str}\n")
            
    print("Log saved successfully.")

def run_merged_simulation():
    print("--- 1. INITIALIZING MICRO-CITY & TRAFFIC ---")
    
    # We choose a random seed in order to compare our models while having the same kind and amount of stations and same traffic 
    np.random.seed(50)
    
    city = CityGrid(global_power_limit=200000)
    generator = TrafficGenerator(city)
    
    all_evs_today = [] 
    crashed_count = 0
    
    # 1. GENERATE THE WHOLE DAY OF TRAFFIC FIRST
    for minute in range(1440):
        new_evs = generator.step(minute)
        for new_ev in new_evs:
            
            new_ev.initial_soc = new_ev.current_soc 
            
            # --- BASELINE 2: SMART HEURISTIC ROUTING ---
            # 1. Figure out how many cars are CURRENTLY parked at each station
            station_loads = {st['id']: 0 for st in city.stations}
            for parked_ev in all_evs_today:
                # Count cars that have arrived but haven't departed yet
                if parked_ev.arrival_time <= minute < parked_ev.departure_time:
                    station_loads[parked_ev.assigned_station_id] += 1
            
            # 2. Score every station based on Distance + Queue Size
            best_station = None
            best_distance = 0
            best_score = float('inf')
            
            for st in city.stations:
                # Calculate Manhattan distance
                diff = np.abs(new_ev.location - st['location'])
                dist = diff[0] + diff[1]
                
                # Calculate Queue (Cars parked vs Physical Plugs)
                total_plugs = st['chargers']['fast'] + st['chargers']['slow']
                current_parked = station_loads[st['id']]
                queue_size = max(0, current_parked - total_plugs)
                
                # THE HEURISTIC MATH: 1 car in queue = 5 units of "distance penalty"
                # This makes the car willing to drive further to avoid a line!
                score = dist + (queue_size * 5.0)
                
                if score < best_score:
                    best_score = score
                    best_station = st
                    best_distance = dist
            
            # 3. Assign to the winning station
            new_ev.assigned_station_id = best_station['id']
            new_ev.target_location = best_station['location']
            
            # Driving Physics (Apply consumption instantly for the math model)
            new_ev._consume_energy(best_distance)
            
            # Charger Type Assignment
            if new_ev.status != 'stranded':
                needed_kwh = max(0.0, (new_ev.target_soc - new_ev.current_soc) * new_ev.battery_capacity)
                available_minutes = new_ev.departure_time - new_ev.arrival_time
                max_slow_kwh = city.charger_specs['slow']['power'] * (available_minutes / 60.0)
                
                # --- NEW: Check if the slow chargers are already full! ---
                total_slow_plugs = best_station['chargers']['slow']
                # Count how many cars are currently parked here and using 'slow'
                current_slow_parked = sum(1 for e in all_evs_today if e.assigned_station_id == best_station['id'] and e.charger_type == 'slow' and e.arrival_time <= minute < e.departure_time)
                
                # Only assign 'slow' if they have time AND there is actually a slow plug physically open
                if max_slow_kwh >= (needed_kwh * 1.25) and current_slow_parked < total_slow_plugs:
                    new_ev.charger_type = 'slow'
                    new_ev.max_charging_power = city.charger_specs['slow']['power']
                else:
                    # Upgrade to Fast Charger (either because they are in a hurry, or the slow plugs are full!)
                    new_ev.charger_type = 'fast'
                    new_ev.max_charging_power = city.charger_specs['fast']['power']
            else:
                crashed_count += 1
                new_ev.charger_type = "N/A" 
            
            all_evs_today.append(new_ev)

    # 2. RUN THE CPLEX OPTIMIZATION
    survivors = [ev for ev in all_evs_today if ev.status != 'stranded']
    solution, vars_dict = None, {}
    
    if survivors:
        solver = CplexSolver(city)
        solution, vars_dict = solver.solve_day(survivors)
        
    # 3. SAVE THE HISTORY TXT FILE
    if all_evs_today:
        save_history_log(all_evs_today, solution, vars_dict, city, "history_optimal.txt")

    # 4. START THE LIVE PLAYBACK (PYGAME)
    print("\n--- 4. STARTING VISUAL PLAYBACK ---")
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
    pygame.display.set_caption("EV Micro-City: Optimal Playback")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial", 18)
    small_font = pygame.font.SysFont("Arial", 14)

    scale = WINDOW_SIZE / city.size 
    active_evs = []
    minute = 0
    visual_crashed_total = 0

    running = True
    while running and minute < 1440:
        mouse_x, mouse_y = pygame.mouse.get_pos()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Spawn cars EXACTLY when they arrived in the generated history
        spawns_this_minute = [ev for ev in all_evs_today if int(ev.arrival_time) == minute]
        for ev in spawns_this_minute:
            # Reset their status for the visualizer so they physically drive
            if ev.charger_type != "N/A":
                ev.status = 'driving' 
                
                # --- FIX 2: Restore their battery for the visualizer! ---
                ev.current_soc = ev.initial_soc 
                
            active_evs.append(ev)

        # Update physics
        for ev in active_evs:
            if ev.status == 'driving':
                ev.drive(speed=0.5) 
            
            # Remove them when their parking time is up
            if minute >= ev.departure_time and ev.status != 'stranded':
                ev.status = 'finished'

        visual_crashed_total = len([ev for ev in active_evs if ev.status == 'stranded'])
        active_evs = [ev for ev in active_evs if ev.status != 'finished']

        # --- DRAWING ---
        screen.fill(BG_COLOR)

        for i in range(int(city.size) + 1):
            pos = i * scale
            pygame.draw.line(screen, GRID_COLOR, (pos, 0), (pos, WINDOW_SIZE))
            pygame.draw.line(screen, GRID_COLOR, (0, pos), (WINDOW_SIZE, pos))

        hovered_station = None

        for st in city.stations:
            x = int(st['location'][0] * scale)
            y = int(st['location'][1] * scale)
            color = HUB_COLOR if st['type'] == "SUPER-HUB" else NORMAL_COLOR
            size = 20 if st['type'] == "SUPER-HUB" else 12
            
            station_rect = pygame.Rect(x - size//2, y - size//2, size, size)
            pygame.draw.rect(screen, color, station_rect)
            
            if station_rect.collidepoint(mouse_x, mouse_y):
                hovered_station = st

        for ev in active_evs:
            x = int(ev.location[0] * scale)
            y = int(ev.location[1] * scale)
            
            if ev.status == 'stranded':
                color = EV_CRASHED; radius = 4
            elif ev.status == 'driving':
                color = EV_DRIVING; radius = 4
            else: 
                # EV is parked at the station
                color = EV_PARKED; radius = 2 
            pygame.draw.circle(screen, color, (x, y), radius)

        # --- DRAW TOOLTIP ---
        if hovered_station:
            total_parked = 0
            total_fast_power = 0.0
            total_slow_power = 0.0
            
            current_timeslot = minute // 15
            
            for ev in active_evs:
                if ev.assigned_station_id == hovered_station['id'] and ev.status not in ['driving', 'stranded']:
                    total_parked += 1 
                    
                    if solution and (ev.id, current_timeslot) in vars_dict:
                        power_drawing = solution.get_value(vars_dict[(ev.id, current_timeslot)])
                        
                        if ev.charger_type == 'fast':
                            total_fast_power += power_drawing
                        else:
                            total_slow_power += power_drawing
            
            total_fast = hovered_station['chargers']['fast']
            total_slow = hovered_station['chargers']['slow']
            grid_limit = hovered_station['p_max']
            
            # 1. Calculate Power Load
            current_power = total_fast_power + total_slow_power
            
            # 2. Calculate Equivalent Plugs in Use
            fast_in_use = min(total_fast, int(np.ceil(total_fast_power / 50.0)))
            slow_in_use = min(total_slow, int(np.ceil(total_slow_power / 11.0)))
            total_plugs_in_use = fast_in_use + slow_in_use
            
            # 3. Calculate Queue/Idle (Cars parked but not actively getting a full plug's worth of power)
            queue_size = max(0, total_parked - total_plugs_in_use)
            
            lines = [
                f"Station {hovered_station['id']} ({hovered_station['type']})",
                f"Power: {current_power:.1f} / {grid_limit:.1f} kW",
                f"Active Fast: {fast_in_use} / {total_fast}",
                f"Active Slow: {slow_in_use} / {total_slow}",
                f"Cars Parked: {total_parked}",
                f"Queued / Idle: {queue_size}"
            ]
            
            box_width = 200
            box_height = len(lines) * 20 + 10
            tip_x = mouse_x + 15
            tip_y = mouse_y + 15
            if tip_x + box_width > WINDOW_SIZE: tip_x = mouse_x - box_width - 10
            if tip_y + box_height > WINDOW_SIZE: tip_y = mouse_y - box_height - 10
            
            s = pygame.Surface((box_width, box_height), pygame.SRCALPHA)
            s.fill(TOOLTIP_BG)
            screen.blit(s, (tip_x, tip_y))
            pygame.draw.rect(screen, TEXT_COLOR, (tip_x, tip_y, box_width, box_height), 1) 
            
            for idx, line in enumerate(lines):
                text_surface = small_font.render(line, True, TEXT_COLOR)
                screen.blit(text_surface, (tip_x + 8, tip_y + 5 + (idx * 20)))

        time_str = f"Time: {minute // 60:02d}:{minute % 60:02d}"
        stats_str = f"Active EVs: {len(active_evs)} | Crashed: {visual_crashed_total}"
        screen.blit(font.render(time_str, True, TEXT_COLOR), (10, 10))
        screen.blit(font.render(stats_str, True, TEXT_COLOR), (10, 35))
        
        pygame.display.flip()
        clock.tick(FPS)
        minute += MINUTES_PER_FRAME

    print("--- Simulation Ended ---")
    pygame.quit()

if __name__ == "__main__":
    run_merged_simulation()