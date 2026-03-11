import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv

# Import your actual backend classes
from citygrid import CityGrid
from traffic_generator import TrafficGenerator

class EVRoutingParallelEnv(ParallelEnv):
    metadata = {"render_modes": ["human"], "name": "ev_routing_parallel_v3"}

    def __init__(self, city_size_km=5, num_stations=12):
        super().__init__()
        
        # Initialize your actual backend simulation
        self.city_grid = CityGrid(size_km=city_size_km, num_stations=num_stations)
        self.traffic_generator = TrafficGenerator(self.city_grid)
        
        self.num_stations = num_stations
        self.max_cars_per_day = 2000 # Buffer for high traffic days
        self.current_minute = 0
        
        # Track active EV objects {agent_id: EV_Object}
        self.active_evs_dict = {}
        
        # RL state trackers
        self.possible_agents = [f"car_{i}" for i in range(self.max_cars_per_day)]
        self.agents = [] 

        # Action Space: Pick a station ID (0 to 11)
        self.action_spaces = {agent: spaces.Discrete(self.num_stations) for agent in self.possible_agents}
        
        # Observation Space: [Car_X, Car_Y, Car_SoC, Queue_0, ... Queue_11]
        obs_size = 3 + self.num_stations 
        self.observation_spaces = {
            agent: spaces.Box(low=0.0, high=1.0, shape=(obs_size,), dtype=np.float32) 
            for agent in self.possible_agents
        }

    def reset(self, seed=None, options=None):
        """Restarts the day at Minute 0."""
        # Reset backend
        self.city_grid = CityGrid(size_km=self.city_grid.size, num_stations=self.num_stations)
        self.traffic_generator = TrafficGenerator(self.city_grid)
        self.current_minute = 0
        self.active_evs_dict = {}
        
        # Fast-forward until the first cars spawn
        self.agents = []
        while not self.agents and self.current_minute < 1440:
            spawned_evs = self.traffic_generator.step(self.current_minute)
            for ev in spawned_evs:
                agent_id = f"car_{ev.id}"
                self.agents.append(agent_id)
                self.active_evs_dict[agent_id] = ev
            if not self.agents:
                self.current_minute += 1

        observations = {agent: self._get_obs(agent) for agent in self.agents}
        infos = {agent: {} for agent in self.agents}
        
        return observations, infos

    def _get_obs(self, agent_id):
        """Pulls exact data from your EV.py and CityGrid.py, then normalizes it."""
        ev = self.active_evs_dict[agent_id]
        
        # Normalize coordinates by city size (e.g., 5km)
        car_x = np.clip(ev.location[0] / self.city_grid.size, 0.0, 1.0)
        car_y = np.clip(ev.location[1] / self.city_grid.size, 0.0, 1.0)
        
        # SoC is already a percentage (0.0 to 1.0)
        car_soc = ev.current_soc
        
        # Normalize queues (assuming a max realistic queue of 20 cars)
        # NOW PULLING DIRECTLY FROM CITYGRID!
        queues = [np.clip(self.city_grid.get_queue(i) / 20.0, 0.0, 1.0) for i in range(self.num_stations)]
        
        return np.array([car_x, car_y, car_soc] + queues, dtype=np.float32)

    def step(self, actions_dict):
        """Routes the EVs and ticks the simulation forward."""
        observations, rewards, terminations, truncations, infos = {}, {}, {}, {}, {}

        # Fictitious play to prevent multi-agent stampedes
        fictitious_queues = {i: 0 for i in range(self.num_stations)}

        # 1. Process all routing decisions made this minute
        for agent_id, chosen_station_id in actions_dict.items():
            ev = self.active_evs_dict[agent_id]
            target_station = self.city_grid.stations[chosen_station_id]
            
            # Use Manhattan distance math to check if car survives the trip
            diff = np.abs(ev.location - target_station['location'])
            distance_km = diff[0] + diff[1]
            energy_needed = distance_km * ev.consumption_per_km 
            
            # Calculate effective queue for reward penalty (Real Queue from Grid + Ghost Cars)
            real_queue = self.city_grid.get_queue(chosen_station_id)
            effective_queue = real_queue + fictitious_queues[chosen_station_id]

            if (ev.current_soc * ev.battery_capacity) < energy_needed:
                # Fatal Penalty: Car died on the way!
                rewards[agent_id] = -100.0
                ev.status = 'stranded'
            else:
                # Success: Positive reward minus the queue penalty
                rewards[agent_id] = 10.0 - (effective_queue * 1.5)
                
                # UPDATE THE ACTUAL BACKEND QUEUE!
                self.city_grid.add_to_queue(chosen_station_id)
                fictitious_queues[chosen_station_id] += 1
                
                ev.status = 'waiting'
                ev.target_location = target_station['location']

            # Agents route once, then terminate
            terminations[agent_id] = True
            truncations[agent_id] = False
            infos[agent_id] = {}
            
            # Clean up the dictionary (car goes to backend logic now)
            del self.active_evs_dict[agent_id]

        # 2. Fast-forward the clock until the next cars spawn or the day ends
        self.agents = []
        self.current_minute += 1
        
        while not self.agents and self.current_minute < 1440:
            spawned_evs = self.traffic_generator.step(self.current_minute)
            for ev in spawned_evs:
                agent_id = f"car_{ev.id}"
                self.agents.append(agent_id)
                self.active_evs_dict[agent_id] = ev
                
            if not self.agents:
                self.current_minute += 1

        # 3. Generate observations for the newly spawned cars
        for new_agent in self.agents:
            observations[new_agent] = self._get_obs(new_agent)

        return observations, rewards, terminations, truncations, infos

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]