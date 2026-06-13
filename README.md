#  Electric Vehicle Taxi Scheduling

A reinforcement learning research project for the **Bachelor's Thesis in Computer Science** at Aristotle University of Thessaloniki. This project investigates intelligent scheduling and charging strategies for a large fleet of electric vehicle taxis using a Proximal Policy Optimization (PPO) agent, benchmarked against a greedy heuristic baseline.

---

##  Overview

Managing a fleet of EV taxis is a complex optimization problem: every taxi must balance serving customers, navigating to charging stations before running out of battery, and repositioning itself in high-demand areas — all simultaneously, across a full 24-hour day.

This project simulates a city of **750 electric taxis** operating on a **20×20 km grid** with **16 charging stations** (4 super-hubs and 12 standard). Two scheduling strategies are evaluated:

- **Greedy Heuristic Baseline** — A rule-based algorithm that sends taxis to the nearest charging station when battery drops below 25%, and rebalances them toward the city center when they drift too far.
- **PPO AI Agent** — A deep reinforcement learning agent trained for 3,000,000 timesteps using Stable-Baselines3, learning to make nuanced decisions from 48 environment features.

Performance is evaluated across **10 distinct demand profiles** (Normal, Commuter, Saturday, Sunday, Low, High Stress, Flattened, Bimodal, Event, Early Spike) and compared on three KPIs: **net profit**, **service rate**, and **average customer wait time**.

An interactive **web-based visualizer** created in assistance with the v0 bot lets you replay any simulation, watch taxis move in real-time on a canvas map, and inspect live metrics including fleet SoC, charging queues, and hourly demand patterns.

---

##  Project Structure

```
Electric-Vehicle-scheduling/
├── environments/
│   ├── ev.py                  # EVTaxi class — state machine for a single taxi
│   ├── citygrid.py            # CityMap — grid, stations, charger management
│   ├── clients.py             # ClientManager — demand generation & waitlist
│   ├── traffic_generator.py   # TrafficGenerator — fleet + demand orchestration
│   └── ev_gym_env.py          # EVFleetEnv — Gymnasium environment for RL training
│
├── baselines/
│   ├── benchmark.py           # GreedyHeuristicBaseline algorithm
│   └── main_simulation.py     # Run baseline across all 10 demand profiles
│
├── reinforcement_learning/
│   ├── train_ppo.py           # Train PPO from scratch (3M steps, 4 parallel envs)
│   ├── resume_train_ppo.py    # Resume training from a saved checkpoint
│   └── evaluate_model.py      # Evaluate PPO across all 10 demand profiles
│
├── server/
│   └── api_server.py          # FastAPI backend — runs simulations on demand
│
├── ev-visualizer/             # React + Vite frontend
│   ├── src/
│   │   ├── App.jsx            # Main app — map, controls, charts, metrics
│   │   ├── main.jsx           # React entry point
│   │   └── index.css          # Base reset styles
│   ├── index.html
│   ├── package.json
│   ├── vite.config.js
│   └── bot-chat/              # Chat log with the v0 bot used to scaffold the frontend (full transparency)
│
├── ppo_model.zip              # Trained PPO model (git-ignored)
├── requirements.txt           # Python dependencies
└── README.md
```

---

##  Getting Started

### Prerequisites

- **Python 3.10+**
- **Node.js 20+** and **npm**

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/Electric-Vehicle-scheduling.git
cd Electric-Vehicle-scheduling
```

### 2. Set Up the Python Environment

```bash
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Key Python dependencies:

| Package | Purpose |
|---|---|
| `gymnasium` | RL environment interface |
| `stable-baselines3` | PPO implementation |
| `numpy` | Numerical computation |
| `fastapi` | API server |
| `uvicorn` | ASGI server |
| `pydantic` | Request validation |

### 3. Set Up the Frontend

```bash
cd ev-visualizer
npm install
```

---

##  Running the Visualizer

The visualizer requires both the FastAPI backend and the Vite development server to run simultaneously.

**Terminal 1 — Start the API server:**

```bash
# From the project root (with venv active)
python server/api_server.py
```

The API will be available at `http://127.0.0.1:8000`. You can verify it's running by visiting `http://127.0.0.1:8000/health`.

**Terminal 2 — Start the frontend:**

```bash
cd ev-visualizer
npm run dev
```

Open your browser at `http://localhost:5173` and you'll see the interactive simulator.

> **Note:** To use the PPO AI agent, you need a trained model file (`ppo_model.zip`) in the project root. See the [Training](#-training-the-ai-agent) section below.

---

##  Environment Design

### City Grid (`citygrid.py`)

The city is a **20×20 km** continuous grid, snapped to a **0.2 km (200 m)** resolution. Distances are computed using Manhattan distance, reflecting real urban driving patterns.

**Charging Infrastructure:**

| Type | Count | Fast Chargers | Slow Chargers | Power Limit |
|---|---|---|---|---|
| Super-Hub | 4 | 7 | 10 | 1500 kW |
| Standard | 12 | 3 | 5 | 300 kW |

**Electricity Pricing:**

| Charger | Peak (09:00–22:00) | Off-Peak |
|---|---|---|
| Fast (50 kW) | €0.65/kWh | €0.50/kWh |
| Slow (11 kW) | €0.45/kWh | €0.35/kWh |

### EV Taxi (`ev.py`)

Each taxi is modeled with a realistic energy model:

- **Battery:** 40 kWh capacity
- **Consumption:** 0.17 kWh/km
- **States:** `IDLE` → `WITH_CUSTOMER` / `MOVING_TO_STATION` / `REBALANCING` → `WAITING_FOR_CHARGER` → `CHARGING` → `IDLE` (or `STRANDED` if battery hits 0)
- **Charging terminates** at 95% SoC (rounded to 100%)

### Demand Profiles (`clients.py`)

Ten hourly demand profiles define how many customer requests arrive each hour of the day:

| # | Profile | Description |
|---|---|---|
| 0 | Normal | Standard weekday with morning peak |
| 1 | Commuter | Heavy morning and evening commuter peaks |
| 2 | Saturday | Afternoon and evening leisure peaks |
| 3 | Sunday | Low, uniform weekend demand |
| 4 | Low | Minimal traffic — efficiency baseline |
| 5 | High Stress | Multiple extreme peaks — stress test |
| 6 | Flattened | No clear peaks, uniform day |
| 7 | Bimodal | Strong AM + PM peaks, quiet noon |
| 8 | Event | Sudden mid-day local spike |
| 9 | Early Spike | Heavy demand in first hours |

Trip types are drawn from {Centre→Centre, Centre→Periphery, Periphery→Centre, Periphery→Periphery} with time-of-day-weighted probabilities. Customers abandon the waitlist after **20 minutes**.

**Fare model:** `max(€4.00, €1.80 + €0.90 × distance_km)`

---

##  Reinforcement Learning

### Observation Space (48 features)

| Index | Feature |
|---|---|
| 0–1 | Cyclical time encoding (sin/cos of minute-of-day) |
| 2 | Taxi state of charge (SoC) |
| 3–4 | Taxi (x, y) position (normalised) |
| 5–20 | Distance to each of 16 stations (normalised) |
| 21–36 | Queue length at each station (normalised) |
| 37–45 | 3×3 spatial demand heatmap of waiting customers |
| 46 | Low-SoC ratio — fraction of fleet below 40% (stampede predictor) |
| 47 | Waitlist pressure — normalised customer backlog |

### Action Space (18 discrete actions)

| Action | Meaning |
|---|---|
| 0–15 | Dispatch to charging station `i` |
| 16 | Stay IDLE (with cooldown) |
| 17 | Rebalance toward city center |

### Reward Function

The reward signal is shaped around four objectives:

1. **Customer service** — +8 per customer served, −4 per abandoned, −5 per newly stranded taxi
2. **Smart charging** — penalties for charging during peak hours with customers waiting; bonuses for proactive overnight charging
3. **Stampede prevention** — extra reward for early charging when >25% of fleet drops below 40% SoC
4. **Availability** — small penalties for IDLE/REBALANCING taxis with dangerously low battery

### Safety Override

Both during training and evaluation, a **safety shield** forces emergency charging if any IDLE taxi drops below 20% SoC, using a nearest-available-station heuristic. This prevents excessive stranding without overriding the agent's strategic decisions.

---

## Training the AI Agent

### Train from Scratch

```bash
python reinforcement_learning/train_ppo.py
```

This trains a PPO agent for **3,000,000 steps** across **4 parallel environments** and saves the model as `ppo_model.zip`.

**Hyperparameters:**

| Parameter | Value |
|---|---|
| Policy | MLP 256×256 (actor + critic) |
| Learning rate | 3×10⁻⁴ |
| Steps per update | 512 |
| Batch size | 256 |
| Discount factor γ | 0.995 |
| Entropy coefficient | 0.005 |
| Clip range | 0.2 |
| Epochs per update | 10 |

### Resume Training

```bash
python reinforcement_learning/resume_train_ppo.py
```

Loads `ppo_model.zip` and continues training for an additional 1,000,000 steps, resuming TensorBoard logs seamlessly.

### Monitor with TensorBoard

```bash
tensorboard --logdir ./tensorboard_logs/
```

---

## 📊 Evaluation

Run a full benchmark of the PPO agent across all 10 demand profiles:

```bash
python reinforcement_learning/evaluate_model.py
```

Run the greedy baseline benchmark:

```bash
python baselines/main_simulation.py
```

Both scripts output a results table with columns: **Profile | Profit (€) | Service % | Wait (min)**

---

## Visualizer Features

The React frontend connects to the FastAPI backend and provides:

- **Live canvas map** — 750 taxis rendered in real-time on a 20×20 km grid, colour-coded by state, with glowing indicators for active states
- **Taxi density heatmap** — toggle to see concentration of fleet across the city
- **Per-station queue badges** — real-time queue depth on each charging station marker
- **Playback controls** — play, pause, step forward/back, jump to start/end, 5 playback speeds (0.25×–4×)
- **Pan & zoom** — scroll to zoom, drag to pan, reset button
- **Live metrics panel** — avg SoC gauge, total queue count, fleet health gauges, state distribution bar
- **Sparkline charts** — queue-over-time and avg-SoC-over-time for the full simulation day
- **Hourly queue heatmap** — per-hour peak queue intensity bar chart
- **Day-phase timeline** — visual indicator of night/morning/peak/afternoon/evening phases
- **Day results summary** — net profit, service rate, avg wait, stranded count, derived KPIs
- **Algorithm info card** — architectural details of whichever algorithm is selected
- **Keyboard shortcuts** — Space (play/pause), Arrow keys (step), Home/End (jump)

---

## KPIs

| Metric | Description |
|---|---|
| **Net Profit (€)** | `Σ (revenue − charging_cost − €40 leasing)` per taxi, summed over fleet |
| **Service Rate (%)** | `customers_served / (served + abandoned) × 100` |
| **Avg Wait Time (min)** | Mean total wait from request to pickup |
| **Stranded Taxis** | Taxis that reached 0% SoC during the day |

---

## License

This project was created as an academic thesis submission. All code is available for research and educational purposes.

---

## Author

Developed by Pavlos Margaritis at the **Aristotle University of Thessaloniki** as a final-year bachelor's thesis on reinforcement learning applied to smart EV fleet management.
