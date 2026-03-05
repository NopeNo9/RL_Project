import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import traci
import pandas as pd
import random
from collections import deque

# --------------------- Q-Network ---------------------
class QNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        return self.fc2(x)

# --------------------- DQN Agent ---------------------
class DQNAgent:
    def __init__(self, input_dim, hidden_dim=128, output_dim=2, lr=0.001, gamma=0.99,
                 epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995):
        self.q_network = QNetwork(input_dim, hidden_dim, output_dim)
        self.target_network = QNetwork(input_dim, hidden_dim, output_dim)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.criteria = nn.MSELoss()
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.output_dim = output_dim
        self.memory = deque(maxlen=5000)
        self.batch_size = 64

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.output_dim - 1)
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_network(state)
        return torch.argmax(q_values).item()

    def store(self, transition):
        self.memory.append(transition)

    def train_step(self):
        if len(self.memory) < self.batch_size:
            return None

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states = zip(*batch)

        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(np.array(next_states))

        q_values = self.q_network(states)
        q_selected = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        next_q_values = self.target_network(next_states).detach().max(1)[0]
        target_q = rewards + self.gamma * next_q_values

        loss = self.criteria(q_selected, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def update_target(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

# --------------------- State Representation ---------------------
def get_state(tls_id, current_phase):
    lane_ids = traci.trafficlight.getControlledLanes(tls_id)
    density, queue, wait_time = [], [], []
    for lane in lane_ids:
        density.append(traci.lane.getLastStepVehicleNumber(lane) / max(1, traci.lane.getLength(lane)))
        queue.append(traci.lane.getLastStepHaltingNumber(lane))
        wait_time.append(traci.lane.getWaitingTime(lane))

    # One-hot encoding for phase
    phase_vec = np.array([1, 0]) if current_phase == 0 else np.array([0, 1])

    # Grid image encoding (12x12)
    grid = np.zeros((12, 12))
    for vid in traci.vehicle.getIDList():
        pos = traci.vehicle.getPosition(vid)
        x, y = int(pos[0]) % 12, int(pos[1]) % 12
        grid[y, x] = 1
    image_flat = grid.flatten()

    state = np.concatenate([density, queue, wait_time, phase_vec, image_flat])
    return (state - np.mean(state)) / (np.std(state) + 1e-6)

# --------------------- Reward Function ---------------------
def compute_reward(lane_ids, current_phase, action, prev_vehicle_ids):
    β1, β2, β3, β4, β5, β6 = -0.25, -0.25, -0.25, -5, -1, -1
    delay, wait, queue = 0, 0, 0
    for lane in lane_ids:
        speed = traci.lane.getLastStepMeanSpeed(lane)
        limit = traci.lane.getMaxSpeed(lane)
        delay += 1 - (speed / limit if limit > 0 else 0)
        wait += traci.lane.getWaitingTime(lane)
        queue += traci.lane.getLastStepHaltingNumber(lane)

    current_vehicle_ids = set(traci.vehicle.getIDList())
    passed_vehicles = prev_vehicle_ids - current_vehicle_ids
    Vt = len(passed_vehicles)

    # FIXED: Use only vehicles still in simulation
    Tt = sum(traci.vehicle.getAccumulatedWaitingTime(v) for v in passed_vehicles if v in current_vehicle_ids)

    Ct = 1 if action == 1 else 0
    reward = β1 * delay + β2 * wait + β3 * queue + β4 * Ct + β5 * Vt + β6 * Tt
    return reward, current_vehicle_ids

# --------------------- Main Training Loop ---------------------
def train(num_episodes=250, max_steps=3000):
    tls_id = "J1"
    sumo_config = "5/beta.sumocfg"

    # Initialize input_dim
    traci.start(["sumo", "-c", sumo_config])
    current_phase = traci.trafficlight.getPhase(tls_id)
    input_dim = len(get_state(tls_id, current_phase))
    traci.close()

    agent = DQNAgent(input_dim=input_dim, output_dim=2)

    stats = {
        "Episode": [], "Reward": [], "Loss": [], "WaitingTime": [],
        "AvgWaitingPerVehicle": [], "Throughput": [], "TotalVehicles": [],
        "GreenTime": [], "RedTime": [], "YellowTime": []
    }

    for episode in range(num_episodes):
        if traci.isLoaded():
            traci.close()
        traci.start(["sumo", "-c", sumo_config, "--start"])
        step = 0
        current_phase = traci.trafficlight.getPhase(tls_id)
        state = get_state(tls_id, current_phase)

        total_reward = 0
        total_loss = 0
        total_waiting_time = 0
        total_throughput = 0
        loss_count = 0
        seen_vehicles = set()
        vehicle_waiting_times = {}
        green_time, yellow_time, red_time = 0, 0, 0

        lane_ids = traci.trafficlight.getControlledLanes(tls_id)
        prev_vehicle_ids = set(traci.vehicle.getIDList())

        while traci.simulation.getMinExpectedNumber() > 0 and step < max_steps:
            action = agent.select_action(state)
            if action == 1:
                current_phase = (current_phase + 1) % len(traci.trafficlight.getAllProgramLogics(tls_id)[0].phases)
            traci.trafficlight.setPhase(tls_id, current_phase)
            traci.simulationStep()

            current_vehicles = set(traci.vehicle.getIDList())
            reward, prev_vehicle_ids = compute_reward(lane_ids, current_phase, action, prev_vehicle_ids)

            next_state = get_state(tls_id, current_phase)
            agent.store((state, action, reward, next_state))
            loss = agent.train_step()
            if loss is not None:
                total_loss += loss
                loss_count += 1

            # Stats collection
            for vid in current_vehicles:
                seen_vehicles.add(vid)
                vehicle_waiting_times[vid] = vehicle_waiting_times.get(vid, 0) + traci.vehicle.getWaitingTime(vid)

            state = next_state
            total_reward += reward
            total_waiting_time += sum(traci.vehicle.getWaitingTime(v) for v in current_vehicles)
            total_throughput += sum(1 for v in current_vehicles if traci.vehicle.getSpeed(v) > 0.1)
            step += 1

        agent.update_target()
        agent.epsilon = max(agent.epsilon * agent.epsilon_decay, agent.epsilon_min)

        total_vehicles = len(seen_vehicles)
        avg_waiting_time = sum(vehicle_waiting_times.values()) / total_vehicles if total_vehicles else 0
        avg_loss = total_loss / loss_count if loss_count else 0

        stats["Episode"].append(episode + 1)
        stats["Reward"].append(total_reward)
        stats["Loss"].append(avg_loss)
        stats["WaitingTime"].append(total_waiting_time)
        stats["AvgWaitingPerVehicle"].append(avg_waiting_time)
        stats["Throughput"].append(total_throughput)
        stats["TotalVehicles"].append(total_vehicles)
        stats["GreenTime"].append(green_time)
        stats["RedTime"].append(red_time)
        stats["YellowTime"].append(yellow_time)

        print(f"Episode {episode+1}: Reward={total_reward:.2f}, AvgLoss={avg_loss:.4f}, AvgWait={avg_waiting_time:.2f}, Throughput={total_throughput}, Vehicles={total_vehicles}")

        traci.close()

    pd.DataFrame(stats).to_csv("dqn_pan_episode_stats_random.csv", index=False)
    torch.save(agent.q_network.state_dict(), "dqn_pan_model.pth")
    print("Training complete. Results saved.")

# --------------------- Run ---------------------
if __name__ == "__main__":
    train()
