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
    def __init__(self, input_dim, hidden_dim=128, output_dim=4, lr=0.001, gamma=0.99,
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
        self.memory = deque(maxlen=10000)
        self.batch_size = 64

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.output_dim - 1)
        state = torch.FloatTensor(np.array(state))
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
def get_state(tls_id):
    lane_ids = traci.trafficlight.getControlledLanes(tls_id)
    density, queue, wait_time, throughput = [], [], [], []
    for lane in lane_ids:
        density.append(traci.lane.getLastStepVehicleNumber(lane) / max(1, traci.lane.getLength(lane)))
        queue.append(traci.lane.getLastStepHaltingNumber(lane))
        wait_time.append(traci.lane.getWaitingTime(lane))
        throughput.append(traci.lane.getLastStepVehicleNumber(lane))
    state = np.array(density + queue + wait_time + throughput)
    return (state - np.mean(state)) / (np.std(state) + 1e-6)

# --------------------- Main Training Loop ---------------------
def train(num_episodes=250, max_steps=3000):
    tls_id = "J1"
    sumo_config = "5/beta.sumocfg"

    # Initialize SUMO once to get input/output dims
    traci.start(["sumo", "-c", sumo_config])
    input_dim = len(get_state(tls_id))
    phases = traci.trafficlight.getAllProgramLogics(tls_id)[0].phases
    output_dim = len(phases)
    traci.close()

    agent = DQNAgent(input_dim=input_dim, output_dim=output_dim)

    stats = {
        "Episode": [],
        "Reward": [],
        "Loss": [],
        "WaitingTime": [],
        "AvgWaitingPerVehicle": [],
        "Throughput": [],
        "TotalVehicles": [],
        "GreenTime": [],
        "RedTime": [],
        "YellowTime": []
    }

    for episode in range(num_episodes):
        if traci.isLoaded():
            traci.close()
        traci.start(["sumo", "-c", sumo_config, "--start"])

        state = get_state(tls_id)
        total_reward = 0
        total_loss = 0
        loss_count = 0
        total_waiting_time = 0
        total_throughput = 0
        seen_vehicles = set()
        vehicle_waiting_times = dict()
        green_time, yellow_time, red_time = 0, 0, 0

        phases = traci.trafficlight.getAllProgramLogics(tls_id)[0].phases
        is_green = [any(c in p.state for c in "gG") for p in phases]
        is_yellow = [any(c in p.state for c in "yY") for p in phases]

        step = 0
        while traci.simulation.getMinExpectedNumber() > 0 and step < max_steps:
            action = agent.select_action(state)
            traci.trafficlight.setPhase(tls_id, action)
            traci.simulationStep()

            current_phase = traci.trafficlight.getPhase(tls_id)
            if is_yellow[current_phase]: yellow_time += 1
            elif is_green[current_phase]: green_time += 1
            else: red_time += 1

            vehicle_ids = traci.vehicle.getIDList()
            for vid in vehicle_ids:
                seen_vehicles.add(vid)
                vehicle_waiting_times[vid] = vehicle_waiting_times.get(vid, 0) + traci.vehicle.getWaitingTime(vid)

            waiting = sum(traci.vehicle.getWaitingTime(vid) for vid in vehicle_ids)
            throughput = sum(1 for vid in vehicle_ids if traci.vehicle.getSpeed(vid) > 0.1)
            reward = throughput - 0.5 * waiting

            next_state = get_state(tls_id)
            agent.store((state, action, reward, next_state))
            loss = agent.train_step()
            if loss is not None:
                total_loss += loss
                loss_count += 1

            state = next_state
            total_reward += reward
            total_waiting_time += waiting
            total_throughput += throughput
            step += 1

        agent.update_target()
        agent.epsilon = max(agent.epsilon * agent.epsilon_decay, agent.epsilon_min)

        total_vehicles = len(seen_vehicles)
        avg_waiting_time = sum(vehicle_waiting_times.values()) / total_vehicles if total_vehicles > 0 else 0
        avg_loss = total_loss / loss_count if loss_count > 0 else 0

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

        print(f"""
Episode {episode + 1} Summary:
  Total Reward           : {total_reward:.2f}
  Average Loss           : {avg_loss:.4f}
  Total Waiting Time     : {total_waiting_time:.2f}
  Average Waiting/Vehicle: {avg_waiting_time:.2f}
  Total Throughput       : {total_throughput}
  Total Unique Vehicles  : {total_vehicles}
  Actual Green Time      : {green_time} steps
  Actual Red Time        : {red_time} steps
  Actual Yellow Time     : {yellow_time} steps
  Epsilon                : {agent.epsilon:.4f}
""")

        traci.close()

    # Save stats and model
    df = pd.DataFrame(stats)
    df.to_csv("dqn_episode_stats_super_traffic.csv", index=False)
    torch.save(agent.q_network.state_dict(), "dqn_model.pth")
    print("Training complete. Stats saved to dqn_episode_stats.csv and model saved to dqn_model.pth.")

# --------------------- Run Training ---------------------
if __name__ == "__main__":
    train()
