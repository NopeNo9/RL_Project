import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import traci
import pandas as pd

# --------------------- Policy Network ---------------------
class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return self.softmax(x)

# --------------------- Policy Gradient Agent ---------------------
class PGAgent:
    def __init__(self, input_dim, hidden_dim, output_dim, lr=0.0001):
        self.policy = PolicyNetwork(input_dim, hidden_dim, output_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.log_probs = []
        self.rewards = []

    def select_action(self, state):
        state = torch.FloatTensor(state)
        probs = self.policy(state)
        dist = Categorical(probs)
        action = dist.sample()
        self.log_probs.append(dist.log_prob(action))
        return action.item()

    def update_policy(self, gamma=0.9):
        R = 0
        returns = []
        for r in reversed(self.rewards):
            R = r + gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        loss = sum(-log_prob * ret for log_prob, ret in zip(self.log_probs, returns))
        # Optional: encourage exploration with entropy
        # entropy = sum(-prob * prob.log() for prob in self.log_probs)
        # loss -= 0.01 * entropy

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.log_probs = []
        self.rewards = []

# --------------------- State and Reward ---------------------
def get_full_state(tls_id):
    lane_ids = traci.trafficlight.getControlledLanes(tls_id)
    vehicle_counts = np.array([traci.lane.getLastStepVehicleNumber(lane) for lane in lane_ids], dtype=np.float32)
    waiting_times = np.array([traci.lane.getWaitingTime(lane) for lane in lane_ids], dtype=np.float32)
    queue_lengths = np.array([traci.lane.getLastStepHaltingNumber(lane) for lane in lane_ids], dtype=np.float32)
    state = np.concatenate([vehicle_counts, waiting_times, queue_lengths])
    return (state - np.mean(state)) / (np.std(state) + 1e-6)

def get_vehicle_counts(tls_id):
    lane_ids = traci.trafficlight.getControlledLanes(tls_id)
    return np.array([traci.lane.getLastStepVehicleNumber(lane) for lane in lane_ids], dtype=np.float32)

def compute_reward(waiting_times):
    return -np.mean(waiting_times)

# --------------------- Main Training Loop ---------------------
def train(num_episodes=250, max_steps=3000, max_lane_capacity=20):
    tls_id = "J1"
    sumo_config = "5/beta.sumocfg"

    if traci.isLoaded():
        traci.close()
    traci.start(["sumo", "-c", sumo_config])
    input_dim = len(get_full_state(tls_id))
    output_dim = 2  # NSG, EWG
    traci.close()

    agent = PGAgent(input_dim, hidden_dim=256, output_dim=output_dim)
    phase_map = {0: 2, 1: 0}  # 0 = NSG → phase 2, 1 = EWG → phase 0
    last_phase = -1

    # Logging
    episode_rewards = []
    episode_waiting_times = []
    avg_waiting_times = []
    episode_throughputs = []
    total_vehicles_list = []

    for episode in range(num_episodes):
        if traci.isLoaded():
            traci.close()
        traci.start(["sumo", "-c", sumo_config, "--start"])

        total_reward = 0
        total_waiting_time = 0
        total_throughput = 0
        vehicle_waiting_times = {}
        seen_vehicles = set()
        action_counts = {0: 0, 1: 0}
        step = 0
        state = get_full_state(tls_id)

        try:
            while traci.simulation.getMinExpectedNumber() > 0 and step < max_steps:
                action = agent.select_action(state)
                action_counts[action] += 1
                new_green = phase_map[action]
                print(f"Episode {episode+1}, Step {step}: Action = {action} → Phase {new_green}")

                # Yellow transition
                if last_phase != -1 and new_green != last_phase:
                    yellow_phase = last_phase + 1
                    traci.trafficlight.setPhase(tls_id, yellow_phase)
                    for _ in range(3):
                        traci.simulationStep()
                        step += 1

                # Green phase
                traci.trafficlight.setPhase(tls_id, new_green)
                for _ in range(10):
                    traci.simulationStep()
                    step += 1

                    vehicle_ids = traci.vehicle.getIDList()
                    for vid in vehicle_ids:
                        seen_vehicles.add(vid)
                        wait = traci.vehicle.getWaitingTime(vid)
                        vehicle_waiting_times[vid] = vehicle_waiting_times.get(vid, 0) + wait

                    total_waiting_time += sum(traci.vehicle.getWaitingTime(vid) for vid in vehicle_ids)
                    total_throughput += sum(1 for vid in vehicle_ids if traci.vehicle.getSpeed(vid) > 0.1)

                last_phase = new_green
                waiting = [traci.lane.getWaitingTime(lane) for lane in traci.trafficlight.getControlledLanes(tls_id)]
                reward = compute_reward(waiting)
                agent.rewards.append(reward)
                total_reward += reward

                if episode > 50 and np.any(get_vehicle_counts(tls_id) > max_lane_capacity):
                    print(f"Episode {episode+1}: Terminated early due to congestion")
                    break

                state = get_full_state(tls_id)

        except Exception as e:
            print(f"Episode {episode+1} failed: {e}")

        finally:
            agent.update_policy()
            total_vehicles = len(seen_vehicles)
            avg_wait = sum(vehicle_waiting_times.values()) / total_vehicles if total_vehicles > 0 else 0

            print(f"""
Episode {episode + 1} Summary:
  Total Reward           : {total_reward:.2f}
  Total Waiting Time     : {total_waiting_time:.2f}
  Average Waiting/Vehicle: {avg_wait:.2f}
  Total Throughput       : {total_throughput}
  Total Unique Vehicles  : {total_vehicles}
  Action Distribution    : {action_counts}
""")

            episode_rewards.append(total_reward)
            episode_waiting_times.append(total_waiting_time)
            avg_waiting_times.append(avg_wait)
            episode_throughputs.append(total_throughput)
            total_vehicles_list.append(total_vehicles)

            traci.close()

    df = pd.DataFrame({
        "Episode": list(range(1, num_episodes + 1)),
        "Reward": episode_rewards,
        "WaitingTime": episode_waiting_times,
        "AvgWaitingPerVehicle": avg_waiting_times,
        "Throughput": episode_throughputs,
        "TotalVehicles": total_vehicles_list
    })
    df.to_csv("pg_balint_episode_stats_super_traffic.csv", index=False)
    print("Training complete. Stats saved to pg_balint_fixed_episode_stats.csv")

# --------------------- Run ---------------------
if __name__ == "__main__":
    train()
