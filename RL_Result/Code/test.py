import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import traci
import pandas as pd

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    def __init__(self, input_dim, hidden_dim, output_dim, lr=0.001):
        self.policy = PolicyNetwork(input_dim, hidden_dim, output_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.log_probs = []
        self.rewards = []
        self.output_dim = output_dim

    def select_action(self, state):
        state = torch.FloatTensor(state)
        probs = self.policy(state)
        dist = Categorical(probs)
        action = dist.sample()
        self.log_probs.append(dist.log_prob(action))
        return action.item() % self.output_dim  # Ensure action is within valid range

    def update_policy(self, gamma=0.99, entropy_coef=0.1):
        R = 0
        returns = []
        for r in reversed(self.rewards):
            R = r + gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)

        loss = 0
        entropy = 0
        for log_prob, ret in zip(self.log_probs, returns):
            loss += -log_prob * ret
            entropy += -log_prob.exp() * log_prob

        total_loss = loss - entropy_coef * entropy.sum()
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        self.log_probs = []
        self.rewards = []

# --------------------- State Representation ---------------------
def get_state(tls_id):
    lane_ids = traci.trafficlight.getControlledLanes(tls_id)
    density = []
    queue = []
    wait_time = []
    throughput = []

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

    # Initialize SUMO to get phase information
    if traci.isLoaded():
        traci.close()
    try:
        traci.start(["sumo", "-c", sumo_config])
        input_dim = len(get_state(tls_id))
        phases = traci.trafficlight.getAllProgramLogics(tls_id)[0].phases
        output_dim = len(phases)
        print("Phase Definitions:")
        for i, phase in enumerate(phases):
            print(f"Phase {i}: State = {phase.state}, Duration = {phase.duration}")
        traci.close()
    except Exception as e:
        print("Failed to start SUMO initially:", e)
        return

    hidden_dim = 128
    agent = PGAgent(input_dim, hidden_dim, output_dim)

    episode_rewards = []
    episode_waiting_times = []
    episode_throughputs = []
    total_vehicles_list = []
    avg_waiting_times = []
    green_times = []
    red_times = []
    yellow_times = []

    for episode in range(num_episodes):
        if traci.isLoaded():
            traci.close()

        try:
            traci.start(["sumo", "-c", sumo_config, "--start"])
            state = get_state(tls_id)
            total_reward = 0
            total_waiting_time = 0
            total_throughput = 0
            seen_vehicles = set()
            vehicle_waiting_times = {}
            green_time_actual = 0
            red_time_actual = 0
            yellow_time_actual = 0
            step = 0

            # Define green and yellow phases
            phases = traci.trafficlight.getAllProgramLogics(tls_id)[0].phases
            is_green = [("g" in p.state.lower() or "G" in p.state.lower()) for p in phases]  # Simplified to detect any green
            is_yellow = [("y" in p.state.lower() or "Y" in p.state.lower()) for p in phases]

            while traci.simulation.getMinExpectedNumber() > 0 and step < max_steps:
                action = agent.select_action(state)
                print(f"Episode {episode + 1}, Step {step}: Selected action = {action}")
                traci.trafficlight.setPhase(tls_id, action)
                traci.simulationStep()
                next_state = get_state(tls_id)

                # Get current phase and log state
                current_phase = traci.trafficlight.getPhase(tls_id)
                phase_state = phases[current_phase].state
                print(f"Step {step}: Current phase = {current_phase}, State = {phase_state}")

                # Accumulate time based on phase
                if is_yellow[current_phase]:
                    yellow_time_actual += 1
                    print(f"Step {step}: Yellow phase")
                elif is_green[current_phase]:
                    green_time_actual += 1
                    print(f"Step {step}: Green phase")
                else:
                    red_time_actual += 1
                    print(f"Step {step}: Red phase")

                # Collect vehicle data
                vehicle_ids = traci.vehicle.getIDList()
                for vid in vehicle_ids:
                    seen_vehicles.add(vid)
                    if vid not in vehicle_waiting_times:
                        vehicle_waiting_times[vid] = traci.vehicle.getWaitingTime(vid)
                    else:
                        vehicle_waiting_times[vid] += traci.vehicle.getWaitingTime(vid)

                step_waiting_time = sum(traci.vehicle.getWaitingTime(vid) for vid in vehicle_ids)
                step_throughput = sum(1 for vid in vehicle_ids if traci.vehicle.getSpeed(vid) > 0.1)
                reward = step_throughput - 0.5 * step_waiting_time

                agent.rewards.append(reward)
                total_reward += reward
                total_waiting_time += step_waiting_time
                total_throughput += step_throughput

                state = next_state
                step += 1

            agent.update_policy()

            total_vehicles = len(seen_vehicles)
            avg_waiting_time = sum(vehicle_waiting_times.values()) / total_vehicles if total_vehicles > 0 else 0

            green_times.append(green_time_actual)
            red_times.append(red_time_actual)
            yellow_times.append(yellow_time_actual)
            episode_rewards.append(total_reward)
            episode_waiting_times.append(total_waiting_time)
            episode_throughputs.append(total_throughput)
            total_vehicles_list.append(total_vehicles)
            avg_waiting_times.append(avg_waiting_time)

            print(f"""
Episode {episode + 1} Summary:
  Total Reward           : {total_reward:.2f}
  Total Waiting Time     : {total_waiting_time:.2f}
  Average Waiting/Vehicle: {avg_waiting_time:.2f}
  Total Throughput       : {total_throughput}
  Total Unique Vehicles  : {total_vehicles}
  Actual Green Time      : {green_time_actual} steps
  Actual Red Time        : {red_time_actual} steps
  Actual Yellow Time     : {yellow_time_actual} steps
""")

        except Exception as e:
            print(f"Episode {episode + 1} failed: {e}")

        finally:
            if traci.isLoaded():
                traci.close()

    # Save episode statistics
    data = {
        "Episode": list(range(1, num_episodes + 1)),
        "Reward": episode_rewards,
        "WaitingTime": episode_waiting_times,
        "AvgWaitingPerVehicle": avg_waiting_times,
        "Throughput": episode_throughputs,
        "TotalVehicles": total_vehicles_list,
        "GreenTime": green_times,
        "RedTime": red_times,
        "YellowTime": yellow_times
    }
    df = pd.DataFrame(data)
    df.to_csv("pg_episode_stats_fork_1.csv", index=False)
    print("PG episode statistics saved to pg_episode_stats.csv")

# --------------------- Run Training ---------------------
if __name__ == "__main__":
    train()