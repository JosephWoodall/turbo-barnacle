import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from src.reinforcement_learning.route_optimization.road_network.road_network_environment import RoadNetworkEnvironment

class DQNAgent(nn.Module):
    def __init__(self, num_cities, learning_rate=0.001, discount_factor=0.9, epsilon=0.1):
        super(DQNAgent, self).__init__()
        self.num_cities = num_cities
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon

        self.fc1 = nn.Linear(num_cities, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, num_cities)

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def choose_action(self, state):
        if random.random() < self.epsilon:
            # Explore: choose a random action
            action = random.randint(0, self.num_cities - 1)
        else:
            # Exploit: choose the action with the highest Q-value
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.forward(state_tensor)
            action = q_values.argmax().item()
        return action

    def update_q_values(self, states, actions, rewards, next_states, dones):
        states_tensor = torch.FloatTensor(states)
        actions_tensor = torch.LongTensor(actions)
        rewards_tensor = torch.FloatTensor(rewards)
        next_states_tensor = torch.FloatTensor(next_states)

        q_values = self.forward(states_tensor)
        q_values = q_values.gather(1, actions_tensor.unsqueeze(1)).squeeze(1)

        next_q_values = self.forward(next_states_tensor)
        next_max_q_values = next_q_values.max(1)[0].detach()
        target_q_values = rewards_tensor + self.discount_factor * next_max_q_values * (1 - dones)

        loss = F.mse_loss(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

# DQN Training
env = RoadNetworkEnvironment()
agent = DQNAgent(env.num_cities)

num_episodes = 1000
batch_size = 32
replay_memory = []

for episode in range(num_episodes):
    env.reset()
    done = False
    while not done:
        state = env.current_city
        action = agent.choose_action(state)
        next_state, reward, done = env.step(action)
        replay_memory.append((state, action, reward, next_state, done))

        if len(replay_memory) >= batch_size:
            # Update the agent using experience replay
            batch = random.sample(replay_memory, batch_size)
            states, actions, rewards, next_states, dones = zip(*batch)
            agent.update_q_values(states, actions, rewards, next_states, dones)

# Testing the learned policy
env.reset()
done = False
while not done:
    state = env.current_city
    action = agent.choose_action(state)
    next_state, reward, done = env.step(action)
    print("Current City:", env.cities[state])
    print("Action:", env.cities[action])
    print("Reward:", reward)
    print("Next City:", env.cities[next_state])
    print("------------")
