import numpy as np

from src.reinforcement_learning.route_optimization.road_network.road_network_environment import RoadNetworkEnvironment

class QLearningAgent:
    def __init__(self, num_cities, learning_rate=0.1, discount_factor=0.9, epsilon=0.1):
        self.num_cities = num_cities
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.q_table = np.zeros((num_cities, num_cities))

    def choose_action(self, state):
        if np.random.uniform(0, 1) < self.epsilon:
            # Explore: choose a random action
            action = np.random.randint(0, self.num_cities)
        else:
            # Exploit: choose the action with the highest Q-value
            action = np.argmax(self.q_table[state])
        return action

    def update_q_table(self, state, action, reward, next_state):
        current_q = self.q_table[state, action]
        max_q = np.max(self.q_table[next_state])
        new_q = (1 - self.learning_rate) * current_q + self.learning_rate * (reward + self.discount_factor * max_q)
        self.q_table[state, action] = new_q

# Q-learning Training
env = RoadNetworkEnvironment()
agent = QLearningAgent(env.num_cities)

num_episodes = 1000
for episode in range(num_episodes):
    env.reset()
    done = False
    while not done:
        state = env.current_city
        action = agent.choose_action(state)
        next_state, reward, done = env.step(action)
        agent.update_q_table(state, action, reward, next_state)

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
