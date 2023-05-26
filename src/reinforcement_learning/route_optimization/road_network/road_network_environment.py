import numpy as np

class RoadNetworkEnvironment:
    def __init__(self):
        self.num_cities = 3
        self.distances = np.array([
            [0, 5, 2],  # Distance from City A to City B and City C
            [5, 0, 3],  # Distance from City B to City A and City C
            [2, 3, 0]   # Distance from City C to City A and City B
        ])
        self.cities = ['A', 'B', 'C']

    def reset(self):
        # Randomly select starting and destination cities
        self.start_city = np.random.randint(0, self.num_cities)
        self.dest_city = np.random.randint(0, self.num_cities)
        
        while self.start_city == self.dest_city:
            self.dest_city = np.random.randint(0, self.num_cities)
        
        self.current_city = self.start_city

    def step(self, action):
        # Take the action (move to the selected city)
        self.current_city = action
        
        # Calculate the reward based on the distance to the destination city
        reward = -self.distances[self.current_city, self.dest_city]
        
        # Check if the destination city is reached
        done = (self.current_city == self.dest_city)
        
        # Return the next state (current city), reward, and done flag
        return self.current_city, reward, done

    def get_valid_actions(self):
        # Return the valid actions (neighboring cities)
        return [city for city in range(self.num_cities) if city != self.current_city]

# Usage example
env = RoadNetworkEnvironment()

# Reset the environment to initialize the starting and destination cities
env.reset()

# Run a sample episode in the environment
done = False
while not done:
    # Get the valid actions and choose an action
    valid_actions = env.get_valid_actions()
    action = np.random.choice(valid_actions)
    
    # Take a step in the environment
    next_state, reward, done = env.step(action)
    
    # Print the current state, action, reward, and next state
    print("Current City:", env.cities[env.current_city])
    print("Action:", env.cities[action])
    print("Reward:", reward)
    print("Next City:", env.cities[next_state])
    print("------------")
