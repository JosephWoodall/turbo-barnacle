import numpy as np

class Maze:
    def __init__(self, size=10, num_obstacles=10, reward_location=None):
        self.size = size
        self.num_obstacles = num_obstacles
        self.reward_location = reward_location
        self.obstacles = []
        self.reset()

    def reset(self):
        self.player_location = (0, 0)
        self.obstacles = self.generate_obstacles()
        if self.reward_location is None:
            self.reward_location = self.generate_reward()
        else:
            self.reward_location = tuple(self.reward_location)
        self.state = self.get_state()

    def generate_obstacles(self):
        obstacles = []
        while len(obstacles) < self.num_obstacles:
            obstacle = np.random.randint(0, self.size, size=2)
            if obstacle not in obstacles and obstacle != (0, 0):
                obstacles.append(obstacle)
        return obstacles

    def generate_reward(self):
        reward = np.random.randint(0, self.size, size=2)
        while reward in self.obstacles or reward == (0, 0):
            reward = np.random.randint(0, self.size, size=2)
        return reward

    def get_state(self):
        state = np.zeros((self.size, self.size))
        state[self.player_location] = 1
        state[self.reward_location] = 2
        for obstacle in self.obstacles:
            state[obstacle] = -1
        return state

    def move_player(self, action):
        new_location = tuple(np.array(self.player_location) + np.array(action))
        if self.valid_location(new_location):
            self.player_location = new_location
        self.state = self.get_state()

    def valid_location(self, location):
        if location[0] < 0 or location[0] >= self.size or location[1] < 0 or location[1] >= self.size:
            return False
        if location in self.obstacles:
            return False
        return True

    def check_reward(self):
        if self.player_location == self.reward_location:
            return True
        return False
