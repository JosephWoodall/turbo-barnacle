import torch
import torch.nn.functional as F
import numpy as np

class RLAgent:
    """ """
    def __init__(self, env, model, optimizer, gamma=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995):
        self.env = env
        self.model = model
        self.optimizer = optimizer
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def act(self, state):
        """

        :param state: 

        """
        if np.random.rand() <= self.epsilon:
            return self.env.action_space.sample()
        state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        q_values = self.model(state)
        action = q_values.argmax(dim=1).item()
        return action

    def train(self, num_episodes=1000, batch_size=32, memory_size=10000):
        """

        :param num_episodes: Default value = 1000)
        :param batch_size: Default value = 32)
        :param memory_size: Default value = 10000)

        """
        memory = []
        for episode in range(num_episodes):
            state = self.env.reset()
            done = False
            total_reward = 0
            while not done:
                action = self.act(state)
                next_state, reward, done, _ = self.env.step(action)
                total_reward += reward
                memory.append((state, action, reward, next_state, done))
                if len(memory) > memory_size:
                    del memory[0]
                state = next_state
                if len(memory) >= batch_size:
                    self._experience_replay(memory, batch_size)
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
            print("Episode:", episode, "Total Reward:", total_reward)

    def _experience_replay(self, memory, batch_size):
        """

        :param memory: param batch_size:
        :param batch_size: 

        """
        batch = np.random.choice(len(memory), batch_size, replace=False)
        state_batch = torch.tensor([memory[i][0] for i in batch], dtype=torch.float32, device=self.device)
        action_batch = torch.tensor([memory[i][1] for i in batch], dtype=torch.long, device=self.device)
        reward_batch = torch.tensor([memory[i][2] for i in batch], dtype=torch.float32, device=self.device)
        next_state_batch = torch.tensor([memory[i][3] for i in batch], dtype=torch.float32, device=self.device)
        done_batch = torch.tensor([memory[i][4] for i in batch], dtype=torch.float32, device=self.device)

        q_values = self.model(state_batch).gather(1, action_batch.unsqueeze(1)).squeeze(1)
        next_q_values = self.model(next_state_batch).max(1)[0]
        expected_q_values = reward_batch + self.gamma * next_q_values * (1 - done_batch)
        loss = F.mse_loss(q_values, expected_q_values.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
