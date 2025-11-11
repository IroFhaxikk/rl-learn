import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm


class PolicyNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=1)


class Config:
    def __init__(self):
        self.gamma = 0.99
        self.learning_rate = 2e-4
        self.env = gym.make("CartPole-v1")
        self.n_train_episode = 5000
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.input_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.n
        self.hidden_dim = 256
        self.filepath  = r"E:\rl-learn\savafiles\policy.pth"


class Agent:
    def __init__(self, config: Config):
        self.config = config
        self.policy_net = PolicyNet(config.input_dim, config.hidden_dim, config.action_dim).to(config.device)

        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=config.learning_rate)

        self.transition = None

    def take_action(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.config.device)
        probs = self.policy_net(state)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item()

    def update(self):
        states = torch.tensor(self.transition['states'], dtype=torch.float32).to(self.config.device)
        actions = torch.tensor(self.transition['actions'], dtype=torch.int64).to(self.config.device)
        rewards = self.transition['rewards']

        # 计算折扣回报 G_t
        G = 0
        returns = []

        for r in reversed(rewards):
            G = r + self.config.gamma * G
            returns.insert(0, G)

        returns = torch.tensor(returns, dtype=torch.float).to(self.config.device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        loss_total = 0

        probs     = self.policy_net(states)                 # shape [N, action_dim]
        dist      = torch.distributions.Categorical(probs)
        log_probs = dist.log_prob(actions)
        loss      = -(log_probs * returns).mean()
        self.optimizer.zero_grad()
        # for state, action, Gt in zip(states, actions, returns):
        #     state = torch.tensor([state], dtype=torch.float).to(self.config.device)

        #     probs = self.policy_net(state)
        #     dist = torch.distributions.Categorical(probs)

        #     action = torch.tensor(action).to(self.config.device)
        #     log_prob = dist.log_prob(action)

        #     loss = -log_prob * Gt
        #     loss_total += loss

        loss.backward()
        self.optimizer.step()

        return loss.item()

    def train(self):
        return_windows = []

        for episode in tqdm(range(self.config.n_train_episode)):
            state, _ = self.config.env.reset()
            terminated = False
            truncated = False
            
            self.transition = {
                'states': [],
                'actions': [],
                'rewards': [],
                'next_states': [],
                'dones': []
            }

            # -------- 交互 --------
            while not (terminated or truncated):
                action = self.take_action(state)
                next_state, reward, terminated, truncated, _ = self.config.env.step(action)

                self.transition['states'].append(state)
                self.transition['actions'].append(action)
                self.transition['rewards'].append(reward)
                self.transition['next_states'].append(next_state)
                self.transition['dones'].append(terminated or truncated)

                state = next_state

            # -------- 更新 --------
            loss = self.update()
            episode_return = sum(self.transition['rewards'])

            # -------- 记录回报 --------
            return_windows.append(episode_return)
            if len(return_windows) > 200:
                return_windows.pop(0)

            # -------- 打印 --------
            if episode % 200 == 0 and episode != 0:
                avg_return = np.mean(return_windows)
                print(
                    f"Episode: {episode} | "
                    f"Return: {episode_return:.1f} | "
                    f"Loss: {loss:.3f} | "
                    f"AvgReturn(200): {avg_return:.1f}"
                )

        self.save_model()


    def save_model(self):
        torch.save(self.policy_net.state_dict(),self.config.filepath)
        print(f"Model paramaters save into {self.config.filepath}")

if __name__ == "__main__":
    cfg = Config()
    agent = Agent(cfg)
    agent.train()
