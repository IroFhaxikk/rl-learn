import torch.nn as nn
import torch.optim as optim
from config import Config
import numpy as np
import random
from tqdm import tqdm
'''
注意传入参数，主要是输入的层数和输出的动作
'''
# class Model(nn.modules):
    
#     def __init__(self):
#         super(Model,self).__init__()
#         pass
    

#     def forward(self,x):
#         return x

'''

Q_learning算法中无需torch
包含更新，计算，初始化
采取epsilon-greedy
以概率 1 - ɛ 进行利用 （即我们的智能体选择状态-动作对值最高的动作）。
以概率 ɛ ：进行探索 （尝试随机动作）。
'''
class Qlearn():

    def __init__(self,config:Config):

        self.config              = config
        self.lr                  = config.lr
        self.gamma               = config.gamma
        self.n_training_episodes = config.n_training_episodes
        self.max_steps           = config.max_steps
        self.decay_rate          = config.decay_rate
        self.max_epsilon         = config.max_epsilon
        self.min_epsilon         = config.min_epsilon
        self.Qtable              = config.Qtable
        self.env                 = config.env
        self.current_epsilon     = self.max_epsilon
        self.episode_steps       = []
        self.episode_reward      = []
        self.Qtabel_changes      = []
        self.Qtable_initial()

    def Qtable_initial(self):
        self.Qtable = np.zeros((self.env.observation_space.n,self.env.action_space.n))


    def select_action(self,state):
        action = np.argmax(self.Qtable[state][:])
        random_number = random.uniform(0,1)
        if random_number < self.current_epsilon:
            action = self.env.action_space.sample()
        else:
            action = np.argmax(self.Qtable[state])
        return action
    
    def Qlearn_update(self):
        for episode in tqdm(range(self.n_training_episodes)):
            # if episode == 0:
            #     print("Train Start")
            self.current_epsilon = self.min_epsilon + ( self.max_epsilon - self.min_epsilon ) * np.exp(-self.decay_rate * episode)
            state,info  = self.env.reset()
            step                = 0
            episode_reward      = 0
            update_count        = 0
            total_Qtable_change = 0
            truncated           = False
            done                = False
            while not done and step < self.max_steps:
                # print("Train Start")
                action = self.select_action(state=state)
                # print(episode,action)
                new_state, reward, done, truncated, info = self.env.step(action)
                step           = step + 1
                episode_reward = reward + episode_reward

                old_Qtable = self.Qtable[state][action].copy()

                self.Qtable[state][action] = self.Qtable[state][action] + self.lr * ( reward + self.gamma * np.max(self.Qtable[new_state]) - self.Qtable[state][action])
                
                Qtable_change       = abs(self.Qtable[state][action] - old_Qtable)
                total_Qtable_change = total_Qtable_change + Qtable_change
                update_count += 1

                if done is True or truncated is True:
                    break
                state = new_state
            self.episode_reward.append(episode_reward)
            self.episode_steps.append(step)
            if episode % 500 == 0:
                print(episode_reward,self.current_epsilon)
        
        # return False
    def print_info(self):
        print("\n========== Qlearn Parameters ==========")
        print(f"Learning rate (lr):           {self.lr}")
        print(f"Discount factor (gamma):      {self.gamma}")
        print(f"Training episodes:            {self.n_training_episodes}")
        print(f"Max steps per episode:        {self.max_steps}")
        print(f"Epsilon max:                  {self.max_epsilon}")
        print(f"Epsilon min:                  {self.min_epsilon}")
        print(f"Decay rate:                   {self.decay_rate}")
        print(f"Environment:                  {self.env}")
        print(f"Q-table shape:                {self.Qtable}")
        print("========================================\n")

