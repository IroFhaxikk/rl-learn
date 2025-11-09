from config import Config
from model import Model
from collections import deque
import cv2
import random
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import tqdm
import pickle
import os
class Trainer():

    def __init__(self, config:Config, model_class):  # 这里接收的是类，不是实例

        self.config = config
        self.env = self.config.env
        self.device = self.config.device
        self.current_epsilon = 1
        self.replay_buffer = deque(maxlen=self.config.repalyBufferSize)
        self.frame_stack = deque(maxlen=4)
        in_channels = 4
        n_actions = self.env.action_space.n

        # 修正：创建模型实例
        self.target_net = model_class(in_channels, n_actions).to(self.config.device)
        self.q_net = model_class(in_channels, n_actions).to(self.config.device)
        
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=self.config.learning_rate)

        self.initial_frame_stack()

     #初始化4帧队列
    def initial_frame_stack(self):
        state, _        = self.env.reset()
        processed_frame = self.process_frame(state)
        self.frame_stack.clear()
        for _ in range(4):
            self.frame_stack.append(processed_frame)

    #预处理一帧
    #input--->frame
    def process_frame(self,state):

        gray_frame       = cv2.cvtColor(state,cv2.COLOR_RGB2GRAY)
        resize_frame     = cv2.resize(gray_frame,(84,84))
        normalized_frame = resize_frame / 255.0
        return normalized_frame.astype(np.float32)
    
    def get_current_framestack(self):
        return np.array(self.frame_stack)   #shape is (4,84,84)
    
    #入队一个状态
    def replaybuffer_push(self,state,action,reward,next_state,done):

        experience = (state,action,reward,next_state,done)
        self.replay_buffer.append(experience)

    #从经验回放中抽batch_size出来
    def replayBuffer_Sample(self):
        if len(self.replay_buffer) < self.config.batch_size:
            return None

        
        batch = random.sample(self.replay_buffer,self.config.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states      = torch.FloatTensor(np.array(states)).to(self.device)
        actions     = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards     = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones       = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        return states,actions,rewards,next_states,dones
    
    #选择动作根据ε-greedy策略选择
    def select_action(self,state):
        random_number = random.random()
        if random_number < self.current_epsilon:
            action = self.env.action_space.sample()
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_value      = self.q_net(state_tensor)
                action       = q_value.argmax(1).item()
        return action
    

    #衰减epsilon,根据轮数
    def decay_epsilon(self,episode):
        self.current_epsilon = self.config.min_epsilon + ( self.config.max_epsilon - self.config.min_epsilon ) * np.exp(-self.config.decay_rate * episode)
    
    def save_replay_buffer(self, filepath=r"C:\Users\Administrator\Desktop\rl-learn\savafiles\replay_buffer.pkl"):
        """保存经验回放缓冲区到文件"""
        try:
            with open(filepath, 'wb') as f:
                # 将 deque 转换为 list 再保存
                buffer_list = list(self.replay_buffer)
                pickle.dump(buffer_list, f)
            print(f"💾 经验回放缓冲区已保存到 {filepath}，包含 {len(buffer_list)} 条经验")
            return True
        except Exception as e:
            print(f" 保存经验回放缓冲区失败: {e}")
            return False
    
    def load_replay_buffer(self, filepath=r"C:\Users\Administrator\Desktop\rl-learn\savafiles\replay_buffer.pkl"):
        """从文件加载经验回放缓冲区"""
        try:
            if os.path.exists(filepath):
                with open(filepath, 'rb') as f:
                    buffer_list = pickle.load(f)
                self.replay_buffer.clear()
                self.replay_buffer.extend(buffer_list)
                print(f"📥 经验回放缓冲区已从 {filepath} 加载，包含 {len(self.replay_buffer)} 条经验")
                return True
            else:
                print(f" 经验回放缓冲区文件 {filepath} 不存在")
                return False
        except Exception as e:
            print(f" 加载经验回放缓冲区失败: {e}")
            return False
    
    def prefill_relaybuffer(self, force_refill=False):
        """预填充经验回放缓冲区，如果已有数据则跳过"""
        # 检查是否已经有足够的数据
        if not force_refill and len(self.replay_buffer) >= self.config.repalyBufferSize * 0.8:
            print(f" 经验回放缓冲区已有 {len(self.replay_buffer)} 条数据，跳过预填充")
            return
        
        print(" 正在预填充经验回放缓冲区...")
        self.initial_frame_stack()
        state = self.get_current_framestack()
        
        fill_target = self.config.repalyBufferSize
        current_size = len(self.replay_buffer)
        
        # 只填充到目标大小
        for i in range(fill_target - current_size):
            action = self.env.action_space.sample()
            next_state_frame, reward, done, truncated, info = self.env.step(action)
            processed_frame = self.process_frame(next_state_frame)
            self.frame_stack.append(processed_frame)
            next_state = self.get_current_framestack()
            
            self.replaybuffer_push(state, action, reward, next_state, done)
            
            if done:
                self.initial_frame_stack()
                state = self.get_current_framestack()
            else:
                state = next_state
            
            # 显示进度
            if (i + 1) % 1000 == 0:
                print(f"  填充进度: {current_size + i + 1}/{fill_target}")
        
        print(f" 经验回放预填充完成，当前总量: {len(self.replay_buffer)} 条")
                
    def train(self):
        # 换到 config 的初始 epsilon
        self.current_epsilon = getattr(self.config, "max_epsilon", 1.0)

        # 优化器与损失（你在 __init__ 已创建 optimizer / loss_fn，可用 self.optimizer, self.loss_fn）
        # 如果想用 Huber loss：loss_fn = F.smooth_l1_loss
        loss_fn = F.smooth_l1_loss  # 更稳健也常用

        global_step = 0  # 用全局步数来控制 target 更新、保存等

        for episode in tqdm.tqdm(range(self.config.n_training_episodes)):
            # reset 环境并初始化 frame stack（使 state 为 4 帧堆叠）
            obs, info = self.env.reset()
            self.initial_frame_stack()
            state = self.get_current_framestack()  # numpy (4,84,84)

            episode_reward = 0.0
            done = False
            truncated = False
            step = 0    
            total_loss = 0.0

            # 把 q_net 设为训练模式（dropout/bn 等）
            self.q_net.train()
            while not done and not truncated:
                # ---------------- 1) 选择动作（ε-greedy）
                action = self.select_action(state)   # 传入 state（4帧堆叠），不是单帧 obs

                # ---------------- 2) 与环境交互，更新 frame stack
                next_obs, reward, done, truncated, info = self.env.step(action)
                processed_obs = self.process_frame(next_obs)
                self.frame_stack.append(processed_obs)
                next_state = self.get_current_framestack()

                # ---------------- 3) 存经验
                self.replaybuffer_push(state, action, reward, next_state, done)

                # ---------------- 4) 训练：从 replay buffer 采样并更新网络
                batch = self.replayBuffer_Sample()
                if batch is not None:
                    states, actions, rewards, next_states, dones = batch
                    # states: (B,4,84,84) tensors on device

                    # Double DQN 目标计算
                    # current Q for taken actions
                    q_values = self.q_net(states).gather(1, actions)  # (B,1)

                    # online net selects the best next action
                    next_actions = self.q_net(next_states).argmax(1).unsqueeze(1)  # (B,1)

                    # target net evaluates that action
                    next_q_values = self.target_net(next_states).gather(1, next_actions)  # (B,1)

                    # compute TD target (detach next_q_values)
                    target_q = rewards + (1.0 - dones) * self.config.gamma * next_q_values.detach()  # (B,1)

                    # loss (Huber)
                    loss = loss_fn(q_values, target_q)

                    # backward + step
                    self.optimizer.zero_grad()
                    loss.backward()
                    # 可选：梯度裁剪
                    if getattr(self.config, "max_grad_norm", None) is not None:
                        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), self.config.max_grad_norm)
                    self.optimizer.step()
                    total_loss += loss.item()
                # ---------------- 5) 更新计数、target 网络、epsilon、state
                step += 1
                global_step += 1
                self.decay_epsilon(episode)
                episode_reward += reward
                state = next_state

                # 用全局步数更新 target_net（更均匀），使用 config.target_update_freq（默认1000）
                if getattr(self.config, "target_update_freq", 1000) > 0:
                    if global_step % self.config.target_update_freq == 0:
                        self.target_net.load_state_dict(self.q_net.state_dict())

                if done or truncated:
                    break

            if episode % 10 == 0 and episode > 0:
                avg_loss = total_loss / (step + 1e-8)
                print(f"\nEpisode {episode:6d} | "
                        f"Reward: {episode_reward:.2f} | "
                        f"Epsilon: {self.current_epsilon:.4f} | "
                        f"Loss: {avg_loss:.6f} | "
                        f"ReplayBuffer: {len(self.replay_buffer)}")
                
