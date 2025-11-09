import torch
import gymnasium as gym
import cv2
import time
import numpy as np
from collections import deque
import ale_py

class Visualization:
    def __init__(self, env_name, model_class, path_model, render_mode="human", device="cpu", frame_stack=4):
        """
        可视化已训练强化学习模型在环境中的表现（兼容图像输入模型）
        """
        self.env = gym.make(env_name, render_mode=render_mode)
        self.device = torch.device(device)
        self.frame_stack = frame_stack
        self.frames = deque(maxlen=frame_stack)

        # 初始化模型（假设输入通道数固定为 frame_stack）
        n_actions = self.env.action_space.n
        self.model = model_class(in_channels=frame_stack, n_actions=n_actions).to(device)
        self.model.load_state_dict(torch.load(path_model, map_location=device))
        self.model.eval()

    def process_frame(self, frame):
        """灰度化 + 缩放 + 归一化"""
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (84, 84))
        return (resized / 255.0).astype(np.float32)

    def reset_stack(self):
        """初始化帧堆叠"""
        obs, _ = self.env.reset()
        frame = self.process_frame(obs)
        self.frames.clear()
        for _ in range(self.frame_stack):
            self.frames.append(frame)
        return np.array(self.frames)

    def select_best_action(self, state):
        """选择最优动作"""
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.model(state_t)
        return q_values.argmax(dim=1).item()

    def run(self, max_steps=1000, sleep_time=0.02):
        """执行可视化"""
        state = self.reset_stack()
        total_reward = 0.0

        for step in range(max_steps):
            action = self.select_best_action(state)
            next_obs, reward, done, truncated, _ = self.env.step(action)
            next_frame = self.process_frame(next_obs)
            self.frames.append(next_frame)
            next_state = np.array(self.frames)

            total_reward += reward
            self.env.render()
            time.sleep(sleep_time)

            if done or truncated:
                print(f"🎯 Episode finished at step {step+1}, total reward={total_reward:.2f}")
                state = self.reset_stack()
                total_reward = 0.0
            else:
                state = next_state

        self.env.close()


from model import Model

# 加载模型
viz = Visualization(
    env_name="ALE/SpaceInvaders-v5",
    model_class=Model,
    path_model=r"C:\Users\Administrator\Desktop\rl-learn\doubledqn_model_final.pth",
    render_mode="human",
    device="cuda" if torch.cuda.is_available() else "cpu"
)

# 运行可视化
viz.run(max_steps=2000, sleep_time=0.01)
