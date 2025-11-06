import matplotlib.pyplot as plt
import time
import numpy as np
import gymnasium as gym
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
class Visualization:
    def __init__(self, env: gym.Env, Qtable):
        self.env = env
        self.Qtable = Qtable

    def visual(self):
        """可视化智能体的决策过程"""
        state, info = self.env.reset()
        truncated = False
        done = False
        step = 0
        total_reward = 0
        
        # 创建图形窗口
        plt.ion()  # 开启交互模式
        fig, ax = plt.subplots(figsize=(16, 16))
        
        while not done and not truncated and step < 100:  # 防止无限循环
            # 选择动作
            action = np.argmax(self.Qtable[state])
            action_names = ['←left', '↓down', '→right', '↑up']
            
            # 执行动作
            next_state, reward, done, truncated, info = self.env.step(action)
            step += 1
            total_reward += reward
            
            # 渲染环境
            img = self.env.render()
            
            # 更新图像
            ax.clear()
            ax.imshow(img)
            ax.set_title(f'step: {step} | action: {action_names[action]} | reward: {reward} | total reward: {total_reward}')
            ax.axis('off')
            
            plt.pause(1)  # 暂停1秒
            plt.draw()
            
            # 更新状态
            state = next_state
            
            # 如果回合结束，显示结果
            if done or truncated:
                result = "Success! " if reward > 0 else "Failed! "
                ax.set_title(f'{result} | total step: {step} | total reward: {total_reward}')
                plt.pause(2)  # 最终结果暂停2秒
                break
        
        plt.ioff()  # 关闭交互模式
        plt.show()
        self.env.close()



