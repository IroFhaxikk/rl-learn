'''
cofig类
配置默认参数
(想到在说)
'''
import gymnasium as gym
import torch
import ale_py
class Config():

    learning_rate       = 2.5e-4  # 更稳妥的 Atari 设定
    gamma               = 0.99    #折扣因子
    batch_size          = 64
    decay_rate          = 0.001  #（保留，若使用步数线性衰减则不再使用）
    max_epsilon         = 1.0
    min_epsilon         = 0.01
    n_training_episodes = 5000
    repalyBufferSize    = 1000
    target_update_freq  = 10_000  # 以步数为尺度更新 target

    # 新增：更稳定的训练控制
    train_freq          = 4        # 每隔多少步更新一次
    learning_starts     = 20_000   # 前若干步仅收集经验，不训练
    epsilon_decay_steps = 1_000_000  # ε 线性从 max→min 的步数
    max_grad_norm       = 10.0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = gym.make("ALE/SpaceInvaders-v5", render_mode="rgb_array")
    
