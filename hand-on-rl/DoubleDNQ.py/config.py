'''
cofig类
配置默认参数
(想到在说)
'''
import gymnasium as gym
import torch
import ale_py
class Config():

    learning_rate       = 0.00025  #学习率
    gamma               = 0.99    #折扣因子
    batch_size          = 512
    decay_rate          = 0.001
    max_epsilon         = 1
    min_epsilon         = 0.005
    n_training_episodes = 25000
    repalyBufferSize    = 25_000
    target_update_freq  = 500
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = gym.make("ALE/SpaceInvaders-v5", render_mode="rgb_array")
    
