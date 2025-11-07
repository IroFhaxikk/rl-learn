'''
cofig类
配置默认参数
(想到在说)
'''
import gymnasium as gym

class config():

    learning_rate       = 0.001   #学习率
    gamma               = 0.99    #折扣因子
    batch_size          = 64
    decay_rate          = 0.0001
    max_epsilon         = 1
    min_epsilon         = 0.005
    n_training_episodes = 100000

    env = gym.make("Atari",)

    state_dim = 
    
