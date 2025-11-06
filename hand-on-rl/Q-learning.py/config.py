'''
cofig类
配置默认参数
(想到在说)
'''
import gymnasium as gym

# ----------------------------
# 强化学习训练相关参数
# ----------------------------
class Config:
    def __init__(self):
        """
        Q-learning 配置参数
        """


        # 总训练轮数（Episode 数）
        # 每一次 Episode = 从环境 reset → 玩到 done=True
        self.n_training_episodes = 100000  

        # 每一局最多走多少步（防止死循环）
        self.max_steps = 300  


        # ----------------------------
        # epsilon-greedy 策略参数
        # ----------------------------

        # 最大探索概率（训练初期，100% 随机动作）
        self.max_epsilon = 1.0  
        # 最小探索概率（收敛后仍然保留少量随机性）
        self.min_epsilon = 0.01  
        # epsilon 的衰减速度，控制探索减少的速度
        # 数值越大，探索减少越快；越小则减少得越慢
        self.decay_rate = 0.00005
        # ----------------------------
        # Q-learning 基础参数
        # ----------------------------
        # 折扣因子 gamma：表示未来奖励的重要程度
        self.gamma = 0.99  
        # 学习率 alpha（用于控制 Q 值更新速度）
        self.lr = 0.8  
        # ----------------------------
        # Qtable（Q值表）相关（在 model.py 中动态生成）
        # ----------------------------

        # 环境对象（由 main.py 或 model.py 注入，不在这里创建）
        self.env : gym.Env = gym.make(
            "FrozenLake-v1", 
            map_name    = "8x8",
            is_slippery = False,
            render_mode = "rgb_array")

        # Qtable 将在训练开始时根据:
        #   Qtable = np.zeros((state_space, action_space))
        # 自动生成，因此这里不写死
        self.Qtable = None  
