# rl-learn
Uestc研一，自学强化学习ing。记录自己学习的时候遇到的困难，理解，进程等等，欢迎大神指导交流。 

先运用强化学习完成一些游戏，最后看转向机械臂、机械狗或者人形机器人
# Q-learning算法
这个算法我是用冰湖游戏进行复刻，算是强化学习的入门算法，可以利用该算法熟悉一下整个流程、gym库的一些使用方法，主要是使用epsilon-greedy 策略选择随机探索还是利用策略。

Qtable--->每个状态对应的动作，我这里使用了8*8的冰湖游戏，state就是64个，action->上下左右，
Qtable.shape--->（64，4）

for 训练步长
    action = ε-greedy策略返回的动作
    
    while 游戏自己结束或外部结束（前者后者的区别要了解）
        更新Qtable
        Qtable(s,a) = Qtable(s,a) + learning_rate *(reward + gamma * max(Qtable(s')) - Qtable)

# Double-DNQ

主要理解经验回放和目标网络