import config
import model
import train
import gymnasium
'''
可视化
input:
env_name:游戏环境
model:训练好的模型
目标：利用已经训练好的模型，将游戏的当前环境作为输入，选择当前最好的动作执行，可视化游戏进程
'''
class Visualization():
    def __init__(self,env_name,model):

        self.env = gymnasium.make(env_name)
        self.model = model

    def select_best_action(self):
        pass