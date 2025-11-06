from config import Config
from model import Qlearn
from visualization import Visualization

config = Config()
agent = Qlearn(config)
agent.print_info()
agent.Qlearn_update()
vis = Visualization(config.env,agent.Qtable)
vis.visual()
