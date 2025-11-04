import torch.nn as nn
'''
注意传入参数，主要是输入的层数和输出的动作
'''
class Model(nn.modules):
    
    def __init__(self):
        super(Model,self).__init__()

    def forward(self,x):
        return x
