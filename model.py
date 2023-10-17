import layer

"""
1. 三层全连接层 激活函数（ReLU）

2. softmax 

3. 三层全连接层 + softmax（激活函数sigmoid）+ 交叉熵损失

L2正则化

batch GD, SGD, mini-batch GD 
"""

class FCmodel(object):
    def __init__(self, batch_size=20, input_size=784, hidden1=256, hidden2=64, outsize=10, lr=0.4,
                 epoch=30, print_iter=100):
        self.batch_size = batch_size
        self.input_size = input_size
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.outsize = outsize
        self.lr = lr
        self.epoch = epoch
        self.print_iter = print_iter

    def fully_connected_model(self):
        self.fc1 = layer.Net(self.input_size, self.hidden1)
        self.relu1 = layer.ReLU()
        # self.sigmoid1 = layer.Sigmoid()
        self.fc2 = layer.Net(self.hidden1, self.hidden2)
        self.relu2 = layer.ReLU()
        # self.sigmoid2 = layer.Sigmoid()
        self.fc3 = layer.Net(self.hidden2, self.outsize)
        # self.softmax = layer.Softmax()

