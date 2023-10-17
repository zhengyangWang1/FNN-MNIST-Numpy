import numpy as np


# 全连接神经网络
class Net(object):
    def __init__(self, num_input, num_output):
        self.num_input = num_input
        self.num_output = num_output
        self.w = np.random.normal(loc=0.0, scale=0.01, size=(self.num_input, self.num_output))  # 随机初始化参数 假设为(n, m)
        self.bias = np.zeros([1, self.num_output])  # 初始化为0  (1, m)
        self.input_data = np.zeros(0)
        self.output_data = np.zeros(0)
        self.grad_w = np.zeros(0)
        self.grad_b = np.zeros(0)

    def forward(self, input_data):
        self.input_data = input_data  # 假设input_data = (1, n)
        self.output_data = np.matmul(self.input_data, self.w) + self.bias  # (1, n) * (n, m) = (1, m) m为下一层的输入维度
        return self.output_data

    def backward(self, grad):
        self.grad_w = np.dot(self.input_data.T, grad)  # (n, 1) * (1, m) = (n, m)
        self.grad_b = np.sum(grad, axis=0)
        next_grad = np.dot(grad, self.w.T)  # (1, m) * (m, n) = (1, n)
        return next_grad

    def backward_with_l2(self, grad, lamb, batch_size):
        self.grad_w = np.dot(self.input_data.T, grad) + (lamb / batch_size) * self.w
        self.grad_b = np.sum(grad, axis=0)
        next_grad = np.dot(grad, self.w.T)
        return next_grad

    def update(self, lr):
        self.w = self.w - lr * self.grad_w
        self.bias = self.bias - lr * self.grad_b


# 激活函数ReLU
class ReLU(object):

    def __init__(self):
        self.input_data = np.zeros(0)

    def forward(self, input_data):
        self.input_data = input_data
        output_data = np.maximum(0, input_data)  # (1, n)
        return output_data

    def backward(self, grad):
        next_grad = grad  # (1, n) * (1, n) 逐元素相乘
        next_grad[self.input_data < 0] = 0
        return next_grad

# 激活函数Sigmoid
class Sigmoid(object):
    def __init__(self):
        self.input_data = None

    def forward(self, input_data):
        self.input_data = 1 / (1 + np.exp(-input_data))
        return self.input_data

    def backward(self, grad):
        return grad * self.input_data * (1 - self.input_data)


# Softmax分类器
class Softmax(object):
    def __init__(self):
        self.prob = np.zeros(0)
        self.batch_size = []
        self.label = []

    def forward(self, input_data):
        input_max = np.max(input_data, axis=1, keepdims=True)
        input_exp = np.exp(input_data - input_max)
        self.prob = input_exp / np.sum(input_exp, axis=1, keepdims=True)
        return self.prob

    def get_loss(self, label):  # 计算损失
        self.label = label
        self.batch_size = self.prob.shape[0]
        loss = -np.sum(label * np.log(self.prob + 1e-7)) / self.batch_size
        return loss

    def backward(self):
        grad = (self.prob - self.label) / self.batch_size
        return grad
