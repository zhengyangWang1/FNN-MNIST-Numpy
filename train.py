import model
import numpy as np
from config import batch_size, input_size, hidden1, hidden2, outsize, lr, epoch, print_iter, val_data, val_step, lamb
import matplotlib.pyplot as plt


class Trainer(object):

    def __init__(self):
        self.fc_model = model.FCmodel(batch_size=batch_size, input_size=input_size, hidden1=hidden1, hidden2=hidden2,
                                      outsize=outsize,
                                      lr=lr,
                                      epoch=epoch, print_iter=print_iter)
        self.fc_model.fully_connected_model()
        self.fc1 = self.fc_model.fc1
        self.relu1 = self.fc_model.relu1
        # self.sigmoid1 = self.fc_model.sigmoid1
        self.fc2 = self.fc_model.fc2
        self.relu2 = self.fc_model.relu2
        # self.sigmoid2 = self.fc_model.sigmoid2
        self.fc3 = self.fc_model.fc3
        # self.relu3 = self.fc_model.relu3
        # self.fc4 = self.fc_model.fc4
        # self.softmax = self.fc_model.softmax
        self.batch_size = self.fc_model.batch_size
        self.input_size = self.fc_model.input_size
        self.hidden1 = self.fc_model.hidden1
        self.hidden2 = self.fc_model.hidden2
        self.outsize = self.fc_model.outsize
        self.lr = self.fc_model.lr
        self.epoch = self.fc_model.epoch
        self.print_iter = self.fc_model.print_iter
        self.y = []
        self.y_prob = np.zeros(0)

    def MSE_loss(self, y_pre, y):

        loss = 0.5 * np.sum((y_pre - y) ** 2) / batch_size
        grad = (y_pre - y) / batch_size
        return loss, grad

    def MSE_loss_with_l2(self, y_pre, y, lamb):
        cost, grad = self.MSE_loss(y_pre, y)
        l2_cost = (1 / batch_size) * (lamb / 2) * (
                np.sum(np.square(self.fc1.w)) + np.sum(np.square(self.fc2.w)) + np.sum(np.square(self.fc3.w)))
        loss = cost + l2_cost
        return loss, grad

    def forward(self, input_data):  # 神经网络的前向传播
        h1 = self.fc1.forward(input_data)
        h1 = self.relu1.forward(h1)
        # h1 = self.sigmoid1.forward(h1)
        h2 = self.fc2.forward(h1)
        h2 = self.relu2.forward(h2)
        # h2 = self.sigmoid2.forward(h2)
        h3 = self.fc3.forward(h2)
        # prob = self.softmax.forward(h1)
        return h3

    def backward(self, y_pre, y):  # 神经网络的反向传播
        _, grad = self.MSE_loss(y_pre, y)
        # grad = self.softmax.backward()
        dh3 = self.fc3.backward(grad)
        dh2 = self.relu2.backward(dh3)
        # dh2 = self.sigmoid2.backward(dh3)
        dh2 = self.fc2.backward(dh2)
        dh1 = self.relu1.backward(dh2)
        # dh1 = self.sigmoid1.backward(dh2)
        dh1 = self.fc1.backward(dh1)

    def backward_with_l2(self, y_pre, y, lamb):
        _, grad = self.MSE_loss(y_pre, y)
        # grad = self.softmax.backward()
        dh3 = self.fc3.backward_with_l2(grad, lamb, self.batch_size)
        dh2 = self.relu2.backward(dh3)
        # dh2 = self.sigmoid2.backward(dh3)
        dh2 = self.fc2.backward_with_l2(dh2, lamb, self.batch_size)
        dh1 = self.relu1.backward(dh2)
        # dh1 = self.sigmoid1.backward(dh2)
        dh1 = self.fc1.backward_with_l2(dh1, lamb, self.batch_size)

    def update(self, lr):
        self.fc1.update(lr)
        self.fc2.update(lr)
        self.fc3.update(lr)

    def shuffle_data(self, x_train, y_train):
        train_data = np.append(x_train, y_train, axis=1)
        np.random.shuffle(train_data)
        x_train = train_data[:, :-10]
        y_train = train_data[:, -10:]
        return x_train, y_train

    def train_map(self, loss_list, acc_list):
        plt.figure()
        # iterations = range(self.batch_size)
        plt.plot(loss_list, 'b', label='loss')
        plt.plot(acc_list, 'r', label='acc')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend()
        plt.show()

    def train(self, x_train, y_train, x_val=np.zeros(0), y_val=np.zeros(0)):
        max_batch = int(x_train.shape[0] / self.batch_size)
        loss_list = []
        acc_list = []
        print('Start training...')
        idx_val = 0
        for idx_epoch in range(self.epoch):
            x_train, y_train = self.shuffle_data(x_train, y_train)
            loss = 0
            acc = 0
            for idx_batch in range(max_batch):
                batch_x = x_train[idx_batch * self.batch_size:(idx_batch + 1) * self.batch_size, :]
                batch_y = y_train[idx_batch * self.batch_size:(idx_batch + 1) * self.batch_size, :]
                y_pre = self.forward(batch_x)
                # loss = self.softmax.get_loss(batch_y)
                loss, _ = self.MSE_loss(y_pre, batch_y)
                # loss, _ = self.MSE_loss_with_l2(y_pre, batch_y, lamb)
                self.backward(y_pre, batch_y)
                # self.backward_with_l2(y_pre, batch_y, lamb)
                self.update(self.lr)
                if idx_batch % self.print_iter == 0:
                    print('Epoch %d, iter %d, loss: %.6f' % (idx_epoch, idx_batch, loss))
            if val_data:
                if idx_val % val_step == 0:
                    acc = self.val_evaluate(x_val, y_val)
                    print('acc: %.6f' % acc)
                idx_val = idx_val + 1
            loss_list.append(loss)
            acc_list.append(acc)
        self.train_map(loss_list, acc_list)

    def evaluate(self, x_test, y_test):
        y_pre = self.forward(x_test)
        y_pre = np.argmax(y_pre, axis=1)
        y_test = np.argmax(y_test, axis=1)
        accuracy = np.mean(y_pre == y_test)
        print('---------------------------------')
        print('Accuracy in test set: %f' % accuracy)
        print('---------------------------------')

    def val_evaluate(self, x_val, y_val):
        y_pre = self.forward(x_val)
        y_pre = np.argmax(y_pre, axis=1)
        y_test = np.argmax(y_val, axis=1)
        accuracy = np.mean(y_pre == y_test)
        return accuracy
