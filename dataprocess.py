import numpy as np
import gzip
import os
import pickle


def load_data(normalize=True, flatten=False, one_hot_label=False, val_data=True):
    # 用dataset字典保存由4个文件读取得到的np数组
    dataset = {}
    # 若不存在pkl文件，下载文件导入numpy数组，并生成pkl文件
    if not os.path.exists('mnist.pkl'):
        # MNIST数据集的4个文件
        key_file = {
            'train_img': 'train-images-idx3-ubyte.gz', 'train_label': 'train-labels-idx1-ubyte.gz',
            'test_img': 't10k-images-idx3-ubyte.gz', 'test_label': 't10k-labels-idx1-ubyte.gz'
        }
        for _ in key_file:
            # 用二进制只读方式打开.gz文件
            with gzip.open(os.path.join("datasets/", key_file[_]), 'rb') as f:
                # img文件前16个字节不是img数据，跳过读取；label文件前8个不是label数据，跳过读取
                dataset[_] = np.frombuffer(f.read(), np.uint8,
                                           offset=16 if _ == 'train_img' or _ == 'test_img' else 8)
                if _ == 'train_img' or _ == 'test_img':
                    dataset[_] = dataset[_].reshape(-1, 1, 28, 28)
        # 生成mnist.pkl
        print('Creating pickle file ...')
        with open('mnist.pkl', 'wb') as f:
            pickle.dump(dataset, f, -1)
        print('Create finished!')
    # 若存在pkl文件，把pkl文件内容导入numpy数组
    else:
        with open('mnist.pkl', 'rb') as f:
            dataset = pickle.load(f)
    # 标准化处理
    if normalize:
        for _ in ('train_img', 'test_img'):
            dataset[_] = dataset[_].astype(np.float32) / 255.0
    # one_hot_label处理
    if one_hot_label:
        for _ in ('train_label', 'test_label'):
            t = np.zeros((dataset[_].size, 10))
            for idx, row in enumerate(t):
                row[dataset[_][idx]] = 1
            dataset[_] = t
    # 展平处理
    if flatten:
        for _ in ('train_img', 'test_img'):
            dataset[_] = dataset[_].reshape(-1, 784)
    # 划分验证集
    if val_data:
        x_val_data, x_test_data = np.split(dataset['test_img'], 2)
        y_val_data, y_test_data = np.split(dataset['test_label'], 2)
        return dataset['train_img'], dataset['train_label'], x_val_data, y_val_data, x_test_data, y_test_data

    # 返回np数组
    return dataset['train_img'], dataset['train_label'], [], [], dataset['test_img'], dataset['test_label']
