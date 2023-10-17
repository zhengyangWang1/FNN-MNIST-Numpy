# 配置文件

# 数据处理
normalize = True

flatten = True

one_hot_label = True

val_data = True

# 模型参数
lr = 0.5

epoch = 30

batch_size = 20

input_size = 784

hidden1 = 256

hidden2 = 64

outsize = 10

print_iter = 1000

val_step = 1  # 经过几个epoch后进行验证

lamb = 0.001  # 正则化参数
