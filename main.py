from dataprocess import load_data
import train
from config import normalize, flatten, one_hot_label, val_data

if __name__ == '__main__':
    x_train, y_train, x_val, y_val, x_test, y_test = load_data(normalize=normalize, flatten=flatten,
                                                               one_hot_label=one_hot_label, val_data=val_data)
    trainer = train.Trainer()

    # 训练
    trainer.train(x_train, y_train, x_val, y_val)

    # 评估
    trainer.evaluate(x_test, y_test)
