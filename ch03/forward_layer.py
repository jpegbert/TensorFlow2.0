import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf
from tensorflow import keras
# 如果可以翻墙可以使用下面这种方式
# from tensorflow.keras import layers, optimizers, datasets
# 已经有数据使用这种方式
from tensorflow.keras import layers, optimizers

# (x, y), (x_val, y_val) = datasets.mnist.load_data()
# 已经有数据使用这种方式
f = np.load("./mnist.npz")
x, y = f['x_train'], f['y_train']
x_test, y_test = f['x_test'], f['y_test']
f.close()
x = tf.convert_to_tensor(x, dtype=tf.float32) / 255.
y = tf.convert_to_tensor(y, dtype=tf.int32)
y = tf.one_hot(y, depth=10)
print(x.shape, y.shape)
train_dataset = tf.data.Dataset.from_tensor_slices((x, y))
train_dataset = train_dataset.batch(200)

# 利用 Sequential 容器封装 3 个网络层，前网络层的输出默认作为下一层的输入
model = keras.Sequential([ # 3 个非线性层的嵌套模型
    layers.Dense(512, activation='relu'), # 隐藏层1
    layers.Dense(256, activation='relu'), # 隐藏层2
    layers.Dense(10)]) # 输出层，输出节点数为 10

optimizer = optimizers.SGD(learning_rate=0.001)


def train_epoch(epoch):
    # Step4.loop
    for step, (x, y) in enumerate(train_dataset):
        with tf.GradientTape() as tape:
            # 打平操作，[b, 28, 28] => [b, 784]
            x = tf.reshape(x, (-1, 28 * 28))
            # Step1. 得到模型输出output [b, 784] => [b, 10]
            out = model(x)
            # Step2. 先计算残差平方和 [b, 10]，对应tf.square(out - y)
            # 再计算每个样本的平均误差 [b]，对应tf.reduce_sum(tf.square(out - y)) / x.shape[0]
            loss = tf.reduce_sum(tf.square(out - y)) / x.shape[0]

        # Step3. 计算参数梯度 w1, w2, w3, b1, b2, b3
        grads = tape.gradient(loss, model.trainable_variables)
        # w' = w - lr * grad，更新网络参数
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        if step % 100 == 0:
            print(epoch, step, 'loss:', loss.numpy())


def train():
    for epoch in range(30):
        train_epoch(epoch)


if __name__ == '__main__':
    train()
