import tensorflow as tf
from tensorflow import keras


"""
数据统计
"""


def norm():
    """
    向量范数
    在 TensorFlow 中，可以通过 tf.norm(x, ord)求解张量的 L1、L2、∞等范数，其中参数 ord 指定为 1、2 时
    计算 L1、L2 范数，指定为 np.inf 时计算∞ −范数
    """
    x = tf.ones([2, 2])
    print(x)
    norm_l1 = tf.norm(x, ord=1)  # 计算 L1
    print(norm_l1) # tf.Tensor(4.0, shape=(), dtype=float32)

    norm_l2 = tf.norm(x, ord=2)  # 计算 L2 范数
    print(norm_l2) # tf.Tensor(2.0, shape=(), dtype=float32)

    import numpy as np
    inf_norm = tf.norm(x, ord=np.inf)  # 计算∞范数
    print(inf_norm) # tf.Tensor(1.0, shape=(), dtype=float32)


def max_min_mean_sum():
    """
    最值、均值、和
    """
    x = tf.random.normal([4, 10])  # 模型生成概率
    max_ = tf.reduce_max(x, axis=1)  # 统计概率维度上的最大值
    print(max_) # tf.Tensor([0.9729441  0.73061854 2.0896268  1.1048363 ], shape=(4,), dtype=float32)

    min_ = tf.reduce_min(x, axis=1)  # 统计概率维度上的最小值
    print(min_) # tf.Tensor([-0.57958335 -1.6100361  -0.9848029  -2.0508783 ], shape=(4,), dtype=float32)

    mean_ = tf.reduce_mean(x, axis=1)  # 统计概率维度上的均值
    print(mean_) # tf.Tensor([ 0.08778151 -0.28164566 -0.06802277  0.49110407], shape=(4,), dtype=float32)

    # 当不指定 axis 参数时，tf.reduce_*函数会求解出全局元素的最大、最小、均值、和等 数据
    # 统计全局的最大、最小、均值、和，返回的张量均为标量
    max_, min_, mean_ = tf.reduce_max(x), tf.reduce_min(x), tf.reduce_mean(x)
    print(max_) # tf.Tensor(1.7639534, shape=(), dtype=float32)
    print(min_) # tf.Tensor(-1.5667734, shape=(), dtype=float32)
    print(mean_) # tf.Tensor(0.03698604, shape=(), dtype=float32)

    # 在求解误差函数时，通过 TensorFlow 的 MSE 误差函数可以求得每个样本的误差，需 要计算样本的平均误差，
    # 此时可以通过 tf.reduce_mean 在样本数维度上计算均值
    out = tf.random.normal([4, 10])  # 模拟网络预测输出
    y = tf.constant([1, 2, 2, 0])  # 模拟真实标签
    print(y) # tf.Tensor([1 2 2 0], shape=(4,), dtype=int32)
    y = tf.one_hot(y, depth=10)  # one-hot 编码
    print(y)
    loss = keras.losses.mse(y, out)  # 计算每个样本的误差
    print(loss) # tf.Tensor([1.039666  1.2794526 1.009795  1.9772352], shape=(4,), dtype=float32)
    loss = tf.reduce_mean(loss)  # 平均误差，在样本数维度上取均值
    print(loss) # tf.Tensor(1.3265371, shape=(), dtype=float32)

    reduce_sum_ = tf.reduce_sum(out, axis=-1) # 求最后一个维度的和
    print(reduce_sum_) # tf.Tensor([-3.8688679  0.1597758  3.6051762  0.9152107], shape=(4,), dtype=float32)

    out = tf.random.normal([2, 10])
    print(out)
    out = tf.nn.softmax(out, axis=1)  # 通过 softmax 函数转换为概率值
    print(out)

    pred = tf.argmax(out, axis=1)  # 选取概率最大的位置
    print(pred)



def main():
    # norm() # 向量范数
    max_min_mean_sum() # 最值、均值、和


if __name__ == '__main__':
    main()
