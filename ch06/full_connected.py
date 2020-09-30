import tensorflow as tf
from tensorflow_core.python.keras import layers # 导入层模块


"""
全连接层
"""


def fc_by_tensor():
    """
    # 采用张量方式实现全连接
    """
    x = tf.random.normal([2, 784])
    w1 = tf.Variable(tf.random.truncated_normal([784, 256], stddev=0.1))
    b1 = tf.Variable(tf.zeros([256]))
    o1 = tf.matmul(x, w1) + b1  # 线性变换
    o1 = tf.nn.relu(o1)  # 激活函数


def fc_by_layers():
    """
    采用TensorFlow提供的层的方式实现全连接层
    :return:
    """
    x = tf.random.normal([4, 28 * 28])
    # 创建全连接层，指定输出节点数和激活函数
    fc = layers.Dense(512, activation=tf.nn.relu)
    h1 = fc(x)  # 通过 fc 类实例完成一次全连接层的计算，返回输出张量
    print(fc.kernel)  # 获取 Dense 类的权值矩阵
    # 通过类的trainable_variables获取待优化的参数列表
    print(fc.trainable_variables)
    # 获取不需要优化的参数列表
    print(fc.non_trainable_variables)
    # 获取所有参数列表
    print(fc.variables)



def main():
    fc_by_tensor() # 采用张量方式实现全连接
    fc_by_layers() # 采用TensorFlow提供的层的方式实现全连接层


if __name__ == '__main__':
    main()
