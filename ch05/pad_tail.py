import tensorflow as tf
from tensorflow import keras


"""
tensor填充与复制
"""


def pad():
    """
    填充
    :return:
    """
    a = tf.constant([1, 2, 3, 4, 5, 6])  # 第一个句子
    b = tf.constant([7, 8, 1, 6])  # 第二个句子
    b = tf.pad(b, [[0, 2]])  # 句子末尾填充2个0
    print(b)  # tf.Tensor([7 8 1 6 0 0], shape=(6,), dtype=int32)
    c = tf.stack([a, b], axis=0)  # 堆叠合并，创建句子数维度
    print(c)

    """
    # 模拟自然语言处理中句子，每句80个单词，超过80个单词的句子截断，不足80个单词的句子，在末尾补0
    total_words = 10000 # 设定词汇量大小
    max_review_len = 80 # 最大句子长度
    embedding_len = 100 # 词向量长度
    # 加载 IMDB 数据集
    (x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(num_words=total_words)
    # 将句子填充或截断到相同长度，设置为末尾填充和末尾截断方式
    x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=max_review_len, truncating='post', padding='post')
    x_test = keras.preprocessing.sequence.pad_sequences(x_test, maxlen=max_review_len, truncating='post', padding='post')
    print(x_train.shape, x_test.shape) # 打印等长
    """

    # 对图片，一般是28*28，需要扩充为32*32，可以在图片的上下左右各填充2个单元
    x = tf.random.normal([4, 28, 28, 1])
    # 图片上下、左右各填充 2 个单元
    res = tf.pad(x, [[0, 0], [2, 2], [2, 2], [0, 0]])
    print(res)


def tile():
    """
    复制
    :return:
    """
    x = tf.random.normal([4, 32, 32, 3])
    res = tf.tile(x, [2, 3, 3, 1]) # 数据复制


def range_():
    """
    数据限幅
    :return:
    """
    x = tf.range(9)
    print(x) # tf.Tensor([0 1 2 3 4 5 6 7 8], shape=(9,), dtype=int32)
    res = tf.maximum(x, 2) # 下限幅到2
    print(res) # tf.Tensor([2 2 2 3 4 5 6 7 8], shape=(9,), dtype=int32)
    res = tf.minimum(x, 7) # 上限幅到7
    print(res) # tf.Tensor([0 1 2 3 4 5 6 7 7], shape=(9,), dtype=int32)

    """
    # 注：基于tf.maximum()函数，可以实现ReLU函数如下
    def relu():
        return tf.maximum(x, 0.) # 下限幅为0即可
    """
    """
    # 通过组合tf.maximum(x, a)和tf.minimum(x, b)可以实现同时对数据的上下边界限幅即: x∈[a,b]
    tf.minimum(tf.maximum(x, 2), 7)  # 限幅为 2~7 
    """
    """
    # 更方便的，可以使用tf.clip_by_value()函数实现上下限幅
    res = tf.clip_by_value(x, 2, 7)  # 限幅为 2~7
    """


def main():
    # pad() # 填充
    # tile() # 复制
    range_() # 数据限幅


if __name__ == '__main__':
    main()
