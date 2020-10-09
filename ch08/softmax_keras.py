import tensorflow as tf
from tensorflow.keras import layers


# 创建softmax层，并调用__call__完成前向计算
x = tf.constant([2., 1., 0.1]) # 创建输入张量
layer = layers.Softmax(axis=-1) # 创建softmax层
out = layer(x) # 调用softmax前向计算
print(out)

out = tf.nn.softmax(x) # 调用softmax前向计算
print(out)
