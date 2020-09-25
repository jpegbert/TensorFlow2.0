
"""
利用 TensorFlow 自动在线下载 MNIST 数据集，并转换为 Numpy 数组格 式
"""
import os
# 导入TF库
import tensorflow as tf
# 导入TF子库keras
from tensorflow import keras
# 导入 TF 子库等
from tensorflow.keras import layers, optimizers, datasets

# 加载 MNIST 数据集
(x, y), (x_val, y_val) = datasets.mnist.load_data()
# 转换为浮点张量，并缩放到 -1~1
x = 2 * tf.convert_to_tensor(x, dtype=tf.float32) / 255. - 1
# 转换为整形张量
y = tf.convert_to_tensor(y, dtype=tf.int32)
# one-hot 编码，指定类别总数为 10
y = tf.one_hot(y, depth=10)
print(x.shape, y.shape)
# 构建数据集对象
train_dataset = tf.data.Dataset.from_tensor_slices((x, y))
# 批量训练
train_dataset = train_dataset.batch(512)

