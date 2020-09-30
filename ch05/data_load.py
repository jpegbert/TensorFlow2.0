import tensorflow as tf
from tensorflow import keras
from tensorflow_core.python.keras import datasets # 导入经典数据集加载模块


def preprocess(x, y): # 自定义的预处理函数
    # 调用此函数时会自动传入 x,y 对象，shape 为[b, 28, 28], [b]
    # 标准化到 0~1
    x = tf.cast(x, dtype=tf.float32) / 255.
    x = tf.reshape(x, [-1, 28*28])     # 打平
    y = tf.cast(y, dtype=tf.int32)    # 转成整型张量
    y = tf.one_hot(y, depth=10)    # one-hot 编码
    # 返回的 x,y 将替换传入的 x,y 参数，从而实现数据的预处理功能
    return x, y


# 加载mnist数据集
(x, y), (x_test, y_test) = datasets.mnist.load_data()
print('x:', x.shape, 'y:', y.shape, 'x test:', x_test.shape, 'y test:', y_test)

train_db = tf.data.Dataset.from_tensor_slices((x, y)) # 构建 Dataset 对象
train_db = train_db.shuffle(10000) # 随机打散样本，不会打乱样本与标签映射关系
train_db = train_db.batch(128) # 设置批训练，batch size 为 128
train_db = train_db.repeat(20) # 数据集迭代 20 这种方式等同于循环执行20个epoch
# 预处理函数实现在 preprocess 函数中，传入函数名即可
train_db = train_db.map(preprocess)

for epoch in range(20): # 训练 Epoch 数
    for step, (x, y) in enumerate(train_db): # 迭代 Step 数
        pass # training

