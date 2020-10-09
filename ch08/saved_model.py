import os
import tensorflow as tf
from tensorflow.keras import datasets, layers, optimizers, Sequential, metrics, losses
# from tensorflow_core.python.keras import datasets, layers, optimizers, metrics, losses


"""
采用saved_model的方式存储网络
"""


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def preprocess(x, y):
    """
    x is a simple image, not a batch
    """
    x = tf.cast(x, dtype=tf.float32) / 255.
    x = tf.reshape(x, [28 * 28])
    y = tf.cast(y, dtype=tf.int32)
    y = tf.one_hot(y, depth=10)
    return x, y


batchsz = 128
(x, y), (x_val, y_val) = datasets.mnist.load_data()
print('datasets:', x.shape, y.shape, x.min(), x.max())

db = tf.data.Dataset.from_tensor_slices((x, y))
db = db.map(preprocess).shuffle(60000).batch(batchsz)
ds_val = tf.data.Dataset.from_tensor_slices((x_val, y_val))
ds_val = ds_val.map(preprocess).batch(batchsz)


network = Sequential([layers.Dense(256, activation='relu'),
                      layers.Dense(128, activation='relu'),
                      layers.Dense(64, activation='relu'),
                      layers.Dense(32, activation='relu'),
                      layers.Dense(10)])
network.build(input_shape=(None, 28 * 28))
network.summary()

network.compile(optimizer=optimizers.Adam(lr=0.01),
                loss=tf.losses.CategoricalCrossentropy(from_logits=True),
                metrics=['accuracy']
                )

network.fit(db, epochs=3, validation_data=ds_val, validation_freq=2)

network.evaluate(ds_val)


# 保存模型结构与模型参数到文件
tf.saved_model.save(network, 'model-savedmodel')
print('saving savedmodel.')
del network # 删除网络对象

print('load savedmodel from file.') # 从文件恢复网络结构与网络参数
network = tf.saved_model.load('model-savedmodel') # 准确率计量器
acc_meter = metrics.CategoricalAccuracy()
for x, y in ds_val:   # 遍历测试集
    pred = network(x) # 前向计算
    acc_meter.update_state(y_true=y, y_pred=pred) # 更新准确率统计

# 打印准确率
print("Test Accuracy:%f" % acc_meter.result())


