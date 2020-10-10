import tensorflow as tf
from tensorflow.keras import datasets, layers, optimizers, Sequential, metrics
# from tensorflow_core.python.keras import datasets, layers, optimizers, metrics


def preprocess(x, y):
    x = tf.cast(x, dtype=tf.float32) / 255.
    y = tf.cast(y, dtype=tf.int32)
    return x, y


batchsz = 128
(x, y), (x_val, y_val) = datasets.mnist.load_data()
print('datasets:', x.shape, y.shape, x.min(), x.max())

db = tf.data.Dataset.from_tensor_slices((x, y))
db = db.map(preprocess).shuffle(60000).batch(batchsz).repeat(10)

ds_val = tf.data.Dataset.from_tensor_slices((x_val, y_val))
ds_val = ds_val.map(preprocess).batch(batchsz)

network = Sequential([layers.Dense(256, activation='relu'),
                      layers.Dense(128, activation='relu'),
                      layers.Dense(64, activation='relu'),
                      layers.Dense(32, activation='relu'),
                      layers.Dense(10)])
network.build(input_shape=(None, 28 * 28))
network.summary()

optimizer = optimizers.Adam(lr=0.01)
# 创建准确率测量器
acc_meter = metrics.Accuracy()
# 新建平均测量器，适合 Loss 数据
loss_meter = metrics.Mean()

log_dir = "./log/"
# 创建监控类，监控数据将写入 log_dir 目录
summary_writer = tf.summary.create_file_writer(log_dir)

for step, (x, y) in enumerate(db):
    with tf.GradientTape() as tape:
        # [b, 28, 28] => [b, 784]
        x = tf.reshape(x, (-1, 28 * 28))
        # [b, 784] => [b, 10]
        out = network(x)
        # [b] => [b, 10]
        y_onehot = tf.one_hot(y, depth=10)
        # [b]
        loss = tf.reduce_mean(tf.losses.categorical_crossentropy(y_onehot, out, from_logits=True))
        # 记录采样的数据，通过 float()函数将张量转换为普通数值
        loss_meter.update_state(float(loss))

    grads = tape.gradient(loss, network.trainable_variables)
    optimizer.apply_gradients(zip(grads, network.trainable_variables))

    if step % 100 == 0:
        # 打印统计期间的平均 loss
        print(step, 'loss:', loss_meter.result().numpy())
        with summary_writer.as_default():  # 写入环境
            # 当前时间戳 step 上的数据为 loss，写入到名为 train-loss 数据库中
            tf.summary.scalar('train-loss', float(loss_meter.result().numpy()), step=step)
            # 可视化真实标签的直方图分布
            tf.summary.histogram('y-hist', y, step=step)
            # 查看文本信息
            tf.summary.text('loss-text', str(float(loss)))
        # 打印完后，清零测量器
        loss_meter.reset_states()

    # evaluate
    if step % 500 == 0:
        total, total_correct = 0., 0
        # 测量器清零
        acc_meter.reset_states()

        for step, (x, y) in enumerate(ds_val):
            # [b, 28, 28] => [b, 784]
            x = tf.reshape(x, (-1, 28 * 28))
            # [b, 784] => [b, 10]
            out = network(x)

            # [b, 10] => [b]
            pred = tf.argmax(out, axis=1)
            pred = tf.cast(pred, dtype=tf.int32)
            # bool type
            correct = tf.equal(pred, y)
            # bool tensor => int tensor => numpy
            total_correct += tf.reduce_sum(tf.cast(correct, dtype=tf.int32)).numpy()
            total += x.shape[0]
            # 根据预测值与真实值写入测量器
            acc_meter.update_state(y, pred)
        with summary_writer.as_default():  # 写入环境
            # 写入测试准确率
            tf.summary.scalar('test-acc', float(total_correct / total), step=step)
            # 可视化测试用的图片，设置最多可视化 9 张图片
            # tf.summary.image("val-onebyone-images:", val_images, max_outputs=9, step=step)
        print(step, 'Evaluate Acc:', total_correct / total, acc_meter.result().numpy())


"""
log_dir = "./log/"
# 创建监控类，监控数据将写入 log_dir 目录
summary_writer = tf.summary.create_file_writer(log_dir)

with summary_writer.as_default():  # 写入环境
    # 当前时间戳 step 上的数据为 loss，写入到名为 train-loss 数据库中
    tf.summary.scalar('train-loss', float(loss), step=step)

with summary_writer.as_default():  # 写入环境
    # 写入测试准确率
    tf.summary.scalar('test-acc', float(total_correct/total), step=step)
    # 可视化测试用的图片，设置最多可视化 9 张图片
    tf.summary.image("val-onebyone-images:", val_images, max_outputs=9, step=step)

with summary_writer.as_default():
    # 当前时间戳 step 上的数据为 loss，写入到 ID 位 train-loss 对象中
    tf.summary.scalar('train-loss', float(loss), step=step)
    # 可视化真实标签的直方图分布
    tf.summary.histogram('y-hist', y, step=step)
    # 查看文本信息
    tf.summary.text('loss-text', str(float(loss)))
"""
