import tensorflow as tf


"""
tensor比较
"""

out = tf.random.normal([100,10])
out = tf.nn.softmax(out, axis=1) # 输出转换为概率
pred = tf.argmax(out, axis=1) # 计算预测值
print(pred)
y = tf.random.uniform([100],dtype=tf.int64,maxval=10) # 模拟真实标签
out = tf.equal(pred, y) # 预测值与真实值比较，返回布尔类型的张量
out = tf.cast(out, dtype=tf.float32) # 布尔型转 int 型
correct = tf.reduce_sum(out) # 统计true的个数



