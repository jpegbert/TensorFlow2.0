import tensorflow as tf


"""
合并与分割
"""


a = tf.random.normal([4, 35, 8]) # 模拟成绩册 A, 四个班级，每个班35个人，每人8门课程的成绩
b = tf.random.normal([6, 35, 8]) # 模拟成绩册 B, 六个班级，每个班35个人，每人8门课程的成绩
c = tf.concat([a, b], axis=0) # 拼接合并成绩册, 按照第一个维度拼接
print(c.shape) # (10, 35, 8)

a = tf.random.normal([10, 35, 4]) # 保存了10个班级的前四门课程的成绩
b = tf.random.normal([10, 35, 4]) # 保存了10个班级的后四门课程的成绩
c = tf.concat([a, b], axis=2) # 在科目维度上拼接
print(c.shape)

# 堆叠，用于在某一维度插人新的维度
a = tf.random.normal([35, 8])
b = tf.random.normal([35, 8])
c = tf.stack([a, b], axis=0) # 堆叠合并为 2 个班级，班级维度插入
print(c.shape) # (2, 35, 8)

# 同样可以选择在其他位置插入新维度，例如，最末尾插入班级维度
a = tf.random.normal([35, 8])
b = tf.random.normal([35, 8])
c = tf.stack([a, b], axis=-1) # 在末尾插入班级维度
print(c.shape) # (35, 8, 2)

# 采用concat的当时合并
a = tf.random.normal([35, 8])
b = tf.random.normal([35, 8])
c = tf.concat([a, b], axis=0) # 拼接方式合并，没有 2 个班级的概念
print(c.shape) # (70, 8)


# 分割
x = tf.random.normal([10, 35, 8])
# 等长切割为 10 份
result = tf.split(x, num_or_size_splits=10, axis=0)
print(len(result)) # 10
print(result[0].shape) # (1, 35, 8)

# x = tf.random.normal([10,35,8])
# 自定义长度的切割，切割为 4 份，返回 4 个张量的列表 result
result = tf.split(x, num_or_size_splits=[4, 2, 2, 2], axis=0)
print(len(result)) # 4
print(result[0].shape) # (4, 35, 8)

# 使用unstack使用数据在对应维度全部按长度为1分割
result = tf.unstack(x, axis=0) # Unstack 为长度为 1 的张量
print(len(result)) # 10, 返回 10 个张量的列表
print(result[0].shape) # (35, 8)

