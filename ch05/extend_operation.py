import tensorflow as tf


"""
TensorFlow高级操作
"""


def gather():
    """
    现根据索引号收集数据
    """
    # 假设共有 4 个班级，每个班级 35 个学生，8 门科目，保存成绩册的张量 shape 为[4,35,8]
    x = tf.random.uniform([4, 35, 8], maxval=100, dtype=tf.int32) # 成绩册张量
    # 现在需要收集第1~2个班级的成绩册，可以给定需要收集班级的索引号：[0, 1]，并指定班级的维度axis = 0，
    # 通过tf.gather函数收集数据，代码如下:
    res = tf.gather(x, [0, 1], axis=0)  # 在班级维度收集第 1~2 号班级成绩册

    # 收集第 1,4,9,12,13,27 号同学成绩
    res = tf.gather(x, [0, 3, 8, 11, 12, 26], axis=1)

    # 如果需要收集所有同学的第 3 和第 5 门科目的成绩，则可以指定科目维度 axis=2
    res = tf.gather(x, [2, 4], axis=2) # 第3, 5科目的成绩

    # tf.gather 非常适合索引号没有规则的场合，其中索引号可以乱序排列，此时收集的数据也是对应顺序
    a = tf.range(8)
    a = tf.reshape(a, [4, 2]) # 生成张量a
    tf.gather(a, [3, 1, 0, 2], axis=0)  # 收集第 4,2,1,3 号元素

    # 如果希望抽查第[2,3]班级的第[3,4,6,27]号同学的科目 成绩，则可以通过组合多个 tf.gather实现。
    # 首先抽出第[2,3]班级，实现如下：
    students = tf.gather(x, [1, 2], axis=0)  # 收集第 2,3 号班级
    # 再从这 2 个班级的同学中提取对应学生成绩，代码如下：
    # 基于 students 张量继续收集
    res = tf.gather(students, [2, 3, 5, 26], axis=1)  # 收集第 3, 4，6，27号5同学

    # 抽查第二个班级的第二个同学的所有科目，第三个班级的第三个同学的所有科目，第四个班级的第四个同学的所有科目
    res = tf.stack([x[1, 1], x[2, 2], x[3, 3]], axis=0)


def gather_nd():
    """
    指定每次采样点的多维坐标来实现采样多个点
    """
    # 假设共有 4 个班级，每个班级 35 个学生，8 门科目，保存成绩册的张量 shape 为[4,35,8]
    x = tf.random.uniform([4, 35, 8], maxval=100, dtype=tf.int32)  # 成绩册张量
    # 抽查第 2 个班级的第 2 个同学的所有科目，第 3 个班级的 第 3 个同学的所有科目，
    # 第 4 个班级的第 4 个同学的所有科目。那么这 3 个采样点的索引 坐标可以记为：
    # [1,1]、[2,2]、[3,3]，我们将这个采样方案合并为一个 List 参数，即 [[1,1],[2,2],[3,3]]，
    # 通过 tf.gather_nd 函数即可，实现如下：
    res = tf.gather_nd(x, [[1, 1], [2, 2], [3, 3]])
    print(res)

    # 根据多维度坐标收集数据
    res = tf.gather_nd(x, [[1, 1, 2], [2, 2, 3], [3, 3, 4]])
    print(res)


def boolean_mask():
    """
    通过掩码（Mask）的方式进行采样
    """
    # 假设共有 4 个班级，每个班级 35 个学生，8 门科目，保存成绩册的张量 shape 为[4,35,8]
    x = tf.random.uniform([4, 35, 8], maxval=100, dtype=tf.int32)  # 成绩册张量
    # 根据掩码方式采样班级，给出掩码和维度索引
    res = tf.boolean_mask(x, mask=[True, False, False, True], axis=0)
    print(res)

    # 采样第1 4 5 8门科目
    res = tf.boolean_mask(x, mask=[True, False, False, True, True, False, False, True], axis=2)
    print(res)

    # 多维掩码采样
    # tf.boolean_mask(x, [[True, True, False], [False, True, True]])


def main():
    # gather() # 现根据索引号收集数据
    # gather_nd() # 指定每次采样点的多维坐标来实现采样多个点
    # boolean_mask() # 通过掩码（Mask）的方式进行采样
    where() #


if __name__ == '__main__':
    main()
