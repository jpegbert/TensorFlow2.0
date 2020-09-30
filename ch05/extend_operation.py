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


def where():
    a = tf.ones([3, 3]) # 构造全1矩阵
    b = tf.zeros([3, 3]) # 构造全0矩阵

    # 构建采样条件
    cond = tf.constant([[True, False, False], [False, True, False], [True, True, False]])
    res = tf.where(cond, a, b) # 根据条件从a，b中采样, 返回的张量中为 1 的位置全部来自张量 a，返回的张量中为 0 的位置来自张量 b

    res = tf.where(cond)  # 获取 cond 中为 True 的元素索引

    x = tf.random.normal([3, 3])  # 构造 a
    mask = x > 0  # 比较操作，等同于 tf.math.greater()
    # 通过tf.where提取此掩码处True元素的索引坐标：
    indices = tf.where(mask)  # 提取所有大于 0 的元素索引
    # 拿到索引后，通过tf.gather_nd即可恢复出所有正数的元素
    res = tf.gather_nd(x, indices)  # 提取正数的元素值
    # 当我们得到掩码mask之后，也可以直接通过tf.boolean_mask获取所有正数的元素向量:
    res = tf.boolean_mask(x, mask)  # 通过掩码提取正数的元素值


def scatter_nd():
    # 构造需要刷新数据的位置参数，即4 3 1 和7号位置
    indices = tf.constant([[4], [3], [1], [7]])
    # 构造需要写入的数据，4号位置写4.4,3号位置写3.3，以此类推
    updates = tf.constant([4.4, 3.3, 1.1, 7.7])
    # 在长度为8的全0向量上根据indices写入updates数据
    res = tf.scatter_nd(indices, updates, [8])

    # 构造两个写入位置
    indices = tf.constant([[1], [3]])
    updates = tf.constant([ # 构造写入数据，即2个矩阵
        [[5, 5, 5, 5], [6, 0, 6, 6], [7, 7, 7, 7], [8, 8, 8, 8]],
        [[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]]
    ])
    # 在shape为[4, 4, 4]的白板上根据indices写入updates
    res = tf.scatter_nd(indices, updates, [4, 4, 4])


def sinc(x, y):
    z = tf.sqrt(x ** 2 + y ** 2)
    z = tf.sin(z) / z  # sinc 函数实现
    return z


def meshgrid():
    points = [] # 保存所有点的坐标列表
    for x in range(-8, 8, 100): # 循环生成x坐标，100个采样点
        for y in range(-8, 8, 100): # 循环生成y坐标，100个采样点
            # 计算每个点(x,y)处的 sinc 函数值
            z = tf.sqrt(x ** 2 + y ** 2)
            z = tf.sin(z) / z  # sinc 函数实现
            points.append([x, y, z])  # 保存采样点
    # 上面这种方式效率低，采用tf.meshgrid
    x = tf.linspace(-8., 8, 100)  # 设置 x 轴的采样点
    y = tf.linspace(-8., 8, 100)  # 设置 y 轴的采样点
    x, y = tf.meshgrid(x, y)  # 生成网格点，并内部拆分后返回
    print(x.shape, y.shape) # 打印拆分后的所有点的 x,y 坐标张量 shape
    z = tf.sqrt(x ** 2 + y ** 2)
    z = tf.sin(z) / z  # sinc 函数实现
    import matplotlib
    from matplotlib import pyplot as plt
    # 导入 3D 坐标轴支持
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure()
    ax = Axes3D(fig)  # 设置 3D 坐标轴
    # 根据网格点绘制 sinc 函数 3D 曲面
    ax.contour3D(x.numpy(), y.numpy(), z.numpy(), 50)
    plt.show()


def main():
    # gather() # 现根据索引号收集数据
    # gather_nd() # 指定每次采样点的多维坐标来实现采样多个点
    # boolean_mask() # 通过掩码（Mask）的方式进行采样
    # where() # 通过 tf.where(cond, a, b)操作可以根据 cond 条件的真假从参数𝑨或𝑩中读取数据
    # scatter_nd() #
    meshgrid()


if __name__ == '__main__':
    main()
