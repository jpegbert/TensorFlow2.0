"""
TensorFlow 在运行时，默认会占用所有 GPU 显存资源，这是非常不友好的行为，尤其是当计算机同时有多个用户或者程序在使用 GPU
资源时，占用所有 GPU 显存资源会使得 其他程序无法运行。因此，一般推荐设置 TensorFlow 的显存占用方式为增长式占用模式，
即根据实际模型大小申请显存资源，代码实现如下:
"""


import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # 设置GPU为增长式占用
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        # 打印异常
        print(e)
