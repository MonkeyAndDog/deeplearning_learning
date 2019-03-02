"""
    Created by Xiaozhong.
    Copyright (c) 2019/3/2 Xiaozhong. All rights reserved.
"""
import tensorflow as tf
from numpy.random import RandomState

# 分批训练模型时的数据组数
batch_size = 8

# 初始状况下的神经网络权重
w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))

# 输入内容和正确数值
x = tf.placeholder(tf.float32, shape=(None, 2), name='x-input')
y_ = tf.placeholder(tf.float32, shape=(None, 1), name='y-input')

# 定义神经网络向前传播的过程
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

# 损失函数和反向传播算法
y = tf.sigmoid(y)
cross_entropy = -tf.reduce_mean(
    y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0)) + (1 - y_) * tf.log(tf.clip_by_value(1 - y, 1e-10, 1.0))
)
# 优化
train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

# 使用1作为种子进行产生随机数
rdm = RandomState(1)
# 训练数据集大小
dataset_size = 128
# 产生训练数据集合及相应的数据维数 128 * 2
X = rdm.rand(dataset_size, 2)

# 定义规则来产生样本
Y = [[int(x1 + x2 < 1)] for (x1, x2) in X]

# 创建会话来进行运算
with tf.Session() as sess:
    # 初始化各项变量
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    # 训练之前神经网络参数
    print(sess.run(w1))
    print(sess.run(w2))

    # 训练轮数
    SETPS = 5000
    for i in range(SETPS):
        # 每次选取batch_size个数据进行训练
        start = (i * batch_size) % dataset_size
        end = min(start + batch_size, dataset_size)

        # 通过选区的样本训练神经网络并更新参数
        sess.run(train_step, feed_dict={x: X[start:end], y_: Y[start:end]})
        if i % 1000 == 0:
            # 每隔一段时间计算所有数据上的交叉熵并输出
            total_cross_entropy = sess.run(cross_entropy, feed_dict={x: X, y_: Y})
            print("After %d training step(s), cross entropy on all data is %g" % (i, total_cross_entropy))

    print(sess.run(w1))
    print(sess.run(w2))
