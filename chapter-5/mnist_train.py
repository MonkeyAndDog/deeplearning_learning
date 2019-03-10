"""
    Created by Xiaozhong.
    Copyright (c) 2019/3/8 Xiaozhong. All rights reserved.
"""
import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import mnist_inference

# 定义了训练过程
BATCH_SIZE = 100                # 每一批训练数据大小
LEARNING_RATE_BASE = 0.8        # 学习率的基础率
LEARNING_RATE_DECAY = 0.99      # 学习率的衰减速度
REGULARAZTION_RATE = 0.0001     # 正则化
TRAINING_STEPS = 30000          # 训练次数

MOVING_AVERAGE_DECAY = 0.99     # 滑动平均衰减系数

MODEL_SAVE_PATH = "./model3/"   # 模型保存路径
MODEL_NAME = "model.ckpt"       # 模型保存文件


def train(mnist):
    # 定义了输入内容和验证内容的占位符
    x = tf.placeholder(tf.float32, [None, mnist_inference.INPUT_NODE], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, mnist_inference.OUTPUT_NODE], name='y-input')

    # 正则化函数
    regularizer = tf.contrib.layers.l2_regularizer(REGULARAZTION_RATE)

    # 获取到输出的表达
    y = mnist_inference.inference(x, regularizer)

    # 训练步数
    global_step = tf.Variable(0, trainable=False)

    # 滑动平均值
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)

    #
    variable_averages_op = variable_averages.apply(tf.trainable_variables())

    # 交叉熵损失函数和softmax函数处理过后的结果
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    # 损失量函数
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))

    # 学习速率经过指数衰减
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE, # 基础学习率
        global_step,        # 学习步长
        mnist.train.num_examples / BATCH_SIZE,  # 学习次数
        LEARNING_RATE_DECAY # 学习率衰减
    )

    # 对训练函数进行优化，优化方式为梯度下降优化
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

    # 通过反向传播来更新神经网络参数和滑动平均值
    with tf.control_dependencies([train_step, variable_averages_op]):
        train_op = tf.no_op(name='train')

    # 持久化模型
    saver = tf.train.Saver()

    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        # 训练过程
        for i in range(TRAINING_STEPS):
            # 批量读取数据
            xs, ys = mnist.train.next_batch(BATCH_SIZE)

            # 进行计算并取得损失量和步数等信息
            _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: xs, y_: ys})
            if i % 1000 == 0:
                # 输出损失量和其对应的步数
                print("After %d training steps, loss on training batch is %g." % (step, loss_value))

                # 持久化
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)


if __name__ == '__main__':
    mnist = input_data.read_data_sets("./", one_hot=True)
    train(mnist)
