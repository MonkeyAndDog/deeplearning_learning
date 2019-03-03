"""
 Created by Monkey at 2019/3/3
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 输入层节点个数，28*28 = 784
INPUT_NODE = 784
# 输出层节点个数，标识0-9
OUTPUT_NODE = 10

# 每一层的结点个数，该处只有一层
LAYER_NODE = 500

# 一个训练中batch中的训练个数，数字越小，取出来的个数越少，训练过程越接近随机梯度下降，否则接近梯度下降
BATCH_SIZE = 100

# 基础学习率
LEARNING_RATE_BASE = 0.8
# 学习率的衰减率
LEARNING_RATE_DECAY = 0.99

# 描述模型复杂度的正则化项在损失函数中的系数
REGULARIZATION_RATE = 0.0001
# 训练轮数
TRAINING_STEPS = 30000
# 滑动平均衰减率
MOVING_AVERAGE_DECAY = 0.99


# 定义辅助函数，给定神经网络的输入和所有参数，计算神经网络的前向传播结果
# 使用ReLU激活函数的三层全连接神经网络，同时去线性化
def inference(input_tensor, avg_class, weights1, biases1, weights2, biases2):
    # 当没有提供滑动平均类时，直接使用参数当前的取值
    if avg_class == None:
        # 计算隐藏层的前向传播结果，使用了ReLU激活函数
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights1) + biases1)
        return tf.matmul(layer1, weights2) + biases2
    else:
        # 首先使用avg_class.average() 来计算得出变量的滑动平均值，然后计算前向传播的结果
        layer1 = tf.nn.relu(
            tf.matmul(input_tensor, avg_class.average(weights1)) + avg_class.average(biases1)
        )
        return tf.matmul(layer1, avg_class.average(weights2) + avg_class.average(biases2))


# 训练模型
def train(mnist):
    x = tf.placeholder(tf.float32, shape=(None, INPUT_NODE))
    y_ = tf.placeholder(tf.float32, shape=(None, OUTPUT_NODE))

    # 生成隐藏层的层数
    weights1 = tf.Variable(tf.truncated_normal([INPUT_NODE, LAYER_NODE], stddev=1))
    biases1 = tf.Variable(tf.constant(0.1, shape=[LAYER_NODE]))

    # 生成输出层的层数
    weights2 = tf.Variable(tf.truncated_normal([LAYER_NODE, OUTPUT_NODE], stddev=1))
    biases2 = tf.Variable(tf.constant(0.1, shape=[OUTPUT_NODE]))

    y = inference(x, None, weights1, biases1, weights2, biases2)

    # 训练轮数的变量，一般定义训练轮数的变量时，要指定其为trainable=False，不让它参与训练
    global_step = tf.Variable(0, trainable=False)

    # 初始化滑动平均类
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variable_averages_op = variable_averages.apply(tf.trainable_variables())

    # 计算使用了滑动平均之后的前向传播结果
    average_y = inference(x, variable_averages, weights1, biases1, weights2, biases2)

    # 计算交叉熵作为刻画预测值和真实值之间的差距的损失函数
    # 第一个参数是神经网络不包括softmax层的前向传播结果，第二个是训练数据的正确结果
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    # 计算平均交叉熵
    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    # 计算L2正则化损失函数
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    # 计算模型的正则化损失，一般只计算神经网络上的权重的正则化损失，不使用偏置项
    regularization = regularizer(weights1) + regularizer(weights2)
    # 总损失等于交叉熵损失和正则化损失的和
    loss = cross_entropy_mean + regularization

    # 设置指数衰减的学习率
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,  # 基础学习率
        global_step,  # 当前学习的轮数
        mnist.train.num_examples / BATCH_SIZE,  # 训练完所有的数据需要的迭代次数
        LEARNING_RATE_DECAY  # 学习率衰减速度
    )

    # 使用梯度下降优化算法来优化损失函数
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

    # 在训练模型时，既要通过反向传播来更新神经网络中的参数，又要更新每一个参数的滑动模型
    # 下面等价于
    # with tf.control_dependencies([train_step, variable_averages_op]):
    #     train_op = tf.no_op(name='train')
    train_op = tf.group(train_step, variable_averages_op)

    # 检验使用了滑动平均模型的神经网络前向传播是否正常工作。
    # tf.avgmax(average_y, 1) 表示每一个样例的预测答案，其中的average_y是一个BATCH_SIZE * 10的二维数组，每一行代表样例的前向传播结果
    # tf.argmax()第二个参数为表示选取最大值的操作仅在一个维度中进行，即，只在每一行中选取最大值的对应下目标。
    # 经过equal之后得到的结果为长度为batch的一维数组，该数组中的值就表示了每一个阳历对应数字的识别结果
    correct_prediction = tf.equal(tf.argmax(average_y, 1), tf.argmax(y_, 1))

    # 将布尔型转换为实数型进行计算平均值
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        validate_feed = {x: mnist.validation.images, y_: mnist.validation.labels}

        test_feed = {x: mnist.test.images, y_: mnist.test.labels}

        for i in range(TRAINING_STEPS):
            if i % 1000 == 0:
                validate_acc = sess.run(accuracy, feed_dict=validate_feed)
                print("After %d training step(s), validation accuracy using average model is %g " % (i, validate_acc))
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            sess.run(train_op, feed_dict={x: xs, y_: ys})

        test_acc = sess.run(accuracy, feed_dict=test_feed)

        print("After %d training step(s), test accuracy using average model is %g" % (TRAINING_STEPS, test_acc))


# 会报错
# def main(argv = None):
#     mnist = input_data.read_data_sets("", one_hot=True)
#     train(mnist)


if __name__ == '__main__':
    # tf.app.run()
    mnist = input_data.read_data_sets("", one_hot=True)
    train(mnist)
