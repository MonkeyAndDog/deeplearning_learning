"""
    Created by Xiaozhong.
    Copyright (c) 2019/3/8 Xiaozhong. All rights reserved.
"""
import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import mnist_inference
import mnist_train

# 重新运算求出参数内容的时间间隔，次/10S
EVAL_INTERVAL_SECS = 10


# 对结果进行评估
def evaluate(mnist):
    with tf.Graph().as_default() as g:
        x = tf.placeholder(tf.float32, [None, mnist_inference.INPUT_NODE], name='x-input')
        y_ = tf.placeholder(tf.float32, [None, mnist_inference.OUTPUT_NODE], name='y-input')

        # 验证内容
        validate_feed = {
            x: mnist.validation.images,
            y_: mnist.validation.labels
        }

        # 直接获取到运算结果，测试时不关注正则化内容
        y = mnist_inference.inference(x, None)

        # 是否正确预测
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

        # 得分
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # 通过变量重命名的方式来加载模型，这样在前向传播的过程中就不需要调用求滑动平均的函数来获取平均值了。
        # 这样就可以完全共用mnist_inference.py中的前向传播过程
        variable_averages = tf.train.ExponentialMovingAverage(mnist_train.MOVING_AVERAGE_DECAY)

        variables_to_restore = variable_averages.variables_to_restore()

        saver = tf.train.Saver(variables_to_restore)

        # 每10S计算一次正确率
        while True:
            with tf.Session() as sess:
                # get_checkpoint_state()函数会自动扫描checkpoint文件来自动找到目录中最新的模型文件名
                ckpt = tf.train.get_checkpoint_state(mnist_train.MODEL_SAVE_PATH)
                if ckpt and ckpt.model_checkpoint_path:
                    # 加载模型
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    # 通过文件名来获取到训练的轮数
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                    accuracy_score = sess.run(accuracy, feed_dict=validate_feed)
                    print("After %s training step, validation accuracy = %g" % (global_step, accuracy_score))
                else:
                    print("No checkpoint file found!")
                    return
            time.sleep(EVAL_INTERVAL_SECS)


if __name__ == '__main__':
    mnist = input_data.read_data_sets("./", one_hot=True)
    evaluate(mnist)
