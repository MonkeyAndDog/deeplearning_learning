"""
    Created by Xiaozhong.
    Copyright (c) 2019/3/8 Xiaozhong. All rights reserved.
"""

import tensorflow as tf

# 神经网络相关的参数
INPUT_NODE = 784
OUTPUT_NODE = 10
LAYER_NODE = 500


# 创建神经网络相关参数，使用get_variable() 函数进行，当计算图中没有相关参数时，会自动创建
# 在该函数中也会将变量的正则化损失加入损失合集
# 在测试时，会通过保存的模型加载这些变量的取值，更加方便的是，可以在变量加载时将华东平均变量重命名。
def get_weight_variable(shape, regularizer):
    weights = tf.get_variable("weights", shape, initializer=tf.truncated_normal_initializer(stddev=0.1))

    if regularizer != None:
        tf.add_to_collection('losses', regularizer(weights))
    return weights


# 定义前向传播过程
def inference(input_tensor, regularizer):

    # 声明第一层的神经网络参数并完成前向传播过程
    with tf.variable_scope('layer1'):
        # 权值参数
        weights = get_weight_variable([INPUT_NODE, LAYER_NODE], regularizer)
        # 定义偏移量
        biases = tf.get_variable('biases', [LAYER_NODE], initializer=tf.constant_initializer(0.0))
        # 第一层传播结果
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights) + biases)

        # 第二层
        with tf.variable_scope('layer2'):
            weights = get_weight_variable([LAYER_NODE, OUTPUT_NODE], regularizer)
            biases = tf.get_variable('biases', [OUTPUT_NODE], initializer=tf.constant_initializer(0.0))

            layer2 = tf.matmul(layer1, weights) + biases

        # 返回第二层传播结果
        return layer2
