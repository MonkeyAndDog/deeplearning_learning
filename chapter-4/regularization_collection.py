"""
    Created by Xiaozhong.
    Copyright (c) 2019/3/3 Xiaozhong. All rights reserved.
"""
import tensorflow as tf

# 使用l1和l2正则化方式来正则化神经网络参数
weights = tf.constant([[1.0, -2.0], [-3.0, 4.0]])
with tf.Session() as sess:
    print(sess.run(tf.contrib.layers.l1_regularizer(0.5)(weights)))
    print(sess.run(tf.contrib.layers.l2_regularizer(0.5)(weights)))


# 获取神经网络参数的函数，里面使用了随机数产生临时数据，并将相应的正则化方式添加进集合中，
# 集合类似于一种Map数据结构，使用键值对来描述，不过键可以重复出现在集合中
def get_weight(shape, lambda_value):
    var = tf.Variable(tf.random_normal(shape), dtype=tf.float32)
    tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(lambda_value)(var))
    return var


# 初始化各项参数
x = tf.placeholder(dtype=tf.float32, shape=(None, 2))
y_ = tf.placeholder(dtype=tf.float32, shape=(None, 1))
batch_size = 8

# 每一层的节点数
layer_dimension = [2, 10, 10, 10, 1]
# 神经网络层数
n_layers = len(layer_dimension)

# 维护前向传播时最深层的结点，开始的时候就是输入层
cur_layer = x
# 当前层的节点个数
in_dimension = layer_dimension[0]

# 通过循环产生5层全连接神经网络结构
for i in range(1, n_layers):
    out_dimension = layer_dimension[i]
    # 生成当前层的权重变量，并将这个变量的L2正则化损失加入计算图上的集合
    weight = get_weight([in_dimension, out_dimension], 0.001)
    bias = tf.Variable(tf.constant(0.1, shape=[out_dimension]))
    cur_layer = tf.nn.relu(tf.matmul(cur_layer, weight) + bias)
    in_dimension = layer_dimension[i]

# 定义神经网络前向传播的同时已经将所有的L2正则化损失加入了集合
# 该处仅仅计算刻画模型在训练数据上表现的损失函数
mse_lose = tf.reduce_mean(tf.square(y_ - cur_layer))

tf.add_to_collection('losses', mse_lose)

# 将所有的损失函数不同部分添加进来
loss = tf.add_n(tf.get_collection('losses'))
