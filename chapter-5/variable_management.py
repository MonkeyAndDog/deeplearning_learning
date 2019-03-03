"""
    Created by Xiaozhong.
    Copyright (c) 2019/3/3 Xiaozhong. All rights reserved.
"""
import tensorflow as tf

# 当命名空间reuse为默认（False）时，get_variable和Variable效果一样
v = tf.get_variable("v", shape=[1], initializer=tf.constant_initializer(1.0))

# with tf.Session() as sess:
#     init_op = tf.global_variables_initializer()
#     sess.run(init_op)
#     print(sess.run(v))  # 输出[1.]
#     print(v)  # 输出<tf.Variable 'v:0' shape=(1,) dtype=float32_ref>，表明是在空的命名空间中生成的变量
#
# with tf.variable_scope("foo"):
#     v = tf.get_variable("v", [1], initializer=tf.constant_initializer(1.0))
#     print(v)  # 输出<tf.Variable 'foo/v:0' shape=(1,) dtype=float32_ref>，表明是在foo的命名空间中生成的变量
#
# with tf.variable_scope("foo", reuse=True):  # 使用reuse时，使用get_variable()会直接获取相应的变量，如果没有会直接报错
#     v1 = tf.get_variable("v", [1])
#     print(v1 == v)

print("---------------------------------------------------")
print(v.name)
# 嵌套使用命名空间
with tf.variable_scope("root"):
    print(tf.get_variable_scope().reuse)  # False
    v2 = tf.get_variable("v", [1])
    print(v2.name)
    with tf.variable_scope("foo", reuse=True):  # True
        print(tf.get_variable_scope().reuse)

        with tf.variable_scope("bar"):  # False
            print(tf.get_variable_scope().reuse)

    print(tf.get_variable_scope().reuse)  # False


# 对tensorflow_train_network中inference函数的改进，让它可以接受更少的参数
def inference(input_tensor, reuse=False):
    with tf.variable_scope("layer1", reuse=reuse):
        weights = tf.get_variable("weights")
        biases = tf.get_variable("biases")
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights) + biases)
    return layer1
