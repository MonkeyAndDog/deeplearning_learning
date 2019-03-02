"""
 Created by Monkey at 2019/3/2
"""

import tensorflow as tf

w1 = tf.Variable(tf.random_normal((2, 3)))
w2 = tf.Variable(tf.random_normal((3, 1)))

x = tf.constant([[0.7, 0.6]])

a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

with tf.Session() as sess:
    # sess.run(w1.initializer)
    # sess.run(w2.initializer)
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    print(sess.run(y))
