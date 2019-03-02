"""
    Created by Xiaozhong.
    Copyright (c) 2019/3/2 Xiaozhong. All rights reserved.
"""
import tensorflow as tf

v1 = tf.constant([1, 2, 3, 4])
v2 = tf.constant([4, 3, 2, 1])

sess = tf.InteractiveSession()

print(tf.greater(v1, v2).eval())

print(tf.where(tf.greater(v1, v2), v1, v2).eval())
sess.close()
