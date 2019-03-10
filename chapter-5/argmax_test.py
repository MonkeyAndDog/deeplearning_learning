"""
 Created by Monkey at 2019/3/10
"""
import tensorflow as tf
import numpy as np

A = [[1, 3, 4, 5, 6]]
B = [[1, 3, 4], [2, 4, 1]]

with tf.Session() as sess:
    print(sess.run(tf.argmax(A, 0)))
    print(sess.run(tf.argmax(B, 0)))
