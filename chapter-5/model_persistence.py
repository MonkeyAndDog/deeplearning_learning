"""
    Created by Xiaozhong.
    Copyright (c) 2019/3/3 Xiaozhong. All rights reserved.
"""
import tensorflow as tf
from tensorflow.python.framework import graph_util

v1 = tf.get_variable("v1", [1], dtype=tf.float32, initializer=tf.constant_initializer(1.0))
v2 = tf.get_variable("v2", [1], dtype=tf.float32, initializer=tf.constant_initializer(2.0))

result = v1 + v2

init_op = tf.global_variables_initializer()
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init_op)
    saver.save(sess, "./model1/model.ckpt", )

    graph_def = tf.get_default_graph().as_graph_def()

    output_graph_def = graph_util.convert_variables_to_constants(sess, graph_def, ['add'])
    with tf.gfile.GFile("./model2/combined_model.pb", "wb") as f:
        f.write(output_graph_def.SerializeToString())
