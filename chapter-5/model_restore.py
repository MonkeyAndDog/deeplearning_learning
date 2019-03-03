"""
    Created by Xiaozhong.
    Copyright (c) 2019/3/3 Xiaozhong. All rights reserved.
"""

import tensorflow as tf
from tensorflow.python.platform import gfile

v1 = tf.Variable([3.0], name="other-v1")
v2 = tf.Variable([2.0], name="other-v2")

result = v1 + v2

# 当存储的图中变量名称和现有的变量名称不一致时，使用字典来映射
saver = tf.train.Saver({"v1": v1, "v2": v2})

# 在加载模型时，没有进行变量的初始化，而是通过模型加载进来的，所以结果仍然为3.0
with tf.Session() as sess:
    saver.restore(sess, "./model1/model.ckpt")
    print(sess.run(result))

saver2 = tf.train.import_meta_graph("./model1/model.ckpt.meta")
with tf.Session() as sess:
    saver2.restore(sess, "./model1/model.ckpt")
    print(sess.run(tf.get_default_graph().get_tensor_by_name("add:0")))

with tf.Session() as sess:
    model_filename = "./model2/combined_model.pb"
    with gfile.FastGFile(model_filename, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    result = tf.import_graph_def(graph_def, return_elements=["add:0"])
    print(sess.run(result))
