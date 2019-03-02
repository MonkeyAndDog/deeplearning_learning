"""
 Created by Monkey at 2019/3/2
"""

import tensorflow as tf
a = tf.constant([1.0, 2.0], name="a")
b = tf.constant([2.0, 3.0], name="b")
result = a + b

print(a.graph == tf.get_default_graph())

"一个张量中保存了三个数据，名称、维度、数据类型"
print(result)

with tf.Session() as sess:
    print(sess.run(result))

# sess = tf.Session()
# with sess.as_default():
#     print(sess.run(result))


