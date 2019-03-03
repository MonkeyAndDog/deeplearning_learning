"""
 Created by Monkey at 2019/3/3
"""

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("", one_hot=True)

print("Training data size: " + str(mnist.train.num_examples))

print("Validating data size: ", mnist.validation.num_examples)

print("Test data size: ", mnist.test.num_examples)

print("Example train data: ", mnist.train.images[0])

print("Example training data label: ", mnist.train.labels[0])

batch_size = 100
(xs, ys) = mnist.train.next_batch(batch_size)
print("X shape: ", xs.shape)
print("Y shape: ", ys.shape)
print(xs)
print(ys)
