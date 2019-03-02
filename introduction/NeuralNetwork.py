"""
  Created by Xiaozhong on 2019/2/27.
  Copyright (c) 2019/2/27 Xiaozhong. All rights reserved.
"""
import numpy
import scipy
import matplotlib.pyplot


def sigmoid(x):
    return 1 / (1 + numpy.exp(-x))


class NeuralNetwork:
    # 初始化各项参数
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # 设置输入节点，隐藏层节点，输出层节点数据和学习效率
        self.inodes = input_nodes
        self.hnodes = hidden_nodes
        self.onodes = output_nodes

        # 将权重矩阵弄好， wih为输入层到隐藏层的权重， who为隐藏层到输出层的权重
        self.wih = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        self.who = numpy.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))

        # 学习效率
        self.lr = learning_rate

        # 激活函数
        self.activation_function = lambda x: sigmoid(x)
        pass

    def train(self, input_list, target_list):
        inputs = numpy.array(input_list, ndmin=2).T
        target = numpy.array(target_list, ndmin=2).T
        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        output_errors = target - final_outputs
        hidden_errors = numpy.dot(self.who.T, output_errors)

        # 调整神经权重
        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1 - final_outputs)),
                                        numpy.transpose(hidden_outputs))
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1 - hidden_outputs)),
                                        numpy.transpose(inputs))
        pass

    def query(self, input_list):
        inputs = numpy.array(input_list, ndmin=2).T
        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        return final_outputs


input_nodes = 784
hidden_nodes = 100
output_nodes = 10
learning_rate = 0.3
n = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

# 训练
training_data_file = open("mnist_train_100.csv", "r")
training_data_list = training_data_file.readlines()
training_data_file.close()

for record in training_data_list:
    all_values = record.split(',')
    inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
    targets = numpy.zeros(output_nodes) + 0.01
    targets[int(all_values[0])] = 0.99
    n.train(inputs, targets)

    # image_array = numpy.asfarray(all_values[1:]).reshape((28, 28))
    # print(image_array)
    # matplotlib.pyplot.imshow(image_array, cmap='Greys')
    # matplotlib.pyplot.show()

# 测试
test_data_file = open("mnist_test_10.csv", "r")
test_data_list = test_data_file.readlines()
test_data_file.close()
all_values = test_data_list[0].split(',')
print(all_values[0])
image_array = numpy.asfarray(all_values[1:]).reshape((28, 28))
matplotlib.pyplot.imshow(image_array, cmap='Greys')
matplotlib.pyplot.show()
print(n.query((numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01))
