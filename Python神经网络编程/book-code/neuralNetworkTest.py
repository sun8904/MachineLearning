import numpy
import matplotlib.pyplot as plt
import scipy.misc
import pylab
from NeuralNetwork import NeuralNetwork

# simple nerual network setting
input_nodes=3
hidden_nodes=3
output_nodes=3
learning_rate=0.5
# weight = numpy.random.rand(3,3)
# print(weight)
# weight = (numpy.random.rand(3,3) - 0.5)
# print(weight)
n = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)
print(n.query([1.0, 0.5, -1.5]))


# nerual network settings
input_nodes=784
hidden_nodes=100
output_nodes=10
learning_rate=0.3
n = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

# train and test data download source
# https://pjreddie.com/media/files/mnist_test.csv
# https://pjreddie.com/media/files/mnist_train.csv

# load train data
data_file = open("mnist_train.csv",'r')
data_list = data_file.readlines()
data_file.close()

# train data check and pretreatment
# print('data len', len(data_list))
all_values = data_list[0].split(',')
image_array = numpy.asfarray(all_values[1:]).reshape(28, 28)
scaled_input = (image_array / 255.0 * 0.99) + 0.01
# print('data ', scaled_input)
targets = numpy.zeros(output_nodes) + 0.01
targets[int(all_values[0])] = 0.99
# print('targets',targets)
# plt.imshow(image_array, cmap='Greys', interpolation='None')
# pylab.show()


# train nerual network
epochs =1
for e in  range(epochs):
    for record in data_list:
        all_values = record.split(',')
        image_data = all_values[1:]
        if len(all_values[1:]) != 784:
            print(len(all_values[1:]))
        image_array = numpy.asfarray(all_values[1:])
        scaled_input = (image_array / 255.0 * 0.99)+0.01
        # print('data ', scaled_input)
        targets = numpy.zeros(output_nodes)+0.01
        targets[int(all_values[0])] = 0.99
        n.train(scaled_input, targets)
        pass
    pass

# load test data
test_file = open("mnist_test.csv", 'r')
test_list = test_file.readlines()
test_file.close()

# test data[0] result
all_values = data_list[0].split(',')
print('test case result', all_values[0])
print('network result ', n.query((numpy.asfarray(all_values[1:])/255.0*0.99)+0.01))

# test all result
score_card = []
for record in test_list:
    all_values = record.split(',')
    correct_label = int(all_values[0])
    inputs = numpy.asfarray(all_values[1:])/255.0*0.99+0.01
    outputs = n.query(inputs)
    lable = numpy.argmax(outputs)
    if lable == correct_label:
        score_card.append(1)
    else:
        score_card.append(0)
        pass
    pass
score_card_array = numpy.asarray(score_card)
print('right ', score_card_array.sum())
print('total ', score_card_array.size)
print('performance=', score_card_array.sum()/score_card_array.size)


# test user image
# image_array = scipy.misc.imread('test.jpg', flatten=True)
# image_data = 255 - image_array.reshape(784)
# image_data = (image_data/ 255*0.99)+0.01
