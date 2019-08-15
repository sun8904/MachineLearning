import numpy
import matplotlib.pyplot as plt
from NeuralNetwork import NeuralNetwork as neuralNetwork
import scipy.ndimage

import pylab

# number of input, hidden and output nodes
input_nodes = 784
hidden_nodes = 200
output_nodes = 10

# learning rate
learning_rate = 0.01

# create instance of neural network
n = neuralNetwork(input_nodes,hidden_nodes,output_nodes, learning_rate)
# load the mnist training data CSV file into a list
training_data_file = open("mnist_train.csv", 'r')
training_data_list = training_data_file.readlines()
training_data_file.close()


# train the neural network

# epochs is the number of times the training data set is used for training
epochs = 10
for e in range(epochs):
    # go through all records in the training data set
    for record in training_data_list:
        # split the record by the ',' commas
        all_values = record.split(',')
        # scale and shift the inputs
        inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        # create the target output values (all 0.01, except the desired label which is 0.99)
        targets = numpy.zeros(output_nodes) + 0.01
        # all_values[0] is the target label for this record
        targets[int(all_values[0])] = 0.99
        n.train(inputs, targets)

        ## create rotated variations
        # rotated anticlockwise by x degrees
        inputs_plusx_img = scipy.ndimage.interpolation.rotate(inputs.reshape(28, 28), 10, cval=0.01, order=1,
                                                              reshape=False)
        n.train(inputs_plusx_img.reshape(784), targets)
        # rotated clockwise by x degrees
        inputs_minusx_img = scipy.ndimage.interpolation.rotate(inputs.reshape(28, 28), -10, cval=0.01, order=1,
                                                               reshape=False)
        n.train(inputs_minusx_img.reshape(784), targets)

        # rotated anticlockwise by 10 degrees
        # inputs_plus10_img = scipy.ndimage.interpolation.rotate(inputs.reshape(28,28), 10, cval=0.01, order=1, reshape=False)
        # n.train(inputs_plus10_img.reshape(784), targets)
        # rotated clockwise by 10 degrees
        # inputs_minus10_img = scipy.ndimage.interpolation.rotate(inputs.reshape(28,28), -10, cval=0.01, order=1, reshape=False)
        # n.train(inputs_minus10_img.reshape(784), targets)

        pass
    pass


# load test data
test_file = open("mnist_test.csv", 'r')
test_list = test_file.readlines()
test_file.close()

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
# calculate the performance score, the fraction of correct answers
scorecard_array = numpy.asarray(score_card)
print ("performance = ", scorecard_array.sum() / scorecard_array.size)