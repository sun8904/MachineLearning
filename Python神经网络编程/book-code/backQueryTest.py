import numpy
from NeuralNetwork import NeuralNetwork as neuralNetwork
import matplotlib.pyplot as plt
import pylab

input_nodes = 784
hidden_nodes = 200
output_nodes = 10

# learning rate
learning_rate = 0.1

# create instance of neural network
n = neuralNetwork(input_nodes,hidden_nodes,output_nodes, learning_rate)

# load the mnist training data CSV file into a list
training_data_file = open("mnist_train.csv", 'r')
training_data_list = training_data_file.readlines()
training_data_file.close()

# run the network backwards, given a label, see what image it produces
# train nerual network
epochs =1
for e in  range(epochs):
    for record in training_data_list:
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
print('train done')

while True:
    number= input("plenter enter a number:")
    label = int(number)
    if label<0 | label>9:
        break;
    # label to test
    print(label)
    # create the output signals for this label
    targets = numpy.zeros(output_nodes) + 0.01
    # all_values[0] is the target label for this record
    targets[label] = 0.99
    print(targets)

    # get image data
    image_data = n.backquery(targets)
    print('backquery')
    # plot image data
    plt.imshow(image_data.reshape(28,28), cmap='Greys', interpolation='None')
    pylab.show()