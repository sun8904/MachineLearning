import numpy
import scipy.special


class NeuralNetwork(object):
    def __init__(self, inputNodes, hiddenNodes, outputNodes, learningRate):
        self.inputNodes = inputNodes
        self.hiddenNodes = hiddenNodes
        self.outputNodes = outputNodes
        self.learningRate = learningRate

        # self.wih = (numpy.random.rand(self.hiddenNodes,self.inputNodes)-0.5)
        # self.who= (numpy.random.rand(self.outputNodes,self.hiddenNodes)-0.5)
        self.wih = (numpy.random.normal(0.0, pow(self.hiddenNodes, -0.5), (self.hiddenNodes, self.inputNodes)))
        self.who = (numpy.random.normal(0.0, pow(self.outputNodes, -0.5), (self.outputNodes, self.hiddenNodes)))

        self.lr=learningRate

        self.activation_function = lambda x: scipy.special.expit(x)

        pass

    def train(self, inputs_list, targets_list):
        inputs = numpy.array(inputs_list, ndmin=2).T
        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        targets= numpy.array(targets_list, ndmin=2).T
        output_errors = targets-final_outputs
        hidden_errors = numpy.dot(self.who.T, output_errors)
        self.who += self.lr* numpy.dot( (output_errors* final_outputs *(1-final_outputs)),
            numpy.transpose(hidden_outputs))

        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1 - hidden_outputs)),
                                        numpy.transpose(inputs))
        pass

    def query(self, inputs_list):
        inputs = numpy.array(inputs_list, ndmin=2).T
        # print('inputs ',inputs)
        hidden_inputs = numpy.dot(self.wih, inputs)

        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        return final_outputs
    pass



