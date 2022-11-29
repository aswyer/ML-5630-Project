import numpy as np
import math
import random

class NeuralNetworkAlt:

	def __init__(self, numInputNodes, numHiddenNodes, numOutputNodes):
		self.numInputNodes = numInputNodes
		self.numHiddenNodes = numHiddenNodes
		self.numOutputNodes = numOutputNodes

		# weights between each node
		self.weights_input_hidden = np.random.rand(numHiddenNodes, numInputNodes)
		self.weights_hidden_output = np.random.rand(numOutputNodes, numHiddenNodes)

		# bias value for each node (hidden & output)
		self.bias_hidden = np.ones((numHiddenNodes))
		self.bias_output = np.ones((numOutputNodes))

	def sigmoid(self, x):
		return 1/(1+np.exp(-x))

	def feedfoward(self, input):

		# Outputs of "Hidden" Layer
		# dot inputs & input->hidden weights
		hidden = np.matmul(self.weights_input_hidden, input)
		# add hidden layer bias
		hidden_withBias = np.add(hidden, self.bias_hidden)
		# activate using sigmoid
		hidden_activated = self.sigmoid(hidden_withBias)

		# Outputs of "Output" Layer
		# dot hidden outputs & hidden->output weights
		output = np.matmul(self.weights_hidden_output, hidden_activated)
		# add output layer bias
		output_withBias = np.add(output, self.bias_output)
		# activate using sigmoid
		output_activated = self.sigmoid(output_withBias)

		return output_activated

class Neuron:
	def __init__(self, bias):
		self.bias = bias
		self.weights = []

	def calculate_output(self, inputs):
		self.inputs = inputs
		self.output = self.sigmoid(self.calc_net_input)
		return self.outputs

	def calc_net_input(self):
		net_input = 0
		for i in range(len(self.inputs)):
			net_input += self.inputs[i] * self.weights[i]
		return net_input + self.bias

	# where x is the total net input
	def sigmoid(self, x):
		return 1/(1+np.exp(-x))

	# function calculates mse for the individual node	
	def calc_mse(self, target_output):
		return 0.5 * (target_output - self.output) ** 2

	def calc_pd_err_wrt_output(self, target_ouput):
		return -(target_ouput - self.output)

	# calculate partial derivative of total net input with respect to input
	# total net input is the weighted sum of all inputs onto the neuron 
	def calc_pd_tni_wrt_input(self):
		return self.output * (1 - self.output)

	# this function tells how much to change the neurons input to move closer to the expected output
	# calculate partial derivative of error with respect to total net input
	def calc_pd_of_err_wrt_tni(self, target_output):
		return self.calc_pd_err_wrt_output(target_output) * self.calc_pd_tni_wrt_input()

	#gives the partial derivative of total net input with resepct to the weights
	def calc_pd_tni_wrt_weight(self, index):
		return self.inputs[index]

class NeuronLayer:
	def __init__(self, number_neurons, bias):
		
		# sets bias to be the same for all neurons in a layer and randomly initializes it if
		# it has not been initialized yet.
		self.bias = bias if bias else random.random()
		self.neurons = []
		for i in range (number_neurons):
			self.neurons.append(Neuron(self.bias))

		#feeds inputs into the layer and returns outputs
		def feed_forward(self, inputs):
			outputs = []
			for neuron in self.neurons:
				outputs.append(neuron.calculate_output(inputs))
			return outputs

		# TODO: add method to print info of neuron layer
