import numpy as np
import math
import random

class NeuralNetwork:

	def __init__(self, numInputNodes, numHiddenNodes, numOutputNodes):
		self.numInputNodes = numInputNodes
		self.numHiddenNodes = numHiddenNodes
		self.numOutputNodes = numOutputNodes

		# weights between each node
		self.weights_input_hidden = self.randomColumn(numHiddenNodes, numInputNodes)
		self.weights_hidden_output = self.randomColumn(numOutputNodes, numHiddenNodes)

		# bias value for each node (hidden & output)
		self.bias_hidden = self.randomColumn(numHiddenNodes, 1)
		self.bias_output = self.randomColumn(numOutputNodes, 1)

	def randomColumn(self, rows, columns):
		return np.random.rand(rows, columns) * 2 - 1

	def sigmoid(self, x):
		return 1 / (1 + np.exp(-1 * x))

	def feedfoward(self, input):

		# Outputs of "Hidden" Layer
		# multiply inputs & input->hidden weights
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

	# Backpropigation
	def train(self, lr, input, correctOutput):

		# Get current output

		# ---------------- Same code as seen in feedfoward() ------------------
		# Outputs of "Hidden" Layer
		# multiply inputs & input->hidden weights
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
		# ---------------------------------------------------------------------

		# Update Weights

		# output error
		output_errors = correctOutput - output_activated
		# output gradient
		output_gradients = output_activated * (1 - output_activated) 	# sigmoid * (1 - sigmoid)
		output_gradients = np.multiply(output_gradients, output_errors) # * E
		output_gradients = output_gradients * lr						# * lr
		# hidden -> output deltas
		hidden_activated_transposed = np.transpose(hidden_activated)
		weights_hidden_output_deltas = np.matmul(output_gradients, hidden_activated_transposed)
		# update weights	
		self.weights_hidden_output = self.weights_hidden_output + weights_hidden_output_deltas
		# update bias
		self.bias_output = np.add(self.bias_output, output_gradients)


		# hidden error
		weights_hidden_output_transposed = np.transpose(self.weights_hidden_output) 
		hidden_errors = np.matmul(weights_hidden_output_transposed, output_errors) # feed backwards to get error at the hidden layer
		# hidden gradient
		hidden_gradients = hidden_activated * (1 - hidden_activated)
		hidden_gradients = np.multiply(hidden_gradients, hidden_errors)
		hidden_gradients = hidden_gradients * lr
		# input -> hidden deltas
		inputs_transposed = np.transpose(input)
		weights_input_hidden_deltas = np.matmul(hidden_gradients, inputs_transposed)
		# update weights
		self.weights_input_hidden = self.weights_input_hidden + weights_input_hidden_deltas
		if math.isnan(self.weights_input_hidden[0][0]):
			print("NAN")
		# update bias
		self.bias_hidden = np.add(self.bias_hidden, hidden_gradients)

		# return debug value to plot
		flattened = self.weights_input_hidden.flat
		return flattened[::int(np.ceil(len(flattened)/20))]
		# np.random.choice(self.weights_input_hidden.flat, size=3, replace=False)
		# self.weights_input_hidden.flatten()
		# self.weights_hidden_output[7]
		# np.std(self.weights_hidden_output)
		# self.bias_output
		# self.bias_hidden
		# np.sum(np.power(output_error, 2))

		