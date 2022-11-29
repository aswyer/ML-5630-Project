import numpy as np
import math
import random

class NeuralNetwork:

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

	# Backpropigation
	def train(self, lr, input, correctOutput):

		# Get current output

		# ---------------- Same code as seen in feedfoward() ------------------
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
		# ---------------------------------------------------------------------

		# Update Weights

		# Hidden -> Output
		# Error of the output
		output_error = correctOutput - output_activated
		# Weights
		output_activated_derivative = output_activated * (1 - output_activated)					# derivative of current output
		output_gradient = np.matmul(output_error, output_activated_derivative)					# hidden->output error * output layer derivatives
		hidden_activated_transpose = np.transpose(hidden_activated)								# hidden layer output transposed
		hidden_output_delta = lr * output_gradient * hidden_activated_transpose					# calc delta
		self.weights_hidden_output = np.add(self.weights_hidden_output, hidden_output_delta)	# update weights
		# Bias
		output_bias_delta = lr * output_gradient
		self.bias_output = np.add(self.bias_output, output_bias_delta)


		# Input -> Hidden
		# Error of hidden -> output weights
		weights_hidden_output_transposed = np.transpose(self.weights_hidden_output) 
		hidden_error = np.matmul(weights_hidden_output_transposed, output_error)
		# Weights
		hidden_activated_derivative = hidden_activated * (1 - hidden_activated)				# derivative of current hidden output
		hidden_gradient = np.matmul(hidden_error, hidden_activated_derivative)				# input->hidden error * hidden layer derivatives
		inputs_transpose = np.transpose(input)												# input layer output transposed
		input_hidden_delta = lr * hidden_gradient * inputs_transpose						# calc delta
		self.weights_input_hidden = np.add(self.weights_input_hidden, input_hidden_delta)	# update weights
		# Bias
		hidden_bias_delta = lr * hidden_gradient
		self.bias_hidden = np.add(self.bias_hidden, hidden_bias_delta)

		return 
		# np.std(self.weights_hidden_output)
		# self.bias_output
		# self.bias_hidden
		# np.sum(np.power(output_error, 2))

		