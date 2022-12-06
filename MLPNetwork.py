import constants as const
import numpy as np

class MLPNetwork:

	def __init__(self, numInputNodes, numHiddenNodes, numOutputNodes):
		self.numInputNodes = numInputNodes
		self.numHiddenNodes = numHiddenNodes
		self.numOutputNodes = numOutputNodes

		# weights between each node
		self.weights_input_hidden = self.random(numHiddenNodes, numInputNodes)
		self.weights_hidden_output = self.random(numOutputNodes, numHiddenNodes)

		# bias value for each node (hidden & output)
		self.bias_hidden = self.random(numHiddenNodes, 1)
		self.bias_output = self.random(numOutputNodes, 1)

	def random(self, rows, columns):
		return np.random.rand(rows, columns) * 2 - 1

	def activationFunc(self, x):
		return np.maximum(0, x) #relu
		# return 1 / (1 + np.exp(-x)) # sigmoid
	
	# assume x has been run through sigmoid already
	def derivative_activationFunc(self, x):
		# relu
		x[x<=0] = 0
		x[x>0] = 1
		return x

		# sigmoid
		# return x * (1 - x)

	def feedfoward(self, input):

		# format inputs
		input = np.reshape(input, (len(input), 1))

		# Outputs of "Hidden" Layer
		# multiply inputs & input->hidden weights
		hidden = np.matmul(self.weights_input_hidden, input)
		# add hidden layer bias	
		hidden_withBias = np.add(hidden, self.bias_hidden)
		# activate using sigmoid
		hidden_activated = self.activationFunc(hidden_withBias)

		# Outputs of "Output" Layer
		# dot hidden outputs & hidden->output weights
		output = np.matmul(self.weights_hidden_output, hidden_activated)
		# add output layer bias
		output_withBias = np.add(output, self.bias_output)
		# activate using sigmoid
		output_activated = self.activationFunc(output_withBias)
		reshaped = np.reshape(output_activated, (self.numOutputNodes,))
		return reshaped

	# Backpropagation
	def train(self, lr, input, correctOutput):

		# format inputs
		input = np.reshape(input, (len(input), 1))
		correctOutput = np.reshape(correctOutput, (len(correctOutput), 1))

		# ---------------- Same code as seen in feedfoward() ------------------
		# Outputs of "Hidden" Layer
		# multiply inputs & input->hidden weights
		hidden = np.matmul(self.weights_input_hidden, input)
		# add hidden layer bias	
		hidden_withBias = np.add(hidden, self.bias_hidden)
		# activate using sigmoid
		hidden_activated = self.activationFunc(hidden_withBias)

		# Outputs of "Output" Layer
		# dot hidden outputs & hidden->output weights
		output = np.matmul(self.weights_hidden_output, hidden_activated)
		# add output layer bias
		output_withBias = np.add(output, self.bias_output)
		# activate using sigmoid
		output_activated = self.activationFunc(output_withBias)
		# ---------------------------------------------------------------------


		# Update hidden -> output weights
		# output error
		output_errors = correctOutput - output_activated
		# output gradient
		output_gradients = self.derivative_activationFunc(output_activated)
		output_gradients = np.multiply(output_gradients, output_errors) # * E
		output_gradients = output_gradients * lr						# * lr
		# hidden -> output deltas
		hidden_activated_transposed = np.transpose(hidden_activated)
		weights_hidden_output_deltas = np.matmul(output_gradients, hidden_activated_transposed)
		# update weights	
		self.weights_hidden_output = self.weights_hidden_output + weights_hidden_output_deltas
		# update bias
		self.bias_output = np.add(self.bias_output, output_gradients)
		

		# Update input -> hidden weights
		# hidden error
		weights_hidden_output_transposed = np.transpose(self.weights_hidden_output) 
		hidden_errors = np.matmul(weights_hidden_output_transposed, output_errors) # feed backwards to get error at the hidden layer
		# hidden gradient
		hidden_gradients = self.derivative_activationFunc(hidden_activated)
		hidden_gradients = np.multiply(hidden_gradients, hidden_errors)
		hidden_gradients = hidden_gradients * lr
		# input -> hidden deltas
		inputs_transposed = np.transpose(input)
		weights_input_hidden_deltas = np.matmul(hidden_gradients, inputs_transposed)
		# update weights
		self.weights_input_hidden = self.weights_input_hidden + weights_input_hidden_deltas
		# update bias
		self.bias_hidden = np.add(self.bias_hidden, hidden_gradients)


		if const.SHOULD_PLOT_WEIGHTS:
			# return debug values to plot
			ih_flattened = self.weights_input_hidden.flat
			ih_samples = ih_flattened[::int(np.ceil(len(ih_flattened)/const.NUM_WEIGHT_PLOT_SAMPLES))]
			ho_flattened = self.weights_hidden_output.flat
			ho_samples = ho_flattened[::int(np.ceil(len(ho_flattened)/const.NUM_WEIGHT_PLOT_SAMPLES))]
			return (ih_samples, ho_samples)
