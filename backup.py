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