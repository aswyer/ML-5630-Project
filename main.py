from neuralNetwork import NeuralNetwork
import numpy as np

class Main:

	def setup(self):
		print("start setup")
		
		self.network = NeuralNetwork(2,2,1)

		# load images

	def train(self):
		print("start training")

	def test(self):
		print("start testing")

		# test images
		input = np.array([	
			[0],
			[1]
		])
		output = self.network.feedfoward(input);
		print(output)


if __name__ == "__main__":
	main = Main()
	main.setup()
	main.train()
	main.test()