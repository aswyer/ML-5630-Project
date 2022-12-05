import constants as const
from neuralNetworkLibrary import NeuralNetworkLibrary
from baselines import Baselines
import subprocess
import os

def customMLP():
	mlp = NeuralNetworkLibrary()
	mlp.setup()
	for epoch in range(const.EPOCHS):
		mlp.train(epoch)
	mlp.test()
	# if want to test specific image set breakpoint on mlp.showDebugPlot() & run the below code in the debug console:
	# imageData = main.loadImage("positive", "positive_2.png")
	# print(main.network.feedfoward(imageData))
	mlp.showDebugPlot()

def baselines():
	main = Baselines()
	main.setup()
	main.train()
	main.test()

if __name__ == "__main__":
	print(r"""

==================================================	                                       
 ___ ___ ___ ___     _____ __    _____ 
|  _|  _|_  |   |___|     |  |  |  _  |
|_  | . |_  | | |___| | | |  |__|   __|
|___|___|___|___|   |_|_|_|_____|__|   

By: Andrew Sawyer, Fuller Henderson, Luke Robinson
==================================================
	""")

	continueRunning = True
	while continueRunning:
		print("")
		print("💬 What would you like to do? (enter #)")
		print("1. Custom MLP: train & test")
		print("2. Baselines: train & test")
		print("3. Edit Constants File")
		print("4. Quit")
		print("")
		print("=>", end=" ")
		userInput = int(input())
		if userInput == 1:
			print("-------------------")
			print("Running Custom MLP:")
			print("-------------------")
			customMLP()
			print("-------------------")
		elif userInput == 2:
			print("------------------")
			print("Running Baselines:")
			print("------------------")
			baselines()
			print("------------------")
		elif userInput == 3:
			print("")
			constantsFilePath = os.getcwd() + '/constants.py';
			subprocess.call(["open", "-R", constantsFilePath])
		elif userInput == 4:
			exit()
		else:
			continue
