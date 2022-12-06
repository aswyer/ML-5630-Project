from MLPClassifierBaseline import MLPClassifierBaseline
from MLPLibrary import MLPLibrary
import constants as const
import subprocess
import helper
import os

def customMLP():
	mlp = MLPLibrary()

	print("🛠️  Setting Up")
	mlp.setup()

	for epoch in range(const.EPOCHS):
		print(f"🎛️  Training #{epoch+1}/{const.EPOCHS}")
		mlp.train()
	
	print("📊 Testing")
	mlp.test()
	# if want to test specific image set breakpoint above & run the below code in the debug console:
	# imageData = main.loadImage("positive", "positive_2.png")
	# print(main.network.feedfoward(imageData))

	if const.SHOULD_PLOT_WEIGHTS:
		print("")
		print("🔔 Close graph to continue")
		helper.showDebugPlot(mlp.ihWeightSamples, mlp.hoWeightSamples)

def mlpClassifierBaseline():
	mlp = MLPClassifierBaseline()

	print("🛠️  Setting Up")
	mlp.setup()

	print(f"🖼️  Importing Images")
	mlp.loadImages()

	print(f"🎛️  Training...")
	mlp.train()

	print("📊 Testing")
	mlp.test()

	print("")

if __name__ == "__main__":
	print(r"""

==================================================                                                 
 ___ ___ ___ ___           _____ __    _____ 
|  _|  _|_  |   |   ___   |     |  |  |  _  |
|_  | . |_  | | |  |___|  | | | |  |__|   __|
|___|___|___|___|         |_|_|_|_____|__|

by: Andrew Sawyer, Fuller Henderson, Luke Robinson
==================================================
	""")

	continueRunning = True
	while continueRunning:
		print("")
		print("💬 What would you like to do? (enter #)")
		print("1. Custom MLP: train & test")
		print("2. MLPClassifier (sklearn) baseline: train & test")
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
			print("--------------------------")
		elif userInput == 2:
			print("---------------------------------------")
			print("Running sklearn.MLPClassifier Baseline:")
			print("---------------------------------------")
			mlpClassifierBaseline()
			print("------------------")
		elif userInput == 3:
			print("")
			constantsFilePath = os.getcwd() + '/constants.py';
			subprocess.call(["open", "-R", constantsFilePath])
		elif userInput == 4:
			exit()
		else:
			continue
