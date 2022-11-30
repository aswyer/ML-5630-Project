import numpy as np		# arrays & matricies
from tqdm import tqdm	# command line progress bar
from PIL import Image, ImageOps	# loading images
import os				# access to local filess
from enum import Enum	# create enums
import string
import random
import matplotlib.pyplot as plt
 
from neuralNetwork import NeuralNetwork

class Mode(Enum):
    TRAINING = 1
    TESTING = 2

DATASET_FOLDER_NAME = "dataset" #dataset_alt
LEARNING_RATE = 0.5
SIZE_HIDDEN_LAYER = 40
EPOCHS = 1

class Main:

	# static data
	emotionFolderNames = [
		"anger", 
		"contempt", 
		"disgust", 
		"fear", 
		"happiness", 
		"neutrality", 
		"sadness",
		"surprise"
		# "negative", 
		# "positive"
	]
	correctOutput = np.array([
		[1,0,0,0,0,0,0,0], # anger
		[0,1,0,0,0,0,0,0], # contempt
		[0,0,1,0,0,0,0,0], # digust
		[0,0,0,1,0,0,0,0], # fear
		[0,0,0,0,1,0,0,0], # happiness
		[0,0,0,0,0,1,0,0], # neutrality
		[0,0,0,0,0,0,1,0], # sadness
		[0,0,0,0,0,0,0,1]  # surprise
		# [1, 0], # negative
		# [0, 1],  # positive
	])

	def emotionFromOutputArray(self, output):
		highestValueIndex = np.argmax(output)
		return self.emotionFolderNames[highestValueIndex]
		# if output[1] > output[0]:
		# 	return self.emotionFolderNames[1]
		# else:
		# 	return self.emotionFolderNames[0]
	
	def fileNames(self, emotionFolderName, mode: Mode):
		emotionFolderPath = os.getcwd() + '/' + DATASET_FOLDER_NAME + '/' + emotionFolderName

		fileNames = [] 
		for name in os.listdir(emotionFolderPath):
			if name.endswith('.png') is False: continue
			fileNames.append(name)
		
		totalCount = len(fileNames)
		trainingCount = round(totalCount * 0.9)

		if mode is Mode.TRAINING:
			return fileNames[:trainingCount]
		else:
			testingCount = totalCount - trainingCount
			return fileNames[testingCount:]

	def loadImage(self, emotionFolderName, fileName):
			path = DATASET_FOLDER_NAME + '/' + emotionFolderName + '/' + fileName
			
			image = Image.open(path)
			image = ImageOps.grayscale(image)
			array = np.array(image).flatten()
			array = np.reshape(array, (len(array), 1))
			normalized = (array / 255) * 2 - 1
			return normalized


	def setup(self):
		print("🛠️  Setting Up")
		
		# create nn
		sizeOfInputLayer = 224 * 224 # based on image size
		sizeOfOutputLayer = len(self.correctOutput[0])
		self.network = NeuralNetwork(sizeOfInputLayer, SIZE_HIDDEN_LAYER, sizeOfOutputLayer)

	def train(self, epoch):
		print(f"\n🎛️  Training #{epoch + 1}")

		# Get all images
		imageAssets = []

		for (emotionIndex, emotion) in enumerate(self.emotionFolderNames):
			imageFileNames = self.fileNames(emotion, Mode.TRAINING)

			for fileName in imageFileNames:
				imageAssets.append((emotion, emotionIndex, fileName))

		# Shuffle images
		random.shuffle(imageAssets)

		# Setup debug plot
		y = np.empty((0,10), int)

		# Train for each image
		for imageAsset in tqdm(imageAssets, leave=False):
			(emotion, emotionIndex, fileName) = imageAsset
			
			expectedOutput = self.correctOutput[emotionIndex]
			expectedOutput = np.reshape(expectedOutput, (len(expectedOutput), 1))
			
			imageInput = self.loadImage(emotion, fileName)
			debugValue = self.network.train(LEARNING_RATE, imageInput, expectedOutput)

			y = np.append(y, [debugValue], axis=0)

		# Configure & show debug plot
		y = y[::20]
		x = np.arange(0, len(y), 1)
		plot = plt.plot(x,y)
		plt.show()
				

	def test(self):
		print("\n📊 Testing")

		# Stat variables
		numOfImagesTested_total = 0
		numOfCorrectlyClassified_total = 0
		specific_percentages = []

		# Test each emotion
		for emotion in tqdm(self.emotionFolderNames, leave=False):

			# Stats for this specific emotion
			numOfCorrectlyClassified_specific = 0

			# Get image filenames for testing
			imageFileNames = self.fileNames(emotion, Mode.TESTING)
			numOfImagesTested_specific = len(imageFileNames)

			# Go through each file in this emotion
			for fileName in imageFileNames:

				# Load image & process it with the neural network
				imageData = self.loadImage(emotion, fileName)
				output = self.network.feedfoward(imageData)
				
				# Compare to the correct output
				predictedEmotion = self.emotionFromOutputArray(output)
				if predictedEmotion == emotion:
					numOfCorrectlyClassified_specific += 1

			# Update total stats from specifc stats
			numOfImagesTested_total += numOfImagesTested_specific
			numOfCorrectlyClassified_total += numOfCorrectlyClassified_specific

			# Save specific stats
			percentage = (numOfCorrectlyClassified_specific / numOfImagesTested_specific) * 100
			specific_percentages.append(percentage)

		# Print stats
		percentage = (numOfCorrectlyClassified_total / numOfImagesTested_total) * 100
		print(f"Accuracy: {percentage}%")

		for index, emotion in enumerate(self.emotionFolderNames):
			print(f"- {emotion.capitalize()}: {specific_percentages[index]}%")
		


if __name__ == "__main__":
	main = Main()

	print("")
	main.setup()
	for epoch in range(EPOCHS):
		main.train(epoch)
	main.test()
	
	# if want to test specific image set breakpoint on print("") & run the below code in the debug console:
	# imageData = main.loadImage("positive", "positive_2.png")
	# print(main.network.feedfoward(imageData))

	print("")
	plt.show()