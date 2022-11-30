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
LEARNING_RATE = 0.01
SIZE_HIDDEN_LAYER = 25

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
	])

	def emotionFromOutputArray(self, output):
		highestValueIndex = np.argmax(output)
		return self.emotionFolderNames[highestValueIndex]
	
	def fileNames(self, emotionFolderName, mode: Mode):
		emotionFolderPath = os.getcwd() + '/' + DATASET_FOLDER_NAME + '/' + emotionFolderName

		fileNames = [] 
		for name in os.listdir(emotionFolderPath):
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
		sizeOfOutputLayer = len(self.emotionFolderNames)
		self.network = NeuralNetwork(sizeOfInputLayer, SIZE_HIDDEN_LAYER, sizeOfOutputLayer)

	def train(self):
		print("\n🎛️  Training:")

		# Get all images
		imageAssets = []

		for (emotionIndex, emotion) in enumerate(self.emotionFolderNames):
			imageFileNames = self.fileNames(emotion, Mode.TRAINING)

			for fileName in imageFileNames:
				if fileName.endswith('.png') is False: continue
				imageAssets.append((emotion, emotionIndex, fileName))

		# Shuffle images
		random.shuffle(imageAssets)

		# Train for each image
		for imageAsset in tqdm(imageAssets, leave=False):
			(emotion, emotionIndex, fileName) = imageAsset
			
			expectedOutput = self.correctOutput[emotionIndex]
			expectedOutput = np.reshape(expectedOutput, (len(expectedOutput), 1))
			
			imageInput = self.loadImage(emotion, fileName)
			error = self.network.train(LEARNING_RATE, imageInput, expectedOutput)
				

	def test(self):
		print("\n📊 Testing:")

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
				if fileName.endswith('.png') is False:
					continue
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
	main.train()
	main.test()
	print("")
	# plt.show()