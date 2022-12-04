import numpy as np		# arrays & matricies
from tqdm import tqdm	# command line progress bar
from PIL import Image, ImageOps	# loading images
import os				# access to local filess
from enum import Enum	# create enums
import random
import matplotlib.pyplot as plt
import constants as const
 
from neuralNetwork import NeuralNetwork

class Mode(Enum):
    TRAINING = 1
    TESTING = 2

class Main:

	# static data
	emotionFolderNames = [
		"anger", 
		"disgust", 
		"fear", 
		"happy", 
		"neutral", 
		"sad", 
		"surprise"
		# "1", 
		# "2", 
		# "3", 
		# "4", 
		# "5", 
		# "6", 
		# "7",
	]
	correctOutput = np.array([
		[1,0,0,0,0,0,0], # anger
		[0,1,0,0,0,0,0], # disgust
		[0,0,1,0,0,0,0], # fear
		[0,0,0,1,0,0,0], # happy
		[0,0,0,0,1,0,0], # neutral
		[0,0,0,0,0,1,0], # sad
		[0,0,0,0,0,0,1], # surprise
	])

	def emotionFromOutputArray(self, output):
		highestValueIndex = np.argmax(output)
		return self.emotionFolderNames[highestValueIndex]

	
	def fileNames(self, emotionFolderName, mode: Mode):
		emotionFolderPath = os.getcwd() + '/' + const.DATASET_FOLDER_NAME + '/' + emotionFolderName

		fileNames = [] 
		for name in os.listdir(emotionFolderPath):
			if name.endswith('.jpg') is False: continue
			fileNames.append(name)
		
		totalCount = len(fileNames)
		trainingCount = round(totalCount * const.TRAINING_TEST_RATIO)

		random.shuffle(fileNames)

		if mode is Mode.TRAINING:
			return fileNames[:trainingCount]
		else:
			testingCount = totalCount - trainingCount
			return fileNames[testingCount:]

	def loadImage(self, emotionFolderName, fileName):
			path = const.DATASET_FOLDER_NAME + '/' + emotionFolderName + '/' + fileName
			
			image = Image.open(path)
			image = ImageOps.grayscale(image)
			array = np.array(image).flatten()
			array = np.reshape(array, (len(array), 1))
			normalized = (array / 255) * 2 - 1
			return normalized


	def setup(self):
		print("🛠️  Setting Up")
		
		# create nn
		sizeOfInputLayer = pow(const.INPUT_IMAGE_SIZE, 2) # based on image size
		sizeOfOutputLayer = len(self.correctOutput[0])
		self.network = NeuralNetwork(sizeOfInputLayer, const.SIZE_HIDDEN_LAYER, sizeOfOutputLayer)

		# Setup debug plot
		self.ihWeightSamples = np.empty((0,const.NUM_WEIGHT_SAMPLES), int)
		self.hoWeightSamples = np.empty((0,const.NUM_WEIGHT_SAMPLES), int)

	def getImageAssets(self, mode):
		imageAssets = []

		for (emotionIndex, emotion) in enumerate(self.emotionFolderNames):
			imageFileNames = self.fileNames(emotion, mode)

			for fileName in imageFileNames:
				imageAssets.append((emotion, emotionIndex, fileName))

		# Shuffle images
		random.shuffle(imageAssets)

		return imageAssets

	def train(self, epoch):
		print(f"\n🎛️  Training #{epoch + 1}")

		# Get all images
		allImageAssets = self.getImageAssets(Mode.TRAINING)

		# Train for each image
		for i, imageAsset in enumerate(tqdm(allImageAssets, leave=False)):
			(emotion, emotionIndex, fileName) = imageAsset
			
			expectedOutput = self.correctOutput[emotionIndex]
			expectedOutput = np.reshape(expectedOutput, (len(expectedOutput), 1))
			
			imageInput = self.loadImage(emotion, fileName)
			
			# adjust learning rate if LR_INVERSE_SCALING_ON is true
			learningRate = const.MAX_LEARNING_RATE
			if const.LR_INVERSE_SCALING_ON:
				percentageIncomplete = 1 - (i / len(allImageAssets))
				learningRate *= pow(percentageIncomplete, 2)

			(ihWeightSample, hoWeightSample) = self.network.train(learningRate, imageInput, expectedOutput)

			self.ihWeightSamples = np.append(self.ihWeightSamples, [ihWeightSample], axis=0)
			self.hoWeightSamples = np.append(self.hoWeightSamples, [hoWeightSample], axis=0)
				

	def test(self):
		print("\n📊 Testing")

		# Stat variables
		numOfImagesTested_total = 0
		numOfCorrectlyClassified_total = 0
		specific_percentages = []

		squared_error = np.zeros((len(self.emotionFolderNames,)))

		# Test each emotion
		# TODO: Implement MSE as well. Output both MSE & % correct.
		# TODO: update to use getImageAssets()
		for (emotionIndex, emotion) in enumerate(tqdm(self.emotionFolderNames, leave=False)):

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

				# MSE 
				correctOutput = self.correctOutput[emotionIndex]
				error = correctOutput - output
				squared = np.square(error)
				squared_error = np.add(squared_error, squared)
				
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

		# Calc MSE
		mse = squared_error / numOfImagesTested_total

		# Print stats
		percentage = (numOfCorrectlyClassified_total / numOfImagesTested_total) * 100

		print(f"Binary Accuracy: {round(percentage,2)}% (avg)")
		for index, emotion in enumerate(self.emotionFolderNames):
			print(f"- {emotion.capitalize()}: {round(specific_percentages[index],2)}%")

		print(f"\nMSE: {round(mse.mean(),2)} (avg)")
		for index, emotion in enumerate(self.emotionFolderNames):
			print(f"- {emotion.capitalize()}: {round(mse[index],2)}")

	def showDebugPlot(self):
		# Configure & show debug plot
		x = np.arange(0, len(self.ihWeightSamples), 1)

		fig, axs = plt.subplots(2)
		axs[0].plot(x, self.ihWeightSamples)
		axs[1].plot(x, -self.hoWeightSamples)
		axs[0].set_title('input -> hidden weights (sampled)')
		axs[1].set_title('hidden -> output weights (sampled)')
		plt.show()
		


if __name__ == "__main__":
	main = Main()

	print("")
	main.setup()
	for epoch in range(const.EPOCHS):
		main.train(epoch)
	main.test()
	main.showDebugPlot()

	# if want to test specific image set breakpoint on print("") & run the below code in the debug console:
	# imageData = main.loadImage("positive", "positive_2.png")
	# print(main.network.feedfoward(imageData))

	print("")