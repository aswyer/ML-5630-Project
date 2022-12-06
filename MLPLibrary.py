from MLPNetwork import MLPNetwork
import constants as const
from tqdm import tqdm
import numpy as np
import helper

class MLPLibrary:
	
	def setup(self):
		
		# Create network
		sizeOfInputLayer = pow(const.INPUT_IMAGE_LENGTH, 2) # based on image size
		sizeOfOutputLayer = len(const.CORRECT_OUTPUT[0])
		self.network = MLPNetwork(sizeOfInputLayer, const.SIZE_HIDDEN_LAYER, sizeOfOutputLayer)

		# Setup debug plot
		if const.SHOULD_PLOT_WEIGHTS:
			self.ihWeightSamples = []
			self.hoWeightSamples = []

	def train(self):

		# Get all images
		allImageAssets = helper.getImageAssets(helper.ImageUse.TRAINING)

		# Train for each image
		for i, imageAsset in enumerate(tqdm(allImageAssets)):
			(className, classIndex, fileName) = imageAsset

			imageInput = helper.loadImage(className, fileName)

			expectedOutput = const.CORRECT_OUTPUT[classIndex]
			
			# adjust learning rate if LR_INVERSE_SCALING_ON is true
			learningRate = const.MAX_LEARNING_RATE
			if const.LR_INVERSE_SCALING_ON:
				percentageIncomplete = 1 - (i / len(allImageAssets))
				learningRate *= pow(percentageIncomplete, 2)

			if const.SHOULD_PLOT_WEIGHTS:
				(ihWeightSample, hoWeightSample) = self.network.train(learningRate, imageInput, expectedOutput)
				self.ihWeightSamples.append(ihWeightSample)
				self.hoWeightSamples.append(hoWeightSample)
			else:
				self.network.train(learningRate, imageInput, expectedOutput)

	def test(self, mode = helper.ImageUse.TESTING):

		# Test each class
		testImageAssets = helper.getImageAssets(mode)

		# Stat variables
		numOfClasses = len(const.CLASSES)
		squaredError = np.zeros((numOfClasses,))
		correctlyClassified = np.zeros((numOfClasses,))
		totalPerClass = np.zeros((numOfClasses,))

		shouldLeaveProgressBar = mode != helper.ImageUse.TESTING_SAMPLE
		for imageAsset in tqdm(testImageAssets, leave=shouldLeaveProgressBar):
			(className, classIndex, fileName) = imageAsset

			imageData = helper.loadImage(className, fileName)

			# Feedfoward
			output = self.network.feedforward(imageData)

			# Update total
			totalPerClass[classIndex] += 1

			# Binary accuracy
			predictedClass = helper.classFromOutput(output)
			if predictedClass == className:
				correctlyClassified[classIndex] += 1

			# MSE
			correctOutput = const.CORRECT_OUTPUT[classIndex]
			error = correctOutput - output
			squared = np.square(error)
			squaredError = np.add(squaredError, squared)

		numCorrect = np.sum(correctlyClassified)
		numTotal = np.sum(totalPerClass)

		# Calc MSE
		mse = np.divide(squaredError, numTotal)

		# Print stats
		binaryAccuracy = (numCorrect/numTotal) * 100

		if mode == helper.ImageUse.TESTING_SAMPLE:
			sampleAccuracy = round(binaryAccuracy,2)
			print(f'-> Sample Accuracy: {sampleAccuracy}%')

		else:
			binaryAccuracyPerClass = np.divide(correctlyClassified, totalPerClass) * 100
			print(f"Binary Accuracy: {round(binaryAccuracy,2)}% (avg)")
			for index, className in enumerate(const.CLASSES):
				print(f"- {className.capitalize()}: {round(binaryAccuracyPerClass[index],2)}%")

			print(f"MSE: {round(mse.mean(),2)} (avg)")
			for index, className in enumerate(const.CLASSES):
				print(f"- {className.capitalize()}: {round(mse[index],2)}")