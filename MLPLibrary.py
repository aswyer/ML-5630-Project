from tqdm import tqdm
import constants as const
import numpy as np
from MLPNetwork import MLPNetwork
import helper

class MLPLibrary:
	
	def setup(self):
		
		# Create network
		sizeOfInputLayer = pow(const.INPUT_IMAGE_LENGTH, 2) # based on image size
		sizeOfOutputLayer = len(const.CORRECT_OUTPUT[0])
		self.network = MLPNetwork(sizeOfInputLayer, const.SIZE_HIDDEN_LAYER, sizeOfOutputLayer)

		# Setup debug plot
		self.ihWeightSamples = np.empty((0,const.NUM_WEIGHT_PLOT_SAMPLES), int)
		self.hoWeightSamples = np.empty((0,const.NUM_WEIGHT_PLOT_SAMPLES), int)

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

			(ihWeightSample, hoWeightSample) = self.network.train(learningRate, imageInput, expectedOutput)

			self.ihWeightSamples = np.append(self.ihWeightSamples, [ihWeightSample], axis=0)
			self.hoWeightSamples = np.append(self.hoWeightSamples, [hoWeightSample], axis=0)

	def test(self):

		# Test each class
		testImageAssets = helper.getImageAssets(helper.ImageUse.TESTING)

		# Stat variables
		numOfClasses = len(const.CLASSES)
		squaredError = np.zeros((numOfClasses,))
		correctlyClassified = np.zeros((numOfClasses,))
		totalPerClass = np.zeros((numOfClasses,))

		for imageAsset in tqdm(testImageAssets):
			(className, classIndex, fileName) = imageAsset

			imageData = helper.loadImage(className, fileName)

			# Feedfoward
			output = self.network.feedfoward(imageData)

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
		binaryAccuracyPerClass = np.divide(correctlyClassified, totalPerClass) * 100

		print(f"Binary Accuracy: {round(binaryAccuracy,2)}% (avg)")
		for index, className in enumerate(const.CLASSES):
			print(f"- {className.capitalize()}: {round(binaryAccuracyPerClass[index],2)}%")

		print(f"MSE: {round(mse.mean(),2)} (avg)")
		for index, className in enumerate(const.CLASSES):
			print(f"- {className.capitalize()}: {round(mse[index],2)}")