from sklearn.neural_network import MLPClassifier
import constants as const
from tqdm import tqdm
import numpy as np
import helper

class MLPClassifierBaseline:

	def setup(self):
		# create network
		self.network = MLPClassifier(
			solver='sgd',
			activation='logistic', 
			hidden_layer_sizes=(const.SIZE_HIDDEN_LAYER, const.NUM_HIDDEN_LAYERS), 
			learning_rate=('invscaling' if const.LR_INVERSE_SCALING_ON else 'constant'), 
			learning_rate_init=const.MAX_LEARNING_RATE,
			max_iter=const.EPOCHS, 
			random_state=1
		)

		# Setup debug plot
		self.ihWeightSamples = np.empty((0,const.NUM_WEIGHT_PLOT_SAMPLES), int)
		self.hoWeightSamples = np.empty((0,const.NUM_WEIGHT_PLOT_SAMPLES), int)

	def train(self):
		# Get all images
		allImageAssets = helper.getImageAssets(helper.ImageUse.TRAINING)

		# Train for each image
		for i, imageAsset in enumerate(tqdm(allImageAssets, leave=False)):
			(className, classIndex, fileName) = imageAsset
			
			expectedOutput = [const.CORRECT_OUTPUT[classIndex]]
			
			imageInput = helper.loadImage(className, fileName).transpose()
			
			# adjust learning rate if LR_INVERSE_SCALING_ON is true
			learningRate = const.MAX_LEARNING_RATE
			if const.LR_INVERSE_SCALING_ON:
				percentageIncomplete = 1 - (i / len(allImageAssets))
				learningRate *= pow(percentageIncomplete, 2)

			self.network.partial_fit(imageInput, expectedOutput, const.CORRECT_OUTPUT) #TODO: review this

			ih_flattened = self.network.coefs_[0].flat
			ih_samples = ih_flattened[::int(np.ceil(len(ih_flattened)/const.NUM_WEIGHT_PLOT_SAMPLES))]
			ho_flattened = self.network.coefs_[1].flat
			ho_samples = ho_flattened[::int(np.ceil(len(ho_flattened)/const.NUM_WEIGHT_PLOT_SAMPLES))]

			self.ihWeightSamples = np.append(self.ihWeightSamples, [ih_samples], axis=0)
			self.hoWeightSamples = np.append(self.hoWeightSamples, [ho_samples], axis=0)

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

			imageData = helper.loadImage(className, fileName).transpose()

			# Feedfoward
			output = self.network.predict(imageData)[0]

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