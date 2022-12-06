from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPClassifier
import constants as const
from tqdm import tqdm
import numpy as np
import warnings
import helper
import time

class MLPClassifierBaseline:

	def setup(self):
		# create network
		self.network = MLPClassifier(
			hidden_layer_sizes=(const.SIZE_HIDDEN_LAYER,), 
			max_iter=const.EPOCHS,
			activation='logistic', #'relu',
			solver="sgd",
			learning_rate=('invscaling' if const.LR_INVERSE_SCALING_ON else 'constant'), 
			learning_rate_init=const.MAX_LEARNING_RATE,
			random_state=1,
		)

	def loadImages(self):
		# Get all images
		allImageAssets = helper.getImageAssets(helper.ImageUse.TRAINING)

		self.X = []
		self.y = []

		# Train for each image
		for i, imageAsset in enumerate(tqdm(allImageAssets)):
			(className, classIndex, fileName) = imageAsset

			Xi = helper.loadImage(className, fileName)

			self.X.append(Xi)
			self.y.append(className)

	def train(self):
		with warnings.catch_warnings():
			warnings.filterwarnings("ignore", category=ConvergenceWarning, module="sklearn")
			self.network.fit(self.X, self.y)

	def test(self):

		# Test each class
		testImageAssets = helper.getImageAssets(helper.ImageUse.TESTING)

		# Stat variables
		numOfClasses = len(const.CLASSES)
		correctlyClassified = np.zeros((numOfClasses,))
		totalPerClass = np.zeros((numOfClasses,))

		for imageAsset in tqdm(testImageAssets):
			(className, classIndex, fileName) = imageAsset

			imageData = helper.loadImage(className, fileName)
			imageData = np.reshape(imageData, (1,const.INPUT_IMAGE_FLAT_LENGTH))

			# Feedfoward
			output = self.network.predict(imageData)[0]

			# Update total
			totalPerClass[classIndex] += 1

			# Binary accuracy
			if output == className:
				correctlyClassified[classIndex] += 1

		numCorrect = np.sum(correctlyClassified)
		numTotal = np.sum(totalPerClass)

		# Print stats
		binaryAccuracy = (numCorrect/numTotal) * 100
		binaryAccuracyPerClass = np.divide(correctlyClassified, totalPerClass) * 100

		print(f"Binary Accuracy: {round(binaryAccuracy,2)}% (avg)")
		for index, className in enumerate(const.CLASSES):
			print(f"- {className.capitalize()}: {round(binaryAccuracyPerClass[index],2)}%")