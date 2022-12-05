import numpy as np		# arrays & matricies
from tqdm import tqdm	# command line progress bar
from PIL import Image, ImageOps	# loading images
import os				# access to local filess
from enum import Enum	# create enums
import random
import matplotlib.pyplot as plt
import constants as const
from sklearn.neural_network import MLPClassifier

# TODO: Refactor this file to share code with main.py

class Mode(Enum):
    TRAINING = 1
    TESTING = 2

class Baselines:

	def emotionFromOutputArray(self, output):
		highestValueIndex = np.argmax(output)
		return const.CLASSES[highestValueIndex]
	
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
		self.network = MLPClassifier(max_iter=const.EPOCHS, activation='logistic', hidden_layer_sizes=(const.SIZE_HIDDEN_LAYER, const.NUM_HIDDEN_LAYERS), learning_rate=('invscaling' if const.LR_INVERSE_SCALING_ON else 'constant'), learning_rate_init=const.MAX_LEARNING_RATE,random_state=1)

		# Setup debug plot
		self.ihWeightSamples = np.empty((0,const.NUM_WEIGHT_SAMPLES), int)
		self.hoWeightSamples = np.empty((0,const.NUM_WEIGHT_SAMPLES), int)

	def getImageAssets(self):
		imageAssets = []

		for (emotionIndex, emotion) in enumerate(const.CLASSES):
			imageFileNames = self.fileNames(emotion, Mode.TRAINING)

			for fileName in imageFileNames:
				imageAssets.append((emotion, emotionIndex, fileName))

		# Shuffle images
		random.shuffle(imageAssets)

		return imageAssets

	def train(self):
		print(f"🎛️  Training")

		# Get all images
		allImageAssets = self.getImageAssets()

		# Train for each image
		for imageAsset in tqdm(allImageAssets, leave=False):
			(emotion, emotionIndex, fileName) = imageAsset
			
			imageInput = self.loadImage(emotion, fileName).transpose()
			expectedOutput = const.CORRECT_OUTPUT[emotionIndex]
			expectedOutput = self.emotionFromOutputArray(expectedOutput)

			self.network.partial_fit(imageInput, [expectedOutput], const.CLASSES)
				

	def test(self):
		print("📊 Testing")

		# Stat variables
		numOfImagesTested_total = 0
		numOfCorrectlyClassified_total = 0
		specific_percentages = []

		# Test each emotion
		for emotion in tqdm(const.CLASSES, leave=False):

			# Stats for this specific emotion
			numOfCorrectlyClassified_specific = 0

			# Get image filenames for testing
			imageFileNames = self.fileNames(emotion, Mode.TESTING)
			numOfImagesTested_specific = len(imageFileNames)

			# Go through each file in this emotion
			for fileName in imageFileNames:

				# Load image & process it with the neural network
				imageData = self.loadImage(emotion, fileName).transpose()
				
				output = self.network.predict_proba(imageData)
				
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
		print(f"Accuracy: {round(percentage,2)}%")

		for index, emotion in enumerate(const.CLASSES):
			print(f"- {emotion.capitalize()}: {round(specific_percentages[index],2)}%")