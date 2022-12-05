from enum import Enum
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import constants as const
import numpy as np
import random
import os

class ImageUse(Enum):
	TRAINING = 1
	TESTING = 2

def classFromOutput(output):
		highestValueIndex = np.argmax(output)
		return const.CLASSES[highestValueIndex]
	
def fileNames(className, mode: ImageUse):
	# get file names from os
	fileNames = [] 
	classFolderPath = os.getcwd() + '/' + const.DATASET_FOLDER_NAME + '/' + className
	for name in os.listdir(classFolderPath):
		if name.endswith('.jpg') is False: continue # TODO: update to work with any image data type
		fileNames.append(name)
	
	# randomize order of files
	random.shuffle(fileNames)

	# calc num of training / testing
	totalCount = len(fileNames)
	trainingCount = round(totalCount * const.TRAINING_TEST_RATIO)

	if mode is ImageUse.TRAINING:
		return fileNames[:trainingCount]
	else:
		testingCount = totalCount - trainingCount
		return fileNames[testingCount:]

def loadImage(className, fileName):
		path = const.DATASET_FOLDER_NAME + '/' + className + '/' + fileName
		
		image = Image.open(path)
		image = ImageOps.grayscale(image)
		array = np.array(image).flatten()
		array = np.reshape(array, (len(array), 1))
		normalized = (array / 255) * 2 - 1
		
		return normalized

def showDebugPlot(ihWeightSamples, hoWeightSamples):
	# Configure & show debug plot
	x = np.arange(0, len(ihWeightSamples), 1)

	fig, axs = plt.subplots(2)
	axs[0].plot(x, ihWeightSamples)
	axs[1].plot(x, hoWeightSamples)
	axs[0].set_title('Input -> Hidden Weights (sampled)')
	axs[1].set_title('Hidden -> Output Weights (sampled)')
	
	plt.show()

def getImageAssets(mode):
	imageAssets = []

	for (classIndex, className) in enumerate(const.CLASSES):
		imageFileNames = fileNames(className, mode)

		for fileName in imageFileNames:
			imageAssets.append((className, classIndex, fileName))

	# Shuffle images
	random.shuffle(imageAssets)
	
	return imageAssets