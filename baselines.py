######Imports######
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import main
import random
import matplotlib.pyplot as plt
import pandas as pd
import os
import cv2
from pathlib import Path
from PIL import Image, ImageOps
from sklearn.utils import Bunch
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
######Imports######

DATASET_FOLDER_NAME = "dataset"
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
# 
fileNames = []
flatFiles = []
targets = []

def load_images():
	for i, emotion in enumerate(emotionFolderNames):
		emotionFolderPath = os.getcwd() + '/' + DATASET_FOLDER_NAME + '/' + emotion
		for name in os.listdir(emotionFolderPath):
			emotionFolderPath = os.getcwd() + '/' + DATASET_FOLDER_NAME + '/' + emotion
			if name.endswith('.png') is False: continue
			path = DATASET_FOLDER_NAME + '/' + emotion + '/' + name
			image = Image.open(path)
			image = ImageOps.grayscale(image)
			array = np.array(image)
			fileNames.append(array)
			flatFiles.append(array.flatten())
			targets.append(i)

	flat_data = np.array(flatFiles)
	target_array = np.array(targets)
	image_array = np.array(fileNames)

	return Bunch(data = flat_data, target = target_array, target_names = emotionFolderNames)

	
image_dataset = load_images()
X_train, X_test, y_train, y_test = train_test_split(image_dataset.data, image_dataset.target, test_size=0.9, random_state = 109)

rf = RandomForestClassifier()
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
mse_score = mean_squared_error(y_test, y_pred)
r2score = r2_score(y_test, y_pred)
print("MSE is: ", mse_score)
print("R2 score is: ", r2score)