######Imports######
from sklearn.neural_network import MLPClassifier
import numpy as np
import matplotlib.pyplot as plt
import os
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
	]
	
fileNames = []
flatFiles = []
targets = []

def loadImage(self, emotionFolderName, fileName):
		path = DATASET_FOLDER_NAME + '/' + emotionFolderName + '/' + fileName
		
		image = Image.open(path)
		image = ImageOps.grayscale(image)
		array = np.array(image).flatten()
		array = np.reshape(array, (len(array), 1))
		normalized = (array / 255) * 2 - 1
		return normalized
	
image_dataset = load_images()
X_train, X_test, y_train, y_test = train_test_split(image_dataset.data, image_dataset.target, test_size=0.1)

nn = MLPClassifier(hidden_layer_sizes=(100,), activation='logistic')
nn.fit(X_train, y_train)

y_pred = rf.predict(X_test)
mse_score = mean_squared_error(y_test, y_pred)
r2score = r2_score(y_test, y_pred)
print("MSE is: ", mse_score)
print("R2 score is: ", r2score)