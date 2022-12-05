import numpy as np

# DATASET
DATASET_FOLDER_NAME = "dataset_emotion" # dataset_emotion 	| 	dataset_numbers
INPUT_IMAGE_SIZE = 48					# 48				| 	22
CLASSES = ["anger", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
		# ["1", "2", "3", "4", "5", "6", "7"]

# library only supports data in this format currently
CORRECT_OUTPUT = np.array([
	[1,0,0,0,0,0,0], # anger, 1
	[0,1,0,0,0,0,0], # disgust, 2
	[0,0,1,0,0,0,0], # fear, 3
	[0,0,0,1,0,0,0], # happy, 4
	[0,0,0,0,1,0,0], # neutral, 5
	[0,0,0,0,0,1,0], # sad, 6
	[0,0,0,0,0,0,1], # surprise, 7
])

# Config
EPOCHS = 1
MAX_LEARNING_RATE = 0.1 		# Will be used as constant learning rate if LR_INVERSE_SCALING_ON is false
LR_INVERSE_SCALING_ON = False 	# Currently doesn't support multiple epochs for custom MLP. Will cycle scaling for each epoch.
TRAINING_TEST_RATIO = 0.01		# Percent of data to use for training. (1 - TRAINING_TEST_RATIO) will be used for testing.

# Hidden Layer
SIZE_HIDDEN_LAYER = 300
NUM_HIDDEN_LAYERS = 1			# Only for baselines MLPClassifier. Weight graph will only display first two layers of weights.

# Plotting
NUM_WEIGHT_PLOT_SAMPLES = 24 	# Number of weight samples to show in the graph. Only for custom MLP.