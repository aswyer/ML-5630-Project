# DATASET
DATASET_FOLDER_NAME = "dataset_emotion" # dataset_emotion 	| 	dataset_numbers
INPUT_IMAGE_LENGTH = 48					# 48				| 	28

DATASET_DEBUG_SIZE_MULTIPLE = 2/5 # Only use portion of entire data set for debuging. Use 1 for entire dataset.
TRAINING_TEST_RATIO = 0.8		# Percent of data to use for training. (1 - TRAINING_TEST_RATIO) will be used for testing.

CLASSES = ["anger", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
# CLASSES = ["1", "2", "3", "4", "5", "6", "7"]

# Only for Custom MLP. Library only supports data in this format currently
CORRECT_OUTPUT = [
	[1,0,0,0,0,0,0], # anger, 1
	[0,1,0,0,0,0,0], # disgust, 2
	[0,0,1,0,0,0,0], # fear, 3
	[0,0,0,1,0,0,0], # happy, 4
	[0,0,0,0,1,0,0], # neutral, 5
	[0,0,0,0,0,1,0], # sad, 6
	[0,0,0,0,0,0,1], # surprise, 7
]

# CONFIG
EPOCHS = 1						# Higher epochs -> better the accuracy
MAX_LEARNING_RATE = 0.01 		# Will be used as constant learning rate if LR_INVERSE_SCALING_ON is false
LR_INVERSE_SCALING_ON = False 	# Currently doesn't support multiple epochs for custom MLP. Will cycle scaling for each epoch.

# HIDDEN LAYER
SIZE_HIDDEN_LAYER = 300

# PLOTTING
SHOULD_PLOT_WEIGHTS = True
NUM_WEIGHT_PLOT_SAMPLES = 24 	# Number of weight samples to show in the graph. Only for custom MLP.











# HELPER VARIABLES. Don't Edit.
# TODO: Better place for this?
INPUT_IMAGE_FLAT_LENGTH = pow(INPUT_IMAGE_LENGTH, 2)