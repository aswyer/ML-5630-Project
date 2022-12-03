DATASET_FOLDER_NAME = "dataset_emotion" # dataset_emotion 	| 	dataset_numbers
INPUT_IMAGE_SIZE = 48					# 48				| 	22

EPOCHS = 1
MAX_LEARNING_RATE = 1 			# Will be used as constant learning rate if LR_INVERSE_SCALING_ON is false
LR_INVERSE_SCALING_ON = True 	# Currently doesn't support multiple epochs. will cycle scaling for each epoch.

SIZE_HIDDEN_LAYER = 300
NUM_WEIGHT_SAMPLES = 24 		# Number of weight samples to show in the graph

TRAINING_TEST_RATIO = 0.9 		# Percent of data to use for training. (1 - TRAINING_TEST_RATIO) will be used for testing.