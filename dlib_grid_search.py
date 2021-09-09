# import the necessary packages
from sklearn.model_selection import ParameterGrid
from imutils import face_utils
from utils import config
from imutils import paths
import multiprocessing
import numpy as np
import random
import time
import dlib
import json
import cv2
import os

def evaluate_model_acc(xmlPath, predPath):
	# compute and return the error (lower is better) of the shape
	# predictor over our testing path
	return dlib.test_shape_predictor(xmlPath, predPath)

def evaluate_model_speed(predictor, imagePaths, jsonPaths):
	# initialize the list of timings
	timings = []

	# grab paths to all images
	imagePaths = list(paths.list_files(imagePaths))
	imagePaths = sorted(imagePaths)

	jsonPaths = list(paths.list_files(jsonPaths))
	jsonPaths = sorted(jsonPaths)

	# loop over the images
	for jsonPath, imagePath in zip(jsonPaths, imagePaths):
		# load the input image and convert it to grayscale
		image = cv2.imread(imagePath)
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

		# open the json file and extract the bounding boxes
		f = open(jsonPath) 
		data = json.load(f)

		for shape in data['shapes']:
			if shape['label'] == 'face':
				[[xs, ys], [xe, ye]] = shape['points']
				(xs, ys, xe, ye) = (int(xs), int(ys), int(xe), int(ye))
				if (ye - ys) > (xe - xs):
					xs = xs - ((ye - ys) - (xe - xs)) // 2
					xe = xe + ((ye - ys) - (xe - xs)) // 2
				else:
					ys = ys - ((xe - xs) - (ye - ys)) // 2
					ye = ye + ((xe - xs) - (ye - ys)) // 2

				# create dlib rectangle
				rect = dlib.rectangle(left=xs, top=ys, right=xe, bottom=ye)

				# time how long it takes to perform shape prediction
				# using the current shape prediction model
				start = time.time()
				preds = predictor(gray, rect)
				end = time.time()

				#preds = face_utils.shape_to_np(preds)

		# update our timings list
		timings.append(end - start)

	# compute and return the average over the timings
	return np.average(timings)

# define the columns of our output CSV file
cols = [
	"tree_depth",
	"nu",
	"cascade_depth",
	"feature_pool_size",
	"num_test_splits",
	"oversampling_amount",
	"oversampling_translation_jitter",
	"inference_speed",
	"training_time",
	"training_error",
	"val_error",
	"testing_error",
	"model_size"
]

# open the CSV file for writing and then write the columns as the
# header of the CSV file
csv = open(config.LAND_CSV_PATH, "w")
csv.write("{}\n".format(",".join(cols)))

# determine the number of processes/threads to use
procs = multiprocessing.cpu_count()
procs = config.PROCS if config.PROCS > 0 else procs

# initialize the list of dlib shape predictor hyperparameters that
# we'll be tuning over
hyperparams = {
	"tree_depth": list(range(4, 16, 4)),
	"nu": [0.01, 0.1, 0.25],
	"cascade_depth": list(range(10, 30, 5)),
	"feature_pool_size": [100, 250, 500, 750, 1000],
	"num_test_splits": [20, 100, 300],
	"oversampling_amount": [1, 20, 40],
	"oversampling_translation_jitter": [0.0, 0.1, 0.25]
}

# construct the set of hyperparameter combinations and randomly
# sample them as trying to test *all* of them would be
# computationally prohibitive
combos = list(ParameterGrid(hyperparams))
random.shuffle(combos)
sampledCombos = combos[:config.MAX_TRIALS]
print("[INFO] sampling {} of {} possible combinations".format(
	len(sampledCombos), len(combos)))

# loop over our hyperparameter combinations
for (i, p) in enumerate(sampledCombos):
	# log experiment number
	print("[INFO] starting trial {}/{}...".format(i + 1,
		len(sampledCombos)))
	
	# grab the default options for dlib's shape predictor and then
	# set the values based on our current hyperparameter values
	options = dlib.shape_predictor_training_options()
	options.tree_depth = p["tree_depth"]
	options.nu = p["nu"]
	options.cascade_depth = p["cascade_depth"]
	options.feature_pool_size = p["feature_pool_size"]
	options.num_test_splits = p["num_test_splits"]
	options.oversampling_amount = p["oversampling_amount"]
	otj = p["oversampling_translation_jitter"]
	options.oversampling_translation_jitter = otj

	# tell dlib to be verbose when training and utilize our supplied
	# number of threads when training
	options.be_verbose = True
	options.num_threads = procs

	# train the model using the current set of hyperparameters
	start = time.time()
	dlib.train_shape_predictor(config.TRAIN_LAND_PATH,
		config.LAND_MODEL_PATH, options)
	trainingTime = time.time() - start

	# evaluate the model on the training, validation, and testing split
	trainingError = evaluate_model_acc(config.TRAIN_LAND_PATH,
		config.LAND_MODEL_PATH)
	valError = evaluate_model_acc(config.VAL_LAND_PATH,
		config.LAND_MODEL_PATH)
	testingError = evaluate_model_acc(config.TEST_LAND_PATH,
		config.LAND_MODEL_PATH)

	# compute an approximate inference speed using the trained shape
	# predictor
	predictor = dlib.shape_predictor(config.LAND_MODEL_PATH)
	inferenceSpeed = evaluate_model_speed(predictor,
		config.TEST_IMAGE_PATH, config.TEST_JSON_PATH)

	# determine the model size
	modelSize = os.path.getsize(config.LAND_MODEL_PATH)

	# build the row of data that will be written to our CSV file
	row = [
		p["tree_depth"],
		p["nu"],
		p["cascade_depth"],
		p["feature_pool_size"],
		p["num_test_splits"],
		p["oversampling_amount"],
		p["oversampling_translation_jitter"],
		inferenceSpeed,
		trainingTime,
		trainingError,
		valError,
		testingError,
		modelSize,
	]
	row = [str(x) for x in row]

	# write the output row to our CSV file
	csv.write("{}\n".format(",".join(row)))
	csv.flush()

	# delete the temporary shape predictor model
	if os.path.exists(config.LAND_MODEL_PATH):
		os.remove(config.LAND_MODEL_PATH)

# close the output CSV file
print("[INFO] cleaning up...")
csv.close()