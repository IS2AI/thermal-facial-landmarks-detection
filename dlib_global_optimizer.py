# import the necessary packages
from utils import config
from collections import OrderedDict
import multiprocessing
import dlib
import sys
import os

# determine the number of processes/threads to use
procs = multiprocessing.cpu_count()
procs = config.PROCS if config.PROCS > 0 else procs

def test_shape_predictor_params(treeDepth, nu, cascadeDepth,
	featurePoolSize, numTestSplits, oversamplingAmount,
	oversamplingTransJitter, padding, lambdaParam):
	# grab the default options for dlib's shape predictor and then
	# set the values based on our current hyperparameter values,
	# casting to ints when appropriate
	options = dlib.shape_predictor_training_options()
	options.tree_depth = int(treeDepth)
	options.nu = nu
	options.cascade_depth = int(cascadeDepth)
	options.feature_pool_size = int(featurePoolSize)
	options.num_test_splits = int(numTestSplits)
	options.oversampling_amount = int(oversamplingAmount)
	options.oversampling_translation_jitter = oversamplingTransJitter
	options.feature_pool_region_padding = padding
	options.lambda_param = lambdaParam

	# tell dlib to be verbose when training and utilize our supplied
	# number of threads when training
	options.be_verbose = True
	options.num_threads = procs

	# display the current set of options to our terminal
	print("[INFO] starting training...")
	print(options)
	sys.stdout.flush()

	# train the model using the current set of hyperparameters
	dlib.train_shape_predictor(config.TRAIN_LAND_PATH,
		config.LAND_MODEL_PATH, options)

	# take the newly trained shape predictor model and evaluate it on
	# both our training and testing set
	trainingError = dlib.test_shape_predictor(config.TRAIN_LAND_PATH,
		config.LAND_MODEL_PATH)
	testingError = dlib.test_shape_predictor(config.VAL_LAND_PATH,
		config.LAND_MODEL_PATH)

	# display the training and testing errors for the current trial
	print("[INFO] train error: {}".format(trainingError))
	print("[INFO] validation error: {}".format(testingError))
	sys.stdout.flush()

	# return the error on the testing set
	return testingError

# define the hyperparameters to dlib's shape predictor that we are
# going to explore/tune where the key to the dictionary is the
# hyperparameter name and the value is a 3-tuple consisting of the
# lower range, upper range, and is/is not integer boolean,
# respectively
params = OrderedDict([
	("tree_depth", (4, 8, True)),
	("nu", (0.01, 0.05, False)),
	("cascade_depth", (15, 25, True)),
	("feature_pool_size", (500, 1000, True)),
	("num_test_splits", (50, 150, True)),
	("oversampling_amount", (15, 25, True)),
	("oversampling_translation_jitter",  (0.05, 0.15, False)),
	("feature_pool_region_padding", (-0.2, 0.2, False)),
	("lambda_param", (0.01, 0.99, False))
])

# use our ordered dictionary to easily extract the lower and upper
# boundaries of the hyperparamter range, include whether or not the
# parameter is an integer or not
lower = [v[0] for (k, v) in params.items()]
upper = [v[1] for (k, v) in params.items()]
isint = [v[2] for (k, v) in params.items()]

# utilize dlib to optimize our shape predictor hyperparameters
(bestParams, bestLoss) = dlib.find_min_global(
	test_shape_predictor_params,
	bound1=lower,
	bound2=upper,
	is_integer_variable=isint,
	num_function_calls=10)

# display the optimal hyperparameters so we can reuse them in our
# training script
print("[INFO] optimal parameters: {}".format(bestParams))
print("[INFO] optimal error: {}".format(bestLoss))

# delete the temporary model file
os.remove(config.TEMP_MODEL_PATH)