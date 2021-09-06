# USAGE
# python train_dlib_predictor.py --training dataset/gray/train/dlib_landmarks_train.xml 
# --validation dataset/gray/val/dlib_landmarks_val.xml

# import the necessary packages
import multiprocessing
import argparse
import dlib

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-t", "--training", required=True,
	help="path to input training XML file")
ap.add_argument("-v", "--validation", required=True,
    help="path to input validation XML file")
args = vars(ap.parse_args())

# grab the default options for dlib's shape predictor
print("[INFO] setting shape predictor options...")
options = dlib.shape_predictor_training_options()

# define the depth of each regression tree -- there will be a total
# of 2^tree_depth leaves in each tree; small values of tree_depth
# will be *faster* but *less accurate* while larger values will
# generate trees that are *deeper*, *more accurate*, but will run
# *far slower* when making predictions
options.tree_depth = 4

# regularization parameter in the range [0, 1] that is used to help
# our model generalize -- values closer to 1 will make our model fit
# the training data better, but could cause overfitting; values closer
# to 0 will help our model generalize but will require us to have
# training data in the order of 1000s of data points
options.nu = 0.01

# the number of cascades used to train the shape predictor -- this
# parameter has a *dramtic* impact on both the *accuracy* and *output
# size* of your model; the more cascades you have, the more accurate
# your model can potentially be, but also the *larger* the output size
options.cascade_depth = 20

# number of pixels used to generate features for the random trees at
# each cascade -- larger pixel values will make your shape predictor
# more accurate, but slower; use large values if speed is not a
# problem, otherwise smaller values for resource constrained/embedded
# devices
options.feature_pool_size = 750

# selects best features at each cascade when training -- the larger
# this value is, the *longer* it will take to train but (potentially)
# the more *accurate* your model will be
options.num_test_splits = 100

# controls amount of "jitter" (i.e., data augmentation) when training
# the shape predictor -- applies the supplied number of random
# deformations, thereby performing regularization and increasing the
# ability of our model to generalize
options.oversampling_amount = 20

# amount of translation jitter to apply -- the dlib docs recommend
# values in the range [0, 0.5]
options.oversampling_translation_jitter = 0.1

# tell the dlib shape predictor to be verbose and print out status
# messages our model trains
options.be_verbose = True

# number of threads/CPU cores to be used when training -- we default
# this value to the number of available cores on the system, but you
# can supply an integer value here if you would like
options.num_threads = multiprocessing.cpu_count()

# log our training options to the terminal
print("[INFO] shape predictor options:")
print(options)

# train the shape predictor
print("[INFO] training shape predictor...")
dlib.train_shape_predictor(args["training"], 
	"models/dlib_landmark_predictor.dat", options)

# take the newly trained shape predictor model and evaluate it on
# both our training and testing set 
trainingError = dlib.test_shape_predictor(args["training"], 
	"models/dlib_landmark_predictor.dat")
testingError = dlib.test_shape_predictor(args["validation"], 
	"models/dlib_landmark_predictor.dat")

# display the training and testing errors for the current trial
print("[INFO] train error: {}".format(trainingError))
print("[INFO] test error: {}".format(testingError))

