# import the necessary packages
import os

# define the path to the training, validation, and testing landmarks XML files
TRAIN_LAND_PATH = "dataset/gray/train/dlib_landmarks_train.xml"
VAL_LAND_PATH = "dataset/gray/val/dlib_landmarks_val.xml"
TEST_LAND_PATH = "dataset/gray/test/dlib_landmarks_test.xml"

# define the path to the temporary model files
LAND_MODEL_PATH = "models/temp_landmark_predictor.dat"

# define the path to the output CSV file containing the results of
# our experiments
LAND_CSV_PATH = "results/dlib_landmark_trials.csv"

# define the path to the test set that we'll be using to evaluate
# inference speed using the shape predictor
TEST_IMAGE_PATH = "dataset/gray/test/images"
TEST_JSON_PATH = "dataset/gray/test/json"

# define the number of threads/cores we'll be using when trianing our
# shape predictor models
PROCS = -1

# define the maximum number of trials we'll be performing when tuning
# our shape predictor hyperparameters
MAX_TRIALS = 100