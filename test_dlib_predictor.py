# USAGE
# python test_dlib_predictor.py --testing dataset/gray/test/dlib_landmarks_test.xml 
# --model models/dlib_landmarks_predictor.dat

import argparse
import dlib
import os

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-t", "--testing", required=True,
    help="path to input test XML file")
ap.add_argument("-m", "--model", required=True,
    help="path to the face detection model")
args = vars(ap.parse_args())

print("Testing error: {}".format(
    dlib.test_shape_predictor(args["testing"], args["model"])))


