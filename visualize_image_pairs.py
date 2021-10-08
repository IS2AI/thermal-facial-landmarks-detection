# USAGE
# python visualize_image_pairs.py --dataset dataset/ --color iron --set test

# import the necessary packages
from imutils import paths 
import numpy as np
import argparse
import imutils
import json 
import cv2
import os 

# construct argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to the dataset")
ap.add_argument("-c", "--color", type=str, default="iron",
	help="color palette of dataset: gray/iron")
ap.add_argument("-s", "--set", type=str, default="train",
	help="set: train/val/test")
args = vars(ap.parse_args())

# grab paths to the rgb images
rgbPaths = list(paths.list_images(os.path.join(args["dataset"], "rgb", args["set"])))
rgbPaths = sorted(rgbPaths)

# grab paths to the thermal images
thrPaths = list(paths.list_images(os.path.join(args["dataset"], args["color"], 
	args["set"], "images")))
thrPaths = sorted(thrPaths)

count = 1;
# loop over the files
for thrPath, rgbPath in zip(thrPaths, rgbPaths):
	print("[INFO] Processing {}/{} files ({}/{})".format(rgbPath.split("/")[-1], thrPath.split("/")[-1], count, len(rgbPaths)))
	count += 1;

	# load the images
	thr_image = cv2.imread(thrPath)
	rgb_image = cv2.imread(rgbPath)

	# show the image
	cv2.imshow("Image", np.hstack([thr_image, rgb_image]))
	key = cv2.waitKey(0) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break
  		