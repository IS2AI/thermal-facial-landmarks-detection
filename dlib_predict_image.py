# USAGE
# python dlib_predict_image.py --images dataset/gray/test/images/ --models  models/ --upsample 1

# import the necessary packages
from imutils import face_utils
from imutils import paths
import numpy as np
import imutils
import argparse 
import imutils
import time
import dlib
import cv2
import os 

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", required=True,
	help="path to the images")
ap.add_argument("-m", "--models", required=True,
	help="path to the models")
ap.add_argument("-u", "--upsample", type=int, default=0,
	help="# of upsampling times")
args = vars(ap.parse_args())

# load the face detector (HOG-SVM)
print("[INFO] loading dlib thermal face detector...")
detector = dlib.simple_object_detector(os.path.join(args["models"], "dlib_face_detector.svm"))

# load the facial landmarks predictor
print("[INFO] loading facial landmark predictor...")
predictor = dlib.shape_predictor(os.path.join(args["models"], "dlib_landmark_predictor.dat"))

# grab paths to the images
imagePaths = list(paths.list_files(args["images"]))

# loop over the images
for ind, imagePath in enumerate(imagePaths, 1):
	print("[INFO] Processing image: {}/{}".format(ind, len(imagePaths)))
	# load the image
	image = cv2.imread(imagePath)

	# resize the image
	image = imutils.resize(image, width=300)

	# copy the image
	image_copy = image.copy()

	# convert the image to grayscale
	image = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
	
	# detect faces in the image 
	rects = detector(image, upsample_num_times=args["upsample"])	

	for rect in rects:
		# convert the dlib rectangle into an OpenCV bounding box and
		# draw a bounding box surrounding the face
		(x, y, w, h) = face_utils.rect_to_bb(rect)
		cv2.rectangle(image_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)

		# predict the location of facial landmark coordinates, 
		# then convert the prediction to an easily parsable NumPy array
		shape = predictor(image, rect)
		shape = face_utils.shape_to_np(shape)

		# loop over the (x, y)-coordinates from our dlib shape
		# predictor model draw them on the image
		for (sx, sy) in shape:
			cv2.circle(image_copy, (sx, sy), 2, (0, 0, 255), -1)

	# show the image
	cv2.imshow("Image", image_copy)
	key = cv2.waitKey(0) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

