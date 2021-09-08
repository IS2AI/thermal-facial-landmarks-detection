# USAGE
# python unet_predict_image.py -d dataset/gray/test -p  models/unet_model.h5 -v 1

# import the necessary packages
from keras.models import load_model
from imutils import face_utils
from imutils import paths
import numpy as np
import argparse 
import imutils
import time
import dlib
import json
import cv2
import os 

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to the dataset")
ap.add_argument("-p", "--predictor", required=True,
	help="path to the landmark predictor")
ap.add_argument("-v", "--vis", type=int, default=0,
	help="visualize: 0/1")
args = vars(ap.parse_args())

# load our trained shape predictor
print("[INFO] loading facial landmark predictor...")
model = load_model(os.path.join(args["predictor"], "model.h5"))

# paths to the json files
jsonPaths = list(paths.list_files(os.path.join(args["dataset"], "json")))
jsonPaths = sorted(jsonPaths)

# paths to the images
imagePaths = list(paths.list_files(os.path.join(args["dataset"], "images")))
imagePaths = sorted(imagePaths)

prediction_time = []

counter = 1
# loop over the json files
for jsonPath, imagePath in zip(jsonPaths, imagePaths):
	print("[INFO] Processing file: {}/{}".format(counter, len(jsonPaths)))
	counter += 1

	# Opening the json file 
	f = open(jsonPath) 
  
	# returns the json object 
	# as a dictionary 
	data = json.load(f)

	# load the image
	image = cv2.imread(imagePath)
	image = imutils.resize(image, width=464)

	print(image.shape)
	image_batch = np.expand_dims(image, axis = 0)

	# ground truth landmarks coordinates
	ground_truth = []
	
	# Iterating through the shapes
	# in the json file
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

			# draw ground truth bbox
			cv2.rectangle(image, (xs, ys), (xe, ye), (0, 255, 0), 2)

			# time how long the prediction
			# takes
			t1 = time.time()
			predicted = model.predict(image_batch)
			t2 = time.time()
			prediction_time.append(t2-t1)
			print("[INFO] Prediction time: {:.3f} sec".format(t2 - t1))
			print("------------------------")
		else:
			# draw the ground truth landmarks
			for (x, y) in shape['points']:
				(x, y) = (int(x), int(y))
				ground_truth.append([x, y])
				cv2.circle(image, (x, y), 2, (0, 255, 0), -1)
			
	f.close()

	if args["vis"]:
		# predictions are in normalized format
		# revert back to original
		print(predicted.shape)

		# loop over the (x, y)-coordinates from our dlib shape
		# predictor model draw them on the image
		for (px, py) in predicted:
			cv2.circle(image, (px, py), 2, (255, 0, 0), -1)
		
		# show the image
		cv2.imshow("Frame", image)
		key = cv2.waitKey(0) & 0xFF

		# if the `q` key was pressed, break from the loop
		if key == ord("q"):
			break

print("[INFO] Mean prediction time: {:.3f}".format(sum(prediction_time) / len(prediction_time)))