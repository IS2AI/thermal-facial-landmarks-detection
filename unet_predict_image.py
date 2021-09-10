# USAGE
# python unet_predict_image.py --dataset dataset/gray/test --model  models/ 

# import the necessary packages
from tensorflow.keras.models import load_model
from imutils import paths
import numpy as np
import argparse 
import time
import json
import cv2
import os 

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to the dataset")
ap.add_argument("-m", "--models", required=True,
	help="path to the landmark predictor")
args = vars(ap.parse_args())

# load the facial landmarks predictor
print("[INFO] loading facial landmark predictor...")
model = load_model(os.path.join(args["models"], 'unet_model.h5'))

# grab paths to the json files
jsonPaths = list(paths.list_files(os.path.join(args["dataset"], "json")))
jsonPaths = sorted(jsonPaths)

# grab paths to the images
imagePaths = list(paths.list_files(os.path.join(args["dataset"], "images")))
imagePaths = sorted(imagePaths)

count = 1
# loop over the json files and images
for jsonPath, imagePath in zip(jsonPaths, imagePaths):
	print("[INFO] Processing file: {}/{}".format(count, len(imagePaths)))
	count += 1

	# load the image
	image = cv2.imread(imagePath)

	# convert the image to grayscale
	# and make it three chanell
	image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	image = np.dstack([image] * 3)

	# open the json file
	f = open(jsonPath) 
  
	# store the json object as a dictionary 
	data = json.load(f)
	f.close()

	# iterate through the shapes in the json file
	for shape in data['shapes']:
		# extract bounding box coordinates
		if shape['label'] == 'face':
			[[xs, ys], [xe, ye]] = shape['points']
			(xs, ys, xe, ye) = (int(xs), int(ys), int(xe), int(ye))

			# draw the ground-truth bounding box
			cv2.rectangle(image, (xs, ys), (xe, ye), (0, 255, 0), 2)

			# extract the face region
			face_image = image[ys:ye, xs:xe]

			# preprocess the face image
			face_image = cv2.resize(face_image, (256, 256) )
			face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
			face_image = face_image.astype('float32')
			face_image = face_image / 255.0

			# expand dimensions to batch of one image
			face_image = np.expand_dims(face_image, axis = 0)

			# predict landmarks coordinates
			preds = model.predict(face_image)

			# predictions are in normalized format
			# revert back to original
			preds = preds.reshape(54, 2)
			preds[:,1] = preds[:,1] * (xe - xs) + xs
			preds[:,0] = preds[:,0] * (ye - ys) + ys

			# loop over the (x, y)-coordinates from 
			# predictor and draw them on the image
			for (px, py) in preds:
				cv2.circle(image, (int(py), int(px)), 2, (0, 0, 255), -1)
			
	# show the image
	cv2.imshow("Output", image)
	key = cv2.waitKey(0) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break
