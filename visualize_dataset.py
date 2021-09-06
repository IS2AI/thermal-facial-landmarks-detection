# import the necessary packages
from imutils import paths 
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

# grab paths to the json files
jsonPaths = list(paths.list_files(os.path.join(args["dataset"], args["color"], 
	args["set"], "json"), validExts="json"))
jsonPaths = sorted(jsonPaths)

# grab paths to the images
imagePaths = list(paths.list_files(os.path.join(args["dataset"], args["color"], 
	args["set"], "images")))
imagePaths = sorted(imagePaths)

count = 1;
# loop over the files
for imagePath, jsonPath in zip(imagePaths, jsonPaths):
	print("[INFO] Processing {}/{} files ({}/{})".format(imagePath.split("/")[-1], jsonPath.split("/")[-1], count, len(jsonPaths)))
	count += 1;

	# open the json file 
	f = open(jsonPath,) 
  
	# return the json object 
	# as a dictionary 
	data = json.load(f) 

	# load the image
	image = cv2.imread(imagePath)
  
	# loop over the shapes
	for shape in data['shapes']:
		if shape['label'] == 'face':
			# extract coordinates of the bounding box
			# and convert to int
			[[xs, ys], [xe, ye]] = shape['points']
			(xs, ys, xe, ye) = (int(xs), int(ys), int(xe), int(ye))

			# draw the bounding box
			cv2.rectangle(image, (xs, ys), (xe, ye), (0, 255, 0), 2)
		else:
			# loop overt the coordinates of landmarks
			for (x, y) in shape['points']:
				# convert to int type
				(x, y) = (int(x), int(y))

				# draw the point
				cv2.circle(image, (x, y), 2, (0, 0, 0), -1)
	
	# close the json file 
	f.close()

	# show the image
	cv2.imshow("Image", image)
	key = cv2.waitKey(0) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break
  		