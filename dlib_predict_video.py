# USAGE 
# python dlib_predict_video.py --input video/2_0.avi --models  models/ --upsample 1 --output demo/output.mp4

# import the necessary packages
from imutils.video import VideoStream
from imutils import face_utils
import imutils
import numpy as np
import argparse
import time
import dlib
import cv2
import os

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--input", required=True,
	help="path to the input video")
ap.add_argument("-m", "--models", required=True,
	help="path to the models")
ap.add_argument("-u", "--upsample", type=int, default=0,
	help="# of upsampling times")
ap.add_argument("-o", "--output", required=True,
	help="path to output video")
args = vars(ap.parse_args())

# load the face detector (HOG-SVM)
print("[INFO] loading dlib thermal face detector...")
detector = dlib.simple_object_detector(os.path.join(args["models"], "dlib_face_detector.svm"))

# load the facial landmarks predictor
print("[INFO] loading facial landmark predictor...")
predictor = dlib.shape_predictor(os.path.join(args["models"], "dlib_landmark_predictor.dat"))

# initialize the video stream
vs = cv2.VideoCapture(args["input"])

# initialize the video writer
writer = None
(W, H) = (None, None)

while True:
	# read the next frame from the file
	(grabbed, frame) = vs.read()

	# break the loop if the frame 
	# was not grabbed
	if not grabbed:
		break
	
	# resize the frame
	frame = imutils.resize(frame, width=300)

	# copy the frame
	frame_copy = frame.copy()

	# convert the frame to grayscale
	frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)

	# detect faces in the frame
	rects = detector(frame, upsample_num_times=args["upsample"])

	# loop over the bounding boxes
	for rect in rects:
		# convert the dlib rectangle into an OpenCV bounding box and
		# draw a bounding box surrounding the face
		(x, y, w, h) = face_utils.rect_to_bb(rect)
		cv2.rectangle(frame_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)

		# predict the location of facial landmark coordinates then 
		# convert the prediction to an easily parsable NumPy array
		shape = predictor(frame, rect)
		shape = face_utils.shape_to_np(shape)

		# loop over the (x, y)-coordinates from our dlib shape
		# predictor model draw them on the image
		for (sx, sy) in shape:
			cv2.circle(frame_copy, (sx, sy), 2, (255, 0, 0), -1)

	# if the frame dimensions are empty, grab them
	if W is None or H is None:
		(H, W) = frame.shape[:2]

	# check if the video writer is None
	if writer is None:
		# initialize our video writer
		fourcc = cv2.VideoWriter_fourcc(*"XVID")
		writer = cv2.VideoWriter(args["output"], fourcc, 28,
			(frame.shape[1], frame.shape[0]), True)

	# push the frame to the writer 
	writer.write(frame_copy)

	# show the image
	cv2.imshow("Frame", frame_copy)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

# do a bit cleanup 
cv2.destroyAllWindows()
vs.release()
writer.release()