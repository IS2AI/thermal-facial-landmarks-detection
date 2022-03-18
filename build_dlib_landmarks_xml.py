# USAGE
# python build_dlib_landmarks_xml.py --dataset dataset/ --color gray --set val --vis 1

# import the necessary packages
from xml.etree.ElementTree import Element, SubElement, Comment
from xml.dom import minidom
from imutils import paths 
import xml.etree.ElementTree as ET
import numpy as np 
import argparse
import json 
import cv2
import os 


def build_xml_tree(root, path, bbox, points, names):
	image = SubElement(root, 'image', file=path)
	box = SubElement(image, 'box', top=str(bbox[0]), left=str(bbox[1]), width=str(bbox[2]), height=str(bbox[3]))
	part = [Element('part', name=name,  x=str(point[0]), y=str(point[1])) for name, point in zip(names, points)]
	box.extend(part)


def correct_landmarks_order(pts_mirr):
	pts_mirr_c = []
	# chin
	pts_mirr_c.append(pts_mirr[16])
	pts_mirr_c.append(pts_mirr[15])
	pts_mirr_c.append(pts_mirr[14])
	pts_mirr_c.append(pts_mirr[13])
	pts_mirr_c.append(pts_mirr[12])
	pts_mirr_c.append(pts_mirr[11])
	pts_mirr_c.append(pts_mirr[10])
	pts_mirr_c.append(pts_mirr[9])
	pts_mirr_c.append(pts_mirr[8])
	pts_mirr_c.append(pts_mirr[7])
	pts_mirr_c.append(pts_mirr[6])
	pts_mirr_c.append(pts_mirr[5])
	pts_mirr_c.append(pts_mirr[4])
	pts_mirr_c.append(pts_mirr[3])
	pts_mirr_c.append(pts_mirr[2])
	pts_mirr_c.append(pts_mirr[1])
	pts_mirr_c.append(pts_mirr[0])

	# left eyebrow
	pts_mirr_c.append(pts_mirr[26])
	pts_mirr_c.append(pts_mirr[25])
	pts_mirr_c.append(pts_mirr[24])
	pts_mirr_c.append(pts_mirr[23])
	pts_mirr_c.append(pts_mirr[22])

	# right eyebrow
	pts_mirr_c.append(pts_mirr[21])
	pts_mirr_c.append(pts_mirr[20])
	pts_mirr_c.append(pts_mirr[19])
	pts_mirr_c.append(pts_mirr[18])
	pts_mirr_c.append(pts_mirr[17])

	# nose bridge
	pts_mirr_c.append(pts_mirr[27])
	pts_mirr_c.append(pts_mirr[28])
	pts_mirr_c.append(pts_mirr[29])
	pts_mirr_c.append(pts_mirr[30])

	# nose tip
	pts_mirr_c.append(pts_mirr[35])
	pts_mirr_c.append(pts_mirr[34])
	pts_mirr_c.append(pts_mirr[33])
	pts_mirr_c.append(pts_mirr[32])
	pts_mirr_c.append(pts_mirr[31])

	# left eye
	pts_mirr_c.append(pts_mirr[45])
	pts_mirr_c.append(pts_mirr[44])
	pts_mirr_c.append(pts_mirr[43])
	pts_mirr_c.append(pts_mirr[42])
	pts_mirr_c.append(pts_mirr[47])
	pts_mirr_c.append(pts_mirr[46])

	# right eye
	pts_mirr_c.append(pts_mirr[39])
	pts_mirr_c.append(pts_mirr[38])
	pts_mirr_c.append(pts_mirr[37])
	pts_mirr_c.append(pts_mirr[36])
	pts_mirr_c.append(pts_mirr[41])
	pts_mirr_c.append(pts_mirr[40])

	# lips 
	pts_mirr_c.append(pts_mirr[50])
	pts_mirr_c.append(pts_mirr[49])
	pts_mirr_c.append(pts_mirr[48])
	pts_mirr_c.append(pts_mirr[51])
	pts_mirr_c.append(pts_mirr[52])
	pts_mirr_c.append(pts_mirr[53])

	return pts_mirr_c


# construct argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to the dataset")
ap.add_argument("-c", "--color", type=str, default="iron",
	help="color palette of dataset: gray/iron")
ap.add_argument("-s", "--set", type=str, default="train",
	help="train/val/test")
ap.add_argument("-v", "--vis", type=int, default=0,
	help="visualize: 0/1")
args = vars(ap.parse_args())

# grab paths to the json files
jsonPaths = list(paths.list_files(os.path.join(args["dataset"], args["color"], 
	args["set"], "json"), validExts="json"))
jsonPaths = sorted(jsonPaths)

# grab paths to the images
imagePaths = list(paths.list_files(os.path.join(args["dataset"], args["color"], 
	args["set"], "images")))
imagePaths = sorted(imagePaths)

# initialize landmarks ids
landmarks_id = ["00", "01", "02", "03", "04", "05", "06", "07", "08", "09", "10", 
		"11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", 
		"22", "23", "24", "25", "26", "27", "28", "29", "30", "31", "32", 
		"33", "34", "35", "36", "37", "38", "39", "40", "41", "42", "43", 
		"44", "45", "46", "47", "48", "49", "50", "51", "52", "53"]

# define the root of the xml structure
root = Element('images')

count = 0
# loop over the files
for jsonPath, imagePath in zip(jsonPaths, imagePaths):
	# extract the files' names
	imageName = imagePath.split("/")[-1]
	jsonName = jsonPath.split("/")[-1]

	count += 1
	print("[INFO] Processing {}/{} files ({}/{})".format(jsonName, imageName, count, len(jsonPaths)))

	assert imageName.split(".")[0] == jsonName.split(".")[0]

	# load the image
	image = cv2.imread(imagePath)
	
	# extract height and width of the image
	(h, w) = image.shape[:2]
		
	# mirror the image
	image_m = cv2.flip(image, flipCode=1)
  
	# lists to store points for xml files
	points = []
	boxes = []

	points_m = []
	boxes_m = []

	# open the json file 
	f = open(jsonPath) 
  
	# return the json object as a dictionary 
	data = json.load(f)

	# iterate through the shapes
	# in the json file
	for shape in data['shapes']:
		if shape['label'] == 'face':
			# extract coordinates of the bouding box
			[[xs, ys], [xe, ye]] = shape['points']
			# convert to int type
			(xs, ys, xe, ye) = (int(xs), int(ys), int(xe), int(ye))

			# make square bounding box
			if (ye - ys) > (xe - xs):
				xs = xs - ((ye - ys) - (xe - xs)) // 2
				xe = xe + ((ye - ys) - (xe - xs)) // 2
			else:
				ys = ys - ((xe - xs) - (ye - ys)) // 2
				ye = ye + ((xe - xs) - (ye - ys)) // 2
			
			# mirror the coordinates
			xe_m = w - xs
			xs_m = w - xe

			# add to the lists
			boxes.append(ys)
			boxes.append(xs)
			boxes.append(xe-xs)
			boxes.append(ye-ys)

			boxes_m.append(ys)
			boxes_m.append(xs_m)
			boxes_m.append(xe_m-xs_m)
			boxes_m.append(ye-ys)

		else:
			# loop over the coordinates of landmarks
			for (x, y) in shape['points']:
				(x, y) = (int(x), int(y))
				points.append([x, y])

				# mirror the coordinates
				xm = w - x
				points_m.append([xm, y])

	# correct the order of the mirrored landmarks 			
	points_m = correct_landmarks_order(points_m)
		
	# write the xml files
	build_xml_tree(root, os.path.join("images", imageName), boxes, points, landmarks_id)
	build_xml_tree(root, os.path.join("images_mirr", imageName), boxes_m, points_m, landmarks_id)

	# save the mirrored images
	#cv2.imwrite(os.path.join(args["dataset"], args["set"], "images_mirr", imageName), image_m)

	# close the file 
	f.close() 

	# visualize the dataset
	if args["vis"]:
		# draw bounding boxes
		cv2.rectangle(image, (xs, ys), (xe, ye), (0, 255, 0), 1)
		cv2.rectangle(image_m, (xs_m, ys), (xe_m, ye), (0, 255, 0), 1)

		# draw facial landmarks
		for p, p_m, l in zip(points, points_m, landmarks_id):
			cv2.circle(image, (p[0], p[1]), 2, (255, 0, 0), -1)
			#cv2.putText(image, l, (p[0], p[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)

			cv2.circle(image_m, (p_m[0], p_m[1]), 2, (255, 0, 0), -1)
			#cv2.putText(image_m, l, (p_m[0], p_m[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)

		#cv2.imwrite("images/{}".format(imageName), np.hstack([image, image_m]))

		# show the images
		cv2.imshow("Images", np.hstack([image, image_m]))
		key = cv2.waitKey(0) & 0xFF

		# if the `q` key was pressed, break from the loop
		if key == ord("q"):
			break
  		
xmlstr = minidom.parseString(ET.tostring(root)).toprettyxml(indent="  ", encoding='UTF-8')
with open(os.path.sep.join([args["dataset"], args["color"], args["set"], "dlib_landmarks_{}.xml".format(args["set"])]), "w") as f:
    f.write(str(xmlstr.decode('UTF-8')))
    f.close()
