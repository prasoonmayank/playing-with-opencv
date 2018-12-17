import cv2
import numpy as np
import matplotlib.pyplot as plt

filename = 'logos/mozilla.jpg'
img = cv2.imread(filename)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def harrisdetector(gray, img):
	dst = cv2.cornerHarris(gray,2,3,0.04)

	# results dilated for making corners
	dst = cv2.dilate(dst, None)

	# Threshold for the optimal value which may vary depending upon the image
	img[dst>0.01*dst.max()]=[0,0,255]

	showimg(img, 'des')

def showimg(img, title):
	cv2.imshow(title,img)
	if cv2.waitKey(0) & 0xff == 27:
		cv2.destroyAllWindows()

def shitomasidetector(gray, img):
	corners = cv2.goodFeaturesToTrack(gray, 25, 0.01, 10)
	corners = np.int0(corners)

	for i in corners:
		x,y = i.ravel()
		cv2.circle(img,(x,y),10,255,-1)

	showimg(img, 'fimg')

def siftdetector(gray, img):
	sift = cv2.SIFT()
	kp = sift.detect(gray, None)

	img = cv2.drawKeypoints(gray, kp)

	cv2.imwrite('sift_keypoints.jpg',img)

def fastdetection(img):
	fast = cv2.FastFeatureDetector()

	kp = fast.detect(img,None)
	img2 = cv2.drawKeypoints(img, kp, color=(255,0,0))

	print "Threshold: ", fast.getInt('threshold')
	print "nonmaxSuppression: ", fast.getBool('nonmaxSuppression')
	print "neighborhood: ", fast.getInt('type')
	print "Total keypoints", len(kp)

	cv2.imwrite('fast_true.png',img2)

	fast.setBool('nonmaxSuppression',0)
	kp = fast.detect(img, None)

	print "Total keypoints without suppression", len(kp)
	img3 = cv2.drawKeypoints(img, kp, color=(255,0,0))

	cv2.imwrite('fast_false.png',img3)

def orbdetector(img):
	orb = cv2.ORB_create()

	kp = orb.detect(img, None)

	kp, des = orb.compute(img, kp)

	img2 = cv2.drawKeypoints(img, kp, None, color=(0,255,0), flags=0)
	showimg(img2, 'img')

orbdetector(img)



