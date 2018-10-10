import cv2
import numpy as numpy
import matplotlib.pyplot as plt

img = cv2.imread('logos.jpg', 0)

def changeimgdetails(img, type):
	laplacian = cv2.Laplacian(img,type)
	sobelx = cv2.Sobel(img,type,1,0,ksize=5)
	sobely = cv2.Sobel(img,type,0,1,ksize=5)

	cv2.imshow('img',img)
	cv2.waitKey(0)
	cv2.imshow('laplacian',laplacian)
	cv2.waitKey(0)
	cv2.imshow('sobelx',sobelx)
	cv2.waitKey(0)
	cv2.imshow('sobely',sobely)
	cv2.waitKey(0)	

changeimgdetails(img, cv2.CV_8U)

