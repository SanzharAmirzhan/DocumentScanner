from operator import itemgetter
from math import sqrt, acos, asin, pi
import numpy as np
import cv2
from PIL import Image

image = cv2.imread('image.jpg')

def getContour(image):
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(gray, (3, 3), 0)
	kernel = np.ones((3,3),np.uint8)
	gray = cv2.erode(gray,kernel,iterations = 1)
	gray = cv2.dilate(gray,kernel,iterations = 1)
	edged = cv2.Canny(gray, 75.0, 200.0)
	_, cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
	cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]

	for c in cnts:
		peri = cv2.arcLength(c, True)
		approx = cv2.approxPolyDP(c, 0.02 * peri, True)

		if len(approx) == 4:
			screenCnt = approx
			return (screenCnt, True)
	return ([], False)

def getDist(a, b):
    return sqrt((a[1] - b[1])**2 + (a[0] - b[0])**2)

def resize(h):
    h = h.reshape((4,2))
    hnew = np.zeros((4,2),dtype = np.float32)
    add = h.sum(1)
    hnew[0] = h[np.argmin(add)]
    hnew[2] = h[np.argmax(add)]
    diff = np.diff(h,axis = 1)
    hnew[1] = h[np.argmin(diff)]
    hnew[3] = h[np.argmax(diff)]
    return hnew

def process(img, points, contour):
    image = img
    angle = 0
    c = round(getDist(points[3], points[1]))
    b = round(getDist(points[3], (points[1][0], points[3][1])))
    deg = acos(b/c)
    angle = (180*deg)/pi
    angle = round(angle,1)

    imageWidth = 400
    imageHeight = 400

    if(getDist(points[0], points[3]) > getDist(points[0], points[2])):
        angle -= 90
        imageHeight = imageWidth * (getDist(points[0], points[3])/getDist(points[0], points[2]))
    else:
	    imageWidth = imageHeight * (getDist(points[0], points[2])/getDist(points[0], points[3]))

    approx = resize(contour)
    pts2 = np.float32([[0,0], [imageWidth,0], [imageWidth,imageHeight], [0, imageHeight]])
    M = cv2.getPerspectiveTransform(approx, pts2)
    dst = cv2.warpPerspective(img.copy(), M, (int(imageWidth), int(imageHeight)))
    return dst.copy()

def getPoints(screenCnt):
	res = [(screenCnt[0][0][0], screenCnt[0][0][1]), (screenCnt[1][0][0], screenCnt[1][0][1]),
	(screenCnt[2][0][0], screenCnt[2][0][1]), (screenCnt[3][0][0], screenCnt[3][0][1])]
	ans = [sorted(res,key=itemgetter(1))[0], sorted(res,key=itemgetter(1))[-1], sorted(res,key=itemgetter(0))[0], sorted(res,key=itemgetter(0))[-1]]
	ans = [(t[1], t[0]) for t in ans]
	return ans.copy()

contours, retCode = getContour(image.copy())
if(retCode == False):
	print("contours not found")
	exit()

points = getPoints(contours.copy())

image2 = image.copy()
cv2.drawContours(image2, [contours], -1, (0, 255, 0), 1)

imageRes = process(image.copy(), points, contours).copy()

imageForPdf = cv2.cvtColor(imageRes.copy(), cv2.COLOR_BGR2GRAY)
imageForPdf = cv2.adaptiveThreshold(imageForPdf,255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,4)

cv2.imshow('processed', imageRes)
cv2.imshow('original', image)
cv2.imshow('forPdf', imageForPdf)
cv2.imshow('contour', image2)
cv2.waitKey(0)
cv2.destroyAllWindows()


cv2.imwrite("image2pdfTemp.jpg", imageForPdf)
im = Image.open("image2pdfTemp.jpg")
im.save("scannedDoc.pdf","PDF", Quality = 100)
