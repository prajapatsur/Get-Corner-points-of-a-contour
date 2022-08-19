import cv2
import numpy as np

img = cv2.imread('cards.webp')

img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
img_blur = cv2.GaussianBlur(img_gray,(3,3),0)
img_canny = cv2.Canny(img_blur,100,200)
img_copy = img.copy()

#detecting contours
contours,heirarchy = cv2.findContours(img_canny,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

#function to get corner points of any contour
def getCornerPoints(contour):
    peri = cv2.arcLength(contour,True)
    approx = cv2.approxPolyDP(contour, 0.02*peri,True)
    return approx

#drawing contours
cv2.drawContours(img,contours,-1,(0,255,0),3,cv2.LINE_AA)

Corner_points = getCornerPoints(contours[0])
print(Corner_points)

cv2.imshow('Contours',img)

cv2.waitKey(0)
cv2.destroyAllWindows()


