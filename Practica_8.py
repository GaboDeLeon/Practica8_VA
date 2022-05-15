import cv2
import numpy as np

img = cv2.imread('Rojo-mochila.jpeg')
img2 = cv2.resize(img, dsize=(480,680),interpolation=cv2.INTER_CUBIC)
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

#Laplaciano
lap = cv2.Laplacian(img2,cv2.CV_64F)
lap = np.uint8(np.absolute(lap))

#Sobel 'X' y 'Y'
sobelX = cv2.Sobel(img2,cv2.CV_64F,1,0)
sobelX = np.uint8(np.absolute(sobelX))

sobelY = cv2.Sobel(img2,cv2.CV_64F,0,1)
sobelY = np.uint8(np.absolute(sobelY))

#Canny
canny = cv2.Canny(img2,30,150)

cv2.imshow('Original',img2)
cv2.imshow('Laplaciano',lap)
cv2.imshow('SobelX',sobelX)
cv2.imshow('SobelY',sobelY)
cv2.imshow('Canny',canny)

cv2.waitKey()


