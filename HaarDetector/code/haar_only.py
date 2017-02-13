# -*- coding: utf-8 -*-
"""
Created on Thu Feb 09 18:23:18 2017

@author: Viktor
"""


import cv2
from matplotlib import pyplot as plt
 
# constants
IMAGE_SIZE = 500.0
MATCH_THRESHOLD = 20
MATCH_UPPER = 100

cv2.ocl.setUseOpenCL(False)

# load haar cascade and street image
stop_cascade = cv2.CascadeClassifier('round_win.xml')
street = cv2.imread('Data/FullIJCNN2013/00612.ppm') #74 #140 #115
 
# do roundabout detection on street image
gray = cv2.cvtColor(street,cv2.COLOR_RGB2GRAY)
roundabouts = stop_cascade.detectMultiScale(
    gray, 
    scaleFactor=1.1, 
    minNeighbors=30
    )

# loop through all detected objects
for (x,y,w,h) in roundabouts:
    cv2.rectangle(street,(x,y),(x+w,y+h),(255,0,0),2)


# show objects on street image
cv2.imshow('test',street)
cv2.waitKey(4000)

cv2.destroyAllWindows()
cv2.waitKey(1)
cv2.waitKey(1)
cv2.waitKey(1)
cv2.waitKey(1)