# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 18:04:15 2017

@author: student
"""

import cv2
from matplotlib import pyplot as plt
 
# constants
IMAGE_SIZE = 500.0
MATCH_THRESHOLD = 20
MATCH_UPPER = 100

cv2.ocl.setUseOpenCL(False)

# load haar cascade and street image
stop_cascade = cv2.CascadeClassifier('cascade_haar.xml')
street = cv2.imread('Data/FullIJCNN2013/00115.ppm') #74 #140 #115
 
# do roundabout detection on street image
gray = cv2.cvtColor(street,cv2.COLOR_RGB2GRAY)
roadsigns = stop_cascade.detectMultiScale(
    gray, 
    scaleFactor=1.1, 
    minNeighbors=30
    )
 
# initialize ORB and BFMatcher
orb = cv2.ORB_create()
bf = cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck=True)
brisk = cv2.BRISK_create()
#freak = cv2.xfeatures2d.FREAK_create() 
 
# find the keypoints and descriptors for roadsign image
stop = cv2.imread('stop.jpeg',0)
kp_r,des_r = orb.detectAndCompute(stop,None)
kp_b,des_b = brisk.detectAndCompute(stop,None)

# loop through all detected objects
for (x,y,w,h) in roadsigns:
        
    #i = 2
    
    #x = roadsigns[i][0]
    #y = roadsigns[i][1]
    #w = roadsigns[i][2]
    #h = roadsigns[i][3]  
    
    # obtain object from street image
    obj = gray[y:y+h,x:x+w]
    ratio = IMAGE_SIZE / obj.shape[1]
    obj = cv2.resize(obj,(int(IMAGE_SIZE),int(obj.shape[0]*ratio)))
    
    # ORB find the keypoints and descriptors for object
    kp_o, des_o = orb.detectAndCompute(obj,None)
    #if len(kp_o) == 0 or des_o == None: continue
        
    # BRISK find the keypoints and descriptors for object
    #kp_o, des_o = brisk.detectAndCompute(obj,None)
    #if len(kp_o) == 0 or des_o == None: continue
    
    img2 = cv2.drawKeypoints(obj, kp_o, None, color=(0,255,0), flags=0)
    plt.imshow(img2), plt.show()
    
    # match descriptors
    matches = bf.match(des_o,des_r)
    
    good = []
    for m in matches:
        if m.distance < 70:
            good.append(m)

    
    print 'good: '+str(len(good))
    
    #matches = sorted(matches, key = lambda x:x.distance)
    #img3 = cv2.drawMatches(stop,kp_b,obj,kp_o,matches, None)
    #plt.imshow(img3),plt.show()    
    
    print 'matches: '+str(len(matches))
    cv2.imshow('test',obj)
    cv2.waitKey(4000)
     
    # draw object on street image, if threshold met
    if(len(matches) >= MATCH_THRESHOLD and len(matches) <= MATCH_UPPER):
        cv2.rectangle(street,(x,y),(x+w,y+h),(255,0,0),2)

print len(roadsigns)
 
# show objects on street image
cv2.imshow('test',street)
cv2.waitKey(4000)

cv2.destroyAllWindows()
cv2.waitKey(1)
cv2.waitKey(1)
cv2.waitKey(1)
cv2.waitKey(1)



img2 = cv2.drawKeypoints(stop, kp_b, None, color=(0,255,0), flags=0)
plt.imshow(img2), plt.show()