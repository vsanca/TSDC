# -*- coding: utf-8 -*-
"""

@author: Viktor
"""

import cv2
import numpy as np

WIN_SIZE = (48,48)
BLOCK_SIZE = (4,4) #(4,4), (8,8)
BLOCK_STRIDE = (4,4)
CELL_SIZE = (4,4) #(4,4), (8,8)
NBINS = 9
DERIV_APERTURE = 1
WIN_SIGMA = -1


SVM_PATH = 'E:\\FTN\\7. Semestar\\SC\\Data\\SVM\\Classifiers\\'
STOP = 'stop.vec'
ROUND = 'round.vec'
TRIANGLE_UP = 'triangle_up.vec'
TRIANGLE_DOWN = 'triangle_down.vec'
SQUARE_ROT = 'square_rot.vec'

IMG_PATH = 'E:\\FTN\\7. semestar\\SC\\Data\\FullIJCNN2013\\'
IMG_FILE = '00789.png' #74 115 140 177 193 198 200 202 809 827
FILES = ['00074.png','00115.png','00140.png','00177.png','00193.png','00198.png','00200.png','00202.png','00340.png','00789.png','00809.png','00827.png']


HIT_THRESHOLD = 0.6
SCALE = 1.02
WIN_STRIDE = (4,4)
PADDING = (10,10)

def read_svm_descriptor(path):

    f = open(path, 'r')
    line = f.readline()

    l = []

    for el in line.split():
        l.append(float(el))
        
    return l
    

def hog_detector_setup(SVM_FILE):
    hog = cv2.HOGDescriptor(WIN_SIZE, BLOCK_SIZE, BLOCK_STRIDE, 
                            CELL_SIZE, NBINS, DERIV_APERTURE, WIN_SIGMA)
    
    svm = np.array(read_svm_descriptor(SVM_PATH+SVM_FILE))
    
    hog.setSVMDetector(svm)
    
    return hog
    
def detect_and_mark(img, hog, color, text):
    (boxes, weights) = hog.detectMultiScale(img, HIT_THRESHOLD, WIN_STRIDE, PADDING, SCALE)
    
    for (x, y, w, h) in boxes:
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, text, (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color,2)
    
        
def test_image(image_path, stp, rnd, tru, trd, sqr):

    img = cv2.imread(image_path)
    
    detect_and_mark(img, stp, (0,0,255), "stop")
    #detect_and_mark(img, rnd, (255,0,0), "round")
    #detect_and_mark(img, tru, (0,255,0), "triangle_up")
    #detect_and_mark(img, trd, (0,255,255), "triangle_down")
    #detect_and_mark(img, sqr, (255,255,0), "square")
        
    cv2.imshow('Image',img)
    
    
stp = hog_detector_setup(STOP)
rnd = hog_detector_setup(ROUND)
tru = hog_detector_setup(TRIANGLE_UP) 
trd = hog_detector_setup(TRIANGLE_DOWN)
sqr = hog_detector_setup(SQUARE_ROT)

test_image(IMG_PATH+FILES[0], stp, rnd, tru, trd, sqr)