# -*- coding: utf-8 -*-
"""

@author: Viktor
"""


import cv2
from matplotlib import pyplot as plt
from skimage import io, color, exposure, transform
import numpy as np
from TrafficSignCNN import load_model

# constants
IMAGE_WIDTH = 680.0
BOX_SIZE = 48
scale_step = 5
MAX_BOX = 30

model = load_model()

#Preprocesiranje slike, 
def image_preprocess(img):
    #Normalizacija histograma
    hsv = color.rgb2hsv(img)
    hsv[:,:,2] = exposure.equalize_hist(hsv[:,:,2])
    img = color.hsv2rgb(hsv)
    
    ratio = img.shape[0]/float(img.shape[1])
    
    #Skaliranje na jedinstvenu velicinu
    img = transform.resize(img, (int(IMAGE_WIDTH*ratio), int(IMAGE_WIDTH)))

    #Prebacivanje ose slike kako bi bila u dobrom formatu za ulaz na CNN
    #img = np.rollaxis(img,-1)

    return img

#wrapper metoda za olaksavanje testiranja, vraca klasu objekta, ako nije pogodjeno -1
def match(model,img, thresh):
    p = model.predict(img.reshape(1,3,48,48))
    if(p.max()>thresh):
        return p.argmax(), p.max()
    else:
        return -1, p.max()


# load street image
full_image = image_preprocess(io.imread('Data/FullIJCNN2013/00115.ppm')) #74 #140 #115
#original_image = io.imread('Data/FullIJCNN2013/00115.ppm')

#ratio = original_image.shape[0]/float(original_image.shape[1])
#Skaliranje na jedinstvenu velicinu
#img = transform.resize(original_image, (int(IMAGE_WIDTH*ratio), int(IMAGE_WIDTH)))


current_box = BOX_SIZE
rects = []

for x in range (0, full_image.shape[1]-current_box, 10):
    for y in range(0, full_image.shape[0]-current_box, 10):
        block = full_image[y:y+current_box,x:x+current_box]
        cls, r = match(model, np.rollaxis(transform.resize(block, (48,48)),-1), 0.88)
        
        if(cls!=-1 and r < 0.92):
            print(cls, r)   
            rects.append((x,y))
            #rect = cv2.rectangle(full_image,(x,y),(x+current_box,y+current_box),(255,0,0),2)
        
 
for rect in rects:
    tmp = cv2.rectangle(full_image,rect,(rect[0]+current_box,rect[1]+current_box),(255,0,0),2)
 
# show objects on street image
io.imshow(full_image)

