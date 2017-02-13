# -*- coding: utf-8 -*-
"""

Tool for converting existing GTSDB annotations to JSON accepted by TensorBox,
in accordance to brainwash dataset example JSON file.

[
    {
        "image_path": $path (string),
        "rects": [
            {
                "x1":$x1 (float),
                "x2":$x2 (float),
                "y1":$y1 (float), 
                "y2":$y2 (float)
            },...
        ]
    },...
]

@author: viktor
"""

JSON_FILE = '/home/student/Desktop/projekat/Data/FullIJCNN2013/train_boxes.json'
GTSDB_FILE = '/home/student/Desktop/projekat/Data/FullIJCNN2013/gt.txt'

GTSDB_FILE_RESIZED = '/home/student/Desktop/projekat/Data/FullIJCNN2013/gt_resized.txt'


IMG_TEST = '/home/student/Desktop/tensorbox/data/brainwash/gtsdb/00000.png'

import matplotlib.pyplot as plt
import matplotlib.image as img

def parse_gtsdb_to_json(gtsdb_path, json_out):
    gtsdb = open(gtsdb_path, 'r')
    json = open(json_out, 'w')
    
    lines_in = []
    
    for line in gtsdb:
        lines_in.append(line)
    
    json.writelines('[\n')
    current_image = ''
    
    for line in lines_in:
        parts = line.split()
        parts = parts[0].split(';')
        
        if(current_image==parts[0]):
            json.writelines('\t\t,{\n')
            json.writelines('\t\t\t"x1":'+str(parts[1])+',\n')
            json.writelines('\t\t\t"x2":'+str(parts[3])+',\n')
            json.writelines('\t\t\t"y1":'+str(parts[2])+',\n')
            json.writelines('\t\t\t"y2":'+str(parts[4])+'\n')            
            json.writelines('\t\t}\n')
        else:
            if(current_image!=''):
                json.writelines('\t\t]\n')
                json.writelines('\t},\n')
                
            current_image=parts[0]
            json.writelines('\t{\n')
            json.writelines('\t\t"image_path": "gtsdb_train/'+parts[0].replace('.ppm','.png')+'",\n')
            json.writelines('\t\t"rects": [\n')
            
            json.writelines('\t\t{\n')
            json.writelines('\t\t\t"x1":'+str(parts[1])+',\n')
            json.writelines('\t\t\t"x2":'+str(parts[3])+',\n')
            json.writelines('\t\t\t"y1":'+str(parts[2])+',\n')
            json.writelines('\t\t\t"y2":'+str(parts[4])+'\n')            
            json.writelines('\t\t}\n')

    json.writelines('\t\t]\n\t}\n')
    json.writelines(']')
    

def resize_boxes(gtsdb_path, gtsdb_path_new, current_size, new_size):
    gtsdb = open(gtsdb_path, 'r')
    gtsdb_new = open(gtsdb_path_new, 'w')    
    
    for line in gtsdb:
        parts = line.split()
        parts = parts[0].split(';')
        
        x1 = parts[1]
        x2 = parts[3]
        y1 = parts[2]
        y2 = parts[4]
        
        x1_n = int(float(x1)/current_size[0]*float(new_size[0]))
        x2_n = int(float(x2)/current_size[0]*float(new_size[0]))
        y1_n = int(float(y1)/current_size[1]*float(new_size[1]))
        y2_n = int(float(y2)/current_size[1]*float(new_size[1]))
        
        gtsdb_new.writelines(parts[0]+';'+str(x1_n)+';'+str(y1_n)+';'+str(x2_n)+';'+str(y2_n)+';'+parts[5]+'\n')
        

# Testing the results on images
import skimage.transform as transform
import cv2 
img = cv2.imread(IMG_TEST)

#img = transform.resize(img, (480,640))

cv2.rectangle(img,(364,246),(383,267),(0,255,0),3)
cv2.imshow('hey',img)