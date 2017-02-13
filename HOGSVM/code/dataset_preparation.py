# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 00:54:09 2017

@author: Viktor
"""

import shutil, os, random

IMAGES_PATH = r'E:\FTN\7. semestar\SC\Data\SVM\Positive_Images\Images'
DESTINATION_PATH = r'E:\FTN\7. semestar\SC\Data\SVM\Positive_Images'
COUNT = 200

# copies single file
def copy_single_file(src, dst):
    shutil.copyfile(src, dst)
    
# copies files from folder
def read_folder_hog(folder_name):
    files = os.listdir(IMAGES_PATH+'\\'+folder_name)
    
    for i in range(0, COUNT):
        tmp = random.choice(files)
        copy_single_file(IMAGES_PATH+'\\'+folder_name+'\\'+tmp, DESTINATION_PATH+'\\'+folder_name+'_'+tmp)
    
# reads all folders
def read_folders(folder_path):
    
    dirs = os.listdir(IMAGES_PATH)
    
    for d in dirs:
        read_folder_hog(d)


    