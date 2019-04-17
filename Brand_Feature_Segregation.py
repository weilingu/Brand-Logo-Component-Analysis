# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 13:51:02 2019

@author: emily.gu
"""

'''
Thses functions aim to segregate the features of an image after the brand name is cropped out
'''
import cv2
import numpy as np
from skimage.filters import sobel
from scipy import ndimage as ndi
import operator
from skimage.morphology import dilation, erosion, disk

'''
This script aims to segregate the features of an image after the brand name is cropped out
'''
'''  
To seperate image into serevel parts, use elevation map:  
Elevation map could transform the image into "higher dimentions"
The backgroung of images will have low pixel values
The foregrand will have high pixel values
Use the difference in values to seperate image into parts, assign a value to each part
'''
def imag_seperation(img):
    
    elevation_map = sobel(img)
    ele_transformed=np.zeros_like(elevation_map)
    ele_transformed[elevation_map<=0.2]=0*1
    ele_transformed[elevation_map>0.2]=1*1
    ele_transformed = ndi.binary_fill_holes(ele_transformed)  # This function will fill the image. Include this can reduce the number of un-necessary segregations. Inlcuding this might make some images less discernation and reduce similarity score
    label,_ = ndi.label(ele_transformed)
    
    return label

'''
resize the images for image comparision
'''
def img_resize(img,size): # input size is in tuple format. 
    
    img = img[~(img==0).all(1)]
    img = img.transpose()
    img = img[~(img==0).all(1)]
    img = (img.transpose())*1
    # append a few rows and columns of zero for rescalling
    img = np.hstack((img,np.zeros((img.shape[0],2))))
    img = np.hstack((np.zeros((img.shape[0],2)),img))
    img = np.vstack((img,np.zeros((2,img.shape[1]))))
    img = np.vstack((np.zeros((2,img.shape[1])),img))
    # rescale the image 
    img=cv2.resize(img,size)
    
    return img

'''
This function is used to remove noise such that images with touching edges can also be segregated
- find the pixel value with highest frequency, use this value to identify whether the inmage has black or white background
- convert the image to binary to prepare for erosion and dilation. 
'''

def noise_remove(img):
    hist = cv2.calcHist([img],[0],None,[256],[0,256])
    background_color, value = max(enumerate(hist), key=operator.itemgetter(1))
    if background_color>=128:  # white background    
        # if white background, use "dilation" to dilate the image and then segregate 
        img=dilation(img,selem=disk(4))
    else: # black background  
        # if black background, use "erosion" to erode the image and then segregate 
        img=erosion(img,selem=disk(4))   
    
    return img

'''
# This code allows one to see each segregated shape from an image for checking purpose. 
img=cv2.imread([img_path],0)
plt.imshow(img)

img_cleaned=noise_remove(img)
image_labels=imag_seperation(img_cleaned)

i=1
size=np.unique(image_labels).shape[0]-1 # do not take the backgound (0) into account
fig, ax = plt.subplots(figsize=(5, 50))
while i<=size:
    partial_image=image_labels==i
    ax = fig.add_subplot(size, 1, i)
    plt.imshow(img_resize(partial_image, (28,28)))
    plt.axis("off")
    i+=1
plt.show()  
'''