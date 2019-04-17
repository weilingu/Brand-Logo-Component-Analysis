# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 15:04:15 2019

@author: emily.gu
"""

'''
This function finds and stores all contours of an image:
'''
import cv2
import operator
import pandas as pd

def img2contour(img):  # the image has to be gray-scale
    ''' 
    given an image: 
        1) convert image to gray scale, identify whether it has black or white background 
        2) improve the contrast of the image for better contour detection
        3) identify the threshold at which background becomes image content
    Note:
    - some images have an outer edge: make sure the outer edge is convert to the background
    - Contour detection operates on binary image data. Need to identify whether the image has black/white blackground:
        1) obtain the pixcel value that has the highest density. The background color value should be somewhere around this pixcel value 
        2) if this pixcel value >=128, then the image has a white background. Otherwise it has a black background

    - if the image has white background, to find the threshold:
        1) locate to the black pixel area
        2) find the peak pixel within the black area
        3) The threshold is the (highest peak pixel value+50) -- we can change the value 50 to make the estimation more accurate
            we can also make it the mid point between the two peaks on two sides of the histogram
    
    - if the image has black background, to find the threshold:
        1) locate to the white pixel area
        2) find the peak pixel within the white area
        3) The threshold is the (highest peak pixel value-50) -- we can change the value 50 to make the estimation more accurate
             we can also make it the mid point between the two peaks on two sides of the histogram
    '''
    # find the pixel value with highest frequency, use that to identify whether the background is black or white:
    hist = cv2.calcHist([img],[0],None,[256],[0,256])
    background_color, value = max(enumerate(hist), key=operator.itemgetter(1))
    
    # some image have boders at the outer edge: remove the boarders by assigning the background value to outer edge
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img[0,j]=background_color
            img[i,0]=background_color
            img[img.shape[0]-1,j]=background_color
            img[i,img.shape[1]-1]=background_color
        
    if background_color>=128:  # white background
        # to increase the contrast of an image, if  background is too bright, try alpha<1
        img = cv2.convertScaleAbs(img, alpha=0.8, beta=0)
        
        # use the updated image to calculate threshold
        hist = cv2.calcHist([img],[0],None,[256],[0,256])
        background_color_update, value = max(enumerate(hist), key=operator.itemgetter(1))
        # after identifying the background, look for the threshold: [(peak pixel) + (peak pixel on the opposite side)]/2
        # use this information for contour detection
        pixel_df=pd.DataFrame(hist)
        # obtain the most frenquent pixel on the "other side" to calculate threshold later
        sliced_pixel=pixel_df.loc[0:129,:]
        pixel_thresh_temp=sliced_pixel.idxmax(axis=0)[0]
        pixel_thresh=(pixel_thresh_temp+background_color_update)/2
        
        ret, thresh = cv2.threshold(img, pixel_thresh, 255,cv2.THRESH_BINARY_INV) # contour detection for white background
        
    else: # black background
        # to increase the contrast of an image, if  foregound is too dark, try alpha>1
        img = cv2.convertScaleAbs(img, alpha=2.2, beta=0)
        hist = cv2.calcHist([img],[0],None,[256],[0,256])
        background_color_update, value = max(enumerate(hist), key=operator.itemgetter(1))
        pixel_df=pd.DataFrame(hist)
        sliced_pixel=pixel_df.loc[129:,:]
        pixel_thresh_temp=sliced_pixel.idxmax(axis=0)[0]
        pixel_thresh=(pixel_thresh_temp+background_color_update)/2
    
        ret, thresh = cv2.threshold(img, pixel_thresh, 255,cv2.THRESH_BINARY) # contour detection for black background
    
    # fill the holes in the image
    # thresh = ndi.binary_fill_holes(thresh)
    # use RETR_EXTERNAL to retrieve only the most external contour
    contours, hierarchy=cv2.findContours(thresh, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    return contours, thresh,ret,hierarchy