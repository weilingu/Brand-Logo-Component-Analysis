# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 11:54:49 2019

@author: emily.gu
"""

import cv2
import pytesseract
from pytesseract import Output
import jellyfish
pytesseract.pytesseract.tesseract_cmd="[path to your downloaded tesseract.exe program.]"

'''
Use pytesseract to identify words/phrases in an image and their corresponding bonding boxes

Caution: 
- pytesseract tries to read every feature as words, including graphic patterns
- to make sure only words/phrases are captured, compare the captured features with company name
  (use string distance "jaro_winkler distance" )
- only words/phrases that fully/partially matches company name are retained 
'''
def brand_name_dect(company_symbol,company,path):
    
    img=cv2.imread(path+company_symbol+".jpg",0)
    image_data = pytesseract.image_to_data(img,output_type=Output.DICT)
    
    phras_recog=[]
    left=[]
    top=[]
    width=[]
    height=[]
    
    phras_recog_clean=[]
    left_clean=[]
    top_clean=[]
    width_clean=[]
    height_clean=[]
    text_idx=0
    
    company_name=list([x.lower() for x in company[company.Symbol==company_symbol].Name.tolist()[0].split()])
    
    while text_idx<len(image_data['level']):
        if not any(s.isalpha() for s in image_data['text'][text_idx]):
            text_idx+=1
        else:
            phras_recog+=[image_data['text'][text_idx]]
            left+=[image_data['left'][text_idx]]
            top+=[image_data['top'][text_idx]]
            width+=[image_data['width'][text_idx]]
            height+=[image_data['height'][text_idx]]
    
            '''
            - This part collects mis-recognized words/phrases 
            - adjust the threshold if exclusing too much useful information
            - try other string comparison methods. eg: damerau_levenshtein_distance
            '''
            word=image_data['text'][text_idx].lower()
            if ( word in company_name or 
                max([jellyfish.jaro_winkler(word,x) for x in company_name])>0.8):
                phras_recog_clean+=[image_data['text'][text_idx]]
                left_clean+=[image_data['left'][text_idx]]
                top_clean+=[image_data['top'][text_idx]]
                width_clean+=[image_data['width'][text_idx]]
                height_clean+=[image_data['height'][text_idx]]
               
            text_idx+=1
    
    
    '''
    - This part wipes out the recognized words/phrases from the image 
    - The output is an image with areas where words/phrases are correctly recognized 
      are turned into background
    '''
    if len(phras_recog_clean)==0:
        print (company_symbol+" Does hnot have brand name recognized on logo")
        phras_recog_clean=""
        if len(phras_recog)==0:
            phras_recog=""
        return img, company_symbol, company_name, phras_recog_clean, phras_recog
    
    word_index=0
    background_color=img[1,1]
    img_out=img
    while word_index<len(phras_recog_clean):
        # choose the color of the first cell to be the background color
        for i in range(left_clean[word_index],left_clean[word_index]+width_clean[word_index]):
            for j in range(top_clean[word_index],top_clean[word_index]+height_clean[word_index]):
                img_out[j,i]=background_color
        word_index+=1
    
    #plt.imshow(img_out)
    #return company_symbol, company_name, phras_recog_clean, phras_recog 
    return img_out, company_symbol, company_name, phras_recog_clean, phras_recog 
