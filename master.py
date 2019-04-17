# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 11:57:30 2019

@author: emily.gu
"""

'''
This script combines all the functions together to create a brand logo feature dataset
'''
import Brand_Name_Detection
import Brand_Feature_Segregation
import Brand_Shape_Contour_Detection
import keras
import glob
import pandas as pd
import cv2
import numpy as np
from keras.models import model_from_json

# Create global file path
path='C:\\Users\\emily.gu\\Desktop\\Image Process\\'
# create path to the image file
img_path='P:\\User\\Colin.Williams\\CS 229\\images\\*'

# create path to company information file. The files should contain at least information on 
# the company name
company_info_path="P:\\User\\Colin.Williams\\CS 229\\companylist.csv"

# create path to processed images (images whose recognized brand names or letters are cropped out) and base shapes
brand_name_processed_img="C:\\Users\\emily.gu\\Desktop\\Image Process\\int\\brand_name_dect\\"
brand_letter_processed_img="C:\\Users\\emily.gu\\Desktop\\Image Process\\int\\brand_letter_dect\\"
base_shape="C:\\Users\\emily.gu\\Desktop\\Image Process\\int\\"
'''
import data and process images, categorized processed images as:
1. success_convert: brand name is fully/partially recognized on the image
2. no_recog_brand_name:  no brand name is recognized on the image
3. failed_convert: images failed to be processed, might due to the format of the image
'''
company_symbol_all=glob.glob(img_path)
path_start,path_end = img_path[:-1] ,'.'
company_symbol=[]
for i in company_symbol_all:
    company_symbol+=[(i.split(path_start))[1].split(path_end)[0]]

company=pd.read_csv(company_info_path )

success_convert={}
no_recog_brand_name={}
failed_convert=[]
for c in company_symbol:
    print(c)
    try:
        output=Brand_Name_Detection.brand_name_dect(c,company)
        if output[3]=="":
            no_recog_brand_name[output[1]]=[output[2],output[3],output[4]]
        else:
            success_convert[output[1]]=[output[2],output[3],output[4]]
        file_name=brand_name_processed_img+output[1]+".jpg"
        cv2.imwrite(file_name, output[0])
        
    except:
        print (c+" failed to convert the image")
        failed_convert+=[c]

# convert the results to dataframe
success_convert_df=pd.DataFrame.from_dict(success_convert,orient='index',columns=['company_name', 'phras_recog_clean','phras_recog']) 
success_convert_df['company']=success_convert_df.index
success_convert_df["Brand_Name_Exist"]=1

no_recog_brand_name_df=pd.DataFrame.from_dict(no_recog_brand_name,orient='index',columns=['company_name', 'phras_recog_clean','phras_recog']) 
no_recog_brand_name_df['company']=no_recog_brand_name_df.index
no_recog_brand_name_df["Brand_Name_Exist"]=0
'''
use the processed image (without brand names) to segregate remaining shapes of the logo
'''        
company_symbol=list(success_convert.keys())+list(no_recog_brand_name.keys()) 
    
#create a dataset to collect the segerated image
shape_w_o_brand_name={}

for company in company_symbol:
    print(company)
 
    img=cv2.imread(brand_name_processed_img+company+".jpg",0)
    
    img_cleaned=Brand_Feature_Segregation.noise_remove(img)
    
    image_labels=Brand_Feature_Segregation.imag_seperation(img_cleaned)    
    

    shape_collect=np.empty([28,28]) # because of this initialization of empty array, each shape collection dataset has an shape with zeros for all entries. Drop it at the end of the loop
    i=1
    size=np.unique(image_labels).shape[0]-1 # do not take the backgound (0) into account
    while i<=size:
        partial_image=image_labels==i
        shape_collect=np.concatenate((shape_collect,Brand_Feature_Segregation.img_resize(partial_image, (28,28))))
        i+=1
    # reshape the recognized shapes to match inputs to the trained CNN model
    shape_collect=shape_collect.reshape(-1,28,28,1)
    shape_collect=shape_collect[1:]
    
    shape_w_o_brand_name[company]={}
    #convert the type of pixcel entries to match the image used to train CNN
    shape_w_o_brand_name[company]['seg_shape']=shape_collect.astype('float32') 
    shape_w_o_brand_name[company]['img_seg_label']=image_labels


'''
identify letters on a logo after brand name is cropped out
uses:
    1. trained CNN letter recognition model
    2. segregated image shapes
'''    
# load the trained CNN model:
with open(path+'CNN_letter_model.json', 'r') as f:
    model = model_from_json(f.read())
model.load_weights(path+"letter_recog_model.h5")
model.compile(loss=keras.losses.categorical_crossentropy,optimizer=keras.optimizers.Adam(),metrics=['accuracy'])

# collect the letters recognized

letter_recog={}

for company in shape_w_o_brand_name.keys():
    print(company)
    if len(shape_w_o_brand_name[company]['seg_shape'])==0:
        letter_recog[company]=[]
        # if no shape is recognized, save the original image
        company_letter_cleaned=cv2.imread(brand_name_processed_img+company+".jpg",0)

    else:
        predicted_classes = model.predict(shape_w_o_brand_name[company]['seg_shape'])
        predicted_label = np.argmax(predicted_classes,axis=1)
        
        tol=0.9 # set the toleration level
        i=0
        max_prob=np.max(predicted_classes,axis=1)
        while i<len(max_prob):
            if max_prob[i]<tol:
                predicted_label[i]=0
            i+=1
        letter_recog[company]=[i for i in predicted_label if i!=0]
        
        '''
        crop out the letters that are recognized from the image as letters
        '''
        # since the image segmentation label starts with background as 0: add 0 at the begining of the predicted label
        predicted_label=[1]+predicted_label.tolist()
        # create a mask matrix, where the background and all the shapes that are recognized as letters are replaced with "0" on the image. 
        # all the shapes that are not recognized as letters are replaced with "1" on the image. 
        recignized_seg_idx=[i for i in range(len(predicted_label)) if predicted_label[i] !=0]
        mask_letter_reduce=np.isin(shape_w_o_brand_name[company]['img_seg_label'],recignized_seg_idx, invert=True)*1
        # reduce the recognized letters to background
        company_name_cropped_logo=cv2.imread(brand_name_processed_img+company+".jpg",0)
        company_letter_cleaned=company_name_cropped_logo*mask_letter_reduce
        
        # increase the contrast to make the remaining foreground standout more:
        mask_background=mask_letter_reduce*255
        company_letter_cleaned=np.minimum(mask_background+company_letter_cleaned,225)
    
    file_name=brand_letter_processed_img+company+".jpg"
    cv2.imwrite(file_name, company_letter_cleaned)

'''
identify remaining shapes in the logo after cropping out brand name and letters
'''
# load shape pictures
shape={}
animal_shapes=glob.glob(base_shape+'animal shapes\\*')
geographical_shapes=glob.glob(base_shape+'geographical shapes\\*')
geometric_shapes=glob.glob(base_shape+'geometric shapes\\*')
shapes_all=animal_shapes+geographical_shapes+geometric_shapes
for img in shapes_all:
    path_start, path_mid,path_end = base_shape ,'\\','.'
    img_name=((img.split(path_start))[1].split(path_mid)[1]).split(path_end)[0]
    shape[img_name]=cv2.imread(img,0)
        
# convert all the shapes to contours, collect the information into a shape dictionary
shape_countour={}
shape_thresh={}
for data_img in shape.keys():
    shape_countour[data_img]=Brand_Shape_Contour_Detection.img2contour(shape[data_img])[0]
    shape_thresh[data_img]=Brand_Shape_Contour_Detection.img2contour(shape[data_img])[1]
    # need to make sure all the base shape has contour 1: 
    if len(shape_countour[data_img])>1:
        print(data_img+" has "+str(len(shape_countour[data_img]))+" segments of contours: Invalid")   
        

# compare each countour with the countour of the base shapes
# could try differnet contour-matching methods
shape_data=dict((key,0) for key in shape_countour.keys())
shape_data=pd.DataFrame(shape_data,index=[0])

ind=1
for company in company_symbol:
    print(company)
    img=cv2.imread(brand_letter_processed_img+company+".jpg",0)
    img_contours=Brand_Shape_Contour_Detection.img2contour(img)[0]
    
    img_shape_match=dict((key,0) for key in shape_countour.keys())
    img_shape_match['company']=company
    for img_cont in img_contours:
        shape_match_score={}
        for shape_cont in shape_countour.keys():
            match_score=cv2.matchShapes(img_cont, shape_countour[shape_cont][0], cv2.CONTOURS_MATCH_I1,0)
            shape_match_score[shape_cont]=match_score
        # in order to avoid the program assigning an arbitrary shape, only recognize the shape if similarity score <=1.5
        if min(shape_match_score.values())<=1.5:
            shape_matched=min(shape_match_score,key=shape_match_score.get)
            img_shape_match[shape_matched]+=1
            
    img_shape_match=pd.DataFrame(img_shape_match,index=[ind])
    ind+=1
    shape_data=shape_data.append(img_shape_match)
shape_data=shape_data.drop([0])

'''
Combine the dataset
'''
# read the company info dataset
company_info=pd.read_csv(company_info_path )
company_info=company_info.rename(index=str, columns={"Symbol":"company"})

# clean up letter_recog dataset:
df={'company': "None"}
for i in range(1,27):
    col=str(i)
    df[col]=0

letter_recog_df = pd.DataFrame(data=df,index=[0])

for company_name in letter_recog.keys():
    df['company']=company_name
    for letter_idx in letter_recog[company_name]:
        idx=str(letter_idx)
        df[idx]+=1
    df_pd=pd.DataFrame(data=df,index=[0])
    
    letter_recog_df=letter_recog_df.append(df_pd)
    # initialize the datafrme 
    for i in range(1,27):
        col=str(i)
        df[col]=0
'''
rename the columns based on the lookup :

'''
letter_lu = { "1": 'a', "2": 'b', "3": 'c', "4": 'd', "5": 'e', "6": 'f', "7": 'g', "8": 'h', "9": 'i', "10": 'j',
                    "11": 'k', "12": 'l', "13": 'm', "14": 'n', "15": 'o', "16": 'p', "17": 'q', "18": 'r', "19": 's', "20": 't',
                    "21": 'u', "22": 'v', "23": 'w', "24": 'x', "25": 'y', "26": 'z'}   
letter_recog_df=letter_recog_df.rename(index=str, columns=letter_lu)

df=success_convert_df.append(no_recog_brand_name_df)
df.index=range(len(df))

df=df.merge(shape_data, how='inner', on="company")
df=df.merge(letter_recog_df, how='inner', on="company")
df=df.merge(company_info, how='inner', on="company")

df.to_csv(path+"combined data.csv")