# Logo Feature Component Analysis

## Overview
There are hundreds of competitors selling substitue goods and services in almost every industry. However, only a few companies stay in the game in the long run. In most cases, the companies that do not make it to the top either operate at low profit margins or go out of business. It's a common phenomenon that the operating profit of the top one company is more than the rest combined. A lot of economic and business theories have been developed to analyze factors that lead to business success. 

This project aims to understand the sucess of a business from a different perspective. Given the similarities of products in the market, would it be possible that the logo design could significantly affect consumer behavior? If so, what features in a logo are the most attractive to consumers?  

The code developed in this project allows you to extract features from a 2D image. Although the primary goal is to analyze brand logo images, this code can easily be adapted to analyze other kinds of 2D images. 

Aftering running the code on provided sample dataset, the output dataset consists of the following information:

  1.	whether a logo contains the brand name
  2.	whether a logo contains letters other than the brand name
  3.	what shape(s) a logo contains 
  4.  other business information such as stock prices, industries, and etc. 



## Code Implementation

After downloading the code, edit the directories as indicated in `master.py` and run `master.py`. 

* `CNN_Letter_Recog_Model.py` Convolutional neural network model for alphabet recognition.
*	`Brand_Name_Detection.py` Detecting presence of brand name in a logo using Python-tesseract.
*	`Brand_Feature_Segregation.py` Segregating a logo into its shape component(s).
*	`Brand_Shape_Contour_Detection.py` Finding all the contour(s) of the shape component(s) of a logo.
*	`master.py` Call the above functions, detect shapes, create final dataset.

Trained CNN letter recognition model and weights. Around 94.36% accuracy on the emnist test data:
* `CNN_letter_model.json` Convolutional neural network architecture
* `letter_recog_model.h5` CNN weights


## Data:

* `data_sample` *Contributors: Colin Williams, Emily Gu*
*  To download `emnist` dataset for CNN model training, follow the installation guide on [python-mnist](https://pypi.org/project/python-mnist/) 

## Image Processing Procedure

- Step 1

   Use `tesseract` to identify words/phrases in the logo. If the identified words/phrases fully/partially match the brand name, they are cropped out of the image

- Step 2

   Identify whether the processed image contains alphabetic letters, the image is segregated into componets and each component is evaluated by a `convolutional neural network`. If a component is identified as an alphabet, it is cropped out of the image. 

- Step 3

   To identify whether the remaining image contains certain shape(s), find all the contours of the image and use `Hu invariants` to identify shape(s) of the contour(s)
   
### Example: American Airlines

* **Original Logo** 

<img src="https://user-images.githubusercontent.com/48388315/56430917-fdd8be00-6295-11e9-9c36-8f6b9514d347.jpg" align="center"  height="400" width="500">

* **Step 1: Crop out identified brand name**

<img src="https://user-images.githubusercontent.com/48388315/56432695-58751880-629c-11e9-9eeb-48d18c6b40ea.jpg" align="center"  height="400" width="500">

* **Step 2: Identify Alphabetic Letters**

1. Image feature segregation

<img src="https://user-images.githubusercontent.com/48388315/56434909-95450d80-62a4-11e9-80ae-66dbb64342bd.png" align="center"  height="200" width="400">

2. The 2nd to the 4th segments are identified as "O", "A", and "A", and are cropped out of the image

<img src="https://user-images.githubusercontent.com/48388315/56435242-ce31b200-62a5-11e9-9374-41b709ecdba5.jpg" align="center"  height="300" width="300">


* **Step 3: Identify contours for shape comparison**

<img src="https://user-images.githubusercontent.com/48388315/56435967-4a2cf980-62a8-11e9-9233-e91b4f076627.jpg" align="center"  height="300" width="300">



