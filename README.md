# Logo-Feature-Component-Analysis

This code allows you to extract features from a 2-D image. The preferred target images are brand logos. 
The output consists of information on:
  1.	whether the logo contains the brand name.
  2.	whether the logo contains letters other than the brand name
  3.	what shape(s) the logo contains 

Although the target images are brand logos, the code can easily be used to analyse other kinds of 2-D images.

# Python Script Description

* `CNN_Letter_Recog_Model.py` Convolutional neural network model for alphabet recognition.
*	`Brand_Name_Detection.py` Detecting presence of brand logos using Python-tesseract.
*	`Brand_Feature_Segregation.py` Segregate a logo into its shape component(s).
*	`Brand_Shape_Contour_Detection.py` Finds all the contour(s) of the shape component(s) of a logo.
*	`master.py` Call the functions, detect shapes, create final dataset.

# Datasets

1.	Shape dataset: Contains shapes of interest. Can be modified.
2.	Brand Information: Data from NASDAQ, cleaned and provided by Colin Williams.
3.	Logo image: 3457 publicly traded companies. Dataset provided by Colin Williams.
