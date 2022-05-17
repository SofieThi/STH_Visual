## Self-assigned project: How well can a model predict which sport is happening in the images?

This repository contains the answers to the following tasks:

-Make the https://www.kaggle.com/code/victorxiao4/sports-tf/data smaller (20 classes instead of 100) by only including the first 20 folders (alphabethical order) in the train, test and valid folders

-Train a sequential model from TensorFlow to predict which of the 20 classes the image belongs to

-Find out how well the model performs

-Generate plot history and save it to output folder


## Structure

This repository has the following directory structure:

| Column | Description|
|--------|:-----------|
```data```| a folder to be used for inputting the data you wish to run
```notebooks``` | Jupyter notebooks in both .ipynb and .html format
```src``` | the .py script version of the assignment
```output``` | the results of inputting the kaggle Sport tf dataset

## Contribution

Sofie Thinggaard au613703

201909063@post.au.dk

Code from VICTOR XIAO https://www.kaggle.com/code/victorxiao4/sports-tf

## Methods

This problem relates to finding how how well a sequential model can predict which sport is happening in sports images. In order to address this problem, I set parameters to be used by the model including normalizing the image sizes and setting the number of classes. Then, load the data with Keras path finder and define training, validation and test set. In the training folder there should be 2634 files, the validation 105 files and test with 101 files - all with 20 classes. Them create the model with hidden layers, maxpooling layers, relu activation layers etc. Now, compile and train the model with 10 epochs. Now, plot history for the model's loss and accucy. Make a function to be able to plot the Training and Validation Accuracy and Training and Validation Loss. Save it to output folder.

## Usage (reproducing results)

In order to run this code, clone the repository and store the data in the data folder. You will need the packages in the requirements.txt document.

To replicate the results choose the flowers dataset from https://www.kaggle.com/code/victorxiao4/sports-tf/data and remove all but the 20 first folders in alphabethical order (keeping the classes: air hockey to cheerleading). This is also in the data folder as a zip file. The code can run with the whole dataset with 100 classes, but it will take a long time.

## Discussion of results

Results: 
