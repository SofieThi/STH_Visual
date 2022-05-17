## Self-assigned project: How well can a model predict which sport is happening in the images?

This repository contains the answers to the following tasks:

-Change the dataset: compile the split dataset from https://www.kaggle.com/code/victorxiao4/sports-tf/data into one big dataset with 5 categories: arm wrestling, barell racing, billiards, bmx and boxing: these are the classes the model has previosuly been best at predicting (with val, test, train split and 20 classes)

-Train a sequential model from TensorFlow to predict which of the 5 classes the images belongs to

-Find out how well the model performs

-Generate plot history and save it to output folder

-Generate classification report and save it to output folder


## Structure

This repository has the following directory structure:

| Column | Description|
|--------|:-----------|
```data```| a folder to be used for inputting the data you wish to run: the small dataset used for this assignment is added as a .zip file
```notebooks``` | Jupyter notebooks in both .ipynb and .html format
```src``` | the .py script version of the assignment
```output``` | the results of inputting the kaggle Sport tf dataset

## Contribution

Sofie Thinggaard au613703

201909063@post.au.dk

Code from VICTOR XIAO https://www.kaggle.com/code/victorxiao4/sports-tf

## Methods

This problem relates to finding how how well a sequential model can predict which sport is happening in sports images. In order to address this problem, I set parameters to be used by the model including normalizing the image sizes and setting the number of classes. Then, load the data with Keras path finder and define training and validation subsets using additional parameters like validation_split and seed from https://errorsfixing.com/is-it-possible-to-split-a-tensorflow-dataset-into-train-validation-and-test-datasets-when-using-image_dataset_from_directory/. Then, create the model with hidden layers, maxpooling layers, relu activation layers etc. Now, compile and train the model with 10 epochs, then plot history for the model's loss and accuracy using a function to plot the Training and Validation Accuracy and Training and Validation Loss. Save it to output folder. Lastly, generate the classification report and save it to an output folder.

## Usage (reproducing results)

The data used is a small subsample of Sports tf from https://www.kaggle.com/code/victorxiao4/sports-tf/data

In order to run this code, clone the repository and unzip the data in the data folder. You will need the packages in the requirements.txt document.

## Discussion of results

Results: getting a classification report and two plot models to see how well the model is performing on the dataset. The model's accuracy is 23% and seems best at predicting images with people playing billiards. Seen with the plots, the model does well on the training data, but is likely overfitting on the training data, seen with the poor scores on the test data. The model cannot generalize that well. A possible solution will be Transfer learning from pre-trained models.

## Link to self-assigned assignment on Github

https://github.com/SofieThi/STH_Visual/tree/main/Self-assigned%20project
