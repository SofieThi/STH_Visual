## Assignment 3 - Transfer learning + CNN classification
The code does the following:

-Load the CIFAR10 dataset

-Use VGG16 to perform feature extraction

-Train a classifier

-Save plots of the loss and accuracy

-Save the classification report



## Structure

This repository has the following directory structure:

| Column | Description|
|--------|:-----------|
```data```| a folder to be used for inputting the data you wish to run
```notebooks``` | Jupyter notebooks in both .ipynb and .html format
```src``` | the .py script version of the assignment
```output``` | the results of inputting the CIFAR_10 dataset. The plots are saved as plots_assign_3.png and the classification report as classification_assign_3.txt

## Contribution

Sofie Thinggaard au613703

201909063@post.au.dk

## Methods

This problem relates to transfer learning and CNN classification. In order to address this problem, load the cifar10 data, normalize, binarize the labels, load and initilize the VGG16 model (pretrained on large amounts of data), disable training of convolutional layers (if it updates it defeats the purpose, meaning no transfer learning), add new classification layers (which correspond to our dataset: label_names), and add everything together. Now, we train the model over 10 epochs. This also gives us knowledge of loss and accuracy. To make our plot of the loss and accuracy we use the premade function: plot_history, and save to output folder. Finally, we generate a classification report and save to output folder.


## Usage (reproducing results)

In order to run this code, clone the repository and store the data in the data folder. You will need the packages in the requirements.txt document.

To replicate the results chose the cifar10 dataset. from tensorflow.keras.datasets import cifar10

## Discussion of results

Results: getting a plot of the accuracy and loss of the model (plots_assign_3.png) and a classification report (classification_assign_3.txt). Our model has an accuracy of 51% but the training curves show that training and test loss could potentially continue to decrease, hence learn more, with more epochs, meaning getting a higher accuracy. 

## Link to assignment 3 on Github

https://github.com/SofieThi/STH_Visual/tree/main/assignment_3
