## Assignment 2 - Image classifier benchmark scripts
The code does the following:

--> One script should be called logistic_regression.py and should do the following:

Load either the MNIST_784 data or the CIFAR_10 data

Train a Logistic Regression model using scikit-learn

Print the classification report to the terminal and save the classification report to out/lr_report.txt

--> Another script should be called nn_classifier.py and should do the following:

Load either the MNIST_784 data or the CIFAR_10 data

Train a Neural Network model using the premade module in neuralnetwork.py

Print output to the terminal during training showing epochs and loss

Print the classification report to the terminal and save the classification report to out/nn_report.txt



## Structure

This repository has the following directory structure:

| Column | Description|
|--------|:-----------|
```data```| a folder to be used for inputting the data you wish to run
```notebooks``` | Jupyter notebooks in both .ipynb and .html format
```src``` | the .py script version of the assignments: nn_classifier.py and logistic_regression.py
```output``` | the results of inputting the CIFAR_10 dataset: nn_report.txt for the nn_classifier.py script and lr_report.txt for the logistic_regression.py script
```utils``` | premade script with premade functions for the neural network model and classifier

## Contribution

Sofie Thinggaard au613703

201909063@post.au.dk

## Methods

This problem relates to training a Logistic Regression model using scikit-learn and a Neural Network model using the premade module in neuralnetwork.py. 

In the logistic_regression.py script we need to load the data and then split it into training and testing. Then, convert all the data to greyscale, normalize the values with min/max normalization, reshape the data. Now, we train a Logistic Regression model using scikit-learn, which will enable us to get predictions and make a classification report.

In the nn_classifier.py script we need to do the same: load the data and then split it into training and testing, convert all the data to greyscale, normalize the values with min/max normalization and finally reshape the data. Now, instead, we use a Neural network classifier. We binarize the labels (0 and 1s) and fit the data to dimensions of the labels. We do this both with the test and train data. Now, we can train the network over 10 epochs. Like with the lr script, this will enable us to get predictions and make a classification report.

## Usage (reproducing results)

In order to run this code, clone the repository and store the data in the data folder. You will need the packages in the requirements.txt document and the premade functions in the utils folder.

To replicate the results load the dataset: cifar10. From tensorflow.keras.datasets import cifar10.

## Discussion of results

Results: getting two classification reports, one for each model. The resulting classification report shows that the Logistic Regression model has an accuracy of 32%. By comparison, the Neural Network model resulted in a score of 38% on accuracy. However, the loss information the epochs shows reveals that the loss function gets lower and lower, which means the model is learning every full pass over the dataset. It should be rapid fall then slower and slower. Adding more epochs could show whether the loss continues to decrease (more learning = higher accuracy).
