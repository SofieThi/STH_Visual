## Assignment 2 - Image classifier benchmark scripts
The code does the following:

One script should be called logistic_regression.py and should do the following:

Load either the MNIST_784 data or the CIFAR_10 data

Train a Logistic Regression model using scikit-learn

Print the classification report to the terminal and save the classification report to out/lr_report.txt

Another scripts should be called nn_classifier.py and should do the following:

Load either the MNIST_784 data or the CIFAR_10 data

Train a Neural Network model using the premade module in neuralnetwork.py

Print output to the terminal during training showing epochs and loss

Print the classification report to the terminal and save the classification report to out/nn_report.txt


## Structure
This repository has the following directory structure:

data: a folder to be used for inputting the data you wish to run

notebooks: Jupyter notebooks in both .ipynb and .html format

src: the .py script version of the assignments: nn_classifier.py and logistic_regression.py

output: the results of inputting the CIFAR_10 dataset
