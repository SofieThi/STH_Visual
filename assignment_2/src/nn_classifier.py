#session_6
import os
import sys
sys.path.append(os.path.join("..","..","CDS-VIS"))

# Import teaching utils
import numpy as np
import utils.classifier_utils as clf_util

# Import sklearn metrics
from sklearn import metrics
from sklearn.datasets import fetch_openml #database with pre-tagged datasets, see https://www.openml.org/
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression #classifier
from sklearn.metrics import accuracy_score #calculating accuracy

#session_7
# path tools
import sys,os
sys.path.append(os.path.join("..", "..", "CDS-VIS"))

# image processing
import cv2

# neural networks with numpy
import numpy as np
from tensorflow.keras.datasets import cifar10 #getting access to complex tools
# cifar10: very famous dataset with low res images for training (0 is airplane) 
# https://keras.io/api/datasets/cifar10/ (32, 32, 3) 32x32 3 color channels
from utils.neuralnetwork import NeuralNetwork #Ross has written it see utils

# machine learning tools
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression

def main():
    #Loading cifar10 data
    print("Loading the cifar10 data")
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    print("cifar10 data has been loaded successfully")

    #Convert all the data to greyscale
    X_train_grey = np.array([cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in X_train])
    X_test_grey = np.array([cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in X_test])

    ##Normalize the values
    #min/max normalization: X1 = X(array)-Xmin (minimum value in array)/Xmax-Xmin
    # scaling
    X_train_scaled = (X_train_grey - X_train_grey.min())/(X_train_grey.max() - X_train_grey.min())
    X_test_scaled = (X_test_grey - X_test_grey.min())/(X_test_grey.max() - X_test_grey.min())

    #Reshaping the data
    nsamples, nx, ny = X_train_scaled.shape
    X_train_dataset = X_train_scaled.reshape((nsamples,nx*ny))
    nsamples, nx, ny = X_test_scaled.shape
    X_test_dataset = X_test_scaled.reshape((nsamples,nx*ny))

    #Neural network classifier
    #do something else with the labels: binarizing the labels (0 and 1s)
    y_train = LabelBinarizer().fit_transform(y_train) #fit data to dimensions of labels
    #then convert to 0 and 1s like above

    y_test = LabelBinarizer().fit_transform(y_test) #test data as well

    print("[INFO] training network...") #feedback on what the model is doing
    input_shape = X_train_dataset.shape[1]
    nn = NeuralNetwork([input_shape, 64, 10]) #neural network, hidden layer of 64 nodes, 
    #output layer of 10 (airplane etc.)
    #can add two hidden layers like input_shape, 128, 32, 10
    print(f"[INFO] {nn}")
    nn.fit(X_train_dataset, y_train, epochs=10, displayUpdate=1) #a full pass over dataset: getting update
    #after each epoch, the loss function gets lower and lower = learning every time
    #should be rapid fall then slower and slower

    predictions = nn.predict(X_test_dataset)

    labels = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"] 
    #labels from the webpage

    print("Generating report...")

    y_pred = predictions.argmax(axis=1)
    report = classification_report(y_test.argmax(axis=1), y_pred, target_names = labels)

    print(report)

    g = open("../../cds-visual/Assignments/output/nn_report.txt",'w')
    print(report, file=g)

    print("Done! Report has been generated and saved in the output folder as nn_report.txt")
    
if __name__=="__main__":
    main()