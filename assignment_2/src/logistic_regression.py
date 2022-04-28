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
from sklearn.model_selection import train_test_split #splitting into test and train
from sklearn.linear_model import LogisticRegression #classifier
from sklearn.metrics import accuracy_score #calculating accuracy
from sklearn.metrics import classification_report #report

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


def main():
    #Loading CIFAR_10
    #Printing to terminal
    print("CIFAR_10 data is being loaded - Please Wait...")

    (X_train, y_train), (X_test, y_test) = cifar10.load_data() #Splitting data up into train and test data. 

    print("CIFAR_10 data has been loaded successfully")

    #Convert all the data to greyscale
    X_train_grey = np.array([cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in X_train])
    X_test_grey = np.array([cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in X_test])
    
    # scaling
    X_train_scaled = (X_train_grey - X_train_grey.min())/(X_train_grey.max() - X_train_grey.min())
    X_test_scaled = (X_test_grey - X_test_grey.min())/(X_test_grey.max() - X_test_grey.min())

    #Reshaping the data
    nsamples, nx, ny = X_train_scaled.shape
    X_train_dataset = X_train_scaled.reshape((nsamples,nx*ny))

    nsamples, nx, ny = X_test_scaled.shape
    X_test_dataset = X_test_scaled.reshape((nsamples,nx*ny))

    ##Simple logistic regression classifier
    #Train a Logistic Regression model using scikit-learn
    clf = LogisticRegression(penalty='none', 
                             tol=0.1, 
                             solver='saga',
                             multi_class='multinomial').fit(X_train_dataset, y_train)

    ##Get predictions and make classification report
    y_pred = clf.predict(X_test_dataset)

    #labels from the CIFAR homepage
    labels = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

    report = classification_report(y_test, y_pred, target_names = labels) #getting airplane etc that we defined with labels
    print(report)

    f = open("../../cds-visual/Assignments/output/lr_report.txt",'w') #saving in this folder as lr_report.txt
    print(report, file=f)

    print("Done! Report has been generated and saved in the output folder as lr_report.txt")

if __name__=="__main__":
    main()