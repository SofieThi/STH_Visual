#packages
import os

# tf tools
import tensorflow as tf

# image processsing
from tensorflow.keras.preprocessing.image import (load_img,
                                                  img_to_array,
                                                  ImageDataGenerator)
# VGG16 model
from tensorflow.keras.applications.vgg16 import (preprocess_input,
                                                 decode_predictions,
                                                 VGG16)
# cifar10 data - 32x32
from tensorflow.keras.datasets import cifar10

# layers
from tensorflow.keras.layers import (Flatten, 
                                     Dense, 
                                     Dropout, 
                                     BatchNormalization)
# generic model object
from tensorflow.keras.models import Model

# optimizers
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.optimizers import SGD

#scikit-learn
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report

# for plotting
import numpy as np
import matplotlib.pyplot as plt

#Plotting function (from session_9) -needed to evaluate model later
def plot_history(H, epochs):
    plt.style.use("seaborn-colorblind")

    plt.figure(figsize=(12,6))
    plt.subplot(1,2,1)
    plt.plot(np.arange(0, epochs), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, epochs), H.history["val_loss"], label="val_loss", linestyle=":")
    plt.title("Loss curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(np.arange(0, epochs), H.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, epochs), H.history["val_accuracy"], label="val_acc", linestyle=":")
    plt.title("Accuracy curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.legend()



def main():
    #Loading CIFAR_10
    print("CIFAR_10 data is being loaded - Please Wait...")

    (X_train, y_train), (X_test, y_test) = cifar10.load_data() #Splitting data up into train and test data. 

    print("CIFAR_10 data has been loaded successfully")
    
    #normalize
    X_train = X_train/255
    X_test = X_test/255
    
    #Binarize labels
    lb = LabelBinarizer()
    y_train = lb.fit_transform(y_train) #fitting and transforing to data
    y_test = lb.fit_transform(y_test)

    #labels for CIFAR-10 dataset in alphabetical order
    label_names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    
    #Check tensor shape
    X_train.shape #correct shape: 50000 images, 32x32X3
    
    # load and initilize VGG16
    #Printing to terminal
    print("model VGG16 is being loaded - Please Wait...")
    model = VGG16(include_top = False, 
                  pooling = "avg", #pooling layer
                  input_shape = (32,32,3))#change input shape
    
    #Disable training of Conv layers
    #disable while training - stay intact during training or else no transfer learning. 
    #if it updates it defeats the purpose
    for layer in model.layers:
        layer.trainable = False
    
    print(model.summary())
    
    #add new classification layers
    flat1 = Flatten()(model.layers[-1].output) #flattening layer, output of the last layer of model
    class1 = Dense(128, activation = "relu")(flat1)
    output = Dense(10, activation = "softmax")(class1)
    
    #adding everything together
    model = Model(inputs = model.inputs, outputs = output)
    
    #summarize
    print(model.summary()) #see new layers
    
    #Compile
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.01, #start learning quickly
        decay_steps=10000, #go down 10000 (big steps)
        decay_rate=0.9) #take smaller steps with each epoch: slowing down

    sgd = SGD(learning_rate= lr_schedule)
    
    model.compile(optimizer=sgd,
                  loss="categorical_crossentropy",
                  metrics=["accuracy"]) #can change this

    #Train model
    print("[INFO] training model...")
    H = model.fit(X_train, y_train,
                  validation_data = (X_test, y_test),
                  batch_size = 128, #how many images we are looking at at a time
                  epochs = 10,
                  verbose = 1) #printed to the screen as it is training
    
    
    # make path to output data
    outpath = os.path.join("..","..","cds-visual","Assignments","output")
    
    #generate and save the plots of the loss and accuracy to a specific folder
    plot_history(H,10)
    plt.savefig(outpath + "/" + "plots_assign_3.png", dpi=300, bbox_inches='tight')
    plt.show()

    print("Done! Plots are generated and saved in the output folder as plots_assign_3.png")
    
    #classification report
    predictions = model.predict(X_test, batch_size = 128)

    report = (classification_report(y_test.argmax(axis=1),
                                predictions.argmax(axis=1), #return the biggest value of the predictions
                                target_names= label_names))
    print(report) #print classification report

    #Save the classification report in specific folder
    g = open("../../cds-visual/Assignments/output/classification_assign_3.txt",'w')
    print(report, file=g)

    print("Done! Report has been generated and saved in the output folder as classification_assign_3.txt")

    
if __name__=="__main__":
    main()