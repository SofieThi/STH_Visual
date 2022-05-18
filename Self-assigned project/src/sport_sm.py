#import
import numpy as np
import os
import pathlib
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import tensorflow_hub as hub
import warnings
warnings.filterwarnings('ignore')
os.environ["KMP_WARNINGS"] = "FALSE" 
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

#making the plot history for the model's loss and accuracy as a function
def plot_history(history, epochs):
    plt.style.use("seaborn-colorblind")
    
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')

    plt.title('Training and Validation Accuracy')
    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Cross Entropy')

    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.savefig(os.path.join("output/self_assigned_plots.png"))

def main():
    # Parameters
    num_classes = 5 #image categories/classes
    batch_size = 64 #how many images we are looking at at a time
    img_height = 224 #image dimensions 224x224
    img_width = 224

    #path to data folder with sports pictures in 5 folders of the 5 different sports
    input_directory = pathlib.Path("data", "sport_data")
    
    # defining and splitting training and validation subsets
    train_ds = tf.keras.utils.image_dataset_from_directory(
        input_directory,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    val_ds = tf.keras.utils.image_dataset_from_directory(
        input_directory,
        validation_split=0.3,
        subset="validation",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    #create model
    model1 = Sequential([
        layers.Rescaling(1./255),
        layers.Conv2D(64, (3,3), padding='same', activation='relu'), #relu activation layers
        layers.MaxPooling2D(),
        layers.Conv2D(128, (3,3), padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(256, (3,3), padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(512, (3,3), padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(512, (3,3), padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(), #flatten layer
        layers.Dense(256, activation='relu'), #hidden layers
        layers.Dense(256, activation='relu'),
        layers.Dense(num_classes) #the number of categories we want to predict
    ])
    
    #compile and train the model
    print("[INFO] training model...")

    model1.compile(optimizer='adam', #adam optimizer from keras
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), #loss function
        metrics=['accuracy']) #accuracy
    history = model1.fit(train_ds, epochs=10, validation_data=val_ds)
    
    #see the layers of the model
    print(model1.summary())
    
    #plot history for the model's loss and accuracy -using the function to save and show the plots
    plot_history(history, 10) #10 epochs
    plt.show()

    print("Done! Plots are generated and saved in the output folder as self_assigned_plots.png")
    
    #generate classification report
    test_label = np.concatenate([y for x, y in val_ds], axis= 0)
    
    predictions = model1.predict(val_ds)
    
    predictions.argmax(axis=1)
    
    labels = train_ds.class_names

    report = classification_report(test_label, 
                                   predictions.argmax(axis=1), #return the biggest value of the predictions
                                   target_names = labels)

    print(report)
    
    #Save the classification report
    g = open("output/self-assigned_cr.txt",'w')
    print(report, file=g)

    print("Done! Report has been generated and saved in the output folder as self-assigned_cr.txt")
    
    
if __name__=="__main__":
    main()  