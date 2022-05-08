# We need to include the home directory in our path, so we can read in our own module.
import os
import sys
sys.path.append(os.path.join("..", "..", "CDS-VIS"))
import cv2
import numpy as np
from utils.imutils import jimshow
from utils.imutils import jimshow_channel
import matplotlib.pyplot as plt
import pandas as pd

def main():
    #Take a user-defined image from the folder
    #Load image
    filename = os.path.join("..","..","CDS-VIS", "flowers", "image_0004.jpg")

    base_image = cv2.imread(filename)

    #show the base/target image
    jimshow(base_image)

    # combining filepaths
    path = os.path.join("..","..","CDS-VIS", "flowers")
    filenames = os.listdir(path)
    for name in filenames: #It finds all filenames of the flower images
        filepath = os.path.join(path, name)
        print(filepath)

    #finding distance scores
    #using compare and normalizations formulas from session_3
    distance_score = [] #empty list

    filenames = os.listdir(path)

    #target image normalization outside of for loop 
    hist1 = cv2.calcHist([base_image], [0,1,2], None, [8,8,8], [0,256, 0,256, 0,256])
    hist1 = cv2.normalize(hist1, hist1, 0,255, cv2.NORM_MINMAX)

    for name in filenames:
        #ignoring non jpgs in the folder
        if name.endswith(".jpg"):
            #target image
            hist1
            # Get the color histogram for one of the image in the folder to compare with the target image
            filepath = os.path.join(path, name)
            other_image = cv2.imread(filepath)
            hist2 = cv2.calcHist([other_image], [0,1,2], None, [8,8,8], [0,256, 0,256, 0,256])
            hist2 = cv2.normalize(hist2, hist2, 0,255, cv2.NORM_MINMAX)
            # Compare target image and one of the images in the folder
            score = round(cv2.compareHist(hist1, hist2, cv2.HISTCMP_CHISQR), 2)
            # Save the distance score
            distance_score.append(score)
        else:
            continue

    print(distance_score) #takes some time!

    #Combining distance scores to the file names of the pictures
    list_context=list(zip(filenames, distance_score)) #zip: combining two lists, creating a new list with tuples

    # Converting lists of tuples into pandas Dataframe
    dframe = pd.DataFrame(list_context, columns=['File names', 'Distance score']) #2 columns
    # Printing the data
    print(dframe)

    # Finding the 3 most similar images (the 3 with the lowest distance score)
    ranked = sorted(distance_score)
    print("The values of the three most similar pictures: " + str(ranked[0:4]) + " (the first one is the target picture itself)")

    # Finding the file name of the 3 images that are most similar to the target picture
    image1 = dframe[dframe['Distance score']==ranked[0]]['File names'].values[0] # This is just the target picture
    image1 = (path + "/" + image1) #Add the path to the file name
    #repeating
    image2 = dframe[dframe['Distance score']==ranked[1]]['File names'].values[0]
    image2 = (path + "/" + image2)
    image3 = dframe[dframe['Distance score']==ranked[2]]['File names'].values[0]
    image3 = (path + "/" + image3)
    image4 = dframe[dframe['Distance score']==ranked[3]]['File names'].values[0]
    image4 = (path + "/" + image4)
    
    #getting the filenames of the closest images in descending order into a csv
    images = image1, image2, image3, image4

    print(images)

    #into dataframe

    desc_dframe=list(zip(images, ranked)) #zip: combining two lists, creating a new list with tuples

    # Converting lists of tuples into pandas Dataframe
    dataframe = pd.DataFrame(desc_dframe, columns=['File names', 'Sorted Distance score']) #2 columns
    # Printing the data
    print(dataframe)
    
    #saving the dframe in specific folder
    dataframe.to_csv(os.path.join("..","..","cds-visual", "Assignments", "output", "assign_1.csv"))

    print("Done! dframe has been generated and saved in the output folder as assign_1.csv")
    

if __name__=="__main__":
    main()