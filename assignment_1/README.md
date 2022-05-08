## Assignment 1 - Image search

This repository contains the answers to the following tasks:

Take a user-defined image from the folder

Calculate the "distance" between the colour histogram of that image and all of the others

Find which 3 image are most "similar" to the target image.

Save an image which shows the target image, the three most similar, and the calculated distance score.

Save a CSV which has one column for the filename and three columns showing the filenames of the closest images in descending order


## Structure

This repository has the following directory structure:

| Column | Description|
|--------|:-----------|
```data```| a folder to be used for inputting the data you wish to run
```notebooks``` | Jupyter notebooks in both .ipynb and .html format
```src``` | the .py script version of the assignment
```output``` | the results of inputting the flowers dataset 

## Contribution

Sofie Thinggaard au613703

201909063@post.au.dk

## Methods

This problem relates to finding how similar a given image is to others. In order to address this problem, we calculate the "distance" between the colour histogram (distribution of pixel's intensity) of a chosen image and all of the other images in the flower folder. Firstly, we define an image from the folder and then find all filenames of the flower images in the folder (defined as filepath). Next, we do image normalization of our target image. Then, in a for loop, we ignore non jpgs in the folder, get the color histogram for one of the image in the folder to compare with the target image, normalize the other image, compare target image and one of the images in the folder and save distance scores to an empty list. Next, we combine distance scores to the file names of the pictures and convert it into a dataframe. We sort it to find the 3 most similar images. Then, we find the file name of the 3 images that are most similar to the target picture. Finally, we get the filenames of these 3 closest images in descending order into a csv in an output folder.

## Usage (reproducing results)

To replicate the results choose the folder: flowers, and chose the target image "image_0004.jpg"

## Discussion of results

Results: getting a .csv file with a column for the filename and three columns showing the filenames of the three closest images in descending order. The first image (image_0004.jpg) is the traget image itself, therefore it makes sense its score is 0,0. The most similar image to the target image is image_0591.jpg, the second most similar image is image_0591.jpg and the next image that looks most like the target image is image_0566.jpg
