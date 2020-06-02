# Dogs-vs-Cats-Classifier
Convolutional Neural Network to classify dogs or cats

## Introduction

A fun project to implement a binary classifier to classify dog/cat images.

## Prepare the data-set

The data-set can been obtained from [Kaggle](https://www.kaggle.com/c/dogs-vs-cats/data). 

The training data contains 25,000 labeled images of dogs and cats, each category has 12,500 images accordingly. 
The test data contains 12,500 unlabeled images.

Download the train.zip and test1.zip files, unzip them, and put the training data in (**./data/train/**) folder, and the testing data in the (**./data/test1/**) folder.

## Notebooks

**models.py** - A basic CNN (6-layer) 
**training.py** - Script to prepocess the data, train and save the model.
**Prediction.py** - Notebook to test the data

## Further improvements (To do)

1. Will try Transfer learning using any one of the pretrained architectures and finetune the model.
2. Instance Segementation using Mask-RCNN.
