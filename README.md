# Find the nose!

## Table of contents

* [Introduction](#introduction)
* [Structure](#structure)
* [Usage](#usage)
* [Preview](#preview)
* [Technologies](#technologies)

## Introduction

The goal of this project is to train a neural network responsible for finding the tip of the nose person displayed in the picture. During the process the two architectures were tested:
1. Cascade model with architecture following [paper](http://mmlab.ie.cuhk.edu.hk/archive/CNN_FacePoint.htm).
2. Small model that uses pre-trained ResNet as a backbone.
Both models were trained on modified [FK dataset](https://www.kaggle.com/tarunkr/facial-keypoints-68-dataset). (To the original dataset, there were added, cropped copies of original pictures. Models trained on extended that way dataset performed significantly better on real world data.) 

## Structure
* [notebooks](notebooks/) cotains three notebooks:
  * [test.ipynb](notebooks/test.ipynb) -> load pre-trained Cascade model and test it on a few pictures
  * [model.ipynb](notebooks/model.ipynb) -> build, train and compare both models
  * [data.ipynb](notebooks/data.ipynb) -> load and build dataset
* [models](models/) contains classes representing both networks
* [saved models](saved_models/) contains saved pre-trained Cascade model
* [data](data/) contains small sample of preprocessed dataset
* [utility](utility/) contains utilility functions
* [sprawozdanie.pdf](sprawozdanie.pdf) contains short, written in polish, report describing preprocessing, augmentation technics and used models

## Usage

1. Make sure your environment meets requirements listed in [requirements.txt](requirements.txt). (**pip install -r requirements.txt**)
2. To test the model on your own pictures save them under [pictures](pictures/) directory and change file_name in [test.ipynb](notebooks/test.ipynb).
3. You can use pre-trained Cascade model as in notebook [test.ipynb](notebooks/test.ipynb).
4. To repeat training download [FK dataset](https://www.kaggle.com/tarunkr/facial-keypoints-68-dataset) and save it as "Facial Keypoints" under [data](data/) directory. Then run notebook [data.ipynb](notebooks/data.ipynb) to build dataset. After that you can run [model.ipynb](notebooks/model.ipynb) to train and evaluate models. 


## Preview

### Raw data

![Alt text](readme_pictures/raw_data.png?raw=true)

### Preprocessing

![Alt text](readme_pictures/preprocessing.png?raw=true)

### Augumentation

#### Cropping

![Alt text](readme_pictures/aug1.png?raw=true)

#### Add noise

![Alt text](readme_pictures/aug2.png?raw=true)

#### Horizontal flip

![Alt text](readme_pictures/aug_3.png?raw=true)

### Predictions on test data

![Alt text](readme_pictures/test_predictions.png?raw=true)

## Technologies
* Pandas
* Numpy
* PIL
* Tensorflow
* Keras
