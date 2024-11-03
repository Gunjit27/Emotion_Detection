# Emotion Detection using Deep Learning

## Overview
This project utilizes deep learning techniques to detect and classify emotions from images. The model is trained on a dataset of facial expressions and predicts one of the seven moods: angry, disgust, fear, happy, neutral, sad, and surprise.

## Table of Contents
- [Requirements](#requirements)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Results](#results)

## Requirements
- Python 3.x
- TensorFlow
- OpenCV
- NumPy
- scikit-learn

## Dataset
The dataset used for training and validation consists of images organized into folders named after the corresponding emotions. The images are resized to 48x48 pixels and converted to grayscale.

## Model Architecture
The model is a convolutional neural network (CNN) comprising the following layers:
- Convolutional layers
- MaxPooling layers
- Dropout layers
- Fully connected layers

The output layer uses a softmax activation function for multi-class classification.

## Training
The model is trained for 100 epochs with an Adam optimizer and categorical cross-entropy loss. Training and validation accuracy are logged during the training process.

## Results
The model achieves an accuracy of around 94.2% on the validation dataset after 100 epochs. Detailed training logs are available in the training output.

