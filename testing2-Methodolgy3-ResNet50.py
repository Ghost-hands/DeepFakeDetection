import tensorflow as tf
from tensorflow import keras
import random
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from pathlib import Path
import imghdr
from keras import regularizers
import time
import cv2
import torch
import shutil
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications import VGG19,VGG16
from tensorflow.keras import layers, models
from sklearn.metrics import  accuracy_score, roc_auc_score,  confusion_matrix, classification_report
from keras.layers import Input, Dense, GlobalAveragePooling2D, BatchNormalization, Flatten, Dropout, MaxPooling2D, GlobalMaxPooling2D, Conv2D
from keras import regularizers, Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

#STEP 1 - LOADING THE DATA AND PRE-PROCESSING:
###############################################################################################################################################################################################
train_dir = '\\Users\\Ahmed\\Desktop\\Practicecoding\\Mypythonprojects\\Datasets\\GRADPROJECT\\Dataset2\\Dataset\\Train'
test_dir = '\\Users\\Ahmed\\Desktop\\Practicecoding\\Mypythonprojects\\Datasets\\GRADPROJECT\\Dataset2\\Dataset\\Test'
validation_dir = '\\Users\\Ahmed\\Desktop\\Practicecoding\\Mypythonprojects\\Datasets\\GRADPROJECT\\Dataset2\\Dataset\\Validation'
#HISTORY OF PRE-PROCESSING DONE IN THE PROJECT:
#-normalizing all the images in the dataset: Implemented in the final version of the project, as it is an essential step in reducing the overall complexity of the process which makes it more efficient.
#-Re-sizing images, downsizing more specifically: Implemented in the final version of the project, as it as well reduces the overall complexitiy of the program, as the lower the image size the less time it takes to train. Efficiency increases.
#-image augmentation using image data generator (no other options. just rescaling):

IMG_SIZE = (224, 224) #size = (128,128) greatly reduces training time per epoch.
batch_size = 64


'''
WITHOUT IMAGE AUGMENTATION:

def normalize_image(image, labels):
  image = tf.cast(image, tf.float32) / 255.0
  return image, labels


train_data = tf.keras.utils.image_dataset_from_directory(directory=train_dir,
                                                         labels='inferred',
                                                         label_mode='categorical',
                                                         class_names= ['Fake', 'Real'],
                                                         color_mode='rgb',
                                                         image_size=IMG_SIZE,
                                                         batch_size=batch_size,
                                                         shuffle= True)

validation_data = tf.keras.utils.image_dataset_from_directory(
    directory=validation_dir,
    labels='inferred',
    label_mode='categorical',
    class_names=['Fake', 'Real'],
    color_mode='rgb',
    image_size=IMG_SIZE,
    batch_size=batch_size,
    shuffle= True)

test_data = tf.keras.utils.image_dataset_from_directory(
    directory=test_dir,
    labels='inferred',
    label_mode='categorical',
    class_names= ['Fake', 'Real'],
    color_mode='rgb',
    image_size=IMG_SIZE,
    batch_size=batch_size,
    shuffle= False)
'''



'''

'''
#IMAGE AUGMENTATION:

#Image data augmentation using image data generator:
#Image data augmentation generates batches of tensor image data in real-time, to add diversity and supplement the size and quality of the dataset.
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
 rescale=1 / 255.0,
brightness_range=[0.4,1.5],
zoom_range=0.3,
horizontal_flip=True,
vertical_flip=True,
width_shift_range=0.2,
height_shift_range=0.2,
rotation_range=30,
fill_mode='nearest'
 )

test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1 / 255.0)

valid_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1 / 255.0)



train_generator = train_datagen.flow_from_directory(
    directory=train_dir,
    target_size=IMG_SIZE,
    color_mode="rgb",
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=True
)

validation_generator = valid_datagen.flow_from_directory(
    directory=validation_dir,
    target_size=IMG_SIZE,
    color_mode="rgb",
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=True
)
test_generator = test_datagen.flow_from_directory(
    directory=test_dir,
    target_size=IMG_SIZE,
    color_mode="rgb",
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=False
)



#STEP 2 - BUILDING MODEL + TRAINING MODEL:
##################################################################################################################################################################################################################################
base_model=ResNet50(input_shape=(224,224,3),weights='imagenet', include_top=False)


for layer in base_model.layers:
  layer.trainable=False
x = base_model.output
#hidden layers on top of the base of the resnet50 model:
x = GlobalMaxPooling2D() (x)
x = Dense(512, activation="relu")(x)
x = Dropout(0.5)(x)
x = Dense(256, activation="relu")(x)
x = Dropout(0.5)(x)
predictions = Dense(2, activation='softmax')(x) #Output layer calibrated to our purpose of predicting whether a picture of a person's face is deepfake or not.

model = Model(inputs=base_model.input, outputs=predictions)



print(model.summary())

#keras.utils.plot_model(model, to_file='ResNet50model.png', show_shapes=True) #Plotting the structure model. Generates a .png image of the structure of the model.

model.compile(loss = tf.keras.losses.CategoricalCrossentropy(),
              optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0001), #Default learning rate for adam optimizer is 0.001
              metrics = ['accuracy']
              )


model_history = model.fit(train_generator, epochs= 5, validation_data=validation_generator)

#STEP 3 - TESTING AND EVALUATING THE MODEL:
############################################################################################################################################################################################################################################################
test_accuracy = model.evaluate(test_generator)
print(f'test accuracy: {test_accuracy[1]}')
validation_accuracy = model.evaluate(validation_generator)
print(f'validation accuracy: {validation_accuracy[1]}')

#TESTING TABLE (Where i record the change in test and validation accuracies based on parameters changed):
#1-model with batch size= 64, learning rate= 0.001 achieves training accuracy= 0.8349/83.49%,test accuracy= 0.7281/72.81%,validation accuracy = 0.8187/81.87%. after 10 epochs. (image size = 128,128) (Normalization + resizing)
#2-model with batch size= 64, learning rate= 0.0001, dropout = 0.5 (between dense layers) achieves training accuracy= 80s,test accuracy= ,validation accuracy = 80s. after 10 epochs. (image size = 128,128) (Normalization + resizing) TERMINATED at 8th epoch. decline. stays in the 80s.
#3-model with batch size= 64, learning rate= 0.0001 , dropout = 0.5 (between dense layers) achieves training accuracy= 0.8195/81.95%,test accuracy= 0.7513/75.13%,validation accuracy = 0.8181/81.81%. after 5 epochs. (image size = 224,224) (Normalization + resizing)
#4-model with batch size= 64, learning rate= 0.0001 , dropout = 0.5 (between dense layers) achieves training accuracy= 69%,test accuracy= ,validation accuracy = 69%. after 5 epochs. (image size = 224, 224) (Normalization + resizing + image augmentation) TERMINATED in the 5th epoch, due to stagnation.
#5-model with batch size= 64, learning rate= 0.0001 , dropout = 0.5 (between dense layers) achieves training accuracy= 0.5572/55.72%,test accuracy= ,validation accuracy = 0.5803/58.03%. after 5 epochs. (image size = 224,224) (Normalization + resizing + image augmentation(random brightness + random zoom + random rotations + random shifts + random flips)) TERMINATED in the 4th epoch, due to slow progress and having sufficient data for comparison with previous models.

#plotting the accuracies changing over the epochs (Current trend of the graph is that the accuracy gradually increases with each passing epoch):
import seaborn as sns
accuraciesepochs = model_history.history['accuracy']
epochs = range(0,5)
sns.lineplot(x = epochs, y =accuraciesepochs)
plt.show()

'''
#Showing the overall data prediction for each class:
preds = model.predict(test_data)
predictions = np.argmax(preds, axis=1)
actual_labels =  np.array([])
for x, y in test_data:
  actual_labels = np.concatenate([actual_labels, np.argmax(y.numpy(), axis=-1)])

confusion = confusion_matrix(actual_labels, predictions) #The confusion matrix showcases the wrong and correct predictions for the total images in each class in the dataset.

#A visual representation of the confusion matrix using the seaborn library and matplotlib:
class_names = ['Fake', 'Real']
plt.figure(figsize=(8, 6))
sns.heatmap(confusion, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
'''




'''

'''
#IMAGE AUGMENTATION TAILORED CONFUSION MATRIX:
#Showing the overall data prediction for each class:
preds = model.predict(test_generator)
predictions = np.argmax(preds, axis=1)

#Actual labels:
actual_labels = test_generator.classes

confusion = confusion_matrix(actual_labels, predictions) #The confusion matrix showcases the wrong and correct predictions for the total images in each class in the dataset.

#A visual representation of the confusion matrix using the seaborn library and matplotlib:
class_names = list(test_generator.class_indices.keys())
plt.figure(figsize=(8, 6))
sns.heatmap(confusion, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

#Accuracy_Score:
print(accuracy_score(actual_labels, predictions))


#Roc_AUC_score:
print(roc_auc_score(actual_labels, predictions, average=None))

#Classification report:
print(classification_report(actual_labels, predictions, target_names=class_names))




#STEP 4 - SAVING THE WEIGHTS OF THE TRAINED MODEL:
############################################################################################################################################################################################################
model.save('ResNet50Model1.h5')
#HISTORY OF MODELS SAVED:
#ResNet50Model1 (first model with overall 90% accuracy)=