import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.layers import Input, Dense, GlobalAveragePooling2D, BatchNormalization, Flatten, Dropout, MaxPooling2D
from sklearn.metrics import  accuracy_score, roc_auc_score,  confusion_matrix, classification_report
from tensorflow import keras
import random
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from pathlib import Path
import imghdr
from keras import regularizers, Model
import time
import cv2
from tensorflow.keras.applications import Xception


#STEP 1 - LOADING THE DATA AND PRE-PROCESSING:
###############################################################################################################################################################################################
train_dir = '\\Users\\Ahmed\\Desktop\\Practicecoding\\Mypythonprojects\\Datasets\\GRADPROJECT\\Dataset2\\Dataset\\Train'#Not relative. requires a fixed path for the code to work properly.
test_dir = '\\Users\\Ahmed\\Desktop\\Practicecoding\\Mypythonprojects\\Datasets\\GRADPROJECT\\Dataset2\\Dataset\\Test'#Not relative. requires a fixed path for the code to work properly.
validation_dir = '\\Users\\Ahmed\\Desktop\\Practicecoding\\Mypythonprojects\\Datasets\\GRADPROJECT\\Dataset2\\Dataset\\Validation'#Not relative. requires a fixed path for the code to work properly.
#HISTORY OF PRE-PROCESSING DONE IN THE PROJECT:
#-normalizing all the images in the dataset: Implemented in the final version of the project, as it is an essential step in reducing the overall complexity of the process which makes it more efficient.
#-Re-sizing images, downsizing more specifically: Implemented in the final version of the project, as it as well reduces the overall complexitiy of the program, as the lower the image size the less time it takes to train. Efficiency increases.
#-image augmentation using image data generator (random brightness, random zoom, random flips, random shifts and random rotations):
#-image augmentation using image data generator (just the rescaling option. No other options):


IMG_SIZE = (128, 128) #reducing image size from 256,256 (image size that i used in my sequential model) to 224,224 reduces training time.
batch_size = 64

'''
#WITHOUT IMAGE AUGMENTATION:

def normalize_image(image, labels):
  image = tf.cast(image, tf.float32) / 255.0
  return image, labels

#CATEGORICAL:
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



#applying normalization function on all images of the dataset:
train_data = train_data.map(normalize_image)
validation_data = validation_data.map(normalize_image)
test_data = test_data.map(normalize_image)
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

'''


'''
#MOBILENET MODEL MAKE UP:
base_model=keras.applications.MobileNet(input_shape=(128,128,3),weights='imagenet', include_top=False)#input of the model calibrated to fit our particular image size.
#Rather than use the pre-trained model known as Xception model as the base, MobileNet model was used instead.
#As Xception proved to be too inefficient with its long training time per epoch (about 1 hour and 30 minutes per epoch, if not more)
#MobileNet model takes about 22 minutes per epoch. A great improvement over the Xception model, so it was picked.

for layer in base_model.layers:
  layer.trainable=False
x = base_model.output
#hidden layers on top of the base of the mobilenet model:
keras.layers.Conv2D(filters=8, kernel_size=5, activation='relu'),
keras.layers.MaxPooling2D(),
keras.layers.BatchNormalization(),

keras.layers.Conv2D(filters=16, kernel_size=4, activation='relu'),
keras.layers.MaxPooling2D(),
keras.layers.BatchNormalization(),
keras.layers.Dropout(0.1),

keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'),
keras.layers.MaxPooling2D(),
keras.layers.BatchNormalization(),
keras.layers.Dropout(0.2),

x = Flatten()(x)
x = Dense(100, activation="relu")(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
x = Dense(70, activation="relu")(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
x = Dense(40, activation="relu")(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
x = Dense(10, activation="relu")(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
predictions = Dense(2, activation='softmax')(x) #Output layer calibrated to our purpose of predicting whether a picture of a person's face is deepfake or not.
model = Model(inputs=base_model.input, outputs=predictions)



'''
#XCEPTION MODEL MAKE UP:
base_model=Xception(input_shape=(128,128,3),weights='imagenet', include_top=False)

for layer in base_model.layers:
  layer.trainable=False
x = base_model.output
#hidden layers on top of the base of the mobilenet model:
x = GlobalAveragePooling2D()(x)
x = Dense(32, activation='relu')(x)
x = Dense(16, activation='relu')(x)
predictions = Dense(2, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

'''




print(model.summary())

#keras.utils.plot_model(model, to_file='MobileNetmodel.png', show_shapes=True) #Plotting the structure model. Generates a .png image of the structure of the model.

model.compile(loss = tf.keras.losses.CategoricalCrossentropy()
              ,
              optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0001), #Default learning rate for adam optimizer is 0.001
              metrics = ['accuracy']
              )

#reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.3,
                #patience=5, mode='min', verbose=1,
                #min_lr=1e-4)

model_history = model.fit(train_generator, epochs= 12,  validation_data=validation_generator)


#STEP 3 - TESTING AND EVALUATING THE MODEL:
############################################################################################################################################################################################################################################################
test_accuracy = model.evaluate(test_generator)
print(f'test accuracy: {test_accuracy[1]}')
validation_accuracy = model.evaluate(validation_generator)
print(f'validation accuracy: {validation_accuracy[1]}')

#TESTING TABLE (Where i record the change in test and validation accuracies based on parameters changed):
#1-model with batch size= 32, learning rate= 0.001, dropout= 0.4 achieves training accuracy= 0.7931/79.31% ,test accuracy= 0.6707/67.07% ,validation accuracy = 0.7266/72.66%. after 10 epochs. (image size = 224,224) (Normalization + resizing)
#2-model with batch size= 64, learning rate= 0.0001, dropout= 0.4 achieves training accuracy= 0.9455/94.55% ,test accuracy= 0.8048/80.48% ,validation accuracy = 0.8991/89.91%. after 10 epochs. (image size = 224,224)(Normalization + resizing)
#3-model with batch size= 32, learning rate= 0.0001, dropout= 0.2 achieves training accuracy= 0.9537/95.37% ,test accuracy= 0.7959/79.59% ,validation accuracy = 0.8893/88.93%. after 12 epochs.(image size = 224,224)(Normalization + resizing)
#4-model with batch size= 32, learning rate= 0.0001, dropout= 0.5 achieves training accuracy=  0.9182/91.82%,test accuracy= 0.8004/80.04% ,validation accuracy = 0.8919/89.19%. after 12 epochs.(image size = 224,224)(Normalization + resizing)
#5-model with batch size= 64, learning rate= 0.0001, dropout= 0.1, 0.2 (between convolutional layers) and 0.5 (between dense layers) achieves training accuracy=  0.8782/87.82%,test accuracy=  0.7789/77.89%,validation accuracy = 0.8590/85.90%. after 12 epochs.(image size = 128, 128)(Normalization + resizing)
#6-model with batch size= 64, learning rate= 0.0001, dropout= 0.1, 0.2 (between convolutional layers) and 0.5 (between dense layers) achieves training accuracy=  0.8849/88.49%,test accuracy=  0.7798/77.98%,validation accuracy = 0.8540/85.40%. after 12 epochs.(image size = 128, 128)(Normalization + resizing + image augmentation)
#7-model with batch size= 64, learning rate= 0.0001, dropout= 0.1, 0.2 (between convolutional layers) and 0.5 (between dense layers) achieves training accuracy=  0.7917/79.17%,test accuracy=  0.7615/76.15%,validation accuracy = 0.8013/80.13%. after 12 epochs.(image size = 128, 128)(Normalization + resizing + image augmentation(random brightness + random zoom + random rotations + random shifts + random flips))

#plotting the accuracies changing over the epochs (Current trend of the graph is that the accuracy gradually increases with each passing epoch):
import seaborn as sns
accuraciesepochs = model_history.history['accuracy']
epochs = range(0,12)
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
'''

'''


#Accuracy_Score:
print(accuracy_score(actual_labels, predictions))


#Roc_AUC_score:
print(roc_auc_score(actual_labels, predictions, average=None))

#Classification report:
print(classification_report(actual_labels, predictions, target_names=class_names))


#STEP 4 - SAVING THE WEIGHTS OF THE TRAINED MODEL:
############################################################################################################################################################################################################
model.save('MobileNetModel4.h5')
#HISTORY OF MODELS SAVED:
#-MobileNetModel1 =model #2 with batch size= 64, learning rate= 0.0001, dropout= 0.4 achieves training accuracy= 0.9455/94.55% ,test accuracy= 0.8048/80.48% ,validation accuracy = 0.8991/89.91%. after 10 epochs. (image size = 224,224)(Normalization + resizing)
#-MobileNetModel2 =model #3 with batch size= 32, learning rate= 0.0001, dropout= 0.2 achieves training accuracy= 0.9537/95.37% ,test accuracy= 0.7959/79.59% ,validation accuracy = 0.8893/88.93%. after 12 epochs.(image size = 224,224)(Normalization + resizing)
#-MobileNetModel3 =model #4 with batch size= 32, learning rate= 0.0001, dropout= 0.5 achieves training accuracy=  0.9182/91.82%,test accuracy= 0.8004/80.04% ,validation accuracy = 0.8919/89.19%. after 12 epochs.(image size = 224,224)(Normalization + resizing)
#-MobileNetModel4 (first model with overall 90% accuracy) =