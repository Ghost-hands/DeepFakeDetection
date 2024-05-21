from io import StringIO

import PIL
import tensorflow as tf
from tensorflow import keras
import random
import os
import numpy as np
import collections
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import  accuracy_score, roc_auc_score,  confusion_matrix, classification_report
from tensorflow.keras.utils import plot_model
from pathlib import Path
import imghdr
from keras import regularizers
import time
import cv2




physical_devices = tf.config.list_physical_devices("GPU")
print(physical_devices)

print(tf.test.is_gpu_available())
print(tf.test.is_built_with_cuda())
#testing to see whether the GPU is detected or not, so that it can then be used to train models faster than the CPU.
#Unfortunately, GPU is not detected.

#STEP 1 - LOADING THE DATA AND PRE-PROCESSING:
###############################################################################################################################################################################################
train_dir = '\\Users\\Ahmed\\Desktop\\Practicecoding\\Mypythonprojects\\Datasets\\GRADPROJECT\\Dataset2\\Dataset\\Train' #Not relative. requires a fixed path for the code to work properly.
test_dir = '\\Users\\Ahmed\\Desktop\\Practicecoding\\Mypythonprojects\\Datasets\\GRADPROJECT\\Dataset2\\Dataset\\Test'#Not relative. requires a fixed path for the code to work properly.
validation_dir = '\\Users\\Ahmed\\Desktop\\Practicecoding\\Mypythonprojects\\Datasets\\GRADPROJECT\\Dataset2\\Dataset\\Validation'#Not relative. requires a fixed path for the code to work properly.
#HISTORY OF PRE-PROCESSING DONE IN THE PROJECT:
#-normalizing all the images in the dataset: Implemented in the final version of the project, as it is an essential step in reducing the overall complexity of the process which makes it more efficient.
#-Re-sizing images, downsizing more specifically: Implemented in the final version of the project, as it as well an essential step that reduces the overall complexitiy of the program, as the lower the image size the less time it takes to train. Efficiency increases.
#-image augmentation using image data generator (random brightness, random zoom, random flips, random shifts and random rotations):
#-image augmentation using image data generator (just the rescaling option. No other options):

IMG_SIZE = (256, 256) #was using 256,256 for testing, but decided to lower it to reduce training time. (changes depending on what achieves the best,final result)
batch_size = 32


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
 rescale=1 / 255.0
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
sequentialmodel = keras.models.Sequential ([
#input layer:
keras.layers.Conv2D(filters=8, kernel_size=5, input_shape=(256, 256, 3), activation='relu'), #Input layer.
keras.layers.MaxPooling2D(),
keras.layers.BatchNormalization(),


#The hidden layers:
keras.layers.Conv2D(filters=16, kernel_size=4, activation='relu'), #relu is a non-linear activation function, while Conv2d is linear.
keras.layers.MaxPooling2D(),
keras.layers.BatchNormalization(),
keras.layers.Dropout(0.1),


keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'),
keras.layers.MaxPooling2D(),
keras.layers.BatchNormalization(),
keras.layers.Dropout(0.2),



keras.layers.Conv2D(filters=64, kernel_size=2, activation='relu'),
keras.layers.MaxPooling2D(),
keras.layers.BatchNormalization(),
keras.layers.Dropout(0.3),


keras.layers.Conv2D(filters=128, kernel_size=1, activation='relu'),
keras.layers.MaxPooling2D(),
keras.layers.BatchNormalization(),
keras.layers.Dropout(0.4),


keras.layers.Flatten(),
#keras.layers.Dense(units = 200, activation = 'relu'),
#keras.layers.Dropout(0.5), #the higher the dropout, the more features may be lost.
keras.layers.Dense(units = 150, activation = 'relu'),
keras.layers.Dropout(0.5),
keras.layers.Dense(units = 100, activation = 'relu'),
keras.layers.Dropout(0.5),
keras.layers.Dense(units = 50, activation = 'relu'),
keras.layers.Dropout(0.5),
keras.layers.Dense(units = 2, activation = 'softmax') #output layer. number of units = number of classes to predict. in this case, fake and real, so units = 2.
])

sequentialmodel.summary()

#keras.utils.plot_model(sequentialmodel, to_file='Sequentialmodel.png', show_shapes=True) #Plotting the structure model. Generates a .png image of the structure of the model.


sequentialmodel.compile(loss = tf.keras.losses.CategoricalCrossentropy(),
              optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001), #default learning rate = 0.001.
              metrics = ['accuracy']
)
start = time.time()
model_history = sequentialmodel.fit(train_generator,  epochs= 24, validation_data=validation_generator)
stop = time.time()
print(f'Training took: {(stop-start)/60} minutes')



#STEP 3 - TESTING AND EVALUATING THE MODEL:
############################################################################################################################################################################################################################################################
test_accuracy = sequentialmodel.evaluate(test_generator)
print(f'test accuracy: {test_accuracy[1]}')
validation_accuracy = sequentialmodel.evaluate(validation_generator)
print(f'validation accuracy: {validation_accuracy[1]}')

#TESTING TABLE (Where i record the change in test and validation accuracies based on parameters changed):
#1-model with batch size=32, learning rate = 0.001, dropout = 0.2 (dropout between convolutional layers only) achieves training accuracy= 0.8966/ 89.66% ,test accuracy= 0.8566/85.66% ,validation accuracy = 0.8761/87.61%. after 20 epochs. (image size = 256,256) (Normalization + resizing)
#2-model with batch size=64, learning rate = 0.001, dropout = 0.2 (dropout between convolutional layers only) achieves training accuracy= 0.9037/90.37% ,test accuracy= 0.8130/81.30% ,validation accuracy = 0.8448/84.48% . after 20 epochs. (image size = 256,256) (Normalization + resizing)
#3-model with batch size= 16, learning rate = 0.001, dropout = 0.2 (dropout between convolutional layers only) achieves training accuracy= 0.4986/49.86% ,test accuracy= 0.5036/50.36%  ,validation accuracy = 0.4981/49.81%. after 20 epochs. (image size = 256,256) (Normalization + resizing)
#4-model with batch size= 128, learning rate = 0.0001, dropout = 0.2 (dropout between convolutional layers only) achieves training accuracy= 0.9467/94.67%  ,test accuracy=  0.7616/76.16%  ,validation accuracy = 0.8905/89.05%. after 20 epochs. (image size = 224, 224)/ image size lowered to reduce training time.(Normalization + resizing)
#5-model with batch size=32, learning rate = 0.001, dropout = 0.2 (dropout between convolutional layers only) achieves training accuracy=  0.9040/90.40% ,test accuracy=  0.8416/84.16% ,validation accuracy = 0.8920/89.20%. after 25 epochs. (image size = 224,224) (Normalization + resizing)
#6-model with batch size=32, learning rate = 0.001, dropout = 0.2 (between convolutional layers) and 0.5 (between fully-connected/dense layers) achieves training accuracy= 0.9511/95.11%  ,test accuracy=  0.8588/85.88% ,validation accuracy = 0.8912/89.12%. after 20 epochs. (image size = 256,256) (Normalization + resizing)
#7-model with batch size=32, learning rate = 0.001, dropout = 0.2 (between convolutional layers) and 0.5 (between fully-connected/dense layers) achieves training accuracy= 0.9602/96.02% ,test accuracy=  0.8937/89.37% ,validation accuracy = 0.9467/94.67%. after 19 epochs. (image size = 256,256) (Normalization + resizing)/ higher nodes in dense layers.
#8-model with batch size=32, learning rate = 0.001, dropout = 0.2 (between convolutional layers) and 0.5 (between fully-connected/dense layers) achieves training accuracy=  0.9618/96.18%,test accuracy= 0.8794/87.94%  ,validation accuracy = 0.9471/94.71%. after 19 epochs. (image size = 256,256) (Normalization + resizing)/higher nodes in dense layers.
#9-model with batch size=32, learning rate = 0.001, dropout = 0.2 (between convolutional layers) and 0.5 (between fully-connected/dense layers) achieves training accuracy=  0.9559/95.59%,test accuracy= 0.9071/90.71% ,validation accuracy = 0.9379/93.79%. after 19 epochs. (image size = 256,256) (Normalization + resizing)/higher nodes in dense layers + extra dense layer.
#10-model with batch size=32, learning rate = 0.001, dropout = 0.2 (between convolutional layers) and 0.2 (between fully-connected/dense layers) achieves training accuracy= 0.9777/97.77% ,test accuracy= 0.8751/87.51% ,validation accuracy = 0.9546/95.46%. after 19 epochs. (image size = 256,256) (Normalization + resizing)/higher nodes in dense layers + extra dense layer.
#11-model with batch size=32, learning rate = 0.001, dropout = 0.2 (between convolutional layers) and 0.5 (between fully-connected/dense layers) achieves training accuracy= 0.9527/95.27% ,test accuracy= 0.8738/87.38%,validation accuracy = 0.9400/94%. after 19 epochs. (image size = 256,256) (Normalization + resizing)/higher nodes in dense layers + extra dense layer.
#12-model with batch size=32, learning rate = 0.001, dropout = 0.2 (between convolutional layers) and 0.7 (between fully-connected/dense layers) achieves training accuracy= ,test accuracy= ,validation accuracy = . after 19 epochs. (image size = 256,256) (Normalization + resizing)/higher nodes in dense layers + extra dense layer. TERMINATED due to overfitting.
#13-model with batch size=32, learning rate = 0.001, dropout = 0.2 (between convolutional layers) and 0.5 (between fully-connected/dense layers) achieves training accuracy=  0.8867/88.67%,test accuracy= 0.7583/75.83%,validation accuracy =0.7921/79.21%. after 19 epochs. (image size = 224,224) (Normalization + resizing)/higher nodes in dense layers + extra dense layer.
#14-model with batch size=32, learning rate = 0.001, dropout = 0.2 (between convolutional layers) and 0.5 (between fully-connected/dense layers) achieves training accuracy=  0.9682/96.82%,test accuracy= 0.8494/84.94%, validation accuracy = 0.8811/88.11%. after 19 epochs. (image size = 256,256) (Normalization + resizing)/higher nodes in dense layers + extra dense layer + less conv layers.
#15-model with batch size=32, learning rate = 0.001, dropout = 0.1, 0.2 (between convolutional layers) and 0.35 (between fully-connected/dense layers) achieves training accuracy= 0.9829/98.29% ,test accuracy= 0.8779/87.79%,validation accuracy = 0.9289/92.89%. after 19 epochs. (image size = 256,256) (Normalization + resizing)/higher nodes in dense layers + extra dense layer + less conv layers.
#16-model with batch size=32, learning rate = 0.001, dropout = 0.1, 0.2 (between convolutional layers) and 0.35 (between fully-connected/dense layers) achieves training accuracy=  50%,test accuracy= ,validation accuracy = 50%. after 19 epochs. (image size = 256,256) (Normalization + resizing + image augmentation(random brightness + random zoom + random rotations + random shifts + random flips)) TERMINATED after 2 epochs. No change from 50% in either training or validation.
#17-model with batch size=32, learning rate = 0.001, dropout = 0.1, 0.2 (between convolutional layers) and 0.35 (between fully-connected/dense layers) achieves training accuracy=  0.9860/98.60%,test accuracy= 0.8735/87.35%,validation accuracy = 0.9333/93.33%. after 19 epochs. (image size = 256,256) (Normalization + resizing + image augmentation)
#18-model with batch size=32, learning rate = 0.001, dropout = 0.1, 0.2, 0.3, 0,4 (between convolutional layers) and 0.5 (between fully-connected/dense layers) achieves training accuracy=  0.9561/95.61%,test accuracy= 0.8701/87.01%,validation accuracy = 0.9385/93.85%. after 24 epochs. (image size = 256,256) (Normalization + resizing + image augmentation)


#plotting the accuracies changing over the epochs (Current trend of the graph is that the accuracy gradually increases with each passing epoch):
import seaborn as sns
accuraciesepochs = model_history.history['accuracy']
epochs = range(0,24)
sns.lineplot(x = epochs, y =accuraciesepochs)
plt.show()


'''
#Showing the overall data prediction for each class:
preds = sequentialmodel.predict(test_data)
predictions = np.argmax(preds, axis=1)
predcounter = collections.Counter(predictions)
print("Predictions:",predictions)
print (predcounter)
#1 is REAL. 0 is FAKE.

actual_labels =  np.array([])
for x, y in test_data:
  actual_labels = np.concatenate([actual_labels, np.argmax(y.numpy(), axis=-1)])

print("Actual labels:" ,actual_labels)
actualcounter =collections.Counter(actual_labels)
print(actualcounter)
#1 is REAL. 0 is FAKE.

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
preds = sequentialmodel.predict(test_generator)
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
sequentialmodel.save('Sequentialmodel9.h5')
#HISTORY OF MODELS SAVED:
#-sequentialmodel1 = model #4 with batch size= 128, learning rate = 0.0001, dropout = 0.2 achieves training accuracy= 0.9467/94.67%  ,test accuracy=  0.7616/76.16%  ,validation accuracy = 0.8905/89.05%. after 20 epochs. (image size = 224, 224)
#-sequentialmodel2 =model #5 with batch size=32, learning rate = 0.001, dropout = 0.2 achieves training accuracy=  0.9040/90.40% ,test accuracy=  0.8416/84.16% ,validation accuracy = 0.8920/89.20%. after 25 epochs. (image size = 224,224)
#-sequentialmodel3 =model #6 with batch size=32, learning rate = 0.001, dropout = 0.2 (between convolutional layers) and 0.5 (between fully-connected/dense layers) achieves training accuracy= 0.9511/95.11%  ,test accuracy=  0.8588/85.88% ,validation accuracy = 0.8912/89.12%. after 20 epochs. (image size = 256,256) (Normalization + resizing)
#-sequentialmodel4 =model #7 with batch size=32, learning rate = 0.001, dropout = 0.2 (between convolutional layers) and 0.5 (between fully-connected/dense layers) achieves training accuracy= 0.9602/96.02% ,test accuracy=  0.8937/89.37% ,validation accuracy = 0.9467/94.67%. after 19 epochs. (image size = 256,256) (Normalization + resizing)
#-sequentialmodel5 =model #8 with batch size=32, learning rate = 0.001, dropout = 0.2 (between convolutional layers) and 0.5 (between fully-connected/dense layers) achieves training accuracy=  0.9618/96.18%,test accuracy= 0.8794/87.94%  ,validation accuracy = 0.9471/94.71%. after 19 epochs. (image size = 256,256) (Normalization + resizing)/higher nodes in dense layers.
#-sequentialmodel6 (first 90%+ overall model!) =model #9 with batch size=32, learning rate = 0.001, dropout = 0.2 (between convolutional layers) and 0.5 (between fully-connected/dense layers) achieves training accuracy=  0.9559/95.59%,test accuracy= 0.9071/90.71% ,validation accuracy = 0.9379/93.79%. after 19 epochs. (image size = 256,256) (Normalization + resizing)/higher nodes in dense layers + extra dense layer.
#-sequentialmodel7 =model #10 with batch size=32, learning rate = 0.001, dropout = 0.2 (between convolutional layers) and 0.2 (between fully-connected/dense layers) achieves training accuracy= 0.9777/97.77% ,test accuracy= 0.8751/87.51% ,validation accuracy = 0.9546/95.46%. after 19 epochs. (image size = 256,256) (Normalization + resizing)/higher nodes in dense layers + extra dense layer.
#-sequentialmodel8 =model #11 with batch size=32, learning rate = 0.001, dropout = 0.2 (between convolutional layers) and 0.5 (between fully-connected/dense layers) achieves training accuracy= 0.9527/95.27% ,test accuracy= 0.8738/87.38%,validation accuracy = 0.9400/94%. after 19 epochs. (image size = 256,256) (Normalization + resizing)/higher nodes in dense layers + extra dense layer.
#-sequentialmodel9 (second 90%+ overall model)=