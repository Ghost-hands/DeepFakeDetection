
import time
import tkinter as tk
from tkinter import *
from tkinter import messagebox
from PIL import ImageTk, Image
from tkinter.filedialog import asksaveasfilename, askopenfile
import cv2
import os
from tensorflow.keras.models import load_model
import tensorflow as tf
import numpy as np
import collections

from flask import Flask, render_template, request, redirect, session, url_for
from sqlite3 import *
from flask_mail import Mail, Message
from random import randrange
import pickle



window = tk.Tk()
window.title('Deepfake detection')
window.geometry("1000x1000")
window.resizable(True, True)

#Function to normalize the user inputs be they images or frames extracted from videos:
def normalize_image(image):
  image = tf.cast(image, tf.float32) / 255.0
  return image

IMG_SIZE = (256, 256) #The input shape for model 9 from table 2 in the final report.
batch_size = 1

#The function for detecting whether a single image chosen by the user is fake or real.
def detectimg():
    file_path = askopenfile(mode='r', filetypes=[ ("Image Files","*.png *.jpg *.jpeg")])
    if file_path is not None:
        pass
    #img = ImageTk.PhotoImage(Image.open(file_path.name))
    img = cv2.imread(file_path.name)
    img = cv2.resize(img, (256, 256))
    img = tf.keras.utils.img_to_array(img)
    img = img/255
    img = img.reshape(1,256,256,3)
    model = load_model('Sequentialmodel6.h5')
    prediction = np.argmax(model.predict(img))
    print(prediction)
    if prediction == 1:
        messagebox.showinfo("result of detection", "Image is REAL!")

    else:
        messagebox.showinfo("result of detection", "Image is FAKE!")


#The function to detect whether a video chosen by the user is fake or real.
def detectvid():
    file = askopenfile(mode='r', filetypes=[('Video Files', '.mp4')])
    if file is not None:
        pass
    cam = cv2.VideoCapture(file.name)
    try:
        if not os.path.exists('VideoFrames'):
            os.makedirs('VideoFrames') #Creating a directory to store the extracted frames from the video in.
    except OSError:
        print('Error: Creating directory of VideoFrames')

    currentframe = 0

    while (True):

        # reading from frame
        ret, frame = cam.read()

        if ret:
            # if video is still left continue creating images
            name = './VideoFrames/frame' + str(currentframe) + '.jpg'
            print('Creating...' + name)
            # writing the extracted images
            cv2.imwrite(name, frame)

            # increasing counter so that it will
            # show how many frames are created
            currentframe += 1
        else:
            totalframes = currentframe
            print ("total frames extracted from video = ", totalframes)
            break

    cam.release()
    cv2.destroyAllWindows()
    test_dir = '.\\VideoFrames' #treating th directory as if it was a test directory/ test dataset.
    test_data = tf.keras.utils.image_dataset_from_directory(
        directory=test_dir,
        color_mode='rgb',
        labels= None,
        image_size= IMG_SIZE,
        batch_size = batch_size,
        shuffle= True)
    test_data = test_data.map(normalize_image)
    model = load_model('Sequentialmodel6.h5') #Loading model 9 from table 2 in the final report. The model with the highest performance.
    preds = model.predict(test_data)
    predictions = np.argmax(preds, axis=1)
    print(predictions)
    predcounter = collections.Counter(predictions)
    print ("Fake:",predcounter.get(0))#Counting the amount of the frames that the model predicted as Fake (0)
    print ("Real:", predcounter.get(1))#Counting the amount of the frames that the model predicted as Real (1)

    currentframe1 = 0

    #Deleting the directory where the frames were stored to reduce the inconvenience for the user.
    while (totalframes != currentframe1):
        name1 = './VideoFrames/frame' + str(currentframe1) + '.jpg'
        try:
            os.remove(name1)
        except:
            pass
            currentframe1 += 1
        if totalframes == currentframe1:
            if os.path.isdir("./VideoFrames"):
                os.rmdir("./VideoFrames")

    if predcounter.get(0) != None and predcounter.get(1) != None:
        if predcounter.get(0) > predcounter.get(1):
            messagebox.showinfo("result of detection", "the video is FAKE!")
        elif predcounter.get(1) >= predcounter.get(0):
            messagebox.showinfo("result of detection", "the video is REAL!")
    elif predcounter.get(0) != None and predcounter.get(1) == None:
        messagebox.showinfo("result of detection", "the video is FAKE!")
    elif predcounter.get(0) == None and predcounter.get(1) != None:
        messagebox.showinfo("result of detection", "the video is REAL!")




btn = Button(window, text='Detect deepfake in images.', bd='5', command=detectimg)
btn2 = Button(window, text='Detect deepfake in videos.', bd='5', command=detectvid)

btn.pack(side= tk.TOP)
btn2.pack(side = tk.TOP)





window.mainloop()


