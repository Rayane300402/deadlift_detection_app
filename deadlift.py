import tkinter as tk
import customtkinter as ctk

import pandas as pd
import numpy as np
import pickle

import mediapipe as mp
import cv2
from PIL import Image, ImageTk

from landmarks import landmarks
import os

window = tk.Tk() # Create a window
window.geometry("480x700") # Set the size of the window
window.title("Bottcamp-3-EX3") # Set the title of the window

#setting labels
classLabel = ctk.CTkLabel(window, height = 40, width =120, font=("Arial", 20), text_color="black", padx=10)
classLabel.place(x=10, y=1) #position
classLabel.configure(text="STAGE")
counterLabel = ctk.CTkLabel(window, height = 40, width =120, font=("Arial", 20), text_color="black", padx=10)
counterLabel.place(x=160, y=1) #position
counterLabel.configure(text="REPS")
probLabel = ctk.CTkLabel(window, height = 40, width =120, font=("Arial", 20), text_color="black", padx=10)
probLabel.place(x=300, y=1) #position
probLabel.configure(text="PROB")
classBox = ctk.CTkLabel(window, height = 40, width =120, font=("Arial", 20), text_color="black", padx=10)
classBox.place(x=10, y=41) #position
classBox.configure(text="0")    
counterBox = ctk.CTkLabel(window, height = 40, width =120, font=("Arial", 20), text_color="black", padx=10)
counterBox.place(x=160, y=41) #position
counterBox.configure(text="0")
probBox = ctk.CTkLabel(window, height = 40, width =120, font=("Arial", 20), text_color="black", padx=10)
probBox.place(x=300, y=41) #position
probBox.configure(text="0")
#End of labels

def reset_cpunter():
    global counter
    counter = 0
    
button = ctk.CTkButton(window, text="RESET" ,height = 40, width =120, font=("Arial", 20), text_color="black", fg_color="grey")
button.place(x=10, y=600) #position

frame = tk.Frame(window, width=480, height=480) #camera frame   
frame.place(x=10, y=90) #position
lmain = tk.Label(frame) #for updating labels
lmain.place(x=0, y=0) #position

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

with open('./deadlift.pkl','rb') as f:
    model = pickle.load(f)

cap = cv2.VideoCapture(0) #capture video from camera 0 cz default
current_stage = '' #na3ref is up or down
counter = 0 #counts number of lifts
bodylang_prop = np.array([0,0])
bodylang_class = ''

def detect():
    global current_stage
    global counter
    global bodylang_prop
    global bodylang_class

    ret, frame = cap.read() #read frame from camera
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) #convert to RGB
    results = pose.process(image) #process image
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(106,13,173), thickness=4, circle_radius = 5),
                              mp_drawing.DrawingSpec(color=(255,102,0), thickness=5, circle_radius = 10)) #draw landmarks
    
    try:
        row = np.array([{res.x, res.y, res.z} for res in results.pose_landmarks.landmark]).flatten().tolist()
        x = pd.DataFrame([row], columns=landmarks)
        bodylang_prop = model.predict_proba(x)[0]
        bodylang_class = model.predict(x)[0]

        if bodylang_class == 'down' and bodylang_prop[bodylang_prop.argmax()] > 0.7:
            current_stage = 'down'
        elif bodylang_class == 'up' and current_stage == 'down' and bodylang_prop[bodylang_prop.argmax()] > 0.7: 
            current_stage = 'up'
            counter += 1

    except Exception as e:
        print(e)
        pass
    
    img = image[:, :460, :] #slice the images
    imgarr = Image.fromarray(img) #convert the image pixels to array
    imgtk = ImageTk.PhotoImage(imgarr) 
    lmain.imgtk = imgtk
    lmain.configure(image=imgtk)
    lmain.after(10, detect) #call the function after 10ms

    counterBox.configure(text=counter)
    probBox.configure(text=bodylang_prop[bodylang_prop.argmax()])
    classBox.configure(text=bodylang_class)

detect()
window.mainloop() #run the window


