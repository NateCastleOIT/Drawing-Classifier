import pickle
import os.path

import numpy as np
import PIL
import PIL.Image
import PIL.ImageDraw
import cv2 as cv

# Do no import all
from tkinter import *
import tkinter.messagebox
from tkinter import simpledialog

from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklean.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier


class DrawingClassifier:
    
    def __init__(self)
    
        self.class1, self.class2, self.class3  = None, None, None
        self.class1_counter, self.class2_counter, self.class3_counter = None, None, None
        self.clf = None
        self.proj_name = None
        self.root = None
        self.image1 = None
        
        self.status_label = None
        self.canvas = None
        self.draw = None
        
        self.brush_weight = 15
        
        self.classes_prompt()
        self.init_gui()
        
        
    def classes_prompt(self):
        msg = Tk()
        msg.withdraw()
        
        # set the msg to the parent of the prompt
        self.proj_name = simpledialog.askstring("Project Name", "Please enter you project name down below!", parent=msg)
        
        # Check if the file already exists
        if os.path.exists(self.proj_name):
            with open(f"{self.proj_name}/{self.proj_name}_data.pickle", "rb") as f:
                data = pickle.load(f)
            self.class1 = data['c1']
            self.class2 = data['c2']
            self.class3 = data['c3']
            
            self.class1_counter = data['c1c']
            self.class2_counter = data['c2c']
            self.class3_counter = data['c3c']
            
            self.clf = data['clf']
            self.proj_name = data['pname']
            
        else:
            self.class1 = simpledialog.askstring["Class 1", "What is the first clas called?", parent=msg]
            self.class2 = simpledialog.askstring["Class 2", "What is the second clas called?", parent=msg]
            self.class3 = simpledialog.askstring["Class 3", "What is the third clas called?", parent=msg]
            
            self.class1_counter = 1
            self.class2_counter = 1
            self.class3_counter = 1
            
            self.clf = LinearSVC()
            
            os.mkdir(self.proj_name)
            os.chdir(self.proj_name)
            os.mkdir(self.class1)
            os.mkdir(self.class2)
            os.mkdir(self.class3)
            os.chdir(..)
    
    def init_gui(self):
        WIDTH = 500
        HEIGHT = 500
        WHITE = (255,255,255)
        
        self.root = Tk()
        self.root.title(f"Nate Castle Drawing Classifier Alpha v0.1 - {self.proj_name}")
        
        self.canvas = Canvas(self.root, width=WIDTH-10, height=HEIGHT-10, bg="white")
        self.pack(expand=YES, fill=BOTH)
        # Binds button 1 (mouse 1) being clicked along with motion to the self.paint method
        self.canvas.bind("<B1-Motion>", self.paint)
        
        self.image1 = PIL.Image.new("RGB", (WIDTH, HEIGHT), WHITE)
        self.draw = PIL.ImageDraw.Draw(self.image1)
        
        btn_frame = tkinter.Frame(self.root)
        btn_frame.pack(fill=X, side=BOTTOM)
        btn_frame.columnconfigure(0, weight=1)
        btn_frame.columnconfigure(1, weight=1)
        btn_frame.columnconfigure(2, weight=1)
        
        # buttons to save training data
        class1_btn = Button(btn_frame, text=self.class1, command=lambda: self.save(1))
        class1_btn.grid(row=0, column=0, sticky=W + E)
        
        class2_btn = Button(btn_frame, text=self.class2, command=lambda: self.save(2))
        class2_btn.grid(row=0, column=1, sticky=W + E)
        
        class3_btn = Button(btn_frame, text=self.class2, command=lambda: self.save(3))
        class3_btn.grid(row=0, column=2, sticky=W + E)
        
        bm_btn = Button(btn_frame, text="Brush-", command=self.brushminus)
        bm_btn.grid(row=1, column=0, sticky=W + E)
        
        clr_btn = Button(btn_frame, text="Clear", command=self.clear)
        clr_btn.grid(row=1, column=1, sticky=W + E)
        
        bp_btn = Button(btn_frame, text="Brush+", command=self.brushplus)
        bp_btn.grid(row=1, column=2, sticky=W + E)
        
        train_btn = Button(btn_frame, text="Train Model", command=self.train_model)
        train_btn.grid(row=2, column=0, sticky=W + E)
        
        save_btn = Button(btn_frame, text="Save Model", command=self.save_model)
        save_btn.grid(row=2, column=1, sticky=W + E)
        
        load_btn = Button(btn_frame, text="Load Model", command=self.load_model)
        load_btn.grid(row=2, column=2, sticky=W + E)
        
        change_btn = Button(btn_frame, text="Change Model", command=self.rotate_model)
        change_btn.grid(row=3, column=0, sticky=W + E)
        
        predict_btn = Button(btn_frame, text="Predict", command=self.predict)
        predict_btn.grid(row=3, column=1, sticky=W + E)
        
        save_all_btn = Button(btn_frame, text="Save All", command=self.save_all)
        save_all_btn.grid(row=3, column=2, sticky=W + E)
        
        self.status_label = Label(btn_frame, text=f"Current Model: {type(self.clf).__name__}")
        self.status_label.config(font=("Arial", 10))
        self.status_label.grid(row=4, column=1, sticky=W + E)
        
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.mainloop()
        
    def paint(self, event):
        pass
    
    def save(self, class_num):
        pass
    
    def brushminus(self):
        pass
    
    def brushplus(self):
        pass
    
    def clear(self):
        pass
    
    def train_model(self):
        pass
    
    def save_model(self):
        pass
    
    def load_model(self):
        pass
    
    def rotate_model(self):
        pass
    
    def predict(self):
        pass
    
    def save_all(self):
        pass