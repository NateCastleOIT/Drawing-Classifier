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
from tkinter import filedialog
import tqdm

from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from skimage.transform import rotate, rescale


class DrawingClassifier:
    
    def __init__(self):
    
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
            self.class1 = simpledialog.askstring("Class 1", "What is the first clas called?", parent=msg)
            self.class2 = simpledialog.askstring("Class 2", "What is the second clas called?", parent=msg)
            self.class3 = simpledialog.askstring("Class 3", "What is the third clas called?", parent=msg)
            
            self.class1_counter = 1
            self.class2_counter = 1
            self.class3_counter = 1
            
            self.clf = LinearSVC()
            
            os.mkdir(self.proj_name)
            os.chdir(self.proj_name)
            os.mkdir(self.class1)
            os.mkdir(self.class2)
            os.mkdir(self.class3)
            os.chdir("..")
    
    def init_gui(self):
        WIDTH = 500
        HEIGHT = 500
        WHITE = (255,255,255)
        
        self.root = Tk()
        self.root.title(f"Nate Castle Drawing Classifier Alpha v0.1 - {self.proj_name}")
        
        self.canvas = Canvas(self.root, width=WIDTH-10, height=HEIGHT-10, bg="white")
        self.canvas.pack(expand=YES, fill=BOTH)
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
        
        class3_btn = Button(btn_frame, text=self.class3, command=lambda: self.save(3))
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
        x1, y1 = (event.x - 1), (event.y - 1)
        x2, y2 = (event.x + 1), (event.y + 1)
        
        self.canvas.create_rectangle(x1, y1, x2, y2, fill="black", width=self.brush_weight)
        self.draw.rectangle([x1, y2, x2 + self.brush_weight, y2 + self.brush_weight], fill="black", width=self.brush_weight)
    
    def save(self, class_num):
        self.image1.save("temp.png")
        img = PIL.Image.open("temp.png")
        img.thumbnail((50,50), PIL.Image.LANCZOS)
        
        if class_num == 1:
            img.save(f"{self.proj_name}/{self.class1}/{self.class1_counter}.png", "PNG")
            self.class1_counter += 1
        elif class_num == 2:
            img.save(f"{self.proj_name}/{self.class2}/{self.class2_counter}.png", "PNG")
            self.class2_counter += 1
        elif class_num == 3:
            img.save(f"{self.proj_name}/{self.class3}/{self.class3_counter}.png", "PNG")
            self.class3_counter += 1
            
        self.clear()
    
    def brushminus(self):
        if self.brush_weight > 1:
            self.brush_weight -= 1
    
    def brushplus(self):
        self.brush_weight += 1
    
    def clear(self):
        # Clear the canvas data, set the image to pure white
        self.canvas.delete("all")
        self.draw.rectangle([0,0,1000,1000], fill="white")
    
    def augment_image(self, class_num, x):
        angles = list(range(0, 360, 15)) # Generate angles at 15-degree intervals
        augmented_images = []
        
        self.image1.save("temp.png")
        img = PIL.Image.open("temp.png")
        img.thumbnail((50,50), PIL.Image.LANCZOS)
        
        if class_num == 1:
            # Load the original image outside the loop
            og_img = cv.imread(f"{self.proj_name}/{self.class1}/{x}.png")[:, :, 0]
            og_img = np.array(og_img, dtype=np.uint8)
            og_img.reshape(2500)
            
            for ang in angles:
                #Rotate image
                rotated_img = rotate(og_img, angle=ang)
                rotated_pil = PIL.Image.fromarray(rotated_img, mode='L')
                
                #Save the rotated image
                rotated_pil.save(f"{self.proj_name}/{self.class1}/{x}_{ang}.png", format="PNG")
                augmented_images.append(rotated_pil)
                
                #Flip the image
                flipped_img = np.flip(rotated_img, axis=1)
                flipped_pil = PIL.Image.fromarray(flipped_img, mode='L')
                
                # Save flipped image
                flipped_img.save(f"{self.proj_name}/{self.class1}/{x}_{ang}_flipped.png", format="PNG")
                augmented_images.append(flipped_pil)
                
        if class_num == 2:    
            for ang in angles:
                img.save(f"{self.proj_name}/{self.class2}/{x}_{ang}.png", "PNG")
                rotated_img = cv.imread(f"{self.proj_name}/{self.class2}/{x}.png")[:, :, 0]
                rotated_img.reshape(2500)
                rotated_img = rotate(img, angle=ang)
                augmented_images.append(rotated_img)
                
                flipped_img = img.save(f"{self.proj_name}/{self.class2}/{x}_{ang}_flipped.png")
                flipped_img = cv.imread(f"{self.proj_name}/{self.class2}/{x}_{ang}_flipped.png")[:, :, 0]
                flipped_img.reshape(2500)
                flipped_img = np.flip(rotated_img, axis=1)
                augmented_images.append(flipped_img) # Flip image and save
                
        if class_num == 3:        
            for ang in angles:
                img.save(f"{self.proj_name}/{self.class3}/{x}_{ang}.png", "PNG")
                rotated_img = cv.imread(f"{self.proj_name}/{self.class3}/{x}_{ang}_flipped.png")[:, :, 0]
                rotated_img.reshape(2500)
                rotated_img = rotate(img, angle=ang)
                augmented_images.append(rotated_img)
                
                flipped_img = img.save(f"{self.proj_name}/{self.class3}/{x}_{ang}_flipped.png")
                flipped_img = cv.imread(f"{self.proj_name}/{self.class3}/{x}_{ang}_flipped.png")[:, :, 0]
                flipped_img.reshape(2500)
                flipped_img = np.flip(rotated_img, axis=1)
                augmented_images.append(flipped_img) # Flip image and save
            
        return augmented_images
    
    # TODO: Adding data processing for image resizing
    # TODO: GUI Enhancments
    # TODO: Error Handling
    def train_model(self):
        img_list = []
        class_list = []
        
        print(f"Unique classes: {np.unique(class_list)}")
        
        try:
            
            for x in range(1, self.class1_counter):
                img = cv.imread(f"{self.proj_name}/{self.class1}/{x}.png")[:, :, 0]
                img.reshape(2500)
                
                # Data Augmentation - Rotate and Rescale: 15-degree + flipped
                augmented_imgs = self.augment_image(1, x)
                
                # Add new augmented images
                img_list.extend(augmented_imgs)
                class_list.extend([1] * len(augmented_imgs))
            print(f"Unique classes: {np.unique(class_list)}")
                
            for x in range(1, self.class2_counter):
                img = cv.imread(f"{self.proj_name}/{self.class2}/{x}.png")[:, :, 0]
                img.reshape(2500)
                
                # Data Augmentation - Rotate and Rescale: 15-degree + flipped
                augmented_imgs = self.augment_image(1, x)
                
                # Add new augmented images
                img_list.extend(augmented_imgs)
                class_list.extend([1] * len(augmented_imgs))
            print(f"Unique classes: {np.unique(class_list)}")
            
            for x in range(1, self.class3_counter):
                # Data Augmentation - Rotate and Rescale: 15-degree + flipped
                augmented_imgs = self.augment_image(1, x)
                
                # Add new augmented images
                img_list.extend(augmented_imgs)
                class_list.extend([1] * len(augmented_imgs))
            print(f"Unique classes: {np.unique(class_list)}")
                
            #assemble the np array at the end
            img_list = np.array(img_list).reshape(-1, 2500)
            class_list = np.array(class_list)
            print(f"Unique classes: {np.unique(class_list)}")
            
            self.clf.fit(img_list, class_list)
            
            tkinter.messagebox.showinfo("Nate Castle Drawing Classifier", "Model successsfully trained!", parent=self.root)
            
        except Exception as e:
            print(f"An error occurred: {str(e)}")
    
    def save_model(self):
        file_path = filedialog.asksaveasfilename(defaultextension="pickle")
        with open(file_path, "wb") as f:
            pickle.dump(self.clf, f)
        tkinter.messagebox.showinfo("Nate Castle Drawing Classifier", "Model successsfully saved!", parent=self.root)
    
    def load_model(self):
        file_path = filedialog.askopenfilename()
        with open(file_path, "rb") as f:
            self.clf = pickle.load(f)
        tkinter.messagebox.showinfo("Nate Castle Drawing Classifier", "Model successsfully loaded!", parent=self.root)
    
    def rotate_model(self):
        if isinstance(self.clf, LinearSVC):
            self.clf = KNeighborsClassifier()
        elif isinstance(self.clf, KNeighborsClassifier):
            self.clf = LogisticRegression()
        elif isinstance(self.clf, LogisticRegression):
            self.clf = DecisionTreeClassifier()
        elif isinstance(self.clf, DecisionTreeClassifier):
            self.clf = RandomForestClassifier()
        elif isinstance(self.clf, RandomForestClassifier):
            self.clf = GaussianNB()
        elif isinstance(self.clf, GaussianNB):
            self.clf = LinearSVC()
            
        self.status_label.config(text=f"Current Model: {type(self.clf).__name__}")
            
    
    def predict(self):
        self.image1.save("temp.png")
        img = PIL.Image.open("temp.png")
        img.thumbnail((50,50), PIL.Image.LANCZOS)
        img.save("predictshape.png", "PNG")
        
        img = cv.imread("predictshape.png")[:, :, 0]
        img = img.reshape(2500)
        prediction = self.clf.predict([img])
        if prediction[0] == 1:
            tkinter.messagebox.showinfo("Nate Castle Drawing Classifier", f"The drawing is probably a {self.class1}", parent=self.root)
        elif prediction[0] == 2:
            tkinter.messagebox.showinfo("Nate Castle Drawing Classifier", f"The drawing is probably a {self.class2}", parent=self.root)
        if prediction[0] == 3:
            tkinter.messagebox.showinfo("Nate Castle Drawing Classifier", f"The drawing is probably a {self.class3}", parent=self.root)
        
    def save_all(self):
        data = {"c1": self.class1, "c2": self.class2, "c3": self.class3,
                "c1c": self.class1_counter, "c2c": self.class2_counter, "c3c": self.class3_counter,
                "clf": self.clf, "pname": self.proj_name}
        with open(f"{self.proj_name}/{self.proj_name}_data.pickle", "wb") as f:
            pickle.dump(data, f)
        tkinter.messagebox.showinfo("Nate Castle Drawing Classifier", "Project successfully saved!", parent=self.root)
        
    def on_closing(self):
        answer = tkinter.messagebox.askyesnocancel("Quit?", "Do you want to save you work?", parent=self.root)
        if answer:
            self.save_all()
        self.root.destroy()
        exit()
    
DrawingClassifier()