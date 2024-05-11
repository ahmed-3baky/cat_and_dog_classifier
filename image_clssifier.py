import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.transform import resize
from skimage.color import rgb2gray
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score 
from sklearn.decomposition import PCA
import tkinter as tk
from tkinter import filedialog, Label, Button
from tkinter.messagebox import showerror
from PIL import Image, ImageTk

# if you want to increase the accuarcy add more images

class ImageClassifier:
    
    def __init__(self, data_folder):
        self.data_folder = data_folder
        self.pca = PCA(n_components=50)
        self.svm_model = SVC(kernel='rbf', gamma="scale")
        self.class_names = self.load_class_names()
        self.train_model()

    def load_images_from_folder(self):
        images, labels = [], []
        for label, class_folder in enumerate(os.listdir(self.data_folder)):
            class_path = os.path.join(self.data_folder, class_folder)
            for filename in os.listdir(class_path):
                img_path = os.path.join(class_path, filename)
                img = imread(img_path)
                img_gray = rgb2gray(img)
                img_resized = resize(img_gray, (100, 100))
                images.append(img_resized)
                images.append(np.fliplr(img_resized))  
                labels.append(label)
                labels.append(label)  
        return np.array(images), np.array(labels)

    def train_model(self):
        images, labels = self.load_images_from_folder()
        X_train, X_test, y_train, self.y_test = train_test_split(images, labels, test_size=0.2, random_state=43)
        X_train_pca = self.pca.fit_transform(X_train.reshape(X_train.shape[0], -1))
        self.X_test_pca = self.pca.transform(X_test.reshape(X_test.shape[0], -1))

        # Use GridSearchCV for hyperparameter tuning
        parameters = {'C': [0.1, 1, 10, 100], 'gamma': ['scale', 'auto', 0.1, 1, 10, 100]}
        grid = GridSearchCV(SVC(kernel='rbf'), parameters, refit=True, verbose=2)
        grid.fit(X_train_pca, y_train)
        self.svm_model = grid.best_estimator_

        self.evaluate_accuracy()

    def evaluate_accuracy(self):
        y_pred = self.svm_model.predict(self.X_test_pca)
        self.accuracy = accuracy_score(self.y_test, y_pred)

    def predict_image(self, image_path):
        img = imread(image_path)
        img_gray = rgb2gray(img)
        img_resized = resize(img_gray, (100, 100))
        img_pca = self.pca.transform(img_resized.flatten().reshape(1, -1))
        prediction = self.svm_model.predict(img_pca)
        return prediction, self.accuracy, img_resized

    def load_class_names(self):
        return [folder for folder in os.listdir(self.data_folder)]

from PIL import Image, ImageTk

class ImageClassifierGUI:
    def __init__(self, win, classifier):
        self.win = win
        self.classifier = classifier
        win.title("Image Classifier")

        # Set window size
        win.geometry("650x650")

        # Load the background image and set it as the background
        bg_image = Image.open("C:\\Users\\Computec\\Desktop\\midnight-owl-robert-farkas.jpg")  
        bg_image = bg_image.resize((650, 650))
        self.bg_photoimg = ImageTk.PhotoImage(bg_image)
        bg_label = Label(win, image=self.bg_photoimg)
        bg_label.place(x=0, y=0)

        # Create a frame to hold the buttons and labels
        frame = tk.Frame(win)
        frame.place(relx=0.5, rely=0.5, anchor='center')

        self.label = Label(frame, text="Select an image:" , disabledforeground="")
        self.label.pack()

        self.select_button = Button(frame, text="Select Image", command=self.select_image , height= 3 , width=10)
        self.select_button.pack()

        self.predict_button = Button(frame, text="Predict", command=self.predict , height= 3 , width=10)
        self.predict_button.pack()

        self.prediction_label = Label(frame, text="")
        self.prediction_label.pack()



    def select_image(self):
            self.image_path =  filedialog.askopenfilename()
            self.label.config(text="Selected image: " + os.path.basename(self.image_path))

    def predict(self):

                prediction, accuracy, img_resized = self.classifier.predict_image(self.image_path)
                predicted_class = self.classifier.class_names[prediction[0]]
                self.prediction_label.config(text=f"Predicted class: {predicted_class}, Accuracy: {accuracy:.2f}")
                # Display the grayscale image along with the predicted class
                plt.imshow(img_resized, cmap='gray')
                plt.title(f"Predicted class: {predicted_class}, Accuracy: {accuracy:.2f}")
                plt.axis('off')
                plt.show()



def execute_clssifier():
    root = tk.Tk()
    classifier = ImageClassifier(data_folder='C:\\Users\\Computec\\Desktop\\image classifier\\fina\\images')
    app = ImageClassifierGUI(root, classifier)
    root.mainloop()
