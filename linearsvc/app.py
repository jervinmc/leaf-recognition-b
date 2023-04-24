# Importing required libraries
import cv2
import numpy as np
import os
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Defining the path of the dataset
dataset_path = "test1"

# Defining the image size for processing
image_size = (64, 64)

# Creating an empty list to store the image data and labels
data = []
labels = []

# Looping through each subdirectory of the dataset
for sub_folder in os.listdir(dataset_path):
    sub_folder_path = os.path.join(dataset_path, sub_folder)

    # Looping through each file in the subdirectory
    for file_name in os.listdir(sub_folder_path):
        file_path = os.path.join(sub_folder_path, file_name)

        # Reading the image and resizing it
        image = cv2.imread(file_path)
        image = cv2.resize(image, image_size)

        # Flattening the image data and adding it to the list
        image_data = np.ravel(image)
        data.append(image_data)

        # Adding the label of the subdirectory to the labels list
        labels.append(sub_folder)

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Creating a Linear SVC object and training it on the training data
svc = LinearSVC(random_state=42)
svc.fit(X_train, y_train)

# Predicting the labels of the testing data
y_pred = svc.predict(X_test)

# Calculating the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)