import os
import cv2
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

data_dir = "datasets"
categories = ["calamondin", "clementine", "dayap","lemon","orange","tangerine"]
img_size = 100

X = []
y = []

for category in categories:
    folder_path = os.path.join(data_dir, category)
    img_names = os.listdir(folder_path)
    
    for img_name in img_names:
        print(category)
        img_path = os.path.join(folder_path, img_name)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (img_size, img_size))
        
        X.append(img)
        y.append(categories.index(category))

X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)
nbins = 256
X_train_counts = np.apply_along_axis(lambda x: np.histogram(x, bins=nbins)[0], axis=1, arr=X_train_flat)
X_test_counts = np.apply_along_axis(lambda x: np.histogram(x, bins=nbins)[0], axis=1, arr=X_test_flat)
clf = MultinomialNB()
clf.fit(X_train_counts, y_train)
y_pred = clf.predict(X_test_counts)
print("Accuracy of Model:", accuracy_score(y_test, y_pred))

def detect_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (img_size, img_size))
    img_flat = img.reshape(1, -1)
    img_counts = np.apply_along_axis(lambda x: np.histogram(x, bins=nbins)[0], axis=1, arr=img_flat)
    category = categories[clf.predict(img_counts)[0]]
    print("Image detected as:", category)

detect_image('datasets/orange/Orange (1).jpg')