import os
import cv2
import numpy as np
import tensorflow as tf
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
        img_path = os.path.join(folder_path, img_name)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (img_size, img_size))
        
        X.append(img)
        y.append(categories.index(category))

X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train = X_train.reshape(X_train.shape[0], img_size, img_size, 1)
X_test = X_test.reshape(X_test.shape[0], img_size, img_size, 1)

y_train = tf.keras.utils.to_categorical(y_train, len(categories))
y_test = tf.keras.utils.to_categorical(y_test, len(categories))

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(img_size,img_size,1)),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(len(categories), activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

loss, accuracy = model.evaluate(X_test, y_test)
print("Test accuracy:", accuracy)

img_path ='datasets/calamondin/Calamondin (4).jpg'
img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.resize(img, (img_size, img_size))
img = np.expand_dims(img, axis=0)
img = np.expand_dims(img, axis=-1)
prediction = model.predict(img)
category_idx = np.argmax(prediction)
category = categories[category_idx]

print("The image belongs to category:", category)