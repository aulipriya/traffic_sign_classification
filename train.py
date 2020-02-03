# Imports
import tensorflow as tf
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


def load_data(directory):
    directories = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
    labels = []
    images = []
    print(directories)
    for d in directories:
        subdir = os.path.join(directory, d)
        print(subdir)
        filenames = [ os.path.join(subdir, file_name) for file_name in os.listdir(subdir) if file_name.endswith('.ppm')]
        for file_name in filenames:
            images.append(cv2.imread(file_name))
            labels.append(int(d))
    return images, labels


Root = '/data/Belgium_TSC/'
train_directory = os.path.join(Root, 'Training')
images, labels = load_data(train_directory)
images = np.array(images)
labels = np.array(labels)

images28 = [cv2.resize(image, (28, 28), interpolation = cv2.INTER_AREA) for image in images]
images28 = [cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in images28]

model = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28,1)),
  tf.keras.layers.MaxPooling2D((2, 2)),
  tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
  tf.keras.layers.MaxPooling2D((2, 2)),
  tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(62, activation='softmax')
])
model.summary()
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
images28 = np.array(images28)
images_extend = np.expand_dims(images28, -1)
model.fit(images_extend, np.array(labels), epochs=100)