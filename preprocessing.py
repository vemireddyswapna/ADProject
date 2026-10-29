import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

# Dataset path
DATASET_PATH = "dataset"

# Image size
IMG_SIZE = 224

# Class names
classes = [
    "im_Dyskeratotic",
    "im_Koilocytotic",
    "im_Metaplastic",
    "im_Parabasal",
    "im_Superficial-Intermediate"
]

X = []
y = []

print("Loading Images...")

for label, class_name in enumerate(classes):

    class_path = os.path.join(DATASET_PATH, class_name)

    for file in os.listdir(class_path):

        if file.lower().endswith(".bmp"):

            image_path = os.path.join(class_path, file)

            image = cv2.imread(image_path)

            if image is None:
                continue

            image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))

            X.append(image)

            y.append(label)

print("Images Loaded Successfully")

X = np.array(X, dtype="float32") / 255.0
y = np.array(y)

print("Total Images :", len(X))

print("Shape :", X.shape)

print("Labels :", y.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining Images :", len(X_train))
print("Testing Images :", len(X_test))