import os
import cv2
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix

from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint
)

# ------------------------
# Settings
# ------------------------

DATASET_PATH = "dataset"
IMG_SIZE = 224
NUM_CLASSES = 5
BATCH_SIZE = 16

classes = [
    "im_Dyskeratotic",
    "im_Koilocytotic",
    "im_Metaplastic",
    "im_Parabasal",
    "im_Superficial-Intermediate"
]

# ------------------------
# Load Dataset
# ------------------------

X = []
y = []

print("\nLoading Dataset...")

for label, class_name in enumerate(classes):

    class_path = os.path.join(DATASET_PATH, class_name)

    if not os.path.exists(class_path):
        print("Folder not found:", class_path)
        continue

    count = 0

    for file in os.listdir(class_path):

        if file.lower().endswith(".bmp"):

            image_path = os.path.join(class_path, file)

            img = cv2.imread(image_path)

            if img is None:
                continue

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

            X.append(img)
            y.append(label)

            count += 1

    print(class_name, ":", count, "images")

X = np.array(X, dtype="float32")
y = np.array(y)

print("\nTotal Images:", len(X))
print("Image Shape:", X.shape)

# ------------------------
# Train / Validation / Test
# ------------------------

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)

print("\nDataset Split")
print("------------------------")
print("Training Images   :", len(X_train))
print("Validation Images :", len(X_val))
print("Testing Images    :", len(X_test))

# ------------------------
# Data Augmentation
# ------------------------

train_datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.10,
    height_shift_range=0.10,
    zoom_range=0.15,
    horizontal_flip=True,
    vertical_flip=True
)

train_generator = train_datagen.flow(
    X_train,
    y_train,
    batch_size=BATCH_SIZE,
    shuffle=True
)

# ------------------------
# Class Weights
# ------------------------

class_weights_array = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train),
    y=y_train
)

class_weights = dict(
    enumerate(class_weights_array)
)

print("\nClass Weights:")
print(class_weights)

# ------------------------
# EfficientNetB0
# ------------------------

print("\nCreating EfficientNetB0 Model...")

base_model = EfficientNetB0(
    weights="imagenet",
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

# Initially freeze base model
base_model.trainable = False

x = base_model.output

x = GlobalAveragePooling2D()(x)

x = Dropout(0.4)(x)

x = Dense(
    128,
    activation="relu"
)(x)

x = Dropout(0.3)(x)

output = Dense(
    NUM_CLASSES,
    activation="softmax"
)(x)

model = Model(
    inputs=base_model.input,
    outputs=output
)

# ------------------------
# First Training Stage
# ------------------------

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("\nStage 1 Training...")

callbacks = [

    ModelCheckpoint(
        "models/best_cervical_model.keras",
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1
    ),

    EarlyStopping(
        monitor="val_accuracy",
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),

    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.3,
        patience=2,
        min_lr=0.000001,
        verbose=1
    )
]

history1 = model.fit(
    train_generator,
    validation_data=(X_val, y_val),
    epochs=15,
    class_weight=class_weights,
    callbacks=callbacks
)

# ------------------------
# Fine Tuning
# ------------------------

print("\nStarting Fine-Tuning...")

base_model.trainable = True

# Freeze earlier layers
for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.00001
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

history2 = model.fit(
    train_generator,
    validation_data=(X_val, y_val),
    epochs=20,
    class_weight=class_weights,
    callbacks=callbacks
)

# ------------------------
# Final Evaluation
# ------------------------

print("\nEvaluating Model...")

test_loss, test_accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=1
)

print("\n==============================")
print("FINAL TEST ACCURACY :", test_accuracy * 100, "%")
print("==============================")

# ------------------------
# Predictions
# ------------------------

predictions = model.predict(X_test)

predicted_classes = np.argmax(
    predictions,
    axis=1
)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        predicted_classes,
        target_names=[
            "Dyskeratotic",
            "Koilocytotic",
            "Metaplastic",
            "Parabasal",
            "Superficial-Intermediate"
        ]
    )
)

print("\nConfusion Matrix:")
print(
    confusion_matrix(
        y_test,
        predicted_classes
    )
)

# ------------------------
# Save Final Model
# ------------------------

os.makedirs("models", exist_ok=True)

model.save(
    "models/cervical_model.keras"
)

print("\nModel Saved Successfully!")