"""
Trains the CNN used by the web app to classify celestial body images and
saves the result as CNN_Model.h5 in this directory.

This is a script version of train.ipynb with the hardcoded
"/Users/kevinmathew/Documents/UoL CNN/..." paths replaced by paths relative
to this file, so it can run on any machine the repo is checked out on.

Usage:
    python3 train_model.py
"""
import os

from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAINING_DIR = os.path.join(BASE_DIR, "training_data")
VALIDATION_DIR = os.path.join(BASE_DIR, "test_data")
BEST_MODEL_FILE = os.path.join(BASE_DIR, "CNN_best_weights.h5")
MODEL_FILE = os.path.join(BASE_DIR, "CNN_Model.h5")

IMG_WIDTH = IMG_HEIGHT = 256
BATCH_SIZE = 16
EPOCHS = 60


def main():
    train_datagen = ImageDataGenerator(
        rescale=1 / 255.0,
        rotation_range=30,
        zoom_range=0.4,
        horizontal_flip=True,
    )
    train_generator = train_datagen.flow_from_directory(
        TRAINING_DIR,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        target_size=(IMG_HEIGHT, IMG_WIDTH),
    )

    validation_datagen = ImageDataGenerator(rescale=1 / 255.0)
    validation_generator = validation_datagen.flow_from_directory(
        VALIDATION_DIR,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        target_size=(IMG_HEIGHT, IMG_WIDTH),
    )

    # class_indices maps folder name -> index Keras assigns it; this must line
    # up with CLASS_NAMES in app/views.py since the generator sorts classes
    # alphabetically, which already matches that list.
    print("Class indices:", train_generator.class_indices)

    callbacks = EarlyStopping(monitor="val_loss", patience=5, verbose=1, mode="auto")
    best_model = ModelCheckpoint(
        BEST_MODEL_FILE, monitor="val_accuracy", verbose=1, save_best_only=True
    )

    model = Sequential(
        [
            Conv2D(16, (3, 3), activation="relu", input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
            MaxPooling2D(2, 2),
            Conv2D(32, (3, 3), activation="relu"),
            MaxPooling2D(2, 2),
            Conv2D(64, (3, 3), activation="relu"),
            Conv2D(64, (3, 3), activation="relu"),
            MaxPooling2D(2, 2),
            Conv2D(128, (3, 3), activation="relu"),
            Conv2D(128, (3, 3), activation="relu"),
            MaxPooling2D(2, 2),
            Flatten(),
            Dense(512, activation="relu"),
            Dense(512, activation="relu"),
            Dense(12, activation="softmax"),
        ]
    )
    model.summary()

    model.compile(optimizer="Adam", loss="categorical_crossentropy", metrics=["accuracy"])

    model.fit(
        train_generator,
        epochs=EPOCHS,
        verbose=1,
        validation_data=validation_generator,
        callbacks=[callbacks, best_model],
    )

    model.save(MODEL_FILE)
    print(f"Saved trained model to {MODEL_FILE}")


if __name__ == "__main__":
    main()
