import os
import numpy as np
import cv2
import tensorflow
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import layers
from tensorflow.keras.models import Model

# Define image size and batch size
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

# Directory paths
train_dir = r'/Users/nehar/Onedrive/Desktop/NitteHack/trainData'

# Data augmentation and loading
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,  # Use 20% for validation
    horizontal_flip=True,
    zoom_range=0.2,
    shear_range=0.2 

)

# Load the training and validation sets
train_generator = datagen.flow_from_directory(
    train_dir,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

validation_generator = datagen.flow_from_directory(
    train_dir,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

# Check the class indices
print("Class mapping:", train_generator.class_indices)

from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

# Define the ResNet50 backboneinput_tensor = Input(shape=(224, 224, 3))
input_tensor = Input(shape=(224, 224, 3))
backbone = ResNet50(weights='imagenet', include_top=False, input_tensor=input_tensor)

from tensorflow.keras.layers import Flatten

# Load ResNet50 model without the top classification layers (include_top=False)
backbone = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Add classification layers on top of ResNet50
x = Flatten()(backbone.output)
x = Dense(128, activation='relu')(x)
output = Dense(4, activation='softmax')(x)  # Assuming 4 classes

# Create the final model
model = Model(inputs=backbone.input, outputs=output)

# Print the model summary
model.summary()

# Build the model
model = Model(inputs=backbone.input, outputs=output)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Initialize ImageDataGenerator for both training and validation
train_datagen = ImageDataGenerator(
    rescale=1./255,        # Normalize pixel values
    shear_range=0.2,       # Randomly apply shearing transformations
    zoom_range=0.2,        # Randomly zoom in on images
    horizontal_flip=True   # Randomly flip images horizontally
)

val_datagen = ImageDataGenerator(rescale=1./255)  # Only rescale for validation

# Now train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=10,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    verbose=1
)

import keras

# Save the model
model.save('retinanet_resnet50_model.h5')

# Load the trained model
model = keras.models.load_model('retinanet_resnet50_model.h5')

model.save(r'\\Users\\nehar\\OneDrive\\Desktop\\NITTE HACK\\retinanet_resnet50_model.h5')

