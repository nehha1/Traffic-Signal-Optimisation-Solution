import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load the pre-trained ResNet50 model from resnet50.py
MODEL_PATH = 'retinanet_resnet50_model.h5'
model = load_model(MODEL_PATH)

# Define image size and the class labels from the trained model
IMAGE_SIZE = (224, 224)
class_labels = ['Sparse', 'Dense', 'Fire', 'Accident']  # Adjust based on your training

# Function to preprocess the image (resize and normalize)
def preprocess_image(image):
    img_resized = cv2.resize(image, IMAGE_SIZE)
    img_array = np.expand_dims(img_resized, axis=0) / 255.0  # Normalize
    return img_array

# Function to capture images from a camera
def capture_image(camera_index):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"Error: Could not open camera {camera_index}")
        return None
    
    ret, frame = cap.read()
    if not ret:
        print(f"Error: Could not read frame from camera {camera_index}")
        return None
    
    cap.release()
    return frame

# Function to classify the image using the ResNet50 model
def classify_image(image):
    # Preprocess the image before passing it to the model
    img_preprocessed = preprocess_image(image)
    
    # Predict the class
    predictions = model.predict(img_preprocessed)
    predicted_class = np.argmax(predictions, axis=1)[0]  # Get the index of the max confidence score
    return class_labels[predicted_class]

# Function to capture images from 4 different cameras and classify them
def process_intersection():
    # Assuming 4 cameras at the intersection, replace with the actual indices
    camera_indices = [0, 1, 2, 3]
    
    for i, cam_idx in enumerate(camera_indices):
        print(f"Capturing image from camera {i + 1} (index {cam_idx})")
        
        # Capture the snapshot from the current camera
        image = capture_image(cam_idx)
        if image is None:
            print(f"Failed to capture image from camera {i + 1}")
            continue
        
        # Classify the captured image
        classification = classify_image(image)
        print(f"Camera {i + 1} Classification: {classification}")
        
        # Send the classification result to the TrafficSignalLogic module
        # Assuming you have a function to handle traffic signal logic
        send_to_traffic_signal_logic(i + 1, classification)

# Function to send the classification result to TrafficSignalLogic.py
def send_to_traffic_signal_logic(camera_number, classification_result):
    # Placeholder for sending the result to TrafficSignalLogic
    print(f"Sending classification result for camera {camera_number}: {classification_result}")
    # Here, you would implement the logic to interface with TrafficSignalLogic.py
    # For example, you can use sockets, pipes, or any inter-process communication method

# Main function
if __name__ == "__main__":
    # Capture and process snapshots for all four lanes
    process_intersection()
