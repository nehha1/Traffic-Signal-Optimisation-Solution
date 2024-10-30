import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
import pandas as pd
from datetime import datetime

# Define image size and batch size
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

# Define class labels based on your specifications
class_labels = ['dense_traffic', 'sparse_traffic', 'ambulance', 'fire_brigade', 'fire_burst', 'accident', 'truck']

# Directory paths
train_dir = r'/Users/keshavdadhich/Documents/NITTE HACKATHON/data/train'  # Update this path
val_dir = r'/Users/keshavdadhich/Documents/NITTE HACKATHON/data/val'  # Update this path

# Data augmentation and loading
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

# ImageDataGenerator for validation (no augmentation, only rescaling)
val_datagen = ImageDataGenerator(rescale=1./255)

# Load the training and validation sets
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    classes=class_labels
)

validation_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    classes=class_labels
)

# Load ResNet50 model without the top classification layers
backbone = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Add classification layers on top of ResNet50
x = Flatten()(backbone.output)
x = Dense(128, activation='relu')(x)
output = Dense(len(class_labels), activation='softmax')(x)

# Create the final model
model = Model(inputs=backbone.input, outputs=output)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=10,  # You can increase the number of epochs as needed
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    verbose=1
)

# Save the model
model.save('traffic_signal_cnn_model.keras')

# Load the trained model
model = tf.keras.models.load_model('traffic_signal_cnn_model.keras')

# Priority assignment based on class predictions
def get_priority(classification):
    # Mapping classes to priorities
    priority_mapping = {
        'accident': 1,
        'fire_burst': 1,
        'ambulance': 2,
        'fire_brigade': 2,
        'truck': 3,
        'dense_traffic': 4,
        'sparse_traffic': 5,
    }
    return priority_mapping.get(classification, float('inf'))  # Use float('inf') for unknown classes

# Function to predict the class of a new image and assign priority
def predict_image(image_path):
    # Preprocess the image
    img = cv2.imread(image_path)
    img_resized = cv2.resize(img, IMAGE_SIZE)
    img_array = np.expand_dims(img_resized, axis=0) / 255.0

    # Predict the class
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)

    # Get the class labels
    predicted_label = class_labels[predicted_class[0]]
    priority = get_priority(predicted_label)

    return predicted_label, priority  # Return only predicted class and priority

video_path = r'/Users/keshavdadhich/Documents/NITTE HACKATHON/854745-hd_1280_720_50fps.mp4'  # Update this path to your video

# Function to capture frames from video every 5 seconds
def capture_frames_from_video(video_path, output_folder='captured_frames', interval=5):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    results = []  # List to store results

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps * interval)
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            frame_filename = os.path.join(output_folder, f'frame_{frame_count // frame_interval}.jpg')
            cv2.imwrite(frame_filename, frame)
            print(f'Saved: {frame_filename}')
            
            # Predict the class for the captured frame
            predicted_class, priority = predict_image(frame_filename)
            print(f"Predicted class: {predicted_class}, Priority: {priority}")
            results.append((predicted_class, priority))  # Store the result

        frame_count += 1

    cap.release()
    print("Done capturing frames.")
    return results  # Return the collected results

# Example usage
results = capture_frames_from_video(video_path)

# Load the dataset
data = pd.read_csv('Banglore_traffic_Dataset.csv')

# Convert 'Date' to datetime
data['Date'] = pd.to_datetime(data['Date'])

# Select relevant columns for congestion analysis
data = data[['Date', 'Area Name', 'Traffic Volume', 'Congestion Level']]

# Get the current date and month
current_date = datetime.now()
current_day = current_date.day
current_month = current_date.month

# Ask for location input
location_input = input("Enter the location (e.g., 'CMH Road, Indiranagar'): ")

# Get the data for the specified location
location_data = data[data['Area Name'].str.contains(location_input, case=False, na=False)]

if location_data.empty:
    average_congestion = 0  # No data found, set to 0
else:
    # Filter for the same month and day in 2022 and 2023
    location_data_2022 = location_data[(location_data['Date'].dt.year == 2022) & 
                                        (location_data['Date'].dt.month == current_month) & 
                                        (location_data['Date'].dt.day == current_day)]

    location_data_2023 = location_data[(location_data['Date'].dt.year == 2023) & 
                                        (location_data['Date'].dt.month == current_month) & 
                                        (location_data['Date'].dt.day == current_day)]

    # Calculate the average congestion level for both years
    average_congestion_2022 = location_data_2022['Congestion Level'].mean()
    average_congestion_2023 = location_data_2023['Congestion Level'].mean()

    # Calculate the overall average congestion level
    average_congestion = (average_congestion_2022 + average_congestion_2023) / 2 if (not pd.isna(average_congestion_2022) and not pd.isna(average_congestion_2023)) else 0

# Output the average congestion level
print(f"Average Congestion Level for {location_input} on {current_day}/{current_month} (2022 & 2023): {average_congestion:.2f}")


#This is the code snippet for the Signal timings and final output of our project 

results.append((average_congestion))

import time
from enum import Enum


class TrafficStatus(Enum):
    DENSE = 1
    SPARSE = 2
    FIRE = 3
    ACCIDENT = 4
    FIRE_BURST = 5
    FIRE_BRIGADE = 6
    AMBULANCE = 7
    TRUCK = 8


class LaneDirection(Enum):
    NORTH = 1
    SOUTH = 2
    EAST = 3
    WEST = 4


class Lane:
    def __init__(self, direction, status):
        self.direction = direction
        self.status = status


class Signal:
    def __init__(self, lane):
        self.lane = lane
        self.green = False
        self.straight = False
        self.left = False
        self.right = False
        self.duration = 0  # Duration in seconds


class TrafficSignal:
    def __init__(self):
        self.lanes = [
            Lane(LaneDirection.NORTH, TrafficStatus.SPARSE),
            Lane(LaneDirection.SOUTH, TrafficStatus.SPARSE),
            Lane(LaneDirection.EAST, TrafficStatus.SPARSE),
            Lane(LaneDirection.WEST, TrafficStatus.SPARSE)
        ]
        self.signals = [Signal(lane) for lane in self.lanes]

    def update_lane_status(self, lane_direction, status):
        for lane in self.lanes:
            if lane.direction == lane_direction:
                lane.status = status
                break

    def get_signal(self, lane_direction):
        for signal in self.signals:
            if signal.lane.direction == lane_direction:
                return signal

    def emergency_phase(self):
        emergency_priority = {
            TrafficStatus.ACCIDENT: 1,
            TrafficStatus.FIRE_BURST: 1,
            TrafficStatus.AMBULANCE: 2,
            TrafficStatus.FIRE_BRIGADE: 2,
            TrafficStatus.TRUCK: 3,
            TrafficStatus.DENSE: 4,
            TrafficStatus.SPARSE: 5
        }

        # Sort lanes by emergency priority
        emergency_lanes = sorted(
            [lane for lane in self.lanes if lane.status in emergency_priority],
            key=lambda lane: emergency_priority[lane.status]
        )

        phases = []
        for lane in emergency_lanes:
            signal = self.get_signal(lane.direction)
            signal.green = True
            signal.straight = True
            signal.left = True
            signal.right = True
            signal.duration = 30  # Emergency phase lasts for 30 seconds for high-priority emergencies
            phases.append([signal])

        return phases

    def calculate_signals(self):
        emergency_signals = self.emergency_phase()
        if emergency_signals:
            return emergency_signals

        dense_lanes = [lane for lane in self.lanes if lane.status == TrafficStatus.DENSE]
        sparse_lanes = [lane for lane in self.lanes if lane.status == TrafficStatus.SPARSE]

        phases = []
        timer_values = {
            'DENSE_GREEN': 45+i,
            'SPARSE_GREEN': 30+i,
            'YELLOW': 5+i,
            'RED': 30+i
        }

        if len(dense_lanes) == 3:  # All lanes dense
            for lane in self.lanes:
                signal = self.get_signal(lane.direction)
                signal.green = True
                signal.duration = timer_values['DENSE_GREEN']
                signal.straight = True
                signal.left = True
            phases.append([signal for lane in self.lanes])

        elif len(dense_lanes) == 2:  # 2 Dense and 1 Sparse
            for lane in dense_lanes:
                signal = self.get_signal(lane.direction)
                signal.green = True
                signal.duration = timer_values['DENSE_GREEN']
                signal.straight = True
                signal.left = True

            sparse_signal = self.get_signal((set(self.lanes) - set(dense_lanes)).pop().direction)
            sparse_signal.green = False
            sparse_signal.duration = timer_values['RED']
            phases.append([signal for signal in dense_lanes] + [sparse_signal])

        elif len(dense_lanes) == 1:  # 1 Dense and 2 Sparse
            dense_signal = self.get_signal(dense_lanes[0].direction)
            dense_signal.green = True
            dense_signal.duration = timer_values['DENSE_GREEN']
            dense_signal.straight = True
            phases.append([dense_signal])

            for lane in sparse_lanes:
                sparse_signal = self.get_signal(lane.direction)
                sparse_signal.green = False
                sparse_signal.duration = timer_values['RED']
                phases[-1].append(sparse_signal)

        elif len(dense_lanes) == 0:  # All Sparse
            for lane in self.lanes:
                signal = self.get_signal(lane.direction)
                signal.green = True
                signal.duration = timer_values['SPARSE_GREEN']
                signal.straight = True
            phases.append([signal for lane in self.lanes])

        return phases

    def display_phase(self, phase_num, signals):
        print(f"\nPhase {phase_num}:")
        print("=" * 30)

        for signal in signals:
            status = "Red"
            if signal.green:
                status = "Green"
                directions = []
                if signal.straight:
                    directions.append("Straight")
                if signal.left:
                    directions.append("Left")
                if signal.right:
                    directions.append("Right")
                direction_str = ", ".join(directions)
                print(f"Lane {signal.lane.direction.name}: {status} ({direction_str}), Duration: {signal.duration} seconds")
            else:
                print(f"Lane {signal.lane.direction.name}: {status}")
        print("=" * 30)

    def simulate_signals(self):
        phases = self.calculate_signals()

        for idx, phase in enumerate(phases):
            self.display_phase(idx + 1, phase)
            for signal in phase:
                if signal.green:
                    print(f"Turning green light on for {signal.lane.direction.name} lane for {signal.duration} seconds...")
                    time.sleep(signal.duration)  # Simulate time spent in green
                else:
                    print(f"{signal.lane.direction.name} lane is Red.")

for i in range(0,len(results),8):
    traffic_signal = TrafficSignal()
    traffic_signal.update_lane_status(LaneDirection.NORTH, TrafficStatus.results[i])
    traffic_signal.update_lane_status(LaneDirection.SOUTH, TrafficStatus.results[i+2])
    traffic_signal.update_lane_status(LaneDirection.EAST, TrafficStatus.results[i+4])
    traffic_signal.update_lane_status(LaneDirection.WEST, TrafficStatus.results[i+6])
    congestion=average_congestion
    if congestion>75.00:
        i=20
    elif congestion>50.00 & congestion<75.00:
        i=15
    elif congestion>25.00 & congestion<50.00:
        i=10
    elif congestion>0.00 & congestion<25.00:
        i=5
    else :
        i=0
    traffic_signal.simulate_signals()