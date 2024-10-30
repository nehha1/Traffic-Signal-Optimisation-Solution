# Traffic Signal Optimization Model

## Overview

This project implements a machine learning-based model for optimizing traffic signals. The model uses a combination of historical congestion data and real-time camera inputs to dynamically adjust traffic signals, improving traffic flow and reducing congestion.

## Features

- **Real-time Traffic Density Detection**: Utilizes camera feeds to assess current traffic conditions.
- **Historical Data Analysis**: Incorporates past traffic patterns for more informed decision-making.
- **Adaptive Signal Control**: Dynamically adjusts red/green signal timings based on current and predicted traffic flow.

## Model Architecture

The system comprises two main components:

1. **Traffic Density Detection**:
   - **Model**: RESNET50 CNN (Convolutional Neural Network)
   - **Purpose**: Analyzes camera feeds to determine current traffic density in different lanes.

2. **Historical Data Analysis**:
   - **Model**: LSTM (Long Short-Term Memory)
   - **Purpose**: Processes historical congestion data to predict traffic patterns.

## Data Sources

- **Real-time Data**: Camera feeds installed at traffic intersections.
- **Historical Data**: Custom dataset of past traffic patterns.
- **Training Data**: Combination of Kaggle datasets and proprietary data.

## Signal Generation Logic

A custom algorithm processes outputs from both the CNN and LSTM models to determine optimal signal timings for red and green lights.

## Usage

(Provide instructions on how to run the model, including any required inputs or configurations)

## Training

The model is trained using a combination of:
- Kaggle datasets
- Custom proprietary datasets

To train the model:
(Include steps for training or retraining the model)

## Future Improvements

- Integration with GPS data for wider area traffic optimization
- Incorporation of weather data for more accurate predictions
- Expansion to include pedestrian crossing optimization

## Contributors

Neha Naik, Keshav Dadhich
