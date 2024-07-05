# Face Mask Detection

This repository contains code for a real-time face mask detection system using Python, OpenCV, TensorFlow, and Streamlit. The application detects faces through a webcam feed, predicts whether a person is wearing a mask correctly, and sends email alerts for non-compliance.

## Features

- Detects faces in real-time using OpenCV.
- Classifies faces into three categories: "Mask not worn correctly or no mask", "Mask on".
- Sends email alerts using SMTP for mask non-compliance.
- Built with a Streamlit interface for easy deployment and interaction.

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/facemask-detection.git
   cd facemask-detection

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt

## Usage

1. **Run the StreamLit App**
   ```bash
   streamlit run app.py
2. Click on the 'Start Detection' button to begin face mask detection.

## Requirements 
- Python 3.x
- OpenCV
- TensorFlow
- Streamlit
- numpy
  
## Model Archtitecture
The face mask detection model 'face_detection.h5' is a deep learning model trained to classify images into one of three categories based on mask presence and correct positioning. 







   
