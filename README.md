# Background Blurring Web App

This repository contains three separate web applications that demonstrate real-time background blurring using different computer vision technologies: **MediaPipe**, **OpenCV**, and **TensorFlow Lite**. Each folder contains the respective implementation along with necessary scripts and instructions to run the application.

## Table of Contents

- [Background Blurring Web App](#background-blurring-web-app)
  - [Table of Contents](#table-of-contents)
  - [Prerequisites](#prerequisites)
  - [Running the Applications](#running-the-applications)
    - [1. MediaPipe Implementation](#1-mediapipe-implementation)
    - [2. OpenCV Implementation](#2-opencv-implementation)
    - [3. TensorFlow Lite Implementation](#3-tensorflow-lite-implementation)
  - [Notes](#notes)


## Prerequisites

Before running any of the web applications, ensure that you have the following installed:

- **Python 3.8 or above**
- **pip** (Python package installer)
- **Virtual Environment** (optional but recommended)
- **OpenCV** (required for the OpenCV implementation)
- **MediaPipe** (required for the MediaPipe implementation)
- **TensorFlow Lite** (required for the TensorFlow Lite implementation)

Each implementation folder contains a `requirements.txt` file. You can install the dependencies using:

```bash
pip install -r requirements.txt
```
## Running the Applications

### 1. MediaPipe Implementation

The MediaPipe implementation leverages the MediaPipe library to perform real-time background blurring using the selfie segmentation model.

**Steps to run:**

1. **Navigate to the `mediapipe-webapp` directory:**
    ```bash
    cd mediapipe-webapp
    ```

2. **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3. **Run the Flask application:**
    ```bash
    python app.py
    ```

4. **Access the web app:**
    Open your web browser and navigate to `http://127.0.0.1:5000`.

### 2. OpenCV Implementation

The OpenCV implementation uses traditional computer vision techniques to blur the background in real-time.

**Steps to run:**

1. **Navigate to the `opencv-webapp` directory:**
    ```bash
    cd opencv-webapp
    ```

2. **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3. **Run the Flask application:**
    ```bash
    python app.py
    ```

4. **Access the web app:**
    Open your web browser and navigate to `http://127.0.0.1:5000`.

### 3. TensorFlow Lite Implementation

The TensorFlow Lite implementation uses the DeepLabV3 model for semantic segmentation and real-time background blurring.

**Steps to run:**

1. **Navigate to the `tensorflowlite-webapp` directory:**
    ```bash
    cd tensorflowlite-webapp
    ```

2. **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3. **Download and place the `deeplabv3.tflite` model in the directory:**
    Ensure the model is named `deeplabv3.tflite` or update the `app.py` accordingly.
    Add the .tflite file to the folder and change the "model_path" to the name of the .tflite file

4. **Run the Flask application:**
    ```bash
    python app.py
    ```

5. **Access the web app:**
    Open your web browser and navigate to `http://127.0.0.1:5000`.

## Notes

- **Compatibility:** Ensure that your system has a working webcam and that you have the necessary permissions to access it from your browser. The applications have been tested primarily on Windows and macOS.

- **Performance:** The performance of each implementation (MediaPipe, OpenCV, TensorFlow Lite) can vary significantly depending on your system’s hardware, particularly the CPU and GPU. The applications include real-time performance metrics to help you analyze the efficiency of each method.

- **FPS** The FPS changes are not a true measure of the performance with higher or lower cameras. Code is built for the default computer camera and the FPS changes just introduces lag into the video. 

- **TensorFlow Lite Model:** For the TensorFlow Lite implementation, you must download the `deeplabv3.tflite` model and place it in the corresponding directory. Make sure the model file name in the code matches the actual file name.
