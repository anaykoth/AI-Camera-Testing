import cv2
import numpy as np
import time
import tensorflow as tf
import os

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path="deeplabv3.tflite")
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Function to run inference on the TFLite model
def run_inference(frame):
    input_shape = input_details[0]['shape']
    input_data = cv2.resize(frame, (input_shape[2], input_shape[1]))
    input_data = np.expand_dims(input_data, axis=0).astype(np.uint8)

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])[0]
    return output_data

# Function to create a segmentation mask
def create_segmentation_mask(frame):
    seg_map = run_inference(frame)
    mask = seg_map > 0.5  # Thresholding to create a binary mask
    mask = mask.astype(np.uint8) * 255
    return mask

# Function to apply different blurs to the background
def apply_blur(frame, mask, blur_type='gaussian'):
    fg_mask_3d = np.stack([mask] * 3, axis=-1)
    bg_mask_3d = np.stack([cv2.bitwise_not(mask)] * 3, axis=-1)

    if blur_type == 'gaussian':
        blurred_frame = cv2.GaussianBlur(frame, (31, 31), 0)
    elif blur_type == 'median':
        blurred_frame = cv2.medianBlur(frame, 15)
    elif blur_type == 'bilateral':
        blurred_frame = cv2.bilateralFilter(frame, 15, 75, 75)
    elif blur_type == 'box':
        blurred_frame = cv2.blur(frame, (31, 31))
    else:
        blurred_frame = frame

    fg = cv2.bitwise_and(frame, fg_mask_3d)
    bg = cv2.bitwise_and(blurred_frame, bg_mask_3d)

    return cv2.add(fg, bg)

# Initialize webcam
cap = cv2.VideoCapture(0)

prev_time = 0
desired_fps = 10
frame_interval = 1.0 / desired_fps

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Camera frame not captured: Failure")
        break

    current_time = time.time()

    if current_time - prev_time >= frame_interval:
        prev_time = current_time

        # Reduce resolution for speed
        frame_resized = cv2.resize(frame, (640, 360))

        # Create segmentation mask
        mask = create_segmentation_mask(frame_resized)

        # Measure time for each blur
        blur_types = ['box', 'median', 'gaussian', 'bilateral']
        for blur_type in blur_types:
            blur_start_time = time.time_ns()
            blurred_out_image = apply_blur(frame_resized, mask, blur_type)
            blur_end_time = time.time_ns()
            print(f'{blur_type.capitalize()} blur application takes {(blur_end_time - blur_start_time) / 1000000:.4f} ms')

            # Display the result
            cv2.imshow(f'{blur_type.capitalize()} Blur', blurred_out_image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
