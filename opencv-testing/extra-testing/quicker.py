import cv2
import numpy as np
import time

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Function to create a segmentation mask using a simple background subtraction
def create_segmentation_mask(frame, background=None):
    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Use a faster adaptive threshold
    mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    # Apply morphological operations to clean the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    return cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

# Function to apply different blurs to the background
def apply_blur(frame, mask, blur_type='gaussian'):
    # Create an inverted mask
    fg_mask_3d = mask
    bg_mask_3d = cv2.bitwise_not(mask)

    # Apply the chosen blur type
    if blur_type == 'gaussian':
        blurred_frame = cv2.GaussianBlur(frame, (21, 21), 0)  # Reduced kernel size for speed
    elif blur_type == 'median':
        blurred_frame = cv2.medianBlur(frame, 11)  # Reduced kernel size for speed
    elif blur_type == 'bilateral':
        blurred_frame = cv2.bilateralFilter(frame, 9, 50, 50)  # Reduced parameters for speed
    elif blur_type == 'box':
        blurred_frame = cv2.blur(frame, (21, 21))  # Reduced kernel size for speed
    else:
        blurred_frame = frame

    # Blend the original frame with the blurred background
    fg = cv2.bitwise_and(frame, fg_mask_3d)
    bg = cv2.bitwise_and(blurred_frame, bg_mask_3d)

    return cv2.add(fg, bg)

# Main loop
prev_time = 0
desired_fps = 15  # Increase frame rate for real-time performance
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
