import cv2
import numpy as np
import time

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Function to apply blurring to the background
def apply_blur(frame, mask, blur_type='gaussian'):
    # Create a 3-channel mask for blurring
    fg_mask_3d = np.stack([mask] * 3, axis=-1)
    bg_mask_3d = np.stack([cv2.bitwise_not(mask)] * 3, axis=-1)

    # Apply the chosen blur type
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

    # Blend the original frame with the blurred background
    fg = cv2.bitwise_and(frame, fg_mask_3d)
    bg = cv2.bitwise_and(blurred_frame, bg_mask_3d)

    return cv2.add(fg, bg)

# Main loop
prev_time = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Camera frame not captured: Failure")
        break

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply a simple binary threshold to create a segmentation mask
    _, mask = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY_INV)

    # Dilate the mask to improve segmentation quality
    kernel = np.ones((3, 3), np.uint8)
    mask_dilated = cv2.dilate(mask, kernel, iterations=1)

    # Measure time for each blur
    blur_types = ['box', 'median', 'gaussian', 'bilateral']
    for blur_type in blur_types:
        blur_start_time = time.time_ns()
        blurred_out_image = apply_blur(frame, mask_dilated, blur_type)
        blur_end_time = time.time_ns()
        print(f'{blur_type.capitalize()} blur application takes {(blur_end_time - blur_start_time) / 1000000:.4f} ms')

        # Display the result
        cv2.imshow(f'{blur_type.capitalize()} Blur', blurred_out_image)

    # Get FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time else 0
    prev_time = curr_time
    cv2.putText(blurred_out_image, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 192, 255), 2)

    # Break the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
