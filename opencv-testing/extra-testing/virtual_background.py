import cv2
import numpy as np
import time

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Load the background image
bg_image_path = 'C:\Anay Docs\AI Camera Testing\opencv-testing\background.jpg'  # Update with the path to your background image
bg_image = cv2.imread(bg_image_path)

if bg_image is None:
    print('Background image not loaded, using green screen')
    bg_image = np.zeros((480, 640, 3), dtype=np.uint8)  # Assuming default frame size
    bg_image[:] = (0, 255, 0)  # Green color

# Function to apply segmentation mask and background
def apply_virtual_background(frame, bg_image, condition):
    # Resize background image to match frame size
    bg_resized = cv2.resize(bg_image, (frame.shape[1], frame.shape[0]))
    return np.where(condition, frame, bg_resized)

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

    # Create a condition mask for background replacement
    condition = mask_dilated[:, :, None] > 0

    # Apply virtual background
    virtual_start_time = time.time_ns()
    out_image = apply_virtual_background(frame, bg_image, condition)
    virtual_end_time = time.time_ns()
    print(f'Virtual background application takes {(virtual_end_time - virtual_start_time) / 1000000:.4f} ms')

    # Get FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time else 0
    prev_time = curr_time
    cv2.putText(out_image, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 192, 255), 2)
    cv2.imshow('Virtual Background', out_image)

    # Apply blurs and display
    blurs = {
        'Original Frame': frame,
        'Box Blur': cv2.blur(frame, (31, 31)),
        'Median Blur': cv2.medianBlur(frame, 15),
        'Gaussian Blur': cv2.GaussianBlur(frame, (31, 31), 0),
        'Bilateral Filter': cv2.bilateralFilter(frame, 15, 75, 75)
    }

    for blur_name, blurred_frame in blurs.items():
        start_time_ns = time.time_ns()
        out_image = apply_virtual_background(blurred_frame, bg_image, condition)
        end_time_ns = time.time_ns()
        print(f'{blur_name} takes {(end_time_ns - start_time_ns) / 1000000:.4f} ms to finish as measured by the RTC module.')
        cv2.imshow(blur_name, out_image)
        time.sleep(0.001)  # Sleep to allow for accurate measurements

    # Break the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
