import cv2
import numpy as np
import time
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from msrest.authentication import CognitiveServicesCredentials
import io  # Import io for BytesIO

# Azure Credentials
subscription_key = '06187122faad4f34b28bb771f9bf32b4'
endpoint = 'https://caregility-blur-testing.cognitiveservices.azure.com/'

# Initialize Azure Computer Vision Client
client = ComputerVisionClient(endpoint, CognitiveServicesCredentials(subscription_key))

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Load the background image
bg_image_path = r'C:\Anay Docs\AI Camera Testing\azure-vision-project\background.jpg'
bg_image = cv2.imread(bg_image_path)

if bg_image is None:
    print('Background image not loaded, using green screen')
    bg_image = np.zeros((480, 640, 3), dtype=np.uint8)
    bg_image[:] = (0, 255, 0)  # Green color

def get_segmentation_mask(frame):
    # Convert frame to bytes for Azure API
    _, img_encoded = cv2.imencode('.jpg', frame)
    img_bytes = io.BytesIO(img_encoded.tobytes())  # Wrap the bytes in BytesIO

    # Call Azure API for image analysis
    analysis = client.analyze_image_in_stream(img_bytes, visual_features=["Objects"])

    # Create a blank mask
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)

    # Use object detection results to create a mask
    for obj in analysis.objects:
        # Assume objects detected are in the foreground
        x, y, w, h = int(obj.rectangle.x), int(obj.rectangle.y), int(obj.rectangle.w), int(obj.rectangle.h)
        mask[y:y+h, x:x+w] = 255

    # Refine the mask using morphological operations
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=2)
    mask = cv2.dilate(mask, kernel, iterations=2)

    # Use contours to improve mask precision
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    refined_mask = np.zeros_like(mask)

    for cnt in contours:
        cv2.drawContours(refined_mask, [cnt], -1, (255), thickness=cv2.FILLED)

    # Smooth the edges of the mask
    refined_mask = cv2.GaussianBlur(refined_mask, (21, 21), 0)

    return refined_mask

def apply_virtual_background(frame, bg_image, mask):
    bg_resized = cv2.resize(bg_image, (frame.shape[1], frame.shape[0]))
    fg_mask_3d = np.stack([mask]*3, axis=-1)
    bg_mask_3d = np.stack([cv2.bitwise_not(mask)]*3, axis=-1)

    fg = cv2.bitwise_and(frame, fg_mask_3d)
    bg = cv2.bitwise_and(bg_resized, bg_mask_3d)

    return cv2.add(fg, bg)

def apply_blur(frame, mask, blur_type='gaussian'):
    fg_mask_3d = np.stack([mask]*3, axis=-1)
    bg_mask_3d = np.stack([cv2.bitwise_not(mask)]*3, axis=-1)

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

# Main loop
prev_time = 0
desired_fps = 30  # Desired frames per second
frame_interval = 1.0 / desired_fps  # Time interval between frames

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Camera frame not captured: Failure")
        break

    # Measure time before processing
    current_time = time.time()

    if current_time - prev_time >= frame_interval:
        # Update the previous time
        prev_time = current_time

        # Get segmentation mask from Azure
        mask = get_segmentation_mask(frame)

        # Measure time for each blur
        blur_types = ['box', 'median', 'gaussian', 'bilateral']
        for blur_type in blur_types:
            blur_start_time = time.time_ns()
            blurred_out_image = apply_blur(frame, mask, blur_type)
            blur_end_time = time.time_ns()
            print(f'{blur_type.capitalize()} blur application takes {(blur_end_time - blur_start_time) / 1000000:.4f} ms')

            # Display the result
            cv2.imshow(f'{blur_type.capitalize()} Blur', blurred_out_image)

        # Apply virtual background replacement
        virtual_start_time = time.time_ns()
        virtual_bg_out_image = apply_virtual_background(frame, bg_image, mask)
        virtual_end_time = time.time_ns()
        print(f'Virtual background replacement takes {(virtual_end_time - virtual_start_time) / 1000000:.4f} ms')

        # Display the virtual background replacement result
        cv2.imshow('Virtual Background Replacement', virtual_bg_out_image)

    # Break the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
