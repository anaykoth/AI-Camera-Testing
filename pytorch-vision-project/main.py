import cv2
import numpy as np
import torch
from torchvision import transforms, models

# Load the pre-trained DeepLabV3 model
model = models.segmentation.deeplabv3_resnet101(pretrained=True).eval()

print(torch.cuda.is_available())
# Transformation pipeline
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((512, 512)),  # Resize for model input
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load the background image
bg_image_path = 'background.jpg'
bg_image = cv2.imread(bg_image_path)

if bg_image is None:
    print('Background image not loaded, using green screen')
    bg_image = np.zeros((480, 640, 3), dtype=np.uint8)
    bg_image[:] = (0, 255, 0)  # Green color

def get_segmentation_mask(frame):
    input_tensor = transform(frame).unsqueeze(0)
    with torch.no_grad():
        output = model(input_tensor)['out'][0]
    output_predictions = output.argmax(0).byte().cpu().numpy()
    return output_predictions

def apply_virtual_background(frame, bg_image, mask):
    bg_resized = cv2.resize(bg_image, (frame.shape[1], frame.shape[0]))
    fg_mask = mask == 15  # Assuming class 15 corresponds to 'person'
    bg_mask = mask != 15

    fg_mask_3d = np.stack([fg_mask]*3, axis=-1)
    bg_mask_3d = np.stack([bg_mask]*3, axis=-1)

    fg = np.where(fg_mask_3d, frame, 0)
    bg = np.where(bg_mask_3d, bg_resized, 0)
    
    return cv2.add(fg, bg)

def apply_blur_background(frame, mask, blur_type='gaussian'):
    fg_mask = mask == 15  # Assuming class 15 corresponds to 'person'
    bg_mask = mask != 15

    fg_mask_3d = np.stack([fg_mask]*3, axis=-1)
    bg_mask_3d = np.stack([bg_mask]*3, axis=-1)

    if blur_type == 'gaussian':
        blurred_frame = cv2.GaussianBlur(frame, (31, 31), 0)
    elif blur_type == 'median':
        blurred_frame = cv2.medianBlur(frame, 15)
    elif blur_type == 'bilateral':
        blurred_frame = cv2.bilateralFilter(frame, 15, 75, 75)
    else:
        blurred_frame = cv2.blur(frame, (31, 31))

    fg = np.where(fg_mask_3d, frame, 0)
    bg = np.where(bg_mask_3d, blurred_frame, 0)

    return cv2.add(fg, bg)

# Initialize the webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Camera frame not captured: Failure")
        break

    # Get segmentation mask
    mask = get_segmentation_mask(frame)

    # Apply virtual background
    virtual_bg = apply_virtual_background(frame, bg_image, mask)
    cv2.imshow('Virtual Background', virtual_bg)

    # Apply various blurs and display
    blurs = ['gaussian', 'median', 'bilateral', 'box']
    for blur_type in blurs:
        blurred_bg = apply_blur_background(frame, mask, blur_type)
        cv2.imshow(f'{blur_type.capitalize()} Blur', blurred_bg)

    # Break the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
