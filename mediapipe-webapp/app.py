from flask import Flask, render_template, Response, request, jsonify
import cv2
import numpy as np
import mediapipe as mp
import time
import psutil
import subprocess

app = Flask(__name__)

# Initialize MediaPipe
mp_selfie_segmentation = mp.solutions.selfie_segmentation
selfie_segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)

# Global variables for FPS and blur type
current_fps = 30
current_blur_type = 'Original Frame'

# Function to apply blurring and track performance
def process_frame(frame, blur_type):
    frame.flags.writeable = False
    results = selfie_segmentation.process(frame)
    frame.flags.writeable = True

    condition = results.segmentation_mask > 0.1

    if blur_type == 'Box Blur':
        blurred_frame = cv2.blur(frame, (31, 31))
    elif blur_type == 'Median Blur':
        blurred_frame = cv2.medianBlur(frame, 15)
    elif blur_type == 'Gaussian Blur':
        blurred_frame = cv2.GaussianBlur(frame, (31, 31), 0)
    elif blur_type == 'Bilateral Blur':
        blurred_frame = cv2.bilateralFilter(frame, 15, 75, 75)
    else:
        blurred_frame = frame

    out_image = np.where(condition[..., None], frame, blurred_frame)
    return out_image

# # Function to get GPU usage (for Intel GPUs, use a workaround)
# def get_gpu_usage():
#     try:
#         # Use psutil to get GPU usage (if available, otherwise mock data)
#         gpu_info = psutil.sensors_temperatures()
#         if 'coretemp' in gpu_info:
#             # Mock value based on available data; Intel doesn't provide easy GPU monitoring
#             return gpu_info['coretemp'][0].current
#         else:
#             return 0  # No proper GPU info available
#     except Exception as e:
#         print(f"Could not retrieve GPU usage: {e}")
#         return 0

def gen_frames():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break
        
        global current_blur_type
        blur_start_time = time.time_ns()
        out_image = process_frame(frame, current_blur_type)
        blur_end_time = time.time_ns()

        # Calculate performance metrics
        cpu_usage = psutil.cpu_percent()
        memory_usage = psutil.virtual_memory().percent
#        gpu_usage = get_gpu_usage()
        blur_time_ms = (blur_end_time - blur_start_time) / 1000000

        # Add overlay text
        cv2.putText(out_image, f'CPU: {cpu_usage}%', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(out_image, f'Memory: {memory_usage}%', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(out_image, f'Blur Time: {blur_time_ms:.2f}ms', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        ret, buffer = cv2.imencode('.jpg', out_image)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/set_fps', methods=['POST'])
def set_fps():
    global current_fps
    current_fps = int(request.form['fps'])
    return jsonify(success=True)

@app.route('/set_blur_type', methods=['POST'])
def set_blur_type():
    global current_blur_type
    current_blur_type = request.form['blur_type']
    return jsonify(success=True)

@app.route('/performance_stats')
def performance_stats():
    stats = {
        "cpu_usage": psutil.cpu_percent(),
        "memory_usage": psutil.virtual_memory().percent,
#        "gpu_usage": get_gpu_usage(),
        "blur_time_ms": 0  # Replace with actual calculations from your loop
    }
    return jsonify(stats)

if __name__ == '__main__':
    app.run(debug=True)
