from flask import Flask, render_template, Response, request, jsonify
import cv2
import numpy as np
import time
import psutil
import threading
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo

app = Flask(__name__)

# Initialize Detectron2
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Set threshold for this model
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # Use GPU if available
predictor = DefaultPredictor(cfg)

# Global variables for FPS and blur type
current_fps = 30
current_blur_type = 'Original Frame'
blur_times = []
cpu_response_times = []
cpu_usages = []

# Lock for thread-safe updates to shared data
data_lock = threading.Lock()

def reset_metrics():
    global blur_times, cpu_response_times, cpu_usages
    with data_lock:
        blur_times = []
        cpu_response_times = []
        cpu_usages = []

def run_inference(frame):
    outputs = predictor(frame)
    masks = outputs["instances"].pred_masks.cpu().numpy()
    combined_mask = np.zeros(frame.shape[:2], dtype=bool)
    
    # Combine all detected masks
    for mask in masks:
        combined_mask |= mask

    return combined_mask

# Function to apply blurring and track performance
def process_frame(frame, blur_type):
    frame.flags.writeable = False
    start_cpu_time = time.process_time_ns()
    mask = run_inference(frame)
    frame.flags.writeable = True
    end_cpu_time = time.process_time_ns()

    # Start timing the blur operation
    blur_start_time = time.time_ns()

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

    # End timing the blur operation
    blur_end_time = time.time_ns()

    out_image = np.where(mask[:, :, None], frame, blurred_frame)

    # Calculate and store metrics
    blur_time_ms = (blur_end_time - blur_start_time) / 1000000  # Convert to milliseconds
    cpu_response_time_ms = (end_cpu_time - start_cpu_time) / 1000000  # Convert to milliseconds
    cpu_usage = psutil.cpu_percent()

    with data_lock:
        blur_times.append(blur_time_ms)
        cpu_response_times.append(cpu_response_time_ms)
        cpu_usages.append(cpu_usage)

    return out_image, blur_time_ms, cpu_response_time_ms, cpu_usage

def calculate_average_metrics():
    with data_lock:
        if blur_times and cpu_response_times and cpu_usages:
            avg_blur_time = sum(blur_times) / len(blur_times)
            avg_cpu_response_time = sum(cpu_response_times) / len(cpu_response_times)
            avg_cpu_usage = sum(cpu_usages) / len(cpu_usages)
        else:
            avg_blur_time = avg_cpu_response_time = avg_cpu_usage = 0

    return avg_blur_time, avg_cpu_response_time, avg_cpu_usage

def gen_frames():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break
        
        global current_blur_type, current_fps
        out_image, blur_time_ms, cpu_response_time_ms, cpu_usage = process_frame(frame, current_blur_type)

        # Calculate performance metrics
        memory_usage = psutil.virtual_memory().percent

        # Add overlay text
        cv2.putText(out_image, f'CPU Usage: {cpu_usage}%', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(out_image, f'Memory Usage: {memory_usage}%', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(out_image, f'Blur Time: {blur_time_ms:.2f}ms', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(out_image, f'CPU Response Time: {cpu_response_time_ms:.2f}ms', (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        ret, buffer = cv2.imencode('.jpg', out_image)
        frame = buffer.tobytes()

        # Yield the frame and then sleep to match the desired FPS
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        # Add a delay to control the FPS
        time.sleep(1 / current_fps)

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
    avg_blur_time, avg_cpu_response_time, avg_cpu_usage = calculate_average_metrics()

    stats = {
        "cpu_usage": psutil.cpu_percent(),
        "memory_usage": psutil.virtual_memory().percent,
        "blur_time_ms": avg_blur_time,
        "cpu_response_time_ms": avg_cpu_response_time,
        "avg_cpu_usage": avg_cpu_usage
    }
    return jsonify(stats)

@app.route('/reset_metrics', methods=['POST'])
def reset_metrics_route():
    reset_metrics()
    return jsonify(success=True)

if __name__ == '__main__':
    reset_metrics()
    app.run(debug=True)
