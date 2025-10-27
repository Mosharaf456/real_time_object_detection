import os
import cv2
import time 
import torch
import asone
import inspect
import numpy as np

from asone import ASOne
from asone.utils.draw import draw_ui_box

import random as random
from ultralytics import YOLO
from urllib.request import urlretrieve


def download_yolov8_weights():
    weights_dir = "AS-One/asone/weights"
    os.makedirs(weights_dir, exist_ok=True)
    weights_path = os.path.join(weights_dir, "yolov8n.pt")
    
    if not os.path.exists(weights_path):
        print("Downloading YOLOv8 weights...")
        # Download from official Ultralytics release
        url = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt"
        try:
            urlretrieve(url, weights_path)
            print(f"Weights downloaded to: {weights_path}")
        except Exception as e:
            print(f"Download failed: {e}")
            # Fallback: try using gdown if available
            try:
                import gdown
                # Alternative Google Drive link
                gdown_id = "1LZ0B2JWUA1_SD_QR9hY_2uGQeGMQbK-l"
                gdown.download(f"https://drive.google.com/uc?id={gdown_id}", weights_path, quiet=False)
            except:
                print("All download methods failed")
                return None
    
    return weights_path

# Fix for PyTorch 2.6 compatibility with ultralytics
try:
    from ultralytics.nn.tasks import DetectionModel
    torch.serialization.add_safe_globals([DetectionModel])
    torch.serialization.add_safe_globals([torch.nn.modules.container.Sequential])
except Exception as e:
    print(f"PyTorch compatibility fix applied with warning: {e}")
    


def video_detection(path_x='', conf_=0.5, filter_classes=None):
    total_detections = 0
    names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
             'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
             'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
             'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
             'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',  
             'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
             'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
             'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

    colors = [(random.randint(0,255), random.randint(0,255), random.randint(0,255)) for _ in range(1000)]

    if filter_classes:
        filter_classes = [filter_classes]

    weights_path = download_yolov8_weights()
    dt_obj = ASOne(
        tracker=asone.BYTETRACK,
        detector=asone.YOLOV8N_PYTORCH,
        weights=weights_path,
        use_cuda=False
    )

    video = cv2.VideoCapture(path_x)
    nframes = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    start_time = time.time()

    for j in range(nframes):
        ret, img0 = video.read()
        if not ret or img0 is None:
            continue

        fps_x = int((j + 1) / (time.time() - start_time))

        detected = dt_obj.detect(img0, conf_thres=conf_, iou_thres=0.45, filter_classes=filter_classes)
        ouput_detected = detected[0]

        if len(ouput_detected) > 0:
            for det in ouput_detected:
                total_detections += 1
                det = np.array(det).flatten()
                box = det[:4].astype(float).tolist()
                conf = float(det[4])
                cls_id = int(det[5])
                color_idx = cls_id % len(colors)

                label = f"{names[cls_id] if cls_id < len(names) else 'Class'} {conf:.2f}"
                draw_ui_box(img0, box, label, color=colors[color_idx], line_thickness=3)

        yield img0, fps_x, img0.shape, total_detections
    video.release()
    cv2.destroyAllWindows()


def video_detection2(path_x='', conf_=0.5, filter_classes=None):
    """
    Perform object detection on a video file or RTSP stream.

    Args:
        path_x (str): Path to video file or RTSP URL.
        conf_ (float): Confidence threshold (0-1).
        filter_classes (list or None): List of class IDs to filter detections.
    
    Yields:
        tuple: (frame, FPS, frame_shape, total_detections)
    """
    
    names = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
        'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
        'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',  
        'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
        'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
        'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]

    colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(1000)]

    if filter_classes:
        filter_classes = [filter_classes]

    # Load YOLOv8 weights and initialize detector
    weights_path = download_yolov8_weights()
    dt_obj = ASOne(
        tracker=asone.BYTETRACK,
        detector=asone.YOLOV8N_PYTORCH,
        weights=weights_path,
        use_cuda=False
    )

    # Open video or RTSP stream
    video = cv2.VideoCapture(path_x)
    if not video.isOpened():
        raise ValueError(f"Failed to open video or RTSP stream: {path_x}")

    start_time = time.time()
    total_detections = 0

    while True:
        ret, img0 = video.read()
        if not ret or img0 is None:
            # If RTSP stream fails temporarily, just skip this frame
            time.sleep(0.01)
            continue

        # Estimate FPS
        elapsed_time = time.time() - start_time
        fps_x = int((total_detections + 1) / max(elapsed_time, 1e-6))

        # Run detection
        detected = dt_obj.detect(img0, conf_thres=conf_, iou_thres=0.45, filter_classes=filter_classes)
        ouput_detected = detected[0]

        # Draw detections
        if len(ouput_detected) > 0:
            for det in ouput_detected:
                total_detections += 1
                det = np.array(det).flatten()
                box = det[:4].astype(float).tolist()
                conf = float(det[4])
                cls_id = int(det[5])
                color_idx = cls_id % len(colors)
                label = f"{names[cls_id] if cls_id < len(names) else 'Class'} {conf:.2f}"
                draw_ui_box(img0, box, label, color=colors[color_idx], line_thickness=3)

        # Yield the frame, FPS, shape, and total detections
        yield img0, fps_x, img0.shape, total_detections
    # Release resources
    video.release()
    cv2.destroyAllWindows()

