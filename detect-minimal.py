# detect_minimal.py
# Minimal YOLOv5 detection script (no saving, just live detection)

import argparse  # For reading command-line arguments (like --weights, --source)
import os        # For interacting with the operating system (paths, files, etc.)
import sys       # Gives access to system-specific parameters and functions
from pathlib import Path  # Helps in handling file and folder paths easily
import torch     # PyTorch library for deep learning (used to run YOLO model)

# Importing required functions from YOLOv5 utilities
from utils.general import (
    cv2,                   # OpenCV library for image and video processing
    check_img_size,        # Ensures image size is valid for the model
    non_max_suppression,   # Removes duplicate overlapping detections
    scale_boxes,           # Rescales boxes from model size back to original image size
    print_args             # Prints command-line arguments in a readable format
)

from utils.torch_utils import select_device, smart_inference_mode
# select_device → chooses between CPU or GPU for running YOLO
# smart_inference_mode → automatically sets PyTorch to inference (no training)

from models.common import DetectMultiBackend
# DetectMultiBackend → loads YOLO model (supports different formats like .pt, .onnx)

from utils.dataloaders import LoadImages, LoadStreams
# LoadImages → loads images or videos from files
# LoadStreams → loads live video streams (like webcam or RTSP camera)

from ultralytics.utils.plotting import Annotator, colors
# Annotator → helps draw bounding boxes and labels on images
# colors → provides consistent colors for each object class

# ohter imports
import time


import subprocess

# Directly call espeak-ng executable
def speak(msg, speed):
    exe_path = r"C:\Program Files\eSpeak NG\espeak-ng.exe"  # full path to your exe
    try:
        subprocess.Popen([exe_path, "-v", "en", "-s", str(speed), msg])
    except Exception as e:
        print("TTS failed:", e)


# intiaztions
last_print_time = time.time()

# ---------------------- Path setup ----------------------
FILE = Path(__file__).resolve()  # Get the absolute path of this current file
ROOT = FILE.parents[0]           # Get the parent folder (project root)

# Add project root to Python's import path if not already present
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))



@smart_inference_mode()  # Runs model in inference mode (no gradients = faster, less memory)
def run(weights='yolov5s.pt', source='0', imgsz=640, conf_thres=0.25, iou_thres=0.45,
        device='', classes=None, agnostic_nms=False, line_thickness=2):  # Define function with default args for detection
    global last_print_time
    source = str(source)  # Convert input source (like 0, image.jpg, video.mp4) to string
    
    # Check if source is webcam or online stream
    webcam = source.isnumeric() or source.endswith('.streams') or source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))

    # ------------------ Load model ------------------
    device = select_device(device)  # Select GPU if available, else CPU
    model = DetectMultiBackend(weights, device=device)  # Load YOLOv5 model from weights file
    stride, names, pt = model.stride, model.names, model.pt  # Get model stride, class names, and model format type
    imgsz = check_img_size((imgsz, imgsz), s=stride)  # Make sure image size fits model stride properly

    # ------------------ Load input data ------------------
    if webcam:  # If source is webcam or live stream
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)  # Start reading video stream(s)
        bs = len(dataset)  # Set batch size equal to number of streams
    else:  # If source is image or video file
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)  # Load single image/video
        bs = 1  # Only one image/video at a time

    # ------------------ Model warmup ------------------
    model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # Run one dummy inference to make model ready and faster


    for path, im, im0s, vid_cap, s in dataset:  # Loop through each frame/image in the dataset
        im = torch.from_numpy(im).to(device)  # Convert image (NumPy) to PyTorch tensor and move to device (CPU/GPU)
        im = im.half() if model.fp16 else im.float()  # Use float16 if model supports it (faster), else float32
        im /= 255.0  # Normalize pixel values from [0,255] → [0,1]
        if len(im.shape) == 3:  # If image has no batch dimension
            im = im[None]  # Add one (shape becomes [1,3,h,w])

        # ------------------ Inference ------------------
        pred = model(im)  # Run the model on the image → get raw predictions
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms)
        # Apply Non-Max Suppression to remove duplicate boxes and filter low-confidence ones

        # ------------------ Process detections ------------------
        for i, det in enumerate(pred):  # Go through each detection result
            im0 = im0s[i].copy() if webcam else im0s.copy()  # Copy original image frame
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))  # Prepare for drawing boxes

            # Printing
            detections = []
            for *xyxy, conf, cls in reversed(det):
                conf_val = float(conf)
                cls_name = names[int(cls)]
                
                if conf_val > 0.7:
                    detections.append({
                        "class": cls_name,
                        "confidence": conf_val,
                        "bbox": [float(x) for x in xyxy]
                    })
                    color = (0, 255, 0)  # green

                elif 0.5 < conf_val <= 0.7:
                    detections.append({
                        "class": 'Object',
                        "confidence": conf_val,
                        "bbox": [float(x) for x in xyxy]
                    })
                    color = (0, 255, 255)  # yellow
                
                else:
                    continue
            
            # ====== Printing & Speaking every 8 sec ======
            if time.time() - last_print_time >= 8:
                # Keep detections from the current moment only
                current_classes = {d["class"].capitalize() for d in detections if d["confidence"] > 0.5}

                if current_classes:
                    msg = (
                        f"{list(current_classes)[0]} detected."
                        if len(current_classes) == 1
                        else f"{' and '.join(current_classes)} detected."
                    )

                    print(msg)
                    speak(msg, speed=130)

                else:
                    # Optional: you can speak this if nothing found
                    # speak("No object detected.")
                    pass

                last_print_time = time.time()
            # =============================================

            
            # Draw only filtered detections
            for d in detections:
                xyxy = d["bbox"]
                label = f"{d['class']} {d['confidence']:.2f}"
                annotator.box_label(xyxy, label, color=color)  # single color for consistency

            # ------------------ Displa result ------------------
            im0 = annotator.result()  # Get final annotated frame
            cv2.imshow(str(path), im0)  # Show the image or video frame in window
            if cv2.waitKey(1) == ord('q'):  # Wait for 'q' key to quit
                return  # Exit function safely



def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='yolov5s.pt', help='model path')
    parser.add_argument('--source', type=str, default='0', help='file/dir/URL or webcam')
    parser.add_argument('--imgsz', type=int, default=640, help='inference size')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--device', default='', help='cuda device or cpu')
    parser.add_argument('--line-thickness', type=int, default=2, help='bounding box thickness')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class')
    opt = parser.parse_args()
    print_args(vars(opt))
    return opt


def main(opt):
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
