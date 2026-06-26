# Real-Time Object Detection + SORT Tracking (Java)

## Overview
This project reads from a webcam or video file and performs real-time object detection and tracking with OpenCV Java and a YOLO ONNX model.

## Files
- `ObjectDetectionTracking.java` — Java source code for detection and SORT-style tracking.
- `coco.names` — COCO class labels.

## Setup
1. Install OpenCV and make sure the Java bindings are available.
2. Download a YOLO ONNX model such as `yolov8n.onnx` and place it in the project folder.

## Compile
```bash
javac -cp path/to/opencv-<version>.jar ObjectDetectionTracking.java
```

## Run
```bash
java -cp .;path/to/opencv-<version>.jar -Djava.library.path=path/to/opencv/build/java/x64 ObjectDetectionTracking --source 0 --model yolov8n.onnx --classes coco.names --conf 0.35 --nms 0.45
```

To use a video file:
```bash
java -cp .;path/to/opencv-<version>.jar -Djava.library.path=path/to/opencv/build/java/x64 ObjectDetectionTracking --source video.mp4 --model yolov8n.onnx --classes coco.names
```

## Notes
- Replace `path/to/opencv-<version>.jar` and `path/to/opencv/build/java/x64` with your OpenCV installation paths.
- Press `Esc` to stop the video window.
