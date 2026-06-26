# 🚀 Real-Time Object Detection & SORT Tracking (Java)

Detect and track multiple objects in **real time** using **OpenCV Java** and a **YOLO ONNX** model. The application supports both **webcam** and **video file** input while assigning a unique ID to each detected object using a SORT-style tracker.

---

## 📸 Demo

> Add screenshots or a GIF of your project here.

| Detection | Tracking |
|-----------|----------|
| 📦 Bounding Boxes | 🆔 Unique Object IDs |
| 🎯 YOLO Detection | 🚗 Multi-Object Tracking |

---

# ✨ Features

- 🎯 Real-time object detection
- 🆔 SORT-style object tracking
- 📷 Webcam support
- 🎥 Video file support
- 🧠 YOLO ONNX model support
- 📋 COCO class labels
- 📦 Bounding boxes with labels
- 📈 Confidence score display
- ⚡ Fast OpenCV DNN inference
- ⚙️ Adjustable confidence threshold
- 🚀 Adjustable NMS threshold

---

# 📁 Project Structure

```
Project/
│
├── ObjectDetectionTracking.java
├── coco.names
├── yolov8n.onnx
└── README.md
```

---

# 🛠 Requirements

Before running the project, install:

- ☕ Java JDK 11+
- 👁 OpenCV Java
- 📚 OpenCV Java Bindings
- 🤖 YOLO ONNX Model
- 📝 COCO Labels (`coco.names`)

---

# 📥 Setup

## 1️⃣ Install OpenCV

Download and install OpenCV.

Make sure that:

- ✅ OpenCV JAR is added to your project.
- ✅ Native OpenCV library path is configured.

---

## 2️⃣ Download YOLO Model

Download any supported YOLO ONNX model.

Example:

```
yolov8n.onnx
```

Place it inside the project folder.

---

## 3️⃣ Add COCO Labels

Download or copy:

```
coco.names
```

Place it in the project folder.

---

# 🔨 Compile

```bash
javac -cp path/to/opencv-<version>.jar ObjectDetectionTracking.java
```

Example:

```bash
javac -cp opencv-490.jar ObjectDetectionTracking.java
```

---

# ▶️ Run with Webcam

```bash
java -cp .;path/to/opencv-<version>.jar ^
-Djava.library.path=path/to/opencv/build/java/x64 ^
ObjectDetectionTracking ^
--source 0 ^
--model yolov8n.onnx ^
--classes coco.names ^
--conf 0.35 ^
--nms 0.45
```

---

# 🎥 Run with Video

```bash
java -cp .;path/to/opencv-<version>.jar ^
-Djava.library.path=path/to/opencv/build/java/x64 ^
ObjectDetectionTracking ^
--source video.mp4 ^
--model yolov8n.onnx ^
--classes coco.names
```

---

# ⚙️ Command Line Arguments

| Argument | Description | Default |
|-----------|-------------|---------|
| `--source` | Webcam index or video path | `0` |
| `--model` | YOLO ONNX model | Required |
| `--classes` | COCO labels | Required |
| `--conf` | Confidence threshold | `0.35` |
| `--nms` | NMS threshold | `0.45` |

---

# 📺 Output

The application displays:

✅ Bounding Boxes

✅ Class Names

✅ Confidence Score

✅ Tracking IDs

Example:

```
🧍 Person 98%  ID:3
🚗 Car    95%  ID:7
🐶 Dog    91%  ID:2
```

---

# ⌨️ Controls

| Key | Action |
|-----|--------|
| ⎋ Esc | Exit the application |

---

# 📌 Notes

- 📂 Replace OpenCV paths with your installation paths.
- 🤖 Ensure `yolov8n.onnx` is inside the project folder.
- 📋 Ensure `coco.names` is available.
- 💻 Compatible with Windows, Linux, and macOS.
- 🔄 Supports any YOLO model exported to ONNX (with compatible preprocessing).

---

# 🚀 Future Improvements

- 🔥 Deep SORT
- 🚀 ByteTrack
- 🎯 YOLOv11 Support
- ⚡ CUDA GPU Acceleration
- 📊 Object Counting
- 🚦 Line Crossing Detection
- 🗺 Zone Monitoring
- 🎥 Video Recording
- 📈 FPS Counter
- 🌐 RTSP Camera Support

---

# 🖼 Example

```
+--------------------------------------+
| 🧍 Person (98%)      ID: 4           |
|                                      |
|              📦                      |
|                                      |
+--------------------------------------+
```

---

# ❤️ Built With

- ☕ Java
- 👁 OpenCV
- 🤖 YOLO
- 📦 ONNX Runtime / OpenCV DNN
- 🎯 SORT Tracking

---

# 👨‍💻 Author

Made with ❤️ using **Java**, **OpenCV**, and **YOLO ONNX**.
