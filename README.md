# Real-Time-3D-LiDAR-Object-Detection
This repository contains a high-performance gap between 2D AI computer vision and 3D LiDAR data. It uses the **Ouster Python SDK** to ingest live or recorded LiDAR data and runs **YOLOv26** instance segmentation and tracking to detect objects, map them back into 3D space, and calculate their real-world distances and coordinates.

## ✨ Features
* **Multi-Format Support:** Works seamlessly with live Ouster sensors (via IP), or recorded `.pcap` and `.osf` files.
* **Dual-Channel Processing:** Processes both Near-Infrared (Near-IR) and Reflectivity channels independently for robust detection.
* **Image Enhancement:** Automatically applies AutoExposure and Beam Uniformity Correction to make LiDAR data resemble camera data, optimizing YOLO's performance.
* **Spatial Projection:** Calculates the exact real-world median distance (in meters) and 3D XYZ coordinates of detected objects.
* **Persistent Tracking:** Maintains object IDs across sequential frames using YOLO's built-in tracking engine.
* **Dual Visualization Modes:**
  * **OpenCV (2D):** Streams a stacked view of the annotated LiDAR channels in real-time.
  * **Ouster SimpleViz (3D):** Renders the full 3D point cloud with instance IDs, class IDs, and RGB masks natively injected into the scan.

## 🛠️ Prerequisites & Installation

Ensure you have Python 3.8 or newer installed. We highly recommend using a virtual environment.

Install the required dependencies using `pip`:

```bash
pip install ouster-sdk ultralytics torch torchvision opencv-python numpy matplotlib
