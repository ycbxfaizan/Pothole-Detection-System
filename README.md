# Pothole Detection and Classification Using YOLOv8

This repository contains the code, exploratory analysis, and performance evaluations for an automated pothole detection system using the state-of-the-art YOLOv8 deep learning architecture. The project focuses on identifying and localizing both dry and water-filled potholes to assist in proactive road maintenance and improve public safety.

## 🚀 Project Overview

Road infrastructure maintenance is a critical challenge for urban management. This project develops a computer vision-based system that can identify road hazards in real-time. By comparing different scales of the YOLOv8 model (Nano and Small), we evaluate the trade-offs between detection accuracy and computational efficiency, specifically for "AI-on-the-Edge" deployment.

### Key Objectives:
- **Data Acquisition & Preprocessing**: Curation of road surface imagery from Mendeley Data, specifically focusing on multi-class annotations.
- **Exploratory Data Analysis (EDA)**: Detailed statistical analysis of class distributions and bounding box characteristics to ensure model robustness.
- **Model Implementation**: Fine-tuning YOLOv8n (Nano) and YOLOv8s (Small) variants using the PyTorch-based Ultralytics framework.
- **Performance Evaluation**: Comprehensive benchmarking using mAP, Precision, Recall, and Inference Speed (ms) to determine the best candidate for real-time systems.

## 📂 Repository Structure

- **eda_yolo.ipynb**: Contains the full exploratory data analysis workflow, including class frequency visualization, spatial heatmaps of pothole locations, and dataset quality checks.
- **yolov8n.ipynb**: The training and validation pipeline for the YOLOv8 Nano model (~3.2M parameters). Optimized for high FPS on edge devices.
- **yolo_v8s.ipynb**: The training and validation pipeline for the YOLOv8 Small model (~11.1M parameters). Targeted at higher precision in complex road environments.
- **results.csv**: Logged training metrics (Box Loss, Class Loss, mAP) used for generating convergence plots.

## 🛠️ Requirements

The following dependencies are required to run the notebooks:

- **Python**: 3.x
- **Deep Learning**: PyTorch, Ultralytics YOLOv8
- **Data Science**: Pandas, NumPy, Matplotlib, Seaborn
- **Hardware**: NVIDIA GPU with CUDA support is recommended for training.

### Installation


pip install ultralytics torch torchvision torchaudio matplotlib seaborn pandas numpy



## 📈 Model Performance

Based on our experimental results on the validation set (143 images), the models achieved the following performance metrics:

| Model Variant   | Precision | Recall | mAP@50 | Inference Speed |
|-----------------|----------|--------|--------|-----------------|
| YOLOv8n (Nano)  | 0.706    | 0.638  | 0.678  | 7.6 ms          |
| YOLOv8s (Small) | 0.759    | 0.589  | 0.670  | 9.7 ms          |

### Key Finding:
The Nano variant demonstrated superior recall and speed, making it the most suitable candidate for real-time detection where missing a hazard is more critical than a false alarm.

## 📖 Usage

### Clone the Repository:

git clone https://github.com/ycbxfaizan/Pothole-Detection-System/.git.

cd pothole-detection

### 📖 Usage

#### Run EDA:
Open `eda_yolo.ipynb` to visualize class distributions and dataset characteristics.

### Run Yolov8n
!python train_potholes.py \
  --model yolov8n.pt \
  --epochs 500 \
  --imgsz 640 \
  --batch 16 \
  --val_ratio 0.2 \
  --project /content/runs_yolo \
  --name potholes_yolov8n

#### Train Models:
Execute `yolov8n.ipynb` or `yolo_v8s.ipynb` to reproduce the training results.

#### Inference:


from ultralytics import YOLO

model = YOLO('path/to/best.pt')
results = model.predict(source='image_or_video.jpg', conf=0.25)
results[0].show()

## 🔮 Future Work

- **Edge Hardware Integration**: Deploying the Nano model on NVIDIA Jetson or Google Coral for on-vehicle processing.
- **Temporal Tracking**: Implementing multi-frame analysis to reduce flickering and false positives from transient reflections.
- **Multi-Modal Sensing**: Combining RGB data with depth sensors to estimate pothole volume and prioritize repairs based on severity.

---

## 📜 References

This project builds upon existing research in the field, including classical approaches such as:

- Wang, P. et al. (2017). ‘Asphalt pavement pothole detection and segmentation based on wavelet energy field’.

### Dataset:
- Dib, J., Sirlantzis, K. and Howells, G. (2023). Mendeley Data.
