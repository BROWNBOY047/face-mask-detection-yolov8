# 🛡️ Real-Time Face Mask Detection System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-orange?style=for-the-badge&logo=yolo&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-27338e?style=for-the-badge&logo=OpenCV&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Active%20Research-brightgreen?style=for-the-badge)
[![Live Demo](https://img.shields.io/badge/🤗%20HuggingFace%20Space-Live%20Demo%20→-FFD21E?style=for-the-badge)](https://huggingface.co/spaces/BrownBoy47/face-mask-app)

<br/>

> **Final Year Project (FYP) — BS Computer Science, University of Sargodha**
>
> A deep learning-powered, real-time system that detects and classifies face mask usage across three categories: **Proper Mask**, **Improper Mask**, and **No Mask** — even under challenging real-world conditions like occlusion, low light, and low resolution.

</div>

---

## 📌 Table of Contents

- [Project Overview](#-project-overview)
- [Project Objective](#-project-objective)
- [Scope of the Project](#-scope-of-the-project)
- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [Dataset](#-dataset)
- [Model Pipeline](#-model-pipeline)
- [Results & Metrics](#-results--metrics)
- [Insights Gained](#-insights-gained)
- [Installation](#-installation)
- [Usage](#-usage)
- [Edge Deployment](#-edge-deployment)
- [Project Structure](#-project-structure)
- [Future Work](#-future-work)
- [Author](#-author)
- [License](#-license)

---

## 🔍 Project Overview

This project presents a **production-grade, real-time face mask detection system** built on top of the **YOLOv8** object detection framework. It is capable of detecting faces and classifying their mask-wearing status into three categories simultaneously in live video streams, making it suitable for deployment in public health surveillance and safety monitoring environments.

The system was designed with a focus on **real-world robustness** — trained on a custom-annotated and augmented dataset to handle edge cases such as partial occlusion, low-light environments, and varying face orientations. It also integrates **Grad-CAM explainability** to make model decisions transparent and interpretable.

> ⚠️ **Note:** Full dataset, training scripts, and experiment logs are excluded from this repository due to storage constraints. The trained model weights (`best.pt`) are included for direct inference and testing.

---

## 🎯 Project Objective

The objective of this project is to develop a **real-time face mask detection system** capable of accurately identifying whether individuals are wearing masks:

- ✅ **Properly** — mask covering nose and mouth correctly
- ⚠️ **Improperly** — mask worn incorrectly (e.g., below the nose)
- ❌ **Not at all** — no mask present

The system is designed to maintain **high detection accuracy** even under challenging real-world conditions such as:

- 🌑 Low lighting environments
- 📷 Low resolution / blurry frames
- 🙈 Partial occlusion of faces

The end goal is a system that is **efficient enough** for deployment on **edge devices** such as Raspberry Pi or NVIDIA Jetson for real-time, on-device inference.

---

## 📐 Scope of the Project

This project covers the full end-to-end pipeline of a deep learning system:

| Phase | Description |
|-------|-------------|
| **Data Collection & Annotation** | Custom dataset preparation with manual annotations |
| **Data Augmentation** | Flip, rotate, brightness/contrast shifts, mosaic augmentation |
| **Synthetic Data Generation** | GANs used to generate hard-case scenarios (occlusion, low light) |
| **Model Training** | YOLOv8 fine-tuned on custom dataset via Google Colab (T4 GPU) |
| **Model Evaluation** | Precision, Recall, mAP@50, mAP@50-95 |
| **Explainability** | Grad-CAM visualization for model decision interpretation |
| **Optimization** | Model pruning/quantization for edge device deployment |
| **Real-Time Inference** | Live webcam / video stream detection with alert system |
| **Edge Deployment** | Export and deploy on Raspberry Pi / NVIDIA Jetson |

---

## ✨ Key Features

- 🔴 **Real-Time Detection** — Processes live video at high FPS
- 🏷️ **3-Class Classification** — Proper / Improper / No Mask
- 🌒 **Robust to Difficult Conditions** — Low light, occlusion, blur
- 🧬 **Synthetic Data Augmentation** — GAN-generated hard cases for better generalization
- 🔥 **Grad-CAM Explainability** — Visualize what the model "sees"
- ⚡ **Edge-Optimized** — Designed for Raspberry Pi & NVIDIA Jetson deployment
- 🚨 **Real-Time Alerts** — Triggers alerts for improper/missing mask usage
- 📊 **Comprehensive Evaluation** — Multi-metric model assessment

---

## 🏗️ System Architecture

```
Input (Webcam / Video Stream)
        │
        ▼
┌─────────────────────┐
│  Frame Preprocessing │  ← Resize, Normalize, Format
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│   YOLOv8 Backbone   │  ← CSPDarknet + PANet + Detection Head
│  (Custom Fine-Tuned) │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Detection Output   │  ← Bounding Boxes + Class Labels + Confidence
└────────┬────────────┘
         │
         ├──────────────────────────────────┐
         ▼                                  ▼
┌──────────────────┐              ┌──────────────────────┐
│  Grad-CAM Module │              │  Alert System        │
│  (Explainability)│              │  (Improper / No Mask)│
└──────────────────┘              └──────────────────────┘
         │
         ▼
  Annotated Output Frame
  (Displayed / Saved / Streamed)
```

---

## 📦 Dataset

| Property | Details |
|----------|---------|
| **Total Images** | ~2,800+ |
| **Classes** | Proper Mask, Improper Mask, No Mask |
| **Annotation Format** | YOLO `.txt` format (normalized bounding boxes) |
| **Annotation Tool** | LabelImg / Roboflow |
| **Split** | Train / Validation / Test |
| **Augmentation Techniques** | Horizontal flip, random crop, brightness/contrast shift, mosaic, blur |
| **Synthetic Data** | GAN-generated samples for occluded and low-light scenarios |

> **Note:** Dataset is not included in this repository due to size constraints. The model was trained on a custom collected and annotated dataset designed to cover real-world edge cases not present in standard public datasets.

---

## 🔬 Model Pipeline

### Training Configuration
```yaml
# data.yaml
path: ./dataset
train: images/train
val: images/val
test: images/test

nc: 3
names: ['proper_mask', 'improper_mask', 'no_mask']
```

### Model Training *(conducted on Google Colab T4 GPU)*
```python
from ultralytics import YOLO

model = YOLO('yolov8n.pt')  # Load pretrained YOLOv8 nano

results = model.train(
    data='data.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    name='mask_detection',
    patience=20,
    augment=True
)
```

### Inference with Included Weights
```python
from ultralytics import YOLO

# Load the included trained weights
model = YOLO('best.pt')

# Run on webcam
results = model.predict(source=0, show=True, conf=0.5)
```

---

## 📊 Results & Metrics

> Results on validation set after training:

| Metric | Value |
|--------|-------|
| **mAP@50** | *Updating after final training run* |
| **mAP@50-95** | *Updating after final training run* |
| **Precision** | *Updating after final training run* |
| **Recall** | *Updating after final training run* |
| **Inference Speed** | *Target: real-time @ 30+ FPS* |

> 📌 Results will be updated as research experiments conclude. Training is ongoing for performance enhancement.

---

## 💡 Insights Gained

This project provided deep practical insights into applied machine learning:

**1. High Precision ≠ Strong Recall**
> Achieving high precision on a validation set does not guarantee strong recall, especially in challenging real-world scenarios like low light and occlusion. This highlighted the importance of evaluating models with multiple metrics rather than relying on a single score.

**2. Data Quality Over Model Complexity**
> The model's performance was heavily influenced by the quality and diversity of training data. Limited or biased data consistently led to weaker generalization — reinforcing the importance of data augmentation and synthetic data generation.

**3. Accuracy vs. Efficiency Trade-off**
> Optimizing for edge device deployment requires balancing model complexity against inference speed. Reducing model size while maintaining acceptable accuracy is a real engineering challenge, not just a theoretical one.

**4. YOLO Pipeline Mastery**
> Working hands-on with a full YOLO detection pipeline — preprocessing → training → inference → evaluation — provided a thorough understanding of object detection in production settings.

**5. Explainability is Not Optional**
> Integrating Grad-CAM revealed cases where the model was technically "correct" but for the wrong reasons. Explainability methods are critical for building trustworthy AI systems.

**6. Real-World ≠ Benchmark Datasets**
> Performance on curated datasets rarely reflects real-world behavior. Dealing with overfitting, training instability, and limited compute gave a realistic view of the gap between research benchmarks and deployment conditions.

---

## ⚙️ Installation

### Prerequisites
- Python 3.8+
- pip
- (Recommended) CUDA-compatible GPU

### Step 1: Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/face-mask-detection.git
cd face-mask-detection
```

### Step 2: Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate       # Linux/Mac
venv\Scripts\activate          # Windows
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

### Run Real-Time Detection (Webcam)
```bash
python app.py --source 0 --conf 0.5
```

### Run on Video File
```bash
python app.py --source path/to/video.mp4 --conf 0.5
```

### Run on Image
```bash
python app.py --source path/to/image.jpg --conf 0.5
```

### Direct Python Usage
```python
from ultralytics import YOLO

model = YOLO('best.pt')
results = model.predict(source=0, show=True, conf=0.5)
```

---

## 🖥️ Edge Deployment

### Export Model for Edge Devices

```python
from ultralytics import YOLO

model = YOLO('best.pt')

# Export to ONNX (for Jetson / general edge)
model.export(format='onnx')

# Export to TFLite (for Raspberry Pi)
model.export(format='tflite')

# Export to TensorRT (for NVIDIA Jetson - max performance)
model.export(format='engine', device=0)
```

### Raspberry Pi
```bash
pip install ultralytics opencv-python-headless
python app.py --source 0 --weights best_tflite/best_float32.tflite
```

### NVIDIA Jetson
```bash
# Use TensorRT exported engine for maximum FPS
python app.py --source 0 --weights best.engine
```

---

## 📁 Project Structure

```
face-mask-detection/
│
├── app.py                  # Main inference & demo application
├── best.pt                 # Trained YOLOv8 model weights ✅
├── requirements.txt        # Python dependencies
├── .gitignore
├── LICENSE
└── README.md
```

> 📌 **Note:** Full dataset, training scripts, augmentation pipelines, and experiment logs are excluded from this repository due to storage constraints. The trained weights (`best.pt`) are included for direct inference.

---

## 🔮 Future Work

- [ ] Integrate **self-supervised pre-training** for better feature learning with limited labels
- [ ] Expand dataset with more diverse demographics and environments
- [ ] Full **GAN pipeline** for automated hard-case synthetic data generation
- [ ] Complete **Raspberry Pi & Jetson** edge deployment with benchmarked FPS
- [ ] Build a **web dashboard** for remote monitoring and alert management
- [ ] Explore **transformer-based detection heads** (RT-DETR) for comparison
- [ ] Publish results as a **research paper / conference submission**
- [ ] Add multi-camera support for wider surveillance coverage

---

## 👤 Author

**Muhammad Umer**
BS Computer Science — University of Sargodha, Pakistan

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0077B5?style=for-the-badge&logo=linkedin)](https://www.linkedin.com/in/muhammad-umer-b39171337/)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-181717?style=for-the-badge&logo=github)](https://github.com/BROWNBOY047)

> 🔬 Aspiring ML/Deep Learning Researcher | AI/ML Freelance Engineer
> Targeting fully funded PhD programs in AI/ML (Germany 🇩🇪 | Canada 🇨🇦 | USA 🇺🇸)

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

⭐ **If this project helped you, please consider giving it a star!** ⭐

*Built with ❤️ for research, public safety, and the pursuit of knowledge.*

</div>
