# Real-Time Face Mask Detection System (YOLOv8)

## Overview

This project presents a real-time face mask detection system developed as part of my Final Year Project. The system detects whether a person is wearing a mask properly, not wearing a mask, or wearing it incorrectly.

## Features

* Real-time detection using webcam input
* Classification into:

  * mask
  * no_mask
  * incorrect_mask
* Visual bounding boxes with confidence scores
* Deployed using Gradio for live interaction

## Model

* Model: YOLOv8
* Classes: mask, no_mask, incorrect_mask
* Training Data: Custom dataset (~2800+ images)

## Deployment

The system is deployed using Gradio and can run in real-time environments such as Hugging Face Spaces.

## Demo

Live Project Link:
https://huggingface.co/spaces/BrownBoy47/face-mask-app

## Project Status

The system is fully developed and deployed.
Currently working on converting this work into a research paper focusing on performance in challenging conditions (low-light, occlusion).

## Future Work

* Improve recall for difficult cases
* Add synthetic data (GAN-based augmentation)
* Edge deployment optimization (Raspberry Pi / Jetson)
* Model explainability using Grad-CAM

## How to Run

```bash
pip install -r requirements.txt
python app.py
```

## Author

Muhammad Umer (Brown Boy)

