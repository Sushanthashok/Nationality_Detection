### 🌍 NATIONALITY DETECTION

Author: SUSHANTH A

## 🧠 Project Overview

This project implements an advanced deep learning–based multi-task prediction system that analyzes a person’s face and predicts:

🟦 Nationality (Indian, United States, African, Other)

😊 Emotion (7-class MobileNetV2 model)

🎂 Age (rule-based placeholder)

👕 Dress Color (KMeans dominant color detection)

A clean Streamlit UI allows users to upload images, preview detected faces, and view all predicted attributes interactively.

## 🎯 Objectives

Build a full ML pipeline (data → training → evaluation → deployment)

Detect faces reliably using MTCNN

Perform multi-task predictions with conditional logic

Deploy a production-ready Streamlit GUI

Provide internship-ready, industry-standard implementation

## 🏗 System Architecture

```text
Input Image  
   ↓  
MTCNN Face Detection  
   ↓  
Face Crop  
   ↓  
+-----------------------------+
| Multi-task Prediction Block |
+-----------------------------+
       ↓        ↓       ↓       ↓
  Nationality  Emotion  Age   Dress Color
      ↓          ↓       ↓        ↓
  Conditional Output Logic → Final UI Result
```

## 🗂 Dataset Details

# 🔹 1. Nationality Dataset

Created using FairFace (train + val).
Balanced 4-class dataset:

1) Indian

2) United States

3) African

4) Other

Each class: 3000 face images → 12,000 total

## Folder structure:

data/nationality/
    Indian/
    United States/
    African/
    Other/

# 🔹 2. Emotion Dataset (FER-2013)

7 emotion classes

~35k training images

Used for MobileNetV2 training

## 🧪 Model Training

🟦 Nationality Model

Backbone: MobileNetV2

Input: 224×224

Loss: Categorical Crossentropy

Optimizer: Adam

Epochs: 12 + 4 (fine-tuning)

Validation Accuracy: ≈ 48–50%

## 😊 Emotion Model

Backbone: MobileNetV2

Accuracy: ≈ 58%

## 🎨 Dress Color Detection

Torso extraction based on face box

HSV filtering

KMeans clustering

Maps dominant color to 11 named colors

## 🎂 Age Prediction

Simple placeholder generating realistic ages (18–40)

## 🖥 Streamlit Application

Features:

Face detection (MTCNN)

Smart largest-face filtering

Cropped face preview

Nationality prediction

Emotion prediction

Conditional logic for age/dress color

Top-3 nationality scores

Color confidence display

Works fully offline

## 📁 Project Structure

```text

Nationality_detection/
│
├── app.py
├── train_emotion.py
├── train_nationality.py
├── eval_nationality.py
├── build_nationality_dataset.py
│
├── data/
│   ├── nationality/
│   └── emotion/
│
├── models/
│   ├── emotion_mobilenetv2.h5
│   ├── nationality_mobilenetv2.h5
│   └── nationality_labels.json
│
└── README.md
```

## ⚙️ Installation
pip install streamlit tensorflow mtcnn opencv-python pillow numpy seaborn

## ▶️ Run the App
streamlit run app.py

📌 Evaluation (Nationality Model)
Class	Precision	Recall	F1-score
African	1.00	0.04	0.07
Indian	0.37	0.57	0.45
Other	0.31	0.02	0.04
United States	0.38	0.88	0.53

Overall Accuracy: ~38–50%
(Reasonable due to race→nationality label conversion in FairFace.)

## 🧩 Key Features

✔ MTCNN face detection

✔ Nationality classification (4-way)

✔ Emotion recognition (7-way)

✔ Dress color via KMeans clustering

✔ Age estimation placeholder

✔ Conditional logic

✔ Polished Streamlit UI

✔ Offline-ready

✔ Internship-grade system

## 🚀 Future Improvements

Real age regression model

Better nationality dataset

Gender classification

Background removal

TFLite/ONNX optimization

Real-time webcam mode

# Implementation video : [video](https://drive.google.com/file/d/1SDp7J5UU-akh5pnp5u4kxefXmPo7YvSU/view?usp=sharing)

## 🏁 Conclusion

This project demonstrates a complete end-to-end deep learning system, combining:

Computer vision

Multi-task learning

Model training

Dataset engineering

UI/UX design

Deployment skills

