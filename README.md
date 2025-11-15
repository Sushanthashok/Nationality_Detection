# 🧭 Nationality Detection

## 📌 Problem Statement

The objective of this project is to develop an intelligent computer vision system capable of analyzing a person's face and predicting:

Nationality

Emotion

Age (estimated)

Dress color (dominant upper-body region)

The system follows conditional logic:

| Nationality       | Required Predictions      |
| ----------------- | ------------------------- |
| **Indian**        | Emotion, Age, Dress Color |
| **United States** | Emotion, Age              |
| **African**       | Emotion, Dress Color      |
| **Other**         | Emotion only              |

The solution must also include:

A real-time GUI created with Streamlit

Automatic face detection using MTCNN

Ability to handle multiple faces in a single image

This project demonstrates the integration of multi-task deep learning, computer vision pipelines, and practical GUI deployment.

## 📁 Dataset
1. Nationality Dataset – FairFace (Kaggle)

Source: FairFace – A Balanced Race & Gender Dataset

Used for training a 4-class nationality classifier:

Indian

United States

African

Other

Steps performed:

Extracted face images using FairFace labels

Balanced dataset using build_nationality_dataset.py (3000 images per class)

Preprocessed images to 224×224 resolution

## 2. Emotion Dataset – FER2013 (Kaggle)

Used for training a 7-class emotion classifier:

Angry

Disgust

Fear

Happy

Sad

Surprise

Neutral

Preprocessing steps:

Converted pixels → images

Augmentation

Split into training & validation sets

## 3. Additional Components

Age estimation → placeholder (randomized)

Dress color → simple RGB-based region analysis

## 🧠 Methodology

Below is the complete pipeline followed by the system:

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

### 1. Face Detection

MTCNN used to detect bounding boxes

Largest face selected if multiple faces

### 2. Nationality Classification (MobileNetV2)

Trained with transfer learning

Softmax output → top-3 predictions shown

### 3. Emotion Classification (MobileNetV2)

FER2013 dataset

Predicts 7 emotions

### 4. Age Estimation

Lightweight placeholder

Can be upgraded to a regression model

### 5. Dress Color Detection

Extract upper-body ROI

Compute average RGB

Map dominant values → color name

## 📊 Results
Emotion Model (FER2013)

✔ Validation Accuracy: 58%
✔ Good performance on real-world images
✔ Works smoothly with Streamlit

Nationality Model (FairFace)

✔ Validation Accuracy: ~49%
✔ Fine-tuned with MobileNetV2
✔ Balanced dataset improved consistency

Confusion matrix and label distribution files generated:

nationality_confusion_matrix.png

nationality_labels.json

## Streamlit Application Output

The UI displays:

Cropped face

Nationality + confidence

Emotion + confidence

Age (estimated)

Dress color (if applicable)

Top-3 nationality predictions

The system can identify multiple faces in an image.

## 🛠 Technologies Used

Python

TensorFlow / Keras

OpenCV

MTCNN

NumPy

Streamlit

Scikit-learn

Matplotlib / Seaborn

## ▶️ How to Run the App

1. Install dependencies

pip install -r requirements.txt

2. Place models in the models/ folder

models/
 ├── emotion_mobilenetv2.h5
 ├── nationality_mobilenetv2.h5
 └── nationality_labels.json

 3. Run Streamlit

streamlit run app.py

## 📦 Repository Structure

```
Nationality_Detection/
│── app.py
│── train_emotion.py
│── train_nationality.py
│── build_nationality_dataset.py
│── prepare_emotion.py
│── eval_nationality.py
│── models/ (place .h5 files here)
│── data/ (ignored in repository)
│── FairFace/ (local only)
│── README.md
│── requirements.txt
```

## 🎯 Conclusion

This project successfully demonstrates a complete end-to-end AI pipeline combining:

✔ Face detection
✔ Deep learning classification
✔ Multi-task prediction
✔ Conditional output logic
✔ A full GUI-based deployment

It is a strong example of practical computer vision engineering suitable for real-world use cases such as identity analytics, surveillance, and demographic insights.



## Implementation video : [video](https://drive.google.com/file/d/1SDp7J5UU-akh5pnp5u4kxefXmPo7YvSU/view?usp=sharing)


# THANK YOU

