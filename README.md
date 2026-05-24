<p align="center">
  <h1 align="center">🫁 Vocal Biomarkers for Non-Invasive Diagnosis of Pulmonary Disorders</h1>
  <p align="center">A multimodal deep learning framework for automated respiratory disease classification using lung sound biomarkers</p>
</p>

---

## 📌 Overview

Early and accurate diagnosis of respiratory diseases like **COPD**, **Pneumonia**, and **Bronchiectasis** is critical in reducing patient mortality — yet traditional diagnosis requires specialist equipment unavailable in rural areas.

This project proposes a **hybrid CNN-BiLSTM-Attention deep learning model** that classifies pulmonary disorders directly from lung sound recordings, without any invasive procedure. Three acoustic feature modalities — **MFCC**, **Chroma**, and **Mel Spectrogram** — are fused together to capture spectral, harmonic, and temporal patterns in respiratory audio.

---

## ✨ Key Features

- 🔉 **Multimodal acoustic feature extraction** — MFCC + Chroma + Mel Spectrogram fusion
- 🧠 **Hybrid deep learning architecture** — CNN + BiLSTM + Multi-Head Self-Attention
- 🏥 **Patient-level prediction** via majority voting across respiratory segments
- ⚖️ **Class imbalance handling** using balanced class weighting
- 📊 **Severity estimation module** for clinical interpretability
- 🎯 **86.52% accuracy** with ROC-AUC > 0.90 across all disease classes

---

## 🏗️ Model Architecture

```
Audio Input
    │
    ├──► MFCC Branch        ──► CNN + BiLSTM
    ├──► Chroma Branch      ──► Lightweight CNN
    └──► Mel Spectrogram    ──► CNN + Multi-Head Self-Attention
                │
          Feature Fusion (Concatenation)
                │
          Fully Connected Layers + Dropout
                │
          Softmax Output (6 Classes)
```

| Branch | Input | Layers | Purpose |
|--------|-------|--------|---------|
| MFCC | Cepstral coefficients | CNN + BiLSTM | Local spectral + temporal patterns |
| Chroma | Harmonic features | Lightweight CNN | Energy distribution across pitch classes |
| Mel Spectrogram | Time-frequency map | CNN + Self-Attention | Global spectro-temporal patterns |

---

## 📂 Dataset

**ICBHI 2017 Respiratory Sound Database**
> 920 annotated lung sound recordings from 126 patients using various stethoscope devices.
<a href = "https://www.kaggle.com/datasets/vbookshelf/respiratory-sound-database">Dataset</a>

## 📈 Results

| Metric | Score |
|--------|-------|
| ✅ Overall Accuracy | **86.52%** |
| 📉 ROC-AUC (all classes) | **> 0.90** |
| 🎯 COPD Precision & Recall | High (dominant class) |
| 🗳️ Prediction Strategy | Majority voting (patient-level) |

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?style=flat-square&logo=keras&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![Librosa](https://img.shields.io/badge/Librosa-Audio%20Processing-blueviolet?style=flat-square)

---

🔬 Methodology

Preprocessing — Audio resampled to a common rate, trimmed/padded to fixed length, feature-normalized
Feature Extraction — MFCC, Chroma, and Mel Spectrogram extracted per respiratory cycle
Model Training — Adam optimizer + categorical cross-entropy, dropout regularization, early stopping
Class Balancing — Stratified splits + class weight balancing to handle minority classes
Patient-Level Prediction — Majority voting across all respiratory segments of a patient
Evaluation — Accuracy, Precision, Recall, F1-Score, ROC-AUC, Confusion Matrix
---

## 🚀 Future Work

> 📱 **Real-time Deployment** — Deploy as a web or mobile application with digital stethoscope integration
> ⚖️ **Improved Aggregation** — Replace majority voting with weighted voting or attention-based aggregation
> 🗄️ **Expanded Dataset** — Include varied populations and rare disease classes
> 📈 **Severity Prediction** — Improve using longitudinal patient data and risk scoring
