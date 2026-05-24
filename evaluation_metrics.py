import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from tensorflow import keras
from sklearn.preprocessing import label_binarize

from sklearn.metrics import (
    roc_curve, auc,
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# ===================== LOAD SAVED MODEL =====================
model = keras.models.load_model("hbNet.keras")

# ===================== LOAD SAVED VALIDATION DATA =====================
mfcc_val   = np.load("mfcc_val.npy")
chroma_val = np.load("chroma_val.npy")
mel_val    = np.load("mel_val.npy")
y_val      = np.load("y_val.npy")

le = joblib.load("label_encoder.pkl")

# ===================== PREDICTIONS =====================
y_prob = model.predict([mfcc_val, chroma_val, mel_val])
y_pred = np.argmax(y_prob, axis=1)

class_names = le.classes_
n_classes = len(class_names)

# ===================== OVERALL ACCURACY =====================
print("Overall Accuracy:", accuracy_score(y_val, y_pred))

# ===================== CLASSIFICATION REPORT =====================
print("\nClassification Report:\n")
print(classification_report(
    y_val,
    y_pred,
    target_names=class_names,
    digits=4
))


# ===================== CONFUSION MATRIX =====================
cm = confusion_matrix(y_val, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d",
            xticklabels=class_names,
            yticklabels=class_names,
            cmap="Blues")

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()


# ===================== ROC-AUC =====================
y_val_bin = label_binarize(y_val, classes=range(n_classes))

plt.figure(figsize=(8, 6))

for i in range(n_classes):
    fpr, tpr, _ = roc_curve(y_val_bin[:, i], y_prob[:, i])
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, lw=2,
             label=f"{class_names[i]} (AUC = {roc_auc:.3f})")

plt.plot([0, 1], [0, 1], "k--", lw=1)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Multiclass ROC–AUC Curve")
plt.legend(loc="lower right")
plt.grid()
plt.show()
