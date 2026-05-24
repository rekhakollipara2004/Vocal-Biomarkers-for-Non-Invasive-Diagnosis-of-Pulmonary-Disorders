import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

# ================= PATHS =================
BASE_PATH = r"C:\Users\kollipara.Rekha\Documents\Voice based Pulmonary Detection\Project"
DB_PATH = os.path.join(BASE_PATH, "Respiratory_Sound_Database")
AUDIO_PATH = os.path.join(DB_PATH, "audio_and_txt_files")

# ================= LOAD DIAGNOSIS =================
diagnosis = pd.read_csv(
    os.path.join(DB_PATH, "patient_diagnosis.csv"),
    names=["pid", "disease"]
)
diagnosis["pid"] = diagnosis["pid"].astype(int)

print("\nOriginal class distribution:")
print(diagnosis.disease.value_counts())

# ================= BUILD FILE TABLE (ONLY WAV) =================
def extract_pid(filename):
    return int(filename.split("_")[0])

rows = []
for f in os.listdir(AUDIO_PATH):
    if f.endswith(".wav"):
        rows.append({
            "pid": extract_pid(f),
            "filename": f
        })

files_df = pd.DataFrame(rows)

# ================= MERGE =================
data = pd.merge(files_df, diagnosis, on="pid")

# ================= REMOVE ULTRA-RARE CLASSES =================
# COPD dominates, Asthma/LRTI are too small to learn from
MIN_SAMPLES = 5

class_counts = data.disease.value_counts()
valid_classes = class_counts[class_counts >= MIN_SAMPLES].index

data = data[data.disease.isin(valid_classes)]

print("\nAfter removing rare classes:")
print(data.disease.value_counts())

# ================= VISUALIZE IMBALANCE =================
plt.figure(figsize=(8,4))
sns.countplot(x=data.disease)
plt.xticks(rotation=45, fontsize=12)
plt.yticks(fontsize=12)
plt.title("Class Distribution After Cleaning")
plt.tight_layout()
plt.show()

# ================= TRAIN / VALIDATION SPLIT =================
X = data[["filename", "pid"]]
y = data["disease"]

Xtrain, Xval, ytrain, yval = train_test_split(
    X,
    y,
    stratify=y,
    test_size=0.25,
    random_state=42
)

print("\nTrain distribution:")
print(ytrain.value_counts(normalize=True))

print("\nValidation distribution:")
print(yval.value_counts(normalize=True))

# ================= SAVE CSV FILES =================
train_df = Xtrain.copy()
train_df["disease"] = ytrain.values

val_df = Xval.copy()
val_df["disease"] = yval.values

train_df.to_csv("train.csv", index=False)
val_df.to_csv("val.csv", index=False)

print("\n✅ train.csv and val.csv saved")

# ================= COMPUTE CLASS WEIGHTS =================
classes = np.unique(ytrain)

weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=ytrain
)

class_weights = dict(zip(classes, weights))

print("\nClass weights (USE THESE IN model.fit):")
print(class_weights)

# ================= SAVE CLASS WEIGHTS =================
np.save("class_weights.npy", class_weights)
print("\n✅ class_weights.npy saved")
