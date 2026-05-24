import numpy as np
import pandas as pd
import os
import librosa
import joblib
from tensorflow import keras
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate
import warnings

warnings.filterwarnings("ignore")

# ===================== LOAD MODEL =====================
print("Loading model...")
model = keras.models.load_model("hbNet.keras")
le = joblib.load("label_encoder.pkl")
print("Model loaded successfully\n")

# ===================== PATHS =====================
BASE_PATH = r"C:\Users\kollipara.Rekha\Documents\Voice based Pulmonary Detection\Project\Respiratory_Sound_Database"
AUDIO_PATH = os.path.join(BASE_PATH, "audio_and_txt_files")
CSV_PATH = os.path.join(BASE_PATH, "csv", "segment_data.csv")

segment_df = pd.read_csv(CSV_PATH)

# ===================== FEATURE SETTINGS =====================
TARGET_MFCC  = (20, 259)
TARGET_CHROMA = (12, 259)
TARGET_MEL   = (128, 259)

def pad_or_trim(x, target_shape):
    if x.shape[1] < target_shape[1]:
        x = np.pad(x, ((0,0),(0, target_shape[1]-x.shape[1])))
    else:
        x = x[:, :target_shape[1]]
    return x

def extract_features(path):
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return None

    y, sr = librosa.load(path, duration=5)

    # Skip silent audio
    if len(y) == 0 or np.mean(np.abs(y)) < 0.0005:
        print(f"Silent file skipped: {path}")
        return None

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, tuning=0.0)
    mel = librosa.feature.melspectrogram(y=y, sr=sr)

    mfcc   = pad_or_trim(mfcc, TARGET_MFCC)
    chroma = pad_or_trim(chroma, TARGET_CHROMA)
    mel    = pad_or_trim(mel, TARGET_MEL)

    return (
        mfcc[np.newaxis, ..., np.newaxis],
        chroma[np.newaxis, ..., np.newaxis],
        mel[np.newaxis, ..., np.newaxis]
    )

# ===================== PATIENT LEVEL PROCESSING =====================
results = []

print("Starting patient-level prediction...\n")

for pid in segment_df["pid"].unique():

    patient_segments = segment_df[segment_df["pid"] == pid]

    segment_preds = []
    segment_probs = []

    for _, row in patient_segments.iterrows():
        file_path = os.path.join(AUDIO_PATH, row["filename"])

        features = extract_features(file_path)

        if features is None:
            continue

        try:
            mfcc, chroma, mel = features
            prob = model.predict([mfcc, chroma, mel], verbose=0)
            pred_class = np.argmax(prob)

            segment_preds.append(pred_class)
            segment_probs.append(prob[0][pred_class])

        except Exception as e:
            print(f"Prediction error for {file_path}: {e}")
            continue

    if len(segment_preds) == 0:
        print(f"No valid segments for patient {pid}")
        continue

    majority_class = Counter(segment_preds).most_common(1)[0][0]
    mean_prob = np.mean(segment_probs)

    if mean_prob < 0.6:
        severity = "Mild"
    elif mean_prob < 0.8:
        severity = "Moderate"
    else:
        severity = "Severe"

    final_diagnosis = le.inverse_transform([majority_class])[0]

    results.append({
        "Patient ID": pid,
        "Final Diagnosis": final_diagnosis,
        "Mean Confidence": round(mean_prob, 3),
        "Severity Level": severity
    })

# ===================== CHECK IF RESULTS EMPTY =====================
if len(results) == 0:
    print("\n❌ No predictions generated.")
    print("Check:")
    print("1. Audio file paths")
    print("2. segment_data.csv filenames")
    print("3. Model input shape")
    exit()

results_df = pd.DataFrame(results)

# ===================== PRINT TABLE =====================
print("\nPatient-Level Prediction Results:\n")
print(tabulate(results_df, headers="keys", tablefmt="grid"))

# ===================== SAVE CSV =====================
results_df.to_csv("patient_level_results.csv", index=False)
print("\n✅ patient_level_results.csv saved")

# ===================== DIAGNOSIS DISTRIBUTION =====================
plt.figure(figsize=(8,5))
sns.countplot(x=results_df["Final Diagnosis"])
plt.title("Patient-Level Diagnosis Distribution")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ===================== SEVERITY DISTRIBUTION =====================
plt.figure(figsize=(6,4))
sns.countplot(x=results_df["Severity Level"],
              order=["Mild", "Moderate", "Severe"])
plt.title("Severity Level Distribution")
plt.tight_layout()
plt.show()
