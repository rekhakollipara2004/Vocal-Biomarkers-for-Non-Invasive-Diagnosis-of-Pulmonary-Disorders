import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import librosa
import joblib
import os
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# ===================== PATHS =====================
AUDIO_PATH = r"C:\Users\kollipara.Rekha\Documents\Voice based Pulmonary Detection\Project\Respiratory_Sound_Database\audio_and_txt_files"

# ===================== LOAD DATA =====================
train_df = pd.read_csv("train.csv")
val_df   = pd.read_csv("val.csv")

# ===================== LABEL ENCODING =====================
le = LabelEncoder()
y_train = le.fit_transform(train_df.disease)
y_val   = le.transform(val_df.disease)
num_classes = len(le.classes_)

# ===================== FEATURE SHAPES =====================
TARGET_MFCC  = (20, 259)
TARGET_CHROMA = (12, 259)
TARGET_MEL   = (128, 259)

# ===================== AUDIO FEATURE FUNCTIONS =====================
def extract_features(path):
    y, sr = librosa.load(path, duration=5)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    mel = librosa.feature.melspectrogram(y=y, sr=sr)

    return mfcc, chroma, mel

def pad_or_trim(x, target_shape):
    if x.shape[1] < target_shape[1]:
        x = np.pad(x, ((0,0),(0, target_shape[1]-x.shape[1])))
    else:
        x = x[:, :target_shape[1]]
    return x

# ===================== DATASET EXTRACTION =====================
def build_dataset(df):
    mfccs, chromas, mels, valid_idx = [], [], [], []

    for i, row in df.iterrows():
        path = os.path.join(AUDIO_PATH, row.filename)
        try:
            mfcc, chroma, mel = extract_features(path)

            mfcc   = pad_or_trim(mfcc, TARGET_MFCC)
            chroma = pad_or_trim(chroma, TARGET_CHROMA)
            mel    = pad_or_trim(mel, TARGET_MEL)

            mfccs.append(mfcc[..., np.newaxis])
            chromas.append(chroma[..., np.newaxis])
            mels.append(mel[..., np.newaxis])
            valid_idx.append(i)

        except Exception as e:
            print("Skipping:", row.filename)

    return (
        np.array(mfccs),
        np.array(chromas),
        np.array(mels),
        valid_idx
    )

mfcc_train, chroma_train, mel_train, idx_train = build_dataset(train_df)
mfcc_val, chroma_val, mel_val, idx_val = build_dataset(val_df)

y_train = y_train[idx_train]
y_val   = y_val[idx_val]

# ===================== MODEL BLOCKS =====================

# ---- MFCC: CNN + BiLSTM (temporal modeling)
def mfcc_branch(inp):
    x = keras.layers.Conv2D(32, 3, activation="relu", padding="same")(inp)
    x = keras.layers.MaxPooling2D(2)(x)
    x = keras.layers.Conv2D(64, 3, activation="relu", padding="same")(x)

    x = keras.layers.Reshape((x.shape[2], x.shape[1]*x.shape[3]))(x)
    x = keras.layers.Bidirectional(
        keras.layers.LSTM(64, return_sequences=False)
    )(x)

    return x

# ---- Chroma: Pure CNN
def chroma_branch(inp):
    x = keras.layers.Conv2D(16, 5, activation="relu", padding="same")(inp)
    x = keras.layers.MaxPooling2D(2)(x)
    x = keras.layers.Conv2D(32, 3, activation="relu")(x)
    x = keras.layers.GlobalMaxPooling2D()(x)
    return x

# ---- Mel: CNN + Self Attention
def mel_branch(inp):
    x = keras.layers.Conv2D(32, 3, activation="relu", padding="same")(inp)
    x = keras.layers.MaxPooling2D(2)(x)
    x = keras.layers.Conv2D(64, 3, activation="relu", padding="same")(x)
    x = keras.layers.MaxPooling2D(2)(x)

    x = keras.layers.Reshape((x.shape[1]*x.shape[2], x.shape[3]))(x)

    attn = keras.layers.MultiHeadAttention(
        num_heads=4,
        key_dim=64
    )(x, x)

    x = keras.layers.GlobalAveragePooling1D()(attn)
    return x

# ===================== MULTIMODAL MODEL =====================
mfcc_in   = keras.Input(shape=(*TARGET_MFCC,1), name="MFCC_Input")
chroma_in = keras.Input(shape=(*TARGET_CHROMA,1), name="Chroma_Input")
mel_in    = keras.Input(shape=(*TARGET_MEL,1), name="Mel_Input")

mfcc_feat   = mfcc_branch(mfcc_in)
chroma_feat = chroma_branch(chroma_in)
mel_feat    = mel_branch(mel_in)

fusion = keras.layers.Concatenate()([mfcc_feat, chroma_feat, mel_feat])

x = keras.layers.Dense(128, activation="relu")(fusion)
x = keras.layers.Dropout(0.4)(x)
x = keras.layers.Dense(64, activation="relu")(x)

output = keras.layers.Dense(num_classes, activation="softmax")(x)

model = keras.Model(
    inputs=[mfcc_in, chroma_in, mel_in],
    outputs=output,
    name="True_Multimodal_RespiratoryNet"
)

# ===================== COMPILE =====================
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ===================== TRAIN =====================
history = model.fit(
    [mfcc_train, chroma_train, mel_train],
    y_train,
    validation_data=([mfcc_val, chroma_val, mel_val], y_val),
    epochs=50,
    batch_size=16
)

# ===================== SAVE =====================
model.save("hbNet.keras")

np.save("mfcc_val.npy", mfcc_val)
np.save("chroma_val.npy", chroma_val)
np.save("mel_val.npy", mel_val)
np.save("y_val.npy", y_val)
joblib.dump(le, "label_encoder.pkl")

# ===================== PLOT =====================
plt.plot(history.history["accuracy"], label="Train")
plt.plot(history.history["val_accuracy"], label="Validation")
plt.legend()
plt.grid()
plt.show()
