import os
import pandas as pd

BASE_PATH = r"C:\Users\kollipara.Rekha\Documents\Voice based Pulmonary Detection\Project\Respiratory_Sound_Database"
AUDIO_TXT_PATH = os.path.join(BASE_PATH, "audio_and_txt_files")
OUTPUT_DIR = r"C:\Users\kollipara.Rekha\Documents\Voice based Pulmonary Detection\Project\Respiratory_Sound_Database\csv"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------- LOAD DIAGNOSIS ----------------
diagnosis = pd.read_csv(
    os.path.join(BASE_PATH, "patient_diagnosis.csv"),
    names=["pid", "disease"]
)
diagnosis["pid"] = diagnosis["pid"].astype(int)

rows = []

for file in os.listdir(AUDIO_TXT_PATH):
    if not file.endswith(".txt"):
        continue

    pid = int(file.split("_")[0])
    ann = pd.read_csv(
        os.path.join(AUDIO_TXT_PATH, file),
        sep="\t",
        names=["start", "end", "crackles", "wheezes"]
    )
    ann["pid"] = pid
    ann["filename"] = file.replace(".txt", ".wav")
    rows.append(ann)

segment_df = pd.concat(rows).reset_index(drop=True)
segment_df = pd.merge(segment_df, diagnosis, on="pid")

segment_df.to_csv(
    os.path.join(OUTPUT_DIR, "segment_data.csv"),
    index=False
)

print("✅ segment_data.csv created")
print(segment_df.head())

