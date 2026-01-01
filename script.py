import os, shutil, pandas as pd
from sklearn.model_selection import train_test_split

# ==== PATHS (RAW STRINGS) ====
CSV = r"D:\abc\B. Disease Grading\2. Groundtruths\a. IDRiD_Disease Grading_Training Labels.csv"
IMG_DIR = r"D:\abc\B. Disease Grading\1. Original Images\a. Training Set"
DEST = r"D:\abc\dataset\images"

# ==== LOAD CSV ====
df = pd.read_csv(CSV)

# ==== TRAIN–VALIDATION SPLIT ====
train_df, val_df = train_test_split(
    df,
    test_size=0.2,
    stratify=df['Retinopathy grade'],
    random_state=42
)

# ==== CREATE CLASS FOLDERS (0–4) ====
for i in range(5):
    os.makedirs(os.path.join(DEST, "train", str(i)), exist_ok=True)
    os.makedirs(os.path.join(DEST, "val", str(i)), exist_ok=True)

# ==== COPY IMAGES ====
def copy_images(data, folder):
    for _, row in data.iterrows():
        img = row['Image name'] + ".jpg"
        label = str(row['Retinopathy grade'])
        src = os.path.join(IMG_DIR, img)
        dst = os.path.join(DEST, folder, label, img)

        if os.path.exists(src):
            shutil.copy(src, dst)
        else:
            print("Missing:", src)

copy_images(train_df, "train")
copy_images(val_df, "val")

print("Dataset successfully prepared!")
