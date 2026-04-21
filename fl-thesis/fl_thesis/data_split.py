import os
import shutil

import numpy as np
import pandas as pd

# === KONFIGURASI ===
BASE_DIR = "./fl-thesis/fl_thesis/dataset/"  # direktori dataset
CSV_PATH = os.path.join(BASE_DIR, "metadata.csv")

IMAGE_DIRS = [
    os.path.join(BASE_DIR, "imgs_part1"),
    os.path.join(BASE_DIR, "imgs_part2"),
    os.path.join(BASE_DIR, "imgs_part3"),
]

NUM_CLIENTS = 3
OUTPUT_DIR = "output_clients"

BENIGN = ["ACK", "SEK", "NEV"]
MALIGNANT = ["BCC", "SCC", "MEL"]

# === BACA METADATA ===
df = pd.read_csv(CSV_PATH)


# Buat kolom binary_label
def map_label(diag):
    if diag in BENIGN:
        return 0
    elif diag in MALIGNANT:
        return 1
    return None


df["binary_label"] = df["diagnostic"].apply(map_label)
df = df.dropna(subset=["binary_label"])

# --- BANGUN DICTIONARY PENCARIAN GAMBAR ---
image_lookup = {}

print("Membangun indeks gambar dari imgs_part1/2/3 ...")
for folder in IMAGE_DIRS:
    for file in os.listdir(folder):
        if file.endswith(".png"):
            image_lookup[file] = os.path.join(folder, file)

# Tambah kolom path image berdasarkan lookup
df["image_path"] = df["img_id"].apply(lambda x: image_lookup.get(x, None))

missing = df["image_path"].isna().sum()
if missing > 0:
    print(f"PERINGATAN: {missing} gambar tidak ditemukan!")

df = df.dropna(subset=["image_path"])

# Shuffle untuk IID
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

split_size = len(df)

# Split IID
splits = []

for i in range(NUM_CLIENTS - 1):
    splits.append(df.iloc[i * split_size : (i + 1) * split_size].reset_index(drop=True))

splits.append(df.iloc[(NUM_CLIENTS - 1) * split_size :].reset_index(drop=True))

# Bersihkan output
if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)
os.makedirs(OUTPUT_DIR)

# === COPY DATA PER KLIEN ===
for i, split_df in enumerate(splits, start=1):
    client_dir = os.path.join(OUTPUT_DIR, f"client_{i}")
    images_out = os.path.join(client_dir, "images")

    os.makedirs(images_out, exist_ok=True)

    # Simpan metadata
    split_df.to_csv(os.path.join(client_dir, "metadata.csv"), index=False)

    print(f"Menyalin gambar untuk client_{i} ...")

    for _, row in split_df.iterrows():
        src = row["image_path"]
        dst = os.path.join(images_out, os.path.basename(src))
        shutil.copy(src, dst)

print("\n=== SELESAI ===")
print(f"Dataset dibagi menjadi {NUM_CLIENTS} klien di folder: {OUTPUT_DIR}")
