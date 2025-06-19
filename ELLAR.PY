import os
import glob
import cv2
import numpy as np
import csv
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

# -------------------------------------------------------------------
def load_label_file(label_filepath: str, split_name: str, mapping: dict):
    with open(label_filepath, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 2:
                continue
            vid_rel, lbl = parts
            lbl = int(lbl)
            prev = mapping.get(vid_rel)
            if prev and prev[1] in ('train','val'):
                continue
            mapping[vid_rel] = (lbl, split_name)

def compute_snr_and_lux(path: str):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"Cannot open {path}")
    snrs, luxes = [], []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        μ, σ = gray.mean(), gray.std()
        snr = float('inf') if (μ == 0 or σ == 0) else 20.0 * np.log10(μ/σ)
        snrs.append(snr)
        luxes.append(μ)
    cap.release()
    return (float(np.mean(snrs)) if snrs else float('nan'),
            float(np.mean(luxes)) if luxes else float('nan'))

def process_clip(full_path, rel_path, label, split):
    try:
        snr, lux = compute_snr_and_lux(full_path)
        return (rel_path, label, split, snr, lux)
    except Exception as e:
        print(f"⚠️  Failed {rel_path}: {e}")
        return None

# -------------------------------------------------------------------
if __name__ == "__main__":
    # === CONFIG ===
    ROOT_DIR   = r"C:\Users\lenovo\Desktop\projects\ARID-DATASET\ELLAR"
    VIDEOS_DIR = os.path.join(ROOT_DIR, "videos")
    OUTPUT_CSV = os.path.join(ROOT_DIR, "ellar_snr_lux_all_splits.csv")
    MAX_WORKERS = os.cpu_count() or 4
    # ==============

    # 1) Auto-detect label files
    label_files = glob.glob(os.path.join(ROOT_DIR, "ELLAR_label*.txt"))
    if not label_files:
        print(f"❗ No label files matching ELLAR_label*.txt in {ROOT_DIR}")
        exit(1)

    split_map = {}
    for fp in label_files:
        fn = os.path.basename(fp).lower()
        if "train" in fn:
            split_map[fp] = "train"
        elif "val" in fn:
            split_map[fp] = "val"
        else:
            split_map[fp] = "all"

    label_map = {}
    for filepath, split in split_map.items():
        load_label_file(filepath, split, label_map)

    if not label_map:
        print("❗ Found label files, but no entries could be parsed.")
        exit(1)

    # 2) Gather video file paths
    video_tasks = []
    for pat in (os.path.join(VIDEOS_DIR, "**", "*.avi"),
                os.path.join(VIDEOS_DIR, "**", "*.mp4")):
        for full in glob.iglob(pat, recursive=True):
            rel = os.path.relpath(full, VIDEOS_DIR).replace("\\", "/")
            if rel not in label_map:
                print(f"⚠️  No label for {rel}; skipping")
                continue
            lbl, split = label_map[rel]
            video_tasks.append((full, rel, lbl, split))

    if not video_tasks:
        print("❗ No clips to process.")
        exit(1)

    print(f"🔄 Processing {len(video_tasks)} clips using {MAX_WORKERS} threads...")

    # 3) Parallel processing
    entries = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(process_clip, *args) for args in video_tasks]
        for f in as_completed(futures):
            result = f.result()
            if result:
                entries.append(result)

    # 4) Write output
    with open(OUTPUT_CSV, 'w', newline='') as out:
        writer = csv.writer(out)
        writer.writerow(['video_path','label','split','snr_db','lux_estimate'])
        writer.writerows(entries)

    print(f"✅ Done! Processed {len(entries)} clips.")
    print(f"Results saved to:\n   {OUTPUT_CSV}")
