import os
import glob
import cv2
import numpy as np
import csv

def compute_snr_and_lux(avi_path: str):
    cap = cv2.VideoCapture(avi_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file {avi_path}")

    snrs = []
    lux_values = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)

        mean_val = gray.mean()
        std_val = gray.std()

        # SNR computation
        if std_val == 0 or mean_val == 0:
            snr = float('inf')
        else:
            snr = 20.0 * np.log10(mean_val / std_val)
        snrs.append(snr)

        # Lux proxy = mean pixel intensity
        lux_values.append(mean_val)

    cap.release()

    avg_snr = float(np.mean(snrs)) if snrs else float('nan')
    avg_lux = float(np.mean(lux_values)) if lux_values else float('nan')
    return avg_snr, avg_lux

def walk_and_compute_all(root_dir: str):
    results = {}
    pattern = os.path.join(root_dir, '**', '*.avi')
    for avi_path in glob.iglob(pattern, recursive=True):
        rel = os.path.relpath(avi_path, root_dir)
        try:
            snr, lux = compute_snr_and_lux(avi_path)
            results[rel] = (snr, lux)
        except Exception as e:
            print(f"Warning: could not process {rel}: {e}")
    return results

def write_csv(results, out_csv):
    with open(out_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['video', 'snr_db', 'lux_estimate'])
        for video, (snr, lux) in results.items():
            writer.writerow([video, snr, lux])

if __name__ == "__main__":
    ARID_ROOT = r"C:\Users\lenovo\Desktop\projects\ARID-DATASET\archive (1)\clips_v1.5_avi"
    print(f"Scanning ARID videos under:\n  {ARID_ROOT}\n")

    result_dict = walk_and_compute_all(ARID_ROOT)

    if not result_dict:
        print("❗ No videos found!")
        exit(1)

    snrs = np.array([s for s, l in result_dict.values() if np.isfinite(s)])
    luxs = np.array([l for s, l in result_dict.values() if np.isfinite(l)])

    print("\nSNR Statistics (finite only):")
    print(f"  Min:  {snrs.min():.2f} dB")
    print(f"  Max:  {snrs.max():.2f} dB")
    print(f"  Mean: {snrs.mean():.2f} dB")
    print(f"  Std:  {snrs.std():.2f} dB")

    print("\nLux Statistics (finite only):")
    print(f"  Min:  {luxs.min():.2f}")
    print(f"  Max:  {luxs.max():.2f}")
    print(f"  Mean: {luxs.mean():.2f}")
    print(f"  Std:  {luxs.std():.2f}")

    output_path = os.path.join(os.getcwd(), "arid_clip_snr_lux.csv")
    write_csv(result_dict, output_path)
    print(f"\n✅ Per-video SNR & Lux written to: {output_path}")
