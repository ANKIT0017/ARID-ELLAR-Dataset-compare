# import pandas as pd
# import matplotlib.pyplot as plt
# import numpy as np
# from scipy.stats import linregress

# # Load CSV
# df = pd.read_csv("arid_clip_snr_lux.csv")

# # Filter out invalid entries
# df = df.replace([np.inf, -np.inf], np.nan).dropna()

# # Scatter plot
# plt.figure(figsize=(10, 6))
# plt.scatter(df['lux_estimate'], df['snr_db'], alpha=0.5, c='royalblue', label='Videos')

# # Linear regression (trendline)
# slope, intercept, r_value, _, _ = linregress(df['lux_estimate'], df['snr_db'])
# x_vals = np.linspace(df['lux_estimate'].min(), df['lux_estimate'].max(), 100)
# y_vals = slope * x_vals + intercept
# plt.plot(x_vals, y_vals, color='red', linestyle='--', label=f'Trendline (R²={r_value**2:.2f})')

# # Labels and aesthetics
# plt.title("SNR vs Lux Estimate for ARID Dataset Videos", fontsize=14)
# plt.xlabel("Lux Estimate (avg. brightness)", fontsize=12)
# plt.ylabel("SNR (dB)", fontsize=12)
# plt.grid(True, linestyle='--', alpha=0.3)
# plt.legend()
# plt.tight_layout()

# # Save and show
# plt.savefig("snr_vs_lux_plot.png", dpi=300)
# plt.show()

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress
import seaborn as sns

# === Load Data ===
ellar = pd.read_csv("ellar_snr_lux.csv")
arid = pd.read_csv("arid_clip_snr_lux.csv")

ellar = ellar.replace([np.inf, -np.inf], np.nan).dropna()
arid = arid.replace([np.inf, -np.inf], np.nan).dropna()

ellar["dataset"] = "ELLAR"
arid["dataset"] = "ARID"

ellar_df = ellar[["snr_db", "lux_estimate", "dataset"]]
arid_df = arid[["snr_db", "lux_estimate", "dataset"]]
df = pd.concat([ellar_df, arid_df], ignore_index=True)

colors = {"ELLAR": "gold", "ARID": "royalblue"}

# --- 1. Scatter Plot: SNR vs Lux ---
plt.figure(figsize=(10, 6))
for dataset in df["dataset"].unique():
    sub = df[df["dataset"] == dataset]
    plt.scatter(sub["lux_estimate"], sub["snr_db"], alpha=0.4, color=colors[dataset], label=dataset)
    
    slope, intercept, r, _, _ = linregress(sub["lux_estimate"], sub["snr_db"])
    x_vals = np.linspace(sub["lux_estimate"].min(), sub["lux_estimate"].max(), 100)
    y_vals = slope * x_vals + intercept
    plt.plot(x_vals, y_vals, "--", color=colors[dataset], label=f"{dataset} Trend (R²={r**2:.2f})")

plt.xlabel("Lux Estimate")
plt.ylabel("SNR (dB)")
plt.title("SNR vs Lux Estimate — ELLAR vs ARID")
plt.grid(True, linestyle="--", alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("scatter_snr_vs_lux.png", dpi=300)
plt.close()

# --- 2. Histogram: Lux Comparison ---
plt.figure(figsize=(8, 5))
for dataset in df["dataset"].unique():
    sns.histplot(df[df["dataset"] == dataset]["lux_estimate"], bins=40, kde=True,
                 label=dataset, color=colors[dataset], alpha=0.6)
plt.xlabel("Lux Estimate")
plt.ylabel("Count")
plt.title("Lux Distribution — ELLAR vs ARID")
plt.legend()
plt.tight_layout()
plt.savefig("histogram_lux_comparison.png", dpi=300)
plt.close()

# --- 3. Histogram: SNR Comparison ---
plt.figure(figsize=(8, 5))
for dataset in df["dataset"].unique():
    sns.histplot(df[df["dataset"] == dataset]["snr_db"], bins=40, kde=True,
                 label=dataset, color=colors[dataset], alpha=0.6)
plt.xlabel("SNR (dB)")
plt.ylabel("Count")
plt.title("SNR Distribution — ELLAR vs ARID")
plt.legend()
plt.tight_layout()
plt.savefig("histogram_snr_comparison.png", dpi=300)
plt.close()

# --- 4. Boxplots for SNR and Lux ---
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.boxplot(x="dataset", y="snr_db", data=df, palette=colors)
plt.title("Boxplot of SNR (dB)")

plt.subplot(1, 2, 2)
sns.boxplot(x="dataset", y="lux_estimate", data=df, palette=colors)
plt.title("Boxplot of Lux Estimate")

plt.tight_layout()
plt.savefig("boxplot_snr_lux.png", dpi=300)
plt.close()
