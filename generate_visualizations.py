import json
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

os.makedirs("visualisasi", exist_ok=True)

# 1. Load Metrik
with open("model_info.json", "r") as f:
    rf_info = json.load(f)

with open("evaluasi_llm_info.json", "r") as f:
    llm_info = json.load(f)

# Set style
sns.set_theme(style="whitegrid", font="DejaVu Sans")

# ----------------- GRAFIK 1: PERBANDINGAN AKURASI & F1-SCORE -----------------
fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
models = ["Random Forest\n(Library Model)", "Groq Qwen 2.5 32B\n(AI Model Service)"]
accuracies = [rf_info["metrik"]["accuracy"] * 100, llm_info["accuracy"] * 100]
f1_scores = [rf_info["metrik"]["f1_macro"] * 100, llm_info["f1_macro"] * 100]

x = np.arange(len(models))
width = 0.35

rects1 = ax.bar(x - width / 2, accuracies, width, label="Akurasi (%)", color="#1f77b4")
rects2 = ax.bar(x + width / 2, f1_scores, width, label="F1-Macro (%)", color="#ff7f0e")

ax.set_ylabel("Persentase (%)", fontsize=11, fontweight="bold")
ax.set_title("Perbandingan Kinerja: Model Library vs AI Model Service (79 Data Uji)", fontsize=12, fontweight="bold", pad=15)
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=11)
ax.set_ylim(0, 105)
ax.legend(frameon=True, facecolor="white", edgecolor="none")

# Tambahkan label angka di atas bar
for rect in rects1:
    h = rect.get_height()
    ax.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3),
                textcoords="offset points", ha="center", va="bottom", fontweight="bold", fontsize=10)

for rect in rects2:
    h = rect.get_height()
    ax.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3),
                textcoords="offset points", ha="center", va="bottom", fontweight="bold", fontsize=10)

plt.tight_layout()
fig.savefig("visualisasi/perbandingan_performa_model.png")
plt.close(fig)
print("Grafik 1 tersimpan: visualisasi/perbandingan_performa_model.png")

# ----------------- GRAFIK 2: HEATMAP CONFUSION MATRIX -----------------
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), dpi=300)
labels = ["Lulus", "Remedial"]

# Heatmap RF
cm_rf = np.array(rf_info["metrik"]["confusion_matrix"])
sns.heatmap(cm_rf, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axes[0],
            xticklabels=labels, yticklabels=labels, annot_kws={"size": 14, "weight": "bold"})
axes[0].set_title(f"Confusion Matrix: Random Forest\n(Akurasi: {rf_info['metrik']['accuracy']*100:.1f}%)", fontweight="bold")
axes[0].set_ylabel("Aktual", fontweight="bold")
axes[0].set_xlabel("Prediksi", fontweight="bold")

# Heatmap LLM
cm_llm = np.array(llm_info["confusion_matrix"])
sns.heatmap(cm_llm, annot=True, fmt="d", cmap="Oranges", cbar=False, ax=axes[1],
            xticklabels=labels, yticklabels=labels, annot_kws={"size": 14, "weight": "bold"})
axes[1].set_title(f"Confusion Matrix: Groq LLM\n(Akurasi: {llm_info['accuracy']*100:.1f}%)", fontweight="bold")
axes[1].set_ylabel("Aktual", fontweight="bold")
axes[1].set_xlabel("Prediksi", fontweight="bold")

plt.tight_layout()
fig.savefig("visualisasi/confusion_matrix_perbandingan.png")
plt.close(fig)
print("Grafik 2 tersimpan: visualisasi/confusion_matrix_perbandingan.png")

# ----------------- GRAFIK 3: FEATURE IMPORTANCE RANDOM FOREST -----------------
import joblib
pipeline = joblib.load("model_kelulusan.pkl")
rf_clf = pipeline.named_steps["classifier"]
feat_names = ["jam_belajar", "jumlah_kegagalan", "jumlah_absen", "nilai_tugas", "nilai_uts", "keinginan_kuliah", "akses_internet"]
importances = rf_clf.feature_importances_

indices = np.argsort(importances)[::-1]
sorted_names = [feat_names[i] for i in indices]
sorted_importances = importances[indices]

fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
bars = ax.barh(range(len(sorted_names)), sorted_importances[::-1], color="#2ca02c", align="center")
ax.set_yticks(range(len(sorted_names)))
ax.set_yticklabels(sorted_names[::-1], fontsize=10)
ax.set_xlabel("Tingkat Kepentingan Fitur (Feature Importance)", fontsize=11, fontweight="bold")
ax.set_title("Analisis Faktor Paling Berpengaruh Terhadap Kelulusan Siswa", fontsize=12, fontweight="bold", pad=15)

for bar in bars:
    w = bar.get_width()
    ax.annotate(f"{w*100:.1f}%", xy=(w, bar.get_y() + bar.get_height() / 2), xytext=(4, 0),
                textcoords="offset points", ha="left", va="center", fontweight="bold", fontsize=9)

plt.tight_layout()
fig.savefig("visualisasi/feature_importance_rf.png")
plt.close(fig)
print("Grafik 3 tersimpan: visualisasi/feature_importance_rf.png")
