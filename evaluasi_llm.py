import json
import os
import time
import pandas as pd
from dotenv import load_dotenv
from groq import Groq
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

# Muat environment dari root
load_dotenv(dotenv_path="../.env")
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("GROQ_API_KEY tidak ditemukan di file .env!")

client = Groq(api_key=api_key)
MODEL_NAME = "qwen/qwen3.8-27b"

# Baca data test yang sama persis dengan model library
test_df = pd.read_csv("data_test.csv")
output_csv = "hasil_evaluasi_test_llm.csv"

# Load checkpoint jika ada
if os.path.exists(output_csv):
    df_eval = pd.read_csv(output_csv)
    print(f"Melanjutkan dari checkpoint: {len(df_eval)} data sudah dievaluasi.")
else:
    df_eval = pd.DataFrame()

start_idx = len(df_eval)

system_prompt = """Anda adalah sistem pakar prediksi kelulusan siswa.
Tugas Anda adalah memprediksi apakah seorang siswa berstatus 'Lulus' atau 'Remedial' berdasarkan data akademiknya:
- jam_belajar: level 1 (<2 jam/minggu), 2 (2-5 jam/minggu), 3 (5-10 jam/minggu), 4 (>10 jam/minggu)
- jumlah_kegagalan: riwayat kegagalan kelas sebelumnya (0-3)
- jumlah_absen: total hari tidak masuk (0-30+)
- nilai_tugas: nilai tugas / periode 1 (skala 0-20)
- nilai_uts: nilai UTS / periode 2 (skala 0-20, standar kelulusan >= 10)
- keinginan_kuliah: Ya / Tidak
- akses_internet: Ya / Tidak

Aturan Output: Jawab HANYA satu kata: 'Lulus' atau 'Remedial'. Dilarang memberikan tanda baca atau kalimat penjelasan tambahan."""

few_shot = [
    {"role": "user", "content": "jam_belajar: 2, jumlah_kegagalan: 0, jumlah_absen: 2, nilai_tugas: 15, nilai_uts: 16, keinginan_kuliah: Ya, akses_internet: Ya"},
    {"role": "assistant", "content": "Lulus"},
    {"role": "user", "content": "jam_belajar: 1, jumlah_kegagalan: 2, jumlah_absen: 14, nilai_tugas: 6, nilai_uts: 5, keinginan_kuliah: Tidak, akses_internet: Tidak"},
    {"role": "assistant", "content": "Remedial"},
    {"role": "user", "content": "jam_belajar: 3, jumlah_kegagalan: 1, jumlah_absen: 6, nilai_tugas: 11, nilai_uts: 10, keinginan_kuliah: Ya, akses_internet: Ya"},
    {"role": "assistant", "content": "Lulus"},
    {"role": "user", "content": "jam_belajar: 1, jumlah_kegagalan: 3, jumlah_absen: 18, nilai_tugas: 4, nilai_uts: 4, keinginan_kuliah: Tidak, akses_internet: Ya"},
    {"role": "assistant", "content": "Remedial"}
]

print(f"Mulai evaluasi LLM ({MODEL_NAME}) pada {len(test_df) - start_idx} data tersisa...")

for i in range(start_idx, len(test_df)):
    row = test_df.iloc[i]
    user_content = (
        f"jam_belajar: {row['jam_belajar']}, jumlah_kegagalan: {row['jumlah_kegagalan']}, "
        f"jumlah_absen: {row['jumlah_absen']}, nilai_tugas: {row['nilai_tugas']}, "
        f"nilai_uts: {row['nilai_uts']}, keinginan_kuliah: {row['keinginan_kuliah']}, "
        f"akses_internet: {row['akses_internet']}"
    )

    messages = [{"role": "system", "content": system_prompt}] + few_shot + [{"role": "user", "content": user_content}]

    pred = "Remedial"
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            max_tokens=5,
            temperature=0.0
        )
        raw = resp.choices[0].message.content.strip()
        if "lulus" in raw.lower():
            pred = "Lulus"
        elif "remedial" in raw.lower() or "fail" in raw.lower():
            pred = "Remedial"
    except Exception as e:
        print(f"Error data ke-{i+1}: {e}")
        time.sleep(2)

    new_row = row.to_dict()
    new_row["prediksi_llm"] = pred
    df_eval = pd.concat([df_eval, pd.DataFrame([new_row])], ignore_index=True)

    if (i + 1) % 10 == 0 or (i + 1) == len(test_df):
        df_eval.to_csv(output_csv, index=False)
        print(f"Progres: {i+1}/{len(test_df)} selesai.")

    time.sleep(0.5)

# Hitung metrik evaluasi akhir
y_true = df_eval["status_kelulusan"]
y_pred = df_eval["prediksi_llm"]

acc = accuracy_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred, average="macro")
cm = confusion_matrix(y_true, y_pred, labels=["Lulus", "Remedial"])

print("\n=== HASIL EVALUASI AI MODEL SERVICE (GROQ LLM) ===")
print(f"Model    : {MODEL_NAME}")
print(f"Accuracy : {acc:.4f} ({acc*100:.2f}%)")
print(f"F1 Macro : {f1:.4f}")
print("Confusion Matrix (Lulus, Remedial):")
print(cm)
print("\nClassification Report:")
print(classification_report(y_true, y_pred))

# Simpan hasil perbandingan ke evaluasi_summary.json
eval_summary = {
    "model_llm": MODEL_NAME,
    "n_test": len(df_eval),
    "accuracy": float(acc),
    "f1_macro": float(f1),
    "confusion_matrix": cm.tolist()
}

with open("evaluasi_llm_info.json", "w", encoding="utf-8") as f:
    json.dump(eval_summary, f, indent=2)
print("Hasil evaluasi tersimpan di 'evaluasi_llm_info.json'.")
