---
title: EduPredict AI - Prediksi Kelulusan Siswa
emoji: 🎓
colorFrom: blue
colorTo: red
sdk: streamlit
sdk_version: 1.41.0
app_file: app.py
pinned: false
---

# 🎓 EduPredict AI - Deployment Sistem Prediksi Kelulusan Siswa Berbasis AI

Aplikasi web interaktif untuk memprediksi performa dan status kelulusan siswa (**Lulus Tepat Waktu** vs **Remedial / Butuh Bimbingan**) berbasis **Machine Learning (Random Forest)** dan **Cloud AI Model Service (Groq LLM - Qwen 2.5 32B)**.

> **Tugas Praktikum Pertemuan 6 & 7: Penerapan AI di Sektor Kesehatan dan Pendidikan**  
> **Mata Kuliah:** Kecerdasan Artifisial dan Penerapannya (SS1SI-06-REG-01)  
> **Program Studi:** S1 Sistem Informasi, Fakultas Rekayasa Industri, Telkom University Surabaya  
> **Pengembang:** Mochammad Dailam Al Muhibi (NIM: 1204230037)

---

## 🌟 Fitur Utama Aplikasi

1. **Dual Engine Mode:**
   - Model Library: *Random Forest Classifier* (Scikit-Learn).
   - AI Model Service: *Groq Cloud LLM (Qwen 2.5 32B)* via API.
   - Mode Komparasi: Menampilkan perbandingan prediksi kedua model secara berdampingan.
2. **Prediksi Individual:** Input interaktif nilai akademik (skala 0–100), jam belajar, absensi, dan preferensi studi siswa.
3. **Prediksi Massal (Batch CSV):** Fitur upload berkas CSV untuk memproses banyak siswa sekaligus, dilengkapi tombol unduh hasil prediksi.
4. **Sampel Data Siap Unduh:** Berkas `sample_data.csv` berisi 20 sampel data siswa dengan profil yang beragam.
5. **Dashboard Evaluasi:** Visualisasi Confusion Matrix, Akurasi, F1-Macro, dan tabel perbandingan karakteristik operasional.

---

## 📊 Hasil Evaluasi Model (79 Data Uji)

| Metrik Evaluasi | Model Library (Random Forest) | AI Model Service (Groq LLM) |
|---|:---:|:---:|
| **Akurasi** | **88.61%** | **87.34%** |
| **F1-Macro** | **0.8722** | **0.8638** |
| **Confusion Matrix (Lulus, Remedial)** | `[[48, 5], [4, 22]]` | `[[45, 8], [2, 24]]` |
| **Waktu Inferensi** | < 5 ms | ~500 ms |
| **Ketergantungan API** | Mandiri / Offline | Membutuhkan Koneksi Internet & API Key |
