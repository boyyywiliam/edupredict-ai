# 🎓 EduPredict AI - Deployment Sistem Prediksi Kelulusan Siswa Berbasis AI

Aplikasi web interaktif untuk memprediksi performa dan status kelulusan siswa (**Lulus Tepat Waktu** vs **Remedial / Butuh Bimbingan**) berbasis **Machine Learning (Random Forest)** dan **Cloud AI Model Service (Groq LLM - Qwen 2.5 32B)**.

> **Tugas Praktikum Pertemuan 6 & 7: Penerapan AI di Sektor Kesehatan dan Pendidikan**  
> **Mata Kuliah:** Kecerdasan Artifisial dan Penerapannya (SS1SI-06-REG-01)  
> **Program Studi:** S1 Sistem Informasi, Fakultas Rekayasa Industri, Telkom University Surabaya  
> **Pengembang:** Mochammad Dailam Al Muhibi (NIM: 1204230037)

---

## 🌟 Fitur Utama Aplikasi

1. **Dual Engine Mode:**
   - 🌲 **Model Library:** *Random Forest Classifier* (Scikit-Learn) yang telah dilatih secara offline pada 316 data latih.
   - 🤖 **AI Model Service:** *Groq Cloud LLM (Qwen 2.5 32B)* dengan penalaran berbasis *Few-Shot Prompt Engineering*.
   - ⚡ **Mode Komparasi (Dual Model):** Menampilkan perbandingan prediksi kedua model secara berdampingan.
2. **Prediksi Individual (Form Input):** Input interaktif nilai akademik, jam belajar, absensi, dan preferensi studi siswa.
3. **Prediksi Massal (Batch CSV):** Fitur upload file CSV untuk memproses banyak siswa sekaligus, dilengkapi tombol unduh hasil prediksi.
4. **Sampel Data Siap Unduh:** Tombol **Unduh Sampel Data Siswa (sample_data.csv)** langsung dari antarmuka web.
5. **Dashboard Metrik & Evaluasi:** Visualisasi Confusion Matrix, Akurasi, F1-Macro, dan *Feature Importance*.

---

## 📊 Hasil Evaluasi Model (79 Data Uji)

| Metrik Evaluasi | Model Library (Random Forest) | AI Model Service (Groq LLM) |
|---|:---:|:---:|
| **Akurasi** | **88.61%** | 87.34% |
| **F1-Macro** | **0.8722** | 0.8638 |
| **Confusion Matrix (Lulus, Remedial)** | `[[48, 5], [4, 22]]` | `[[45, 8], [2, 24]]` |
| **Waktu Inferensi** | < 2 ms | ~500 ms |
| **Ketergantungan API** | Mandiri / Offline | Membutuhkan Koneksi Internet & API Key |

---

## 🛠️ Cara Menjalankan Secara Lokal

1. **Clone repositori ini:**
   ```bash
   git clone https://github.com/dailam008/edupredict-ai.git
   cd edupredict-ai
   ```

2. **Install dependensi:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Atur API Key Groq:**
   Buat file `.env` di root folder:
   ```env
   GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxxxxxx
   ```

4. **Jalankan aplikasi Streamlit:**
   ```bash
   streamlit run app.py
   ```
   Aplikasi akan terbuka otomatis di peramban pada alamat `http://localhost:8501`.

---

## 🌐 Sumber Dataset
Dataset bersumber dari **UCI Machine Learning Repository**:
*Cortez, P., & Silva, A. (2008). Using Data Mining to Predict Secondary School Student Performance. University of Minho, Portugal.*
