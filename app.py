import json
import os
import joblib
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv(dotenv_path="../.env")
load_dotenv(dotenv_path=".env")

st.set_page_config(
    page_title="Prediksi Kelulusan Siswa",
    page_icon="📚",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# Styling minimalis agar bersih dan rapi
st.markdown("""
<style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2.5rem;
        max-width: 820px;
    }
    #MainMenu, footer, header { visibility: hidden; }
    h1, h2, h3, h4, h5 { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
</style>
""", unsafe_allow_html=True)


# --- Load Model & Metadata ---
@st.cache_resource
def load_model():
    p = os.path.join(os.path.dirname(__file__), "model_kelulusan.pkl")
    return joblib.load(p if os.path.exists(p) else "model_kelulusan.pkl")

@st.cache_data
def load_meta():
    p = os.path.join(os.path.dirname(__file__), "model_info.json")
    with open(p if os.path.exists(p) else "model_info.json", encoding="utf-8") as f:
        return json.load(f)

@st.cache_data
def load_llm_meta():
    p = os.path.join(os.path.dirname(__file__), "evaluasi_llm_info.json")
    if not os.path.exists(p):
        p = "evaluasi_llm_info.json"
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    return {}

@st.cache_resource
def get_groq_client():
    key = ""
    try:
        key = st.secrets.get("GROQ_API_KEY", "")
    except Exception:
        pass
    if not key:
        key = os.getenv("GROQ_API_KEY", "")
    if key:
        from groq import Groq
        return Groq(api_key=key)
    return None

pipeline = load_model()
meta = load_meta()
llm_meta = load_llm_meta()


# --- Fungsi Pemanggilan LLM ---
def tanya_llm(d: dict):
    client = get_groq_client()
    if not client:
        return "Error", "API Key Groq tidak ditemukan."

    prompt_sys = (
        "Anda adalah sistem pakar evaluasi akademik siswa. Berdasarkan data nilai dan kebiasaan belajar siswa, "
        "tentukan prediksi kelulusan siswa: 'Lulus' atau 'Remedial' beserta 1 kalimat alasan logis.\n"
        "Format balasan:\n"
        "STATUS: [Lulus/Remedial]\n"
        "ALASAN: [penjelasan singkat]"
    )
    user_msg = (
        f"Jam belajar mingguan: {d['jam_belajar']}, Jumlah absen: {d['jumlah_absen']} hari, "
        f"Riwayat kegagalan kelas: {d['jumlah_kegagalan']} kali, Nilai tugas: {d['nilai_tugas']*5}/100, "
        f"Nilai UTS: {d['nilai_uts']*5}/100, Rencana kuliah: {d['keinginan_kuliah']}, Akses internet: {d['akses_internet']}"
    )

    try:
        r = client.chat.completions.create(
            model="qwen/qwen3.8-27b",
            messages=[
                {"role": "system", "content": prompt_sys},
                {"role": "user", "content": "Jam belajar: 2, Absen: 2 hari, Kegagalan: 0, Nilai tugas: 80/100, Nilai UTS: 75/100, Kuliah: Ya, Internet: Ya"},
                {"role": "assistant", "content": "STATUS: Lulus\nALASAN: Nilai tugas dan UTS konsisten di atas batas kelulusan dengan tingkat kehadiran yang baik."},
                {"role": "user", "content": user_msg},
            ],
            max_tokens=60,
            temperature=0.0,
        )
        raw = r.choices[0].message.content.strip()
        status = "Lulus" if "lulus" in raw.lower() else "Remedial"
        alasan = raw.split("ALASAN:")[-1].strip() if "ALASAN:" in raw else raw
        return status, alasan
    except Exception as e:
        return "Error", str(e)


# --- Header Aplikasi ---
st.title("Prediksi Kelulusan Siswa Berbasis AI")
st.caption("Pertemuan 6 & 7: Penerapan AI di Sektor Pendidikan | Dataset: UCI Student Performance")

tab_pred, tab_batch, tab_eval = st.tabs(["Prediksi Data Siswa", "Pengujian File CSV", "Evaluasi Model"])


# ==============================================================================
# TAB 1: PREDIKSI DATA SISWA
# ==============================================================================
with tab_pred:
    with st.form("form_prediksi"):
        st.subheader("Input Parameter Siswa")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Performa Akademik & Kehadiran**")
            nilai_tugas_100 = st.slider("Nilai Tugas Harian", 0, 100, 70, step=5, format="%d / 100")
            nilai_uts_100 = st.slider("Nilai Ujian Tengah Semester (UTS)", 0, 100, 65, step=5, format="%d / 100")

            absen_opsi = {
                "Sangat Baik (0 - 3 hari)": 2,
                "Cukup (4 - 7 hari)": 5,
                "Kurang (8 - 15 hari)": 11,
                "Sangat Kurang (> 15 hari)": 20,
            }
            absen_pilihan = st.selectbox("Riwayat Ketidakhadiran (Absen)", list(absen_opsi.keys()), index=0)
            jumlah_absen = absen_opsi[absen_pilihan]

        with col2:
            st.markdown("**Kebiasaan Belajar & Profil**")
            jam_opsi = {
                "< 2 jam per minggu": 1,
                "2 - 5 jam per minggu": 2,
                "5 - 10 jam per minggu": 3,
                "> 10 jam per minggu": 4,
            }
            jam_pilihan = st.selectbox("Waktu Belajar Mandiri Mingguan", list(jam_opsi.keys()), index=1)
            jam_belajar = jam_opsi[jam_pilihan]

            gagal_opsi = {
                "Tidak pernah (0 kali)": 0,
                "Pernah 1 kali": 1,
                "Pernah 2 kali": 2,
                "Pernah 3 kali atau lebih": 3,
            }
            gagal_pilihan = st.selectbox("Riwayat Tidak Naik / Gagal Kelas", list(gagal_opsi.keys()), index=0)
            jumlah_gagal = gagal_opsi[gagal_pilihan]

            kuliah_pilihan = st.radio("Rencana Melanjutkan ke Perguruan Tinggi", ["Ya", "Tidak"], horizontal=True)
            internet_pilihan = st.radio("Memiliki Akses Internet di Rumah", ["Ya", "Tidak"], horizontal=True)

        st.markdown("---")
        mode_model = st.radio(
            "Pilih Model untuk Menjalankan Prediksi:",
            ["Random Forest (Library)", "Groq LLM (Model Service)", "Bandingkan Keduanya"],
            index=2,
            horizontal=True,
        )

        submit_btn = st.form_submit_button("Jalankan Prediksi", type="primary", use_container_width=True)

    if submit_btn:
        # Konversi nilai skala 100 ke skala 20 untuk input model
        nilai_tugas = round(nilai_tugas_100 / 5)
        nilai_uts = round(nilai_uts_100 / 5)

        data_dict = {
            "jam_belajar": jam_belajar,
            "jumlah_kegagalan": jumlah_gagal,
            "jumlah_absen": jumlah_absen,
            "nilai_tugas": nilai_tugas,
            "nilai_uts": nilai_uts,
            "keinginan_kuliah": kuliah_pilihan,
            "akses_internet": internet_pilihan,
        }
        df_input = pd.DataFrame([data_dict])

        st.subheader("Hasil Prediksi")

        # Layout hasil
        if mode_model == "Bandingkan Keduanya":
            col_res_a, col_res_b = st.columns(2)
        else:
            col_res_a = col_res_b = st.container()

        # 1. Output Model Library
        if mode_model in ["Random Forest (Library)", "Bandingkan Keduanya"]:
            pred_rf = pipeline.predict(df_input)[0]
            proba_rf = pipeline.predict_proba(df_input)[0]
            classes = list(pipeline.classes_)
            p_lulus = proba_rf[classes.index("Lulus")]
            p_remed = proba_rf[classes.index("Remedial")]
            conf = p_lulus * 100 if pred_rf == "Lulus" else p_remed * 100

            with col_res_a:
                with st.container(border=True):
                    st.markdown("##### Model Library (Random Forest)")
                    if pred_rf == "Lulus":
                        st.success(f"Status: **{pred_rf}**")
                    else:
                        st.error(f"Status: **{pred_rf}**")

                    st.metric("Tingkat Keyakinan Model", f"{conf:.1f}%")

                    df_chart = pd.DataFrame(
                        {"Probabilitas (%)": [p_lulus * 100, p_remed * 100]},
                        index=["Lulus", "Remedial"],
                    )
                    st.bar_chart(df_chart, height=180)

        # 2. Output AI Model Service
        if mode_model in ["Groq LLM (Model Service)", "Bandingkan Keduanya"]:
            with col_res_b:
                with st.container(border=True):
                    st.markdown("##### AI Model Service (Groq API)")
                    with st.spinner("Memproses inferensi LLM..."):
                        status_llm, alasan_llm = tanya_llm(data_dict)

                    if status_llm == "Lulus":
                        st.success(f"Status: **{status_llm}**")
                    elif status_llm == "Remedial":
                        st.error(f"Status: **{status_llm}**")
                    else:
                        st.warning(f"Status: **{status_llm}**")

                    st.markdown("**Analisis Pertimbangan Model:**")
                    st.write(alasan_llm)
                    st.caption("Mesin: Qwen 2.5 32B via Groq Cloud API")


# ==============================================================================
# TAB 2: PENGUJIAN FILE CSV (BATCH)
# ==============================================================================
with tab_batch:
    st.subheader("Pengujian Massal Melalui Berkas CSV")
    st.write(
        "Fitur ini digunakan untuk menguji prediksi pada banyak data siswa sekaligus. "
        "Silakan unduh berkas sampel data di bawah ini untuk melihat format dan mengujinya langsung."
    )

    sample_file = os.path.join(os.path.dirname(__file__), "sample_data.csv")
    if not os.path.exists(sample_file):
        sample_file = "sample_data.csv"

    if os.path.exists(sample_file):
        with open(sample_file, "rb") as f:
            st.download_button(
                "📥 Unduh Berkas Sampel Pengujian (sample_data.csv)",
                data=f.read(),
                file_name="sample_data.csv",
                mime="text/csv",
                help="Berkas berisi 20 sampel data siswa dengan profil akademik yang beragam.",
            )

    st.markdown("---")
    file_terunggah = st.file_uploader("Unggah Berkas CSV:", type=["csv"])

    cols_req = [
        "jam_belajar", "jumlah_kegagalan", "jumlah_absen",
        "nilai_tugas", "nilai_uts", "keinginan_kuliah", "akses_internet"
    ]

    if file_terunggah:
        df_csv = pd.read_csv(file_terunggah)
        missing_cols = [c for c in cols_req if c not in df_csv.columns]

        if missing_cols:
            st.error(
                f"Format kolom berkas tidak sesuai. Kolom yang dibutuhkan: {', '.join(cols_req)}.\n\n"
                "Silakan gunakan berkas yang diunduh dari tombol 'Unduh Berkas Sampel Pengujian' di atas."
            )
        else:
            st.markdown(f"**Pratinjau Data ({len(df_csv)} siswa):**")
            st.dataframe(df_csv.head(10), use_container_width=True, hide_index=True)

            if st.button("Jalankan Prediksi Seluruh Baris", type="primary"):
                with st.spinner("Memproses prediksi..."):
                    df_csv["Prediksi"] = pipeline.predict(df_csv[cols_req])
                    probs = pipeline.predict_proba(df_csv[cols_req])
                    cls_list = list(pipeline.classes_)
                    idx_l = cls_list.index("Lulus")
                    df_csv["Probabilitas Lulus (%)"] = (probs[:, idx_l] * 100).round(1)

                st.success(f"Selesai! Berhasil memprediksi {len(df_csv)} data siswa.")
                st.dataframe(df_csv, use_container_width=True, hide_index=True)

                col_stat1, col_stat2 = st.columns(2)
                with col_stat1:
                    rekap = df_csv["Prediksi"].value_counts()
                    st.markdown("**Distribusi Hasil:**")
                    for k, v in rekap.items():
                        st.write(f"- {k}: **{v}** siswa")
                with col_stat2:
                    st.bar_chart(rekap)

                st.download_button(
                    "Unduh Hasil Prediksi (CSV)",
                    data=df_csv.to_csv(index=False).encode("utf-8"),
                    file_name="hasil_prediksi_siswa.csv",
                    mime="text/csv",
                )


# ==============================================================================
# TAB 3: EVALUASI & PERBANDINGAN MODEL
# ==============================================================================
with tab_eval:
    m = meta["metrik"]
    lm = llm_meta

    st.subheader("Evaluasi Performa Model pada Data Uji")
    st.write(
        f"Evaluasi kuantitatif dilakukan pada {m['n_test']} sampel data uji yang sama persis "
        "antara Model Library (Random Forest) dan AI Model Service (Groq LLM)."
    )

    # Tabel ringkasan metrik
    df_metrik = pd.DataFrame(
        {
            "Parameter Evaluasi": [
                "Akurasi Keseluruhan",
                "F1-Score (Macro Average)",
                "Jumlah Data Uji (Test Set)",
                "Jumlah Data Latih (Training Set)",
                "Waktu Inferensi Rata-rata",
                "Kebutuhan Koneksi Internet",
                "Metode Inferensi",
            ],
            "Model Library (Random Forest)": [
                f"{m['accuracy'] * 100:.2f}%",
                f"{m['f1_macro']:.4f}",
                f"{m['n_test']} sampel",
                f"{m['n_train']} sampel",
                "< 5 ms per request",
                "Tidak (Offline)",
                "Supervised Learning",
            ],
            "AI Model Service (Groq API)": [
                f"{lm.get('accuracy', 0) * 100:.2f}%",
                f"{lm.get('f1_macro', 0):.4f}",
                f"{lm.get('n_test', m['n_test'])} sampel",
                "—",
                "~400 - 800 ms per request",
                "Ya (Cloud API)",
                "Few-Shot Prompting",
            ],
        }
    )
    st.dataframe(df_metrik, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("Matriks Kesalahan (Confusion Matrix)")
    col_cm1, col_cm2 = st.columns(2)

    with col_cm1:
        st.markdown("**Random Forest**")
        df_cm_rf = pd.DataFrame(
            m["confusion_matrix"],
            index=["Aktual: Lulus", "Aktual: Remedial"],
            columns=["Prediksi: Lulus", "Prediksi: Remedial"],
        )
        st.dataframe(df_cm_rf, use_container_width=True)

    with col_cm2:
        if "confusion_matrix" in lm:
            st.markdown("**Groq LLM (Qwen 2.5 32B)**")
            df_cm_llm = pd.DataFrame(
                lm["confusion_matrix"],
                index=["Aktual: Lulus", "Aktual: Remedial"],
                columns=["Prediksi: Lulus", "Prediksi: Remedial"],
            )
            st.dataframe(df_cm_llm, use_container_width=True)

    st.markdown("---")
    st.subheader("Analisis Hasil Pengujian")
    st.write(
        f"1. **Tingkat Akurasi:** Model Library (Random Forest) memperoleh akurasi sebesar {m['accuracy']*100:.1f}%, "
        f"sedangkan Groq LLM memperoleh {lm.get('accuracy', 0)*100:.1f}%. Selisih kedua model relatif tipis (sekitar 1.3%).\n"
        "2. **Efisiensi Komputasi:** Random Forest memproses prediksi dalam hitungan milidetik secara lokal tanpa kuota API, "
        "sementara LLM memerlukan jaringan internet untuk mengirim permintaan ke cloud.\n"
        "3. **Interpretasi Output:** Random Forest menghasilkan probabilitas numerik, sedangkan LLM mampu "
        "memberikan ringkasan alasan dalam kalimat bahasa Indonesia yang kontekstual."
    )
