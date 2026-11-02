import json
import os
import time
import joblib
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# Konfigurasi Halaman Streamlit
st.set_page_config(
    page_title="EduPredict AI - Prediksi Kelulusan Siswa",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load environment lokal jika ada
load_dotenv(dotenv_path="../.env")
load_dotenv(dotenv_path=".env")


# 1. Cache Muat Model Library (Random Forest)
@st.cache_resource
def load_ml_pipeline():
    model_path = os.path.join(os.path.dirname(__file__), "model_kelulusan.pkl")
    if not os.path.exists(model_path):
        model_path = "model_kelulusan.pkl"
    return joblib.load(model_path)


# Cache Muat Metadata Model
@st.cache_data
def load_model_metadata():
    info_path = os.path.join(os.path.dirname(__file__), "model_info.json")
    if not os.path.exists(info_path):
        info_path = "model_info.json"
    with open(info_path, "r", encoding="utf-8") as f:
        return json.load(f)


# Cache Muat Metadata LLM
@st.cache_data
def load_llm_metadata():
    info_path = os.path.join(os.path.dirname(__file__), "evaluasi_llm_info.json")
    if not os.path.exists(info_path):
        info_path = "evaluasi_llm_info.json"
    if os.path.exists(info_path):
        with open(info_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


# Inisialisasi Kredensial Groq API
def get_groq_api_key():
    # Cek st.secrets (Streamlit Cloud) terlebih dahulu, lalu os.getenv
    try:
        if "GROQ_API_KEY" in st.secrets:
            return st.secrets["GROQ_API_KEY"]
    except Exception:
        pass
    return os.getenv("GROQ_API_KEY", "")


@st.cache_resource
def get_groq_client():
    key = get_groq_api_key()
    if key:
        from groq import Groq

        return Groq(api_key=key)
    return None


pipeline = load_ml_pipeline()
info = load_model_metadata()
llm_info = load_llm_metadata()
groq_client = get_groq_client()

# ================= SIDEBAR =================
with st.sidebar:
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/0/03/Telkom_University_Logo.svg/1200px-Telkom_University_Logo.svg.png",
        width=140,
    )
    st.title("🎓 EduPredict AI")
    st.caption("Penerapan AI Sektor Pendidikan · Pertemuan 6 & 7")

    st.markdown("---")
    st.subheader("👤 Identitas Pengembang")
    st.write("**Nama:** Mochammad Dailam Al Muhibi")
    st.write("**NIM:** 1204230037")
    st.write("**Kelas:** SS1SI-06-REG-01")
    st.write("**Prodi:** S1 Sistem Informasi")
    st.write("**Institusi:** Telkom University")

    st.markdown("---")
    st.subheader("⚙️ Pengaturan Model")
    pilihan_model = st.radio(
        "Pilih Mesin Prediksi:",
        [
            "🌲 Model Library (Random Forest)",
            "🤖 AI Model Service (Groq LLM)",
            "⚡ Bandingkan Keduanya (Dual Model)",
        ],
        index=2,
    )

    st.markdown("---")
    st.subheader("📊 Performa Model (Data Uji)")
    st.metric("Akurasi Random Forest", f"{info['metrik']['accuracy'] * 100:.1f}%")
    st.metric("F1-Macro Random Forest", f"{info['metrik']['f1_macro']:.3f}")
    if llm_info:
        st.metric("Akurasi Groq LLM", f"{llm_info['accuracy'] * 100:.1f}%")
        st.metric("F1-Macro Groq LLM", f"{llm_info['f1_macro']:.3f}")


# ================= HEADER =================
st.title("🎓 Sistem Prediksi Kelulusan Siswa Berbasis AI")
st.markdown(
    """
Aplikasi ini menerapkan pemodelan komparatif antara **Machine Learning Library (Random Forest)** 
dan **Cloud AI Model Service (Groq LLM)** untuk memprediksi apakah seorang siswa berpotensi **Lulus Tepat Waktu** 
atau memerlukan **Program Bimbingan / Remedial**.
*Sumber Dataset: UCI Machine Learning Repository (Student Performance).*
"""
)

tab1, tab2, tab3 = st.tabs(
    [
        "📝 Prediksi Individual (Satu Siswa)",
        "📂 Prediksi Massal (Unggah CSV)",
        "📊 Evaluasi & Perbandingan Model",
    ]
)


# Fungsi Prediksi LLM
def predict_with_llm(data_dict):
    client = get_groq_client()
    if not client:
        return "API Key Belum Dikonfigurasi", "Harap atur GROQ_API_KEY di Streamlit secrets."

    system_prompt = """Anda adalah sistem pakar evaluasi akademik siswa.
Tugas Anda adalah memprediksi status kelulusan siswa: 'Lulus' atau 'Remedial' serta memberikan 1 kalimat alasan ringkas.
Format output WAJIB:
STATUS: [Lulus / Remedial]
ALASAN: [1 kalimat analisis faktor penyebab]"""

    user_text = (
        f"jam_belajar: {data_dict['jam_belajar']}, jumlah_kegagalan: {data_dict['jumlah_kegagalan']}, "
        f"jumlah_absen: {data_dict['jumlah_absen']}, nilai_tugas: {data_dict['nilai_tugas']}, "
        f"nilai_uts: {data_dict['nilai_uts']}, keinginan_kuliah: {data_dict['keinginan_kuliah']}, "
        f"akses_internet: {data_dict['akses_internet']}"
    )

    try:
        resp = client.chat.completions.create(
            model="qwen/qwen3.8-27b",
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": "jam_belajar: 2, jumlah_kegagalan: 0, jumlah_absen: 2, nilai_tugas: 15, nilai_uts: 16, keinginan_kuliah: Ya, akses_internet: Ya",
                },
                {
                    "role": "assistant",
                    "content": "STATUS: Lulus\nALASAN: Nilai tugas dan UTS sangat tinggi dengan kehadiran yang terjaga prima.",
                },
                {"role": "user", "content": user_text},
            ],
            max_tokens=60,
            temperature=0.0,
        )
        content = resp.choices[0].message.content.strip()
        status = "Lulus" if "lulus" in content.lower() else "Remedial"
        alasan = content
        if "ALASAN:" in content:
            alasan = content.split("ALASAN:")[-1].strip()
        return status, alasan
    except Exception as e:
        return "Error", str(e)


# ================= TAB 1: PREDIKSI SATU SISWA =================
with tab1:
    st.subheader("Form Input Parameter Akademik Siswa")
    st.write(
        "Silakan sesuaikan data akademik siswa di bawah ini, kemudian klik tombol **Prediksi Status Kelulusan**."
    )

    with st.form("form_single_pred"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("##### ⏱️ Kebiasaan Belajar & Absensi")
            jam_belajar_map = {
                "< 2 Jam / Minggu (Level 1)": 1,
                "2 - 5 Jam / Minggu (Level 2)": 2,
                "5 - 10 Jam / Minggu (Level 3)": 3,
                "> 10 Jam / Minggu (Level 4)": 4,
            }
            jam_belajar_lbl = st.selectbox(
                "Waktu Belajar Mandiri Mingguan", list(jam_belajar_map.keys()), index=1
            )
            jam_belajar = jam_belajar_map[jam_belajar_lbl]

            jumlah_absen = st.slider("Jumlah Hari Tidak Hadir (Absensi)", 0, 40, 4)

        with col2:
            st.markdown("##### 📝 Nilai Ujian & Riwayat")
            nilai_tugas = st.slider("Nilai Tugas / Periode 1 (Skala 0 - 20)", 0, 20, 14)
            nilai_uts = st.slider("Nilai UTS / Periode 2 (Skala 0 - 20)", 0, 20, 13)
            jumlah_kegagalan = st.slider("Riwayat Kegagalan Kelas Sebelumnya", 0, 3, 0)

        with col3:
            st.markdown("##### 🌐 Faktor Penunjang Siswa")
            keinginan_kuliah = st.radio(
                "Keinginan Melanjutkan Kuliah?", ["Ya", "Tidak"], horizontal=True
            )
            akses_internet = st.radio(
                "Memiliki Akses Internet di Rumah?", ["Ya", "Tidak"], horizontal=True
            )

        submit_btn = st.form_submit_button("🔍 Prediksi Status Kelulusan", use_container_width=True)

    if submit_btn:
        input_dict = {
            "jam_belajar": jam_belajar,
            "jumlah_kegagalan": jumlah_kegagalan,
            "jumlah_absen": jumlah_absen,
            "nilai_tugas": nilai_tugas,
            "nilai_uts": nilai_uts,
            "keinginan_kuliah": keinginan_kuliah,
            "akses_internet": akses_internet,
        }
        input_df = pd.DataFrame([input_dict])

        st.markdown("---")
        st.subheader("🎯 Hasil Prediksi Sistem")

        col_res1, col_res2 = st.columns(2)

        # 1. Hasil Random Forest
        if (
            "Model Library" in pilihan_model
            or "Bandingkan Keduanya" in pilihan_model
        ):
            pred_rf = pipeline.predict(input_df)[0]
            proba_rf = pipeline.predict_proba(input_df)[0]
            classes = list(pipeline.classes_)
            idx_pred = classes.index(pred_rf)
            conf_rf = proba_rf[idx_pred] * 100

            with col_res1:
                st.markdown("#### 🌲 Model Library (Random Forest)")
                if pred_rf == "Lulus":
                    st.success(f"### Status: **LULUS** 🎉")
                else:
                    st.error(f"### Status: **REMEDIAL / BUTUH BIMBINGAN** ⚠️")

                st.write(f"Tingkat Keyakinan (*Confidence*): **{conf_rf:.1f}%**")
                proba_df = pd.DataFrame({"Probabilitas": proba_rf}, index=classes)
                st.bar_chart(proba_df)

        # 2. Hasil LLM
        if "Groq LLM" in pilihan_model or "Bandingkan Keduanya" in pilihan_model:
            with col_res2:
                st.markdown("#### 🤖 AI Model Service (Groq LLM)")
                with st.spinner("Menganalisis data via LLM API..."):
                    pred_llm, alasan_llm = predict_with_llm(input_dict)

                if pred_llm == "Lulus":
                    st.success(f"### Status: **LULUS** 🎉")
                elif pred_llm == "Remedial":
                    st.error(f"### Status: **REMEDIAL / BUTUH BIMBINGAN** ⚠️")
                else:
                    st.warning(f"### Status: {pred_llm}")

                st.info(f"💡 **Analisis LLM:** {alasan_llm}")


# ================= TAB 2: PREDIKSI MASSAL (CSV) =================
with tab2:
    st.subheader("📂 Pengujian Massal Menggunakan Berkas CSV")
    st.write(
        "Anda dapat mengunggah file CSV data siswa untuk memperoleh prediksi secara simultan. "
        "Gunakan berkas sampel resmi di bawah ini untuk mencoba sistem."
    )

    # TOMBOL DOWNLOAD SAMPEL DATA (MEMENUHI SYARAT POIN 5 LMS)
    sample_path = os.path.join(os.path.dirname(__file__), "sample_data.csv")
    if not os.path.exists(sample_path):
        sample_path = "sample_data.csv"

    if os.path.exists(sample_path):
        with open(sample_path, "rb") as f_samp:
            csv_bytes = f_samp.read()
        st.download_button(
            label="📥 Unduh Sampel Data Siswa (sample_data.csv)",
            data=csv_bytes,
            file_name="sample_data.csv",
            mime="text/csv",
            help="Klik untuk mengunduh 10 sampel data uji siap pakai.",
        )
    else:
        st.warning("File sampel data belum tersedia di server.")

    st.markdown("---")
    uploaded_file = st.file_uploader("Unggah file CSV pengujian:", type=["csv"])

    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
        st.markdown("##### 👁️ Pratinjau Data Terunggah")
        st.dataframe(batch_df.head(10), use_container_width=True)

        req_cols = [
            "jam_belajar",
            "jumlah_kegagalan",
            "jumlah_absen",
            "nilai_tugas",
            "nilai_uts",
            "keinginan_kuliah",
            "akses_internet",
        ]
        missing = [c for c in req_cols if c not in batch_df.columns]

        if missing:
            st.error(
                f"Format CSV tidak cocok. Kolom berikut tidak ditemukan: {', '.join(missing)}"
            )
        else:
            if st.button("🚀 Jalankan Prediksi Massal", type="primary"):
                with st.spinner("Memproses prediksi data..."):
                    # Prediksi dengan Random Forest
                    pred_batch_rf = pipeline.predict(batch_df[req_cols])
                    batch_df["Prediksi_RandomForest"] = pred_batch_rf

                    # Tambahkan probabilitas Lulus
                    proba_batch = pipeline.predict_proba(batch_df[req_cols])
                    classes = list(pipeline.classes_)
                    idx_lulus = (
                        classes.index("Lulus") if "Lulus" in classes else 0
                    )
                    batch_df["Probabilitas_Lulus(%)"] = (
                        proba_batch[:, idx_lulus] * 100
                    ).round(1)

                st.success("✅ Prediksi massal berhasil diselesaikan!")
                st.dataframe(batch_df, use_container_width=True)

                col_stat1, col_stat2 = st.columns(2)
                with col_stat1:
                    dist = batch_df["Prediksi_RandomForest"].value_counts()
                    st.write("📊 **Distribusi Hasil Prediksi:**")
                    st.dataframe(dist)
                with col_stat2:
                    st.bar_chart(dist)

                # Tombol Download Hasil
                csv_result = batch_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 Unduh Hasil Prediksi Lengkap (.csv)",
                    data=csv_result,
                    file_name="hasil_prediksi_kelulusan_siswa.csv",
                    mime="text/csv",
                )


# ================= TAB 3: EVALUASI & PERBANDINGAN =================
with tab3:
    st.subheader("📊 Laporan Perbandingan Performa Model")
    st.write(
        "Berikut merupakan hasil pengujian empiris pada 79 data uji (*test set*) yang sama persis "
        "antara **Model Library (Random Forest)** dan **AI Model Service (Groq LLM)**."
    )

    col_m1, col_m2 = st.columns(2)

    with col_m1:
        st.markdown("#### 🌲 Model Library (Random Forest)")
        m_rf = info["metrik"]
        st.markdown(
            f"""
        - **Algoritma:** Random Forest (Scikit-Learn)
        - **Ukuran Data Latih:** {m_rf['n_train']} siswa (80%)
        - **Ukuran Data Uji:** {m_rf['n_test']} siswa (20%)
        - **Akurasi Data Uji:** `{m_rf['accuracy']*100:.2f}%`
        - **F1-Macro Data Uji:** `{m_rf['f1_macro']:.4f}`
        - **Waktu Inferensi:** Sangat Cepat (< 2 milidetik)
        - **Kebutuhan Jaringan:** Offline (Lokal)
        """
        )

        st.markdown("##### Confusion Matrix Random Forest:")
        cm_rf = m_rf["confusion_matrix"]
        cm_rf_df = pd.DataFrame(
            cm_rf,
            index=["Aktual Lulus", "Aktual Remedial"],
            columns=["Prediksi Lulus", "Prediksi Remedial"],
        )
        st.table(cm_rf_df)

    with col_m2:
        st.markdown("#### 🤖 AI Model Service (Groq LLM)")
        if llm_info:
            st.markdown(
                f"""
            - **Model:** `{llm_info['model_llm']}` via Groq API
            - **Pendekatan:** Few-Shot Prompt Engineering
            - **Ukuran Data Uji:** {llm_info['n_test']} siswa (sama persis)
            - **Akurasi Data Uji:** `{llm_info['accuracy']*100:.2f}%`
            - **F1-Macro Data Uji:** `{llm_info['f1_macro']:.4f}`
            - **Waktu Inferensi:** ~400 - 800 milidetik per request
            - **Kebutuhan Jaringan:** Online (Cloud API)
            """
            )

            st.markdown("##### Confusion Matrix Groq LLM:")
            cm_llm = llm_info["confusion_matrix"]
            cm_llm_df = pd.DataFrame(
                cm_llm,
                index=["Aktual Lulus", "Aktual Remedial"],
                columns=["Prediksi Lulus", "Prediksi Remedial"],
            )
            st.table(cm_llm_df)
        else:
            st.info("Data evaluasi LLM sedang diproses...")

    st.markdown("---")
    st.subheader("💡 Kesimpulan Analisis")
    st.markdown(
        """
    1. **Akurasi & Efisiensi:** Model Library (*Random Forest*) unggul dalam kecepatan inferensi dan efisiensi komputasi karena dilatih langsung pada representasi data numerik tabular.
    2. **Kemampuan Interpretatif:** AI Model Service (*LLM*) unggul dalam memberikan penalaran tekstual (*reasoning*) dan penjelasan latar belakang mengapa seorang siswa berpotensi lulus atau memerlukan remedial.
    3. **Rekomendasi Deployment:** Menggabungkan keduanya dalam arsitektur *hybrid* memberikan solusi terbaik: klasifikasi cepat dan terpercaya dengan Random Forest, disertai rekomendasi intervensi edukatif berbasis narasi LLM.
    """
    )
