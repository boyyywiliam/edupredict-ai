import json
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# 1. Load Dataset
df = pd.read_csv("dataset_kelulusan_siswa.csv")

fitur_numerik = ["jam_belajar", "jumlah_kegagalan", "jumlah_absen", "nilai_tugas", "nilai_uts"]
fitur_kategori = ["keinginan_kuliah", "akses_internet"]
target = "status_kelulusan"

X = df[fitur_numerik + fitur_kategori]
y = df[target]

# 2. Split Data (80% Train, 20% Test) dengan Stratified Sampling
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Simpan data train & test ke CSV
train_df = pd.concat([X_train, y_train], axis=1)
test_df = pd.concat([X_test, y_test], axis=1)
train_df.to_csv("data_train.csv", index=False)
test_df.to_csv("data_test.csv", index=False)
print(f"Data latih tersimpan: {len(train_df)} baris")
print(f"Data uji tersimpan: {len(test_df)} baris")

# 3. Pipeline Preprocessing & Model (Random Forest)
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), fitur_numerik),
        ("cat", OneHotEncoder(drop="if_binary", handle_unknown="ignore"), fitur_kategori),
    ]
)

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)),
    ]
)

# 4. Latih Model
pipeline.fit(X_train, y_train)

# 5. Evaluasi pada Test Set
y_pred = pipeline.predict(X_test)
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average="macro")
cm = confusion_matrix(y_test, y_pred, labels=["Lulus", "Remedial"])

print("\n=== HASIL EVALUASI MODEL LIBRARY (RANDOM FOREST) ===")
print(f"Accuracy : {acc:.4f} ({acc*100:.2f}%)")
print(f"F1 Macro : {f1:.4f}")
print("Confusion Matrix (Lulus, Remedial):")
print(cm)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 6. Simpan Model & Info
joblib.dump(pipeline, "model_kelulusan.pkl")

model_info = {
    "nama_model": "Random Forest Classifier (Scikit-Learn)",
    "deskripsi": "Model klasifikasi untuk memprediksi kelulusan siswa berbasis data historis akademik.",
    "fitur_numerik": fitur_numerik,
    "fitur_kategori": fitur_kategori,
    "target": target,
    "kelas": list(pipeline.classes_),
    "rentang_fitur": {col: [float(df[col].min()), float(df[col].max())] for col in fitur_numerik},
    "kategori_fitur": {col: list(df[col].unique()) for col in fitur_kategori},
    "metrik": {
        "n_train": len(X_train),
        "n_test": len(X_test),
        "accuracy": float(acc),
        "f1_macro": float(f1),
        "confusion_matrix": cm.tolist(),
    },
}

with open("model_info.json", "w", encoding="utf-8") as f:
    json.dump(model_info, f, indent=2)

print("\nModel berhasil disimpan ke 'model_kelulusan.pkl' dan 'model_info.json'.")
