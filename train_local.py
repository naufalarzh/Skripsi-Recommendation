# ==============================================================================
# SKRIP PENOLONG: TRAIN 3 MODEL LOKAL & PILIH YANG TERBAIK (ANTI-VERSION-ERROR)
# ==============================================================================
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder

# Impor 3 algoritma & metrik evaluasi
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import joblib

print("⏳ Membaca dataset bersih...")
df = pd.read_csv("exercise_clean_final.csv")

# Gabungkan teks kembali untuk keperluan training
df["text"] = df["Title"].astype(str) + " " + df["Desc"].astype(str)

# ---------------------------------------------------------
# BALANCING DATA (Seperti di Colab)
# ---------------------------------------------------------
print("⚖️ Melakukan balancing data (Maksimal 200 sampel per BodyPart)...")
max_per_class = 200
balanced_parts = []

for bodypart, group in df.groupby("BodyPart"):
    if len(group) > max_per_class:
        group = resample(group, replace=False, n_samples=max_per_class, random_state=42)
    balanced_parts.append(group)

df_balanced = (
    pd.concat(balanced_parts).sample(frac=1, random_state=42).reset_index(drop=True)
)

# Siapkan data X dan y dari data yang sudah seimbang
X = df_balanced[["text", "Type", "Equipment", "Level"]]
y = df_balanced["BodyPart"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------------------------------------------------------
# PEMBUATAN PIPELINE & MODEL POOL
# ---------------------------------------------------------
print("⚙️ Membangun pipeline transformer...")
preprocessor = ColumnTransformer(
    transformers=[
        (
            "text",
            TfidfVectorizer(
                max_features=2000, ngram_range=(1, 2), stop_words="english"
            ),
            "text",
        ),
        ("cat", OneHotEncoder(handle_unknown="ignore"), ["Type", "Equipment", "Level"]),
    ]
)

# Menyiapkan 3 kandidat model
models_pool = {
    "Logistic Regression": Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000, class_weight="balanced", random_state=42
                ),
            ),
        ]
    ),
    "Random Forest": Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=100, class_weight="balanced", random_state=42
                ),
            ),
        ]
    ),
    "Naive Bayes": Pipeline(
        steps=[("preprocessor", preprocessor), ("classifier", MultinomialNB())]
    ),
}

# ---------------------------------------------------------
# TRAINING, EVALUASI, DAN PEMILIHAN OTOMATIS
# ---------------------------------------------------------
best_acc = 0
best_model_name = ""
best_model = None

print("\n🚀 Memulai proses training dan komparasi model...")
for model_name, pipeline in models_pool.items():
    print(f"-> Melatih {model_name}...")
    pipeline.fit(X_train, y_train)

    # Evaluasi akurasi
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"   ✅ Akurasi {model_name}: {acc:.4f}")

    # Kunci model jika akurasinya lebih tinggi
    if acc > best_acc:
        best_acc = acc
        best_model_name = model_name
        best_model = pipeline

print("-" * 60)
print(f"🏆 MODEL TERBAIK: {best_model_name} dengan akurasi {best_acc:.4f}")
print("-" * 60)

# ---------------------------------------------------------
# SIMPAN MODEL TERBAIK UNTUK STREAMLIT
# ---------------------------------------------------------
nama_file_model = "model_klasifikasi_bodypart.pkl"
joblib.dump(best_model, nama_file_model)

print(
    f"💾 BERHASIL! File '{nama_file_model}' (menggunakan algoritma {best_model_name}) sukses dibuat dan siap dijalankan di Streamlit."
)
