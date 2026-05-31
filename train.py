import pandas as pd
import numpy as np
import os
import mlflow
import dagshub
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score
import joblib

# 1. Inisialisasi Integrasi DagsHub & MLflow
# Menggunakan environment variable yang nanti dikirim oleh GitHub Actions
DAGSHUB_REPO = "tembakau_mlops"
DAGSHUB_USER = "yusufpratama17"

if "DAGSHUB_TOKEN" in os.environ:
    os.environ["MLFLOW_TRACKING_USERNAME"] = DAGSHUB_USER
    os.environ["MLFLOW_TRACKING_PASSWORD"] = os.environ["DAGSHUB_TOKEN"]

# Hubungkan MLflow ke server DagsHub secara remote
mlflow.set_tracking_uri(f"https://dagshub.com/{DAGSHUB_USER}/{DAGSHUB_REPO}.mlflow")
mlflow.set_experiment("Sales_Value_Classification_PT_SJT")

with mlflow.start_run():
    print("Memulai proses training model MLOps...")
    
    # 2. Load Data Penjualan Bulanan
    df_tembakau = pd.read_csv('data_penjualan_bulanan.csv')

    # 3. Preprocessing (Label Encoding)
    le_bulan = LabelEncoder()
    df_tembakau['Bulan'] = le_bulan.fit_transform(df_tembakau['Bulan'])

    le_kategori = LabelEncoder()
    df_tembakau['Kategori_Produk'] = le_kategori.fit_transform(df_tembakau['Kategori_Produk'])

    le_produk = LabelEncoder()
    df_tembakau['Nama_Produk'] = le_produk.fit_transform(df_tembakau['Nama_Produk'])

    le_target = LabelEncoder()
    df_tembakau['Target'] = le_target.fit_transform(df_tembakau['Target'])

    # Fitur dan Target
    X = df_tembakau[['Bulan', 'Kategori_Produk', 'Nama_Produk', 'Jumlah_Box', 'Harga_Per_Box']]
    y = df_tembakau['Target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 4. Training Model Random Forest
    n_estimators = 100
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    model.fit(X_train, y_train)

    # 5. Evaluasi Metrik
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')

    print(f"Hasil Training -> Akurasi: {acc*100:.2f}%, F1-Score: {f1:.4f}")

    # 6. Logging Parameter & Metrik ke DagsHub/MLflow
    mlflow.log_param("model_type", "RandomForest")
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("recall", rec)
    mlflow.log_metric("f1_score", f1)

    # 7. Simpan Model dan Encoder Secara Lokal
    joblib.dump(model, 'model_classifier.pkl')
    joblib.dump(le_bulan, 'encoder_bulan.pkl')
    joblib.dump(le_kategori, 'encoder_kategori.pkl')
    joblib.dump(le_produk, 'encoder_produk.pkl')
    
    # Log model ke DagsHub Artifacts
    mlflow.sklearn.log_model(model, "model")
    
    print("Semua parameter, metrik, dan model berhasil dicatat ke DagsHub!")