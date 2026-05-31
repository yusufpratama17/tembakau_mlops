import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time
import requests
from prometheus_client import CollectorRegistry, Counter, Gauge, push_to_gateway

# ==========================================
# 1. SETUP & LOAD MODEL / ENCODER
# ==========================================
st.set_page_config(page_title="PT SJT Sales Dashboard", layout="wide")

@st.cache_resource
def load_assets():
    model = joblib.load('model_classifier.pkl')
    le_bulan = joblib.load('encoder_bulan.pkl')
    le_kategori = joblib.load('encoder_kategori.pkl')
    le_produk = joblib.load('encoder_produk.pkl')
    return model, le_bulan, le_kategori, le_produk

try:
    model, le_bulan, le_kategori, le_produk = load_assets()
except Exception as e:
    st.error(f"Gagal memuat model/encoder. Pastikan file .pkl ada di folder. Error: {e}")

# Mapping Produk & Harga Default per Box
produk_dict = {
    'SKT': ['Sosro Bahu 12', 'Sosro Bahu 16'],
    'SKM': ['Bahamas 12', 'Bahamas 16'],
    'TSC': ['Fines Kasturi', 'Fines Piton']
}

harga_map = {
    'Sosro Bahu 12': 6000000,
    'Sosro Bahu 16': 7500000,
    'Bahamas 12': 10500000,
    'Bahamas 16': 13000000,
    'Fines Kasturi': 5000000,
    'Fines Piton': 5500000
}

# ==========================================
# 2. ANTARMUKA DASHBOARD (UI STREAMLIT)
# ==========================================
st.title("📊 Sales Value Classification Dashboard")
st.subheader("PT Santosa Jaya Tembakau - Sistem Analisis Distributor Bulanan")
st.markdown("---")

# Sidebar untuk Pengaturan Koneksi MLOps Monitoring
st.sidebar.header("⚙️ MLOps Monitoring Config")
ngrok_url = st.sidebar.text_input(
    "Prometheus Gateway URL (Ngrok)", 
    placeholder="https://xxxx-xxxx.ngrok-free.dev"
)

# Layout Form Input Utama
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 📝 Input Data Penjualan")
    
    bulan_pilih = st.selectbox("Pilih Bulan Laporan", [
        'Januari', 'Februari', 'Maret', 'April', 'Mei', 'Juni', 
        'Juli', 'Agustus', 'September', 'Oktober', 'November', 'Desember'
    ])
    
    kategori_pilih = st.selectbox("Kategori Produk Tembakau", list(produk_dict.keys()))
    
    # Menyesuaikan pilihan nama produk berdasarkan kategori yang dipilih
    produk_pilihan_list = produk_dict[kategori_pilih]
    produk_pilih = st.selectbox("Nama Produk Spesifik", produk_pilihan_list)

with col2:
    st.markdown("### 📦 Detail Kuantitas & Harga")
    jumlah_box = st.number_input("Jumlah Penjualan (Per Box)", min_value=1, value=10, step=1)
    
    # Set harga otomatis berdasarkan produk, namun tetap bisa diedit jika ada diskon/perubahan harga
    harga_default = harga_map[produk_pilih]
    harga_box = st.number_input("Harga Per Box (Rp)", min_value=0, value=harga_default, step=50000)
    
    total_nilai = jumlah_box * harga_box
    st.info(f"**Total Nilai Transaksi Bulanan:** Rp {total_nilai:,.0f}")

st.markdown("---")

# ==========================================
# 3. PROSES PREDIKSI & PUSH METRICS
# ==========================================
if st.button("🚀 Proses Analisis Klasifikasi", type="primary"):
    start_time = time.time()
    
    # Jalankan Prediksi Menggunakan Model AI
    try:
        # Transformasi input text menggunakan encoder yang disave
        bulan_encoded = le_bulan.transform([bulan_pilih])[0]
        kategori_encoded = le_kategori.transform([kategori_pilih])[0]
        produk_encoded = le_produk.transform([produk_pilih])[0]
        
        # Buat dataframe untuk prediksi
        input_data = pd.DataFrame([[bulan_encoded, kategori_encoded, produk_encoded, jumlah_box, harga_box]], 
                                  columns=['Bulan', 'Kategori_Produk', 'Nama_Produk', 'Jumlah_Box', 'Harga_Per_Box'])
        
        # Prediksi klasifikasi nilai (0 = HIGH VALUE, 1 = LOW VALUE atau sebaliknya tergantung urutan abjad encoder)
        prediksi_id = model.predict(input_data)[0]
        prediksi_label = le_target.inverse_transform([prediksi_id])[0]
        
        # Hitung Probabilitas Akurasi Prediksi
        probabilitas = model.predict_proba(input_data)[0][prediksi_id] * 100
        latency = time.time() - start_time
        
        # Tampilkan Hasil di Dashboard Web
        st.success("### 📊 Hasil Klasifikasi Sentimen Distributor:")
        c_res1, c_res2, c_res3 = st.columns(3)
        
        with c_res1:
            if prediksi_label == "HIGH VALUE":
                st.metric(label="Kategori Performa Agen", value="🔥 HIGH VALUE")
            else:
                st.metric(label="Kategori Performa Agen", value="📉 LOW VALUE")
                
        with c_res2:
            st.metric(label="Tingkat Keyakinan Model (Confidence)", value=f"{probabilitas:.2f}%")
            
        with c_res3:
            st.metric(label="Kecepatan Latensi Prediksi", value=f"{latency:.4f} detik")
            
        # ==========================================
        # 4. INTEGRASI PROMETHEUS CLIENT (MLOPS GATEWAY)
        # ==========================================
        if ngrok_url:
            try:
                # Setup registry metrik Prometheus
                registry = CollectorRegistry()
                
                # Membuat struktur metrik persis seperti di Pushgateway laporan teman
                c_requests = Counter('app_requests_total', 'Total prediksi yang diproses', registry=registry)
                g_latency = Gauge('inference_latency_seconds', 'Kecepatan pemrosesan model', registry=registry)
                g_high_value = Gauge('model_prediction_high_value', 'Status jika bernilai High Value', registry=registry)
                
                # Isikan nilai ke metrik
                c_requests.inc()
                g_latency.set(latency)
                g_high_value.set(1 if prediksi_label == "HIGH VALUE" else 0)
                
                # Bersihkan URL Ngrok dari slash akhir jika ada
                gateway_clean_url = ngrok_url.strip().rstrip('/')
                
                # Push data metrik ke Prometheus Pushgateway lewat terowongan Ngrok
                push_to_gateway(gateway_clean_url, job='streamlit_dashboard_sjt', registry=registry)
                st.sidebar.success("✅ Metrik MLOps Berhasil Dikirim ke Prometheus!")
            except Exception as prometheus_error:
                st.sidebar.warning(f"⚠️ Gagal push metrik. Periksa kembali URL Ngrok Anda. Detail: {prometheus_error}")
        else:
            st.sidebar.info("ℹ️ Masukkan URL Ngrok di sidebar untuk mulai mengirim metrik ke Grafana lokal.")
            
    except Exception as prediction_error:
        st.error(f"Terjadi kesalahan saat memproses data: {prediction_error}")