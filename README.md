# 🔋 MachineLearning_RUL_Prediction

Aplikasi ini memprediksi Remaining Useful Life (RUL) atau sisa umur pakai baterai berdasarkan fitur-fitur utama hasil pengukuran siklus baterai. Model ini dibangun menggunakan Random Forest Regressor dan telah dilatih pada dataset Battery_RUL.csv.

## Fitur Input

Pengguna diminta memasukkan tiga fitur utama baterai:

- **Maximum Voltage Discharge (V)**: Tegangan maksimum saat baterai didischarge.
- **Minimum Voltage Discharge (V)**: Tegangan minimum saat baterai didischarge.
- **Time at 4.15V (s)**: Lama waktu (dalam detik) baterai berada pada tegangan 4.15V.

## Cara Kerja

1. **Input Data**  
   Pengguna memasukkan nilai fitur melalui form pada aplikasi web (dibangun dengan Streamlit).

2. **Prediksi**  
   Model pipeline yang telah dilatih (`rul_model_pipeline.joblib`) akan memproses input dan memberikan prediksi RUL (dalam satuan siklus).

3. **Output**  
   Hasil prediksi ditampilkan secara interaktif di aplikasi, lengkap dengan indikator progress.

## Model

- **Algoritma**: Random Forest Regressor
- **Fitur yang digunakan**:
  - F3: Maximum Voltage Discharge (V)
  - F4: Minimum Voltage Discharge (V)
  - F5: Time at 4.15V (s)
- **Target**: Remaining Useful Life (RUL) dalam satuan siklus

## Cara Menjalankan

1. Pastikan dependensi sudah terinstall:
   pip install -r requirements.txt

2. Jalankan aplikasi Streamlit:
   streamlit run app.py

3. Buka browser ke alamat yang diberikan oleh Streamlit (biasanya http://localhost:8501).

## File Penting

- `app.py` : Source code aplikasi Streamlit.
- `rul_model_pipeline.joblib` : Model pipeline hasil training.
- `Battery_RUL.csv` : Dataset yang digunakan untuk training.
- `requirements.txt` : Daftar dependensi Python.

---

**Catatan:**  
Model ini hanya akurat untuk rentang data yang mirip dengan data training. Pastikan input berada dalam rentang yang wajar sesuai data baterai yang digunakan saat training.
