# Analisis Sentimen Ulasan Aplikasi Gojek di Google Play Store Menggunakan Algoritma Support Vector Machine

<div align="center">
  <img width="400" height="500" alt="gojek" src="https://github.com/user-attachments/assets/ecd250b7-b045-4d50-a858-d52594412a82" />
</div>

Aplikasi ini sudah di Deploy di Streamlit Cloud. Link Akses : https://sentiment-analysis-gojek-app.streamlit.app/
## Tujuan Penelitian
1. Mengukur kepuasan pengguna secara otomatis
2. Pemantauan reputasi merek (brand monitoring)
3. Identifikasi masalah dan area perbaikan produk
4. Mendukung pengambilan keputusan bisnis

## Tahapan Penelitian
<div align='center'>
  <img width="400" height="7000" alt="Tahapan Penelitian Chart" src="https://github.com/user-attachments/assets/6a15abc2-871b-44ff-b072-3ec824d3618e" />
</div>

## Pipeline
1. Data Preprocessing :
   - Case Folding
   - Menghapus Tanda Baca
   - Normalisasi kalimat (Mengganti dengan kata baku)
     - gak, enggak, ngak, ndak --> tidak
     - pake, make --> memakai
   - Tokenization
   - Menghapus Angka
   - Menghapus Stopwords
2. Ekstraksi Fitur dengan TF-IDF
3. Split data dengan proporsi 80% Train / 20% Test
4. Lakukan pelatihan Model menggunakan Algoritma Support Vector Machine dengan Kernel Linear
5. Evaluasi hasil model dengan Metric Accuracy, Recall, Precision, F1-Score
6. Lakukan Inference / Prediksi
7. Deploy Melalui Streamlit Cloud

## Hasil Evaluasi untuk data Test
- Accuracy : 0.8613
- Precision : 0.8531
- Recall : 0.7155
- F1-Score : 0.7782

