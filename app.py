import re
import os
import nltk
import joblib
import string
import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from indoNLP.preprocessing import replace_slang
from sklearn.metrics import classification_report, confusion_matrix

from io import BytesIO
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle, Spacer, Image

# Atur lokasi download nltk_data secara eksplisit
nltk_data_path = os.path.join(os.getcwd(), 'nltk_data')
if not os.path.exists(nltk_data_path):
    os.mkdir(nltk_data_path)

# Tambahkan path custom ke NLTK
nltk.data.path.append(nltk_data_path)

# Download punkt ke path tersebut
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", download_dir=nltk_data_path)

# DEFINE STOPWORDS
stop_words = {
    'aku', 'saya', 'kamu', 'kau', 'dia', 'mereka', 'kita', 'kami', 
    '-ku', '-mu', '-nya', 'kalian', 'engkau', 'hamba', 'beta', 
    'anda', 'beliau', 'dan', 'atau', 'tetapi', 'namun', 'sebab', 
    'karena', 'jika', 'kalau', 'agar', 'supaya', 'meski', 
    'walaupun', 'sementara', 'sedangkan', 'bahwa', 'sehingga', 
    'sejak', 'sampai', 'ketika', 'sebelum', 'sesudah', 'setelah', 
    'sewaktu', 'di', 'ke', 'dari', 'pada', 'dalam', 'dengan', 
    'tanpa', 'atas', 'bawah', 'antara', 'untuk', 'kepada', 'oleh', 
    'terhadap', 'seperti', 'bagi', 'menurut', 'tentang', 'hingga', 
    'melalui', 'sepanjang', 'versus', 'lah', 'kah', 'pun', 'tah', 
    'per', 'para', 'si', 'sang', 'yang', 'itu', 'ini', 'sini', 
    'situ', 'sana', 'anu', 'hal', 'segala', 'suatu', 'seseorang', 
    'sendiri', 'diri', 'pula', 'juga', 'hanya', 'cuma', 'hampir', 
    'lagi', 'lebih', 'kurang', 'tidak', 'bukan', 'jangan', 'belum', 
    'sudah', 'pernah', 'akan', 'telah', 'hendak', 'musti', 'bisa', 
    'dapat', 'boleh', 'harus', 'perlu', 'mau', 'ingin', 'siap', 
    'masih', 'adalah', 'ialah', 'yaitu', 'adanya', 'bagaimana', 
    'apa', 'apakah', 'kenapa', 'mengapa', 'mana', 'kapan', 'berapa', 
    'siapa', 'saja', 'sama', 'ataukah', 'dengankah', 'dikah', 
    'kahkah', 'kekah', 'olehkah', 'punakah', 'ada', 'adanya', 
    'amat', 'bagai', 'bagaimana', 'bagi', 'bahwa', 'bahwasanya', 
    'bahwasannya', 'baru', 'belum', 'bukan', 'buat', 'dahulu', 
    'dalam', 'dan', 'dengan', 'dia', 'diantara', 'dong', 
    'dua', 'dulu', 'empat', 'enggak', 'enggaknya', 'entah', 'gak', 
    'gua', 'guna', 'gunakan', 'hampir', 'harus', 'hingga', 'ia', 
    'ialah', 'ibu', 'ini', 'itu', 'jadi', 'jika', 'kami', 'kamu', 
    'karena', 'ke', 'kemudian', 'kepada', 'kini', 'kira', 'kita', 
    'lain', 'lalu', 'lama', 'lima', 'lu', 'macam', 
    'mahu', 'maka', 'malah', 'mana', 'masa', 'masih', 'masing', 
    'melainkan', 'melalui', 'memang', 'mempunyai', 'mulai', 'nah', 
    'nak', 'nantinya', 'nanti', 'nya', 'oleh', 'pada', 'padahal', 
    'pak', 'para', 'per', 'percuma', 'pula', 'pun', 'rupanya', 
    'saat', 'saja', 'saling', 'sama', 'sambil', 'sampai', 'sana', 
    'saya', 'se', 'sebab', 'sebagai', 'sebagaimana', 'sebaiknya', 
    'sebaliknya', 'sebanyak', 'sebelum', 'sebelumnya', 'sebenarnya', 
    'secara', 'secukupnya', 'sedang', 'sedangkan', 'sedikit', 
    'sedikitnya', 'segala', 'segalanya', 'segera', 'seharusnya', 
    'sehingga', 'sejak', 'sejenak', 'sekadar', 'sekali', 'sekalian', 
    'sekaligus', 'sekalipun', 'sekarang', 'seketika', 'sekiranya', 
    'sekitar', 'sela', 'selain', 'selalu', 'selama', 'seluruh', 
    'seluruhnya', 'semacam', 'semakin', 'semampu', 'semasa', 
    'semasih', 'semata', 'semaunya', 'sementara', 'sempat', 
    'semua', 'semula', 'sendiri', 'sendirian', 'sendirinya', 
    'seolah', 'seorang', 'sepanjang', 'sepantasnya', 
    'sepantasnyalah', 'seperlunya', 'seperti', 'sepertinya', 
    'serupa', 'sesaat', 'sesama', 'sesampai', 'sesegera', 
    'sesekali', 'seseorang', 'sesuatu', 'sesudah', 'sesudahnya', 
    'setelah', 'setempat', 'setengah', 'seterusnya', 'setiap', 
    'setiba', 'setidaknya', 'seusai', 'sewaktu', 'siap', 'siapa', 
    'siapakah', 'sini', 'sinilah', 'suatu', 'sudah', 'sudahkah', 
    'supaya', 'tadi', 'tadinya', 'tak', 'tanpa', 'tanya', 
    'tanyalah', 'tapi', 'tentu', 'tentulah', 'terdahulu', 
    'terhadap', 'tersebut', 'tersebutlah', 'tertentu', 'tiap', 
    'tiba', 'tidak', 'tidakkah', 'toh', 'tujuh', 'untuk', 'usah', 
    'usai', 'waduh', 'wah', 'wahai', 'walau', 'walaupun', 'wong', 'nya',
    'ya', 'yaitu', 'yakin', 'yakni', 'yang', 'gopay', 'gojek', 'ojek', 'gofood', 'gopood', 'grab', 'goride', 'go-ride'
}

# Load model & tf-idf
linear_svc = joblib.load("svc_linear_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")

label_map = {0: "Negatif", 1: "Positif"}

def simple_tokenize(text):
    return re.findall(r'\b\w+\b', text)

# Preprocessing function
def preprocess(komentar):
    komentar = komentar.lower()
    komentar_norm = replace_slang(komentar)
    tokens = simple_tokenize(komentar_norm)
    tokens = [t for t in tokens if not re.search(r'\d', t)]
    tokens = [t for t in tokens if t not in string.punctuation]
    tokens = [t for t in tokens if t not in stop_words]
    return ' '.join(tokens)

# Prediksi satu komentar
def predict_single(komentar):
    cleaned = preprocess(komentar)
    if not cleaned.strip():
        return "TIDAK DIKETAHUI"
    vector = tfidf.transform([cleaned])
    return str(linear_svc.predict(vector)[0])

# Prediksi untuk banyak komentar dengan progress bar
def predict_bulk(df):
    st.info("🔄 Memproses data... Mohon tunggu sebentar.")
    progress_bar = st.progress(0)

    cleaned_texts = []
    total = len(df)
    for i, komentar in enumerate(df['komentar']):
        cleaned = preprocess(komentar)
        cleaned_texts.append(cleaned)
        if total > 0:
            progress_bar.progress((i + 1) / total)

    df['cleaned'] = cleaned_texts
    X = tfidf.transform(df['cleaned'])
    df['prediksi'] = linear_svc.predict(X)

    progress_bar.empty()
    st.success("✅ Prediksi selesai!")
    return df

# DOWNLOAD REPORT
def generate_pdf_report(df, report_stats=None, cm=None):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    
    # Style untuk teks di tabel agar wrap otomatis
    table_text_style = ParagraphStyle(
        'TableParagraph',
        fontSize=9,
        leading=12,
        wordWrap='CJK',  # penting agar teks panjang bisa dibungkus
    )

    elements = []

    # Judul
    elements.append(Paragraph("Laporan Analisis Sentimen Ulasan Aplikasi Gojek", styles['Title']))
    elements.append(Spacer(1, 12))

    # Tabel Hasil Prediksi
    elements.append(Paragraph("1. Tabel Hasil Prediksi", styles['Heading2']))
    table_data = [["Komentar", "Label Asli", "Prediksi"]]
    for _, row in df.iterrows():
        komentar_paragraph = Paragraph(str(row['komentar']), table_text_style)
        label_paragraph = Paragraph(str(row.get('sentiment', 'Tidak Ada')), table_text_style)
        prediksi_paragraph = Paragraph(str(row['prediksi']), table_text_style)
        table_data.append([komentar_paragraph, label_paragraph, prediksi_paragraph])

    # Atur lebar kolom: komentar lebih lebar
    table = Table(table_data, colWidths=[300, 80, 80])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('LEFTPADDING', (0, 0), (-1, -1), 4),
        ('RIGHTPADDING', (0, 0), (-1, -1), 4),
    ]))
    elements.append(table)
    elements.append(Spacer(1, 12))

    # Statistik evaluasi
    if report_stats:
        elements.append(Paragraph("2. Statistik Evaluasi", styles['Heading2']))
        for label, metrics in report_stats.items():
            elements.append(Paragraph(f"{label}: {metrics}", styles['BodyText']))
        elements.append(Spacer(1, 12))

    # Confusion Matrix sebagai gambar
    if cm is not None:
        elements.append(Paragraph("3. Confusion Matrix", styles['Heading2']))
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Negatif', 'Positif'],
                    yticklabels=['Negatif', 'Positif'], ax=ax)
        ax.set_xlabel("Prediksi")
        ax.set_ylabel("Label Sebenarnya")

        img_buffer = BytesIO()
        plt.savefig(img_buffer, format='png', bbox_inches='tight')
        plt.close(fig)
        img_buffer.seek(0)
        elements.append(Image(img_buffer, width=300, height=300))
        elements.append(Spacer(1, 12))

    doc.build(elements)
    buffer.seek(0)
    return buffer



# UI STREAMLIT
st.title("📱 Analisis Sentimen Ulasan Aplikasi Gojek")
st.sidebar.header("🔍 Pilih metode input")

input_method = st.sidebar.radio("Metode input", ["Input Teks", "Upload File CSV"])

if input_method == "Input Teks":
    komentar = st.text_area("Masukkan ulasan aplikasi:")
    if st.button("Prediksi"):
        if komentar.strip() != "":
            hasil = predict_single(komentar)
            st.success(f"{komentar} adalah komentar **{label_map.get(int(hasil), 'TIDAK DIKETAHUI')}**")
        else:
            st.warning("Input teks tidak boleh kosong!")

elif input_method == "Upload File CSV":
    uploaded_file = st.file_uploader("Unggah file CSV dengan kolom 'komentar':", type=["csv"])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file, encoding='utf-8')
            if 'komentar' not in df.columns:
                st.error("Format file tidak valid! Kolom 'komentar' tidak ditemukan.")
            else:
                st.success("File valid. Memproses data...")
                df_predicted = predict_bulk(df)
                
                st.subheader("📄 Tabel Hasil Prediksi")
                df_display = df_predicted.copy()
                df_display['prediksi'] = df_display['prediksi'].map({0: 'Negatif', 1: 'Positif'})
                st.dataframe(df_display[['komentar', 'sentiment', 'prediksi']])
                
                st.subheader("📊 Statistik Evaluasi (Opsional)")

                # CHECK TRUE LABELS
                if 'sentiment' in df.columns:
                    label_mapping = {'negatif': 0, 'positif': 1}
                    true_labels = df['sentiment'].str.lower().map(label_mapping)
                else:
                    true_labels = None

                # IF TRUE LABELS NOT IN CSV
                if true_labels is not None:
                    st.text("Hasil Evaluasi:")
                    report = classification_report(true_labels, df_predicted['prediksi'], output_dict=True)
                    st.json(report)
                    
                    st.subheader("📉 Confusion Matrix")
                    cm = confusion_matrix(true_labels, df_predicted['prediksi'], labels=[0, 1])
                    fig, ax = plt.subplots()
                    sns.heatmap(cm, 
                                annot=True, 
                                fmt='d', 
                                cmap='Blues', 
                                xticklabels=['Negatif', 'Positif'], 
                                yticklabels=['Negatif', 'Positif'], 
                                ax=ax
                                )
                    ax.set_xlabel("Prediksi")
                    ax.set_ylabel("Label Sebenarnya")

                    st.pyplot(fig)

                st.subheader("📥 Unduh Hasil")
                csv = df_predicted.to_csv(index=False).encode('utf-8')
                st.download_button("💾 Download Hasil", data=csv, file_name="hasil_sentimen.csv", mime='text/csv')

                # Tombol download PDF laporan
                if 'sentiment' in df.columns:
                    pdf_buffer = generate_pdf_report(df_display, report_stats=report, cm=cm)
                else:
                    pdf_buffer = generate_pdf_report(df_display)

                st.download_button(
                label="📄 Download Laporan PDF",
                data=pdf_buffer,
                file_name="laporan_sentimen.pdf",
                mime="application/pdf"
                )
        except Exception as e:
            st.error(f"Terjadi error saat membaca file: {str(e)}")



