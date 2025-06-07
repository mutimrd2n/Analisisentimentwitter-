# 🐦 Analisis Sentimen Twitter dengan Naive Bayes (NB05)

Proyek ini merupakan bagian dari tugas mata kuliah **Artificial Intelligence - Supervised Learning**, dengan fokus pada penerapan algoritma **Naive Bayes** untuk klasifikasi sentimen tweet (positif, negatif, atau netral).

---

## 📊 Deskripsi Proyek

Tujuan dari proyek ini adalah mengembangkan model klasifikasi sentimen dari data tweet terkait maskapai penerbangan di Amerika Serikat. Algoritma yang digunakan adalah **Multinomial Naive Bayes**, yang umum digunakan untuk teks.

---

## 🧠 Algoritma

- **Multinomial Naive Bayes**  
- Cocok untuk data teks (berbasis frekuensi kata)
- Vektorisasi dilakukan dengan **CountVectorizer**

---

## 📁 Dataset

- **Nama**: Twitter US Airline Sentiment Dataset  
- **Jumlah data**: 14.640 tweet  
- **Label**: `positive`, `neutral`, `negative`  
- [Kaggle Link](https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment)

---

## 🔧 Langkah-langkah Utama

1. **Import & Load Dataset**
2. **Preprocessing Teks**
   - Lowercase
   - Hapus URL, mention, hashtag, tanda baca
3. **Ekstraksi Fitur**
   - `CountVectorizer` → representasi teks ke numerik
4. **Split Data**
   - 80% Training / 20% Testing
5. **Training Model**
   - `MultinomialNB` dari `scikit-learn`
6. **Evaluasi Model**
   - Akurasi, precision, recall, f1-score
7. **Visualisasi**
   - Confusion matrix dalam bentuk heatmap

---

## 📈 Hasil Evaluasi

- **Akurasi**: 77.25%
- **F1-Score Kelas Negatif**: 0.86
- **F1-Score Kelas Netral**: 0.49 (masih perlu peningkatan)
- **Confusion Matrix**: tersedia dalam file `confusion_matrix.png`

---

## 📦 Requirements

```bash
pip install -r requirements.txt
