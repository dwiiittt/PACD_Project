# 🔬 Deteksi TB dari Citra X-Ray Menggunakan Teknik Analisis Citra Digital

Proyek Akhir Pengolahan Citra Analisis Digital (PACD) Kelompok 5

## Dibuat oleh
* **Ali Imron Filayaty Faqih**
* **Dwi Gusna**
* **Sapto Nugroho**

## Lencana Proyek

![GitHub language count](https://img.shields.io/github/languages/count/dwiiittt/PACD_Project)
![GitHub top language](https://img.shields.io/github/languages/top/dwiiittt/PACD_Project)
![GitHub repo size](https://img.shields.io/github/repo-size/dwiiittt/PACD_Project)
![GitHub contributors](https://img.shields.io/github/contributors/dwiiittt/PACD_Project)

---

## 💡 Tentang Proyek

Proyek ini bertujuan untuk mengembangkan sistem skrining **Tuberkulosis (TB)** menggunakan pendekatan berbasis **Pengolahan Citra Digital (DIP)** untuk menganalisis citra medis **Radiografi Dada (Chest X-Ray/CXR)**.

Tujuan utama proyek ini adalah menyediakan alat yang cepat dan relatif murah untuk deteksi TB, yang dapat berfungsi sebagai pembaca otomatis (**Computer-Aided Detection/CAD**) untuk meningkatkan efisiensi dan akurasi skrining, terutama di wilayah dengan sumber daya terbatas.

---

## ⚙️ Metodologi dan Alur Sistem (ESFERM)

Sistem deteksi TB ini dikembangkan berdasarkan alur proses yang disebut **ESFERM**, yang merupakan singkatan dari:

1.  **E**nhancement (Prapemrosesan)
2.  **S**egmentation (Segmentasi)
3.  **F**eature **E**xtraction and **R**epresentation (Ekstraksi Fitur dan Representasi)
4.  **M**atching (Klasifikasi)

### Tahapan Utama:

* **Prapemrosesan:** Meliputi peningkatan kontras citra (menggunakan **Gamma Correction** atau **CLAHE**) dan reduksi derau/noise (menggunakan **Filter Gaussian** atau **Filter Bilateral**).
* **Segmentasi:** Proses ini bertujuan mengisolasi area paru-paru (*Region of Interest* / ROI). Metode yang digunakan bervariasi antar model, termasuk **Global Thresholding**, **Adaptive Thresholding**, **Metode Otsu**, dan algoritma **Watershed**. Segmentasi disempurnakan dengan **Operasi Morfologi** dan **Convex Hull**.
* **Ekstraksi Fitur:** Fitur diekstrak dari ROI, mencakup **Intensitas** (rata-rata, standar deviasi, skewness), **Tekstur** (menggunakan **GLCM** dan **LBP**), dan **Bentuk** (mendeteksi properti geometris seperti *eccentricity* dan *solidity*).
    * **Indikator Lesi TB** yang dicari meliputi infiltrat, konsolidasi, kavitas, efusi pleura, fibrosis, dan kalsifikasi.
* **Klasifikasi:** Citra hasil ekstraksi fitur diklasifikasikan menggunakan algoritma **Support Vector Machine (SVM)** untuk penentuan citra normal atau positif TB.

---

## 📊 Hasil Eksperimen

Tiga model dikembangkan dengan pendekatan yang berbeda pada proses pengolahan citra dan pengambilan fitur untuk perbandingan.

| Model | Akurasi | Presisi (Non-TB) | Recall (Non-TB) | F1-Score (Non-TB) | F1-Score (TB) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | 91,79% | 92% | 92% | 92% | 92% |
| **2** | **96,78%** | 97% | 97% | 97% | 92% |
| **3** | 96% | 96% | 96% | 96% | 96% |


**Kesimpulan:** Model 2 menunjukkan akurasi tertinggi, namun secara keseluruhan, Model 3 menghasilkan kinerja F1-Score yang lebih seimbang untuk kedua kelas (Non-TB dan TB).

---

## 💾 Dataset

Penelitian ini menggunakan dataset publik **"Tuberculosis (TB) Chest X-ray Database"** dari platform Kaggle.

| Kategori | Jumlah Citra |
| :---: | :---: |
| Normal | 3.500 |
| TB | 700 |
| **Jumlah Total** | **4.200** |

Dataset dibagi menjadi **80% data pelatihan** dan **20% data pengujian**.

## 🌍 Website
Aplikasi hasil penelitian ini dapat dilihat di https://screeningtb.streamlit.app/
