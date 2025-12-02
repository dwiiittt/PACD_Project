# 🔬 Deteksi TB dari Citra X-Ray Menggunakan Teknik Analisis Citra Digital

Proyek Akhir Pengolahan Citra Analisis Digital (PACD) Kelompok 5

## Dibuat oleh
* [cite_start]**Ali Imron Filayaty Faqih** [cite: 2]
* [cite_start]**Dwi Gusna** [cite: 2]
* [cite_start]**Sapto Nugroho** [cite: 2]

## Lencana Proyek

![GitHub language count](https://img.shields.io/github/languages/count/dwiiittt/PACD_Project)
![GitHub top language](https://img.shields.io/github/languages/top/dwiiittt/PACD_Project)
![GitHub repo size](https://img.shields.io/github/repo-size/dwiiittt/PACD_Project)
![GitHub contributors](https://img.shields.io/github/contributors/dwiiittt/PACD_Project)

---

## 💡 Tentang Proyek

[cite_start]Proyek ini bertujuan untuk mengembangkan sistem skrining **Tuberkulosis (TB)** [cite: 1, 3] [cite_start]menggunakan pendekatan berbasis **Pengolahan Citra Digital (DIP)** untuk menganalisis citra medis **Radiografi Dada (Chest X-Ray/CXR)**[cite: 4, 13].

[cite_start]Tujuan utama proyek ini adalah menyediakan alat yang cepat dan relatif murah untuk deteksi TB, yang dapat berfungsi sebagai pembaca otomatis (**Computer-Aided Detection/CAD**) untuk meningkatkan efisiensi dan akurasi skrining, terutama di wilayah dengan sumber daya terbatas[cite: 13, 15, 16].

---

## ⚙️ Metodologi dan Alur Sistem (ESFERM)

[cite_start]Sistem deteksi TB ini dikembangkan berdasarkan alur proses yang disebut **ESFERM**[cite: 5, 21], yang merupakan singkatan dari:

1.  [cite_start]**E**nhancement (Prapemrosesan) [cite: 5, 21]
2.  [cite_start]**S**egmentation (Segmentasi) [cite: 5, 21]
3.  [cite_start]**F**eature **E**xtraction and **R**epresentation (Ekstraksi Fitur dan Representasi) [cite: 5, 21]
4.  [cite_start]**M**atching (Klasifikasi) [cite: 5]

### Tahapan Utama:

* [cite_start]**Prapemrosesan:** Meliputi peningkatan kontras citra (menggunakan **Gamma Correction** atau **CLAHE**) dan reduksi derau/noise (menggunakan **Filter Gaussian** atau **Filter Bilateral**)[cite: 52, 55, 57, 62, 69].
* **Segmentasi:** Proses ini bertujuan mengisolasi area paru-paru (*Region of Interest* / ROI). [cite_start]Metode yang digunakan bervariasi antar model, termasuk **Global Thresholding**, **Adaptive Thresholding**, **Metode Otsu**, dan algoritma **Watershed**[cite: 74, 76, 79, 82]. [cite_start]Segmentasi disempurnakan dengan **Operasi Morfologi** dan **Convex Hull**[cite: 90, 99].
* [cite_start]**Ekstraksi Fitur:** Fitur diekstrak dari ROI, mencakup **Intensitas** (rata-rata, standar deviasi, skewness), **Tekstur** (menggunakan **GLCM** dan **LBP**), dan **Bentuk** (mendeteksi properti geometris seperti *eccentricity* dan *solidity*)[cite: 104, 109, 111, 112, 121].
    * [cite_start]**Indikator Lesi TB** yang dicari meliputi infiltrat, konsolidasi, kavitas, efusi pleura, fibrosis, dan kalsifikasi[cite: 7].
* [cite_start]**Klasifikasi:** Citra hasil ekstraksi fitur diklasifikasikan menggunakan algoritma **Support Vector Machine (SVM)** untuk penentuan citra normal atau positif TB[cite: 6, 23, 123].

---

## 📊 Hasil Eksperimen

[cite_start]Tiga model dikembangkan dengan pendekatan yang berbeda pada proses pengolahan citra dan pengambilan fitur untuk perbandingan[cite: 8, 49].

| Model | Akurasi | Presisi (Non-TB) | Recall (Non-TB) | F1-Score (Non-TB) | F1-Score (TB) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | 91,79% | 92% | 92% | 92% | 92% |
| **2** | **96,78%** | 97% | 97% | 97% | 92% |
| **3** | 96% | 96% | 96% | 96% | 96% |


---

## 💾 Dataset

[cite_start]Penelitian ini menggunakan dataset publik **"Tuberculosis (TB) Chest X-ray Database"** dari platform Kaggle[cite: 125, 132].

| Kategori | Jumlah Citra |
| :---: | :---: |
| Normal | [cite_start]3.500 [cite: 126, 128] |
| TB | [cite_start]700 [cite: 126, 128] |
| **Jumlah Total** | [cite_start]**4.200** [cite: 128] |

[cite_start]Dataset dibagi menjadi **80% data pelatihan** dan **20% data pengujian**[cite: 129].
