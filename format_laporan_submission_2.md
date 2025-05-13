# Laporan Proyek Machine Learning - Fitria Anggraini

## Project Overview

Perkembangan teknologi digital telah membawa perubahan signifikan dalam dunia literasi dan industri buku. Akses terhadap informasi dan koleksi buku kini semakin mudah melalui platform digital, sehingga jumlah pilihan yang tersedia bagi pembaca meningkat secara drastis. Namun, melimpahnya pilihan ini menimbulkan tantangan berupa information overload, yaitu kondisi di mana pembaca kesulitan menyaring informasi atau buku yang sesuai. Menurut Hariyati & Heriyanto (2021), fenomena ini turut mengubah perilaku pencarian informasi secara luas dan tidak hanya memengaruhi masyarakat umum, tetapi juga ekosistem literasi secara keseluruhan.

Permasalahan information overload ini dirasakan oleh berbagai pihak dalam ekosistem literasi digital. Bagi pembaca, banyaknya pilihan sering kali menyulitkan mereka dalam menemukan buku yang sesuai dengan minat, kebutuhan, atau preferensi pribadi. Proses pencarian buku yang relevan secara manual menjadi tidak efisien, membingungkan, dan dapat menurunkan minat membaca, terutama bagi pengguna yang tidak memiliki referensi khusus atau waktu yang cukup untuk menelusuri seluruh katalog buku (Alkaff dkk., 2020). Di sisi lain, pengelola perpustakaan, penulis, dan penerbit juga menghadapi tantangan dalam memastikan karya mereka dapat menjangkau audiens yang tepat dan meningkatkan sirkulasi buku (Tian dkk., 2019). Salah satu pihak yang terdampak secara langsung oleh ledakan informasi ini adalah pustakawan. Tuntutan peran pustakawan kini semakin kompleks, tidak hanya sebagai pengelola koleksi, tetapi juga sebagai fasilitator dalam membantu pengguna menemukan informasi valid di tengah banjir data (Hariyati & Heriyanto, 2021).

Sistem rekomendasi buku hadir sebagai solusi untuk mengatasi tantangan tersebut. Sistem ini memanfaatkan data perilaku pengguna (riwayat baca, rating, pencarian) dan fitur intrinsik buku (genre, sinopsis, penulis) untuk menyajikan rekomendasi yang dipersonalisasi, sehingga meningkatkan relevansi dan kepuasan pengguna (Kim dkk., 2023). Dua pendekatan utama dalam sistem rekomendasi adalah content-based filtering dan collaborative filtering. Content-based filtering merekomendasikan buku berdasarkan kemiripan atribut konten, sedangkan collaborative filtering mengandalkan pola interaksi antar pengguna untuk mengidentifikasi kelompok pengguna dengan preferensi serupa (Az Zayyad, 2021; Kim dkk., 2023).

Implementasi sistem rekomendasi buku terbukti dapat meningkatkan kepuasan pengguna, mempercepat proses pencarian buku, serta mendukung pengelolaan katalog perpustakaan atau toko buku digital secara lebih efisien. Selain itu, sistem ini juga memberikan peluang baru bagi penulis dan penerbit untuk menjangkau pembaca yang lebih luas dan meningkatkan penjualan atau sirkulasi buku (Abidatillah, 2024).

Berdasarkan urgensi dan manfaat yang ditawarkan, pengembangan sistem rekomendasi buku menjadi salah satu solusi inovatif yang sangat dibutuhkan dalam ekosistem literasi digital saat ini, baik untuk mendukung kebutuhan pembaca, pengelola perpustakaan, maupun pelaku industri buku secara keseluruhan.

## Business Understanding

Untuk menjawab tantangan information overload dalam dunia literasi digital, sistem rekomendasi buku dirancang guna membantu pengguna menemukan bacaan yang sesuai dengan minat dan preferensi mereka. Sistem ini memanfaatkan informasi dari konten buku serta interaksi pengguna sebelumnya agar dapat memberikan rekomendasi yang relevan, personal, dan efisien.

### Problem Statements

- Bagaimana cara memberikan rekomendasi buku yang relevan bagi setiap pengguna berdasarkan data historis?
- Bagaimana membandingkan efektivitas dua pendekatan sistem rekomendasi meggunakan Content-Based Filtering dan Collaborative Filtering dalam menyarankan buku yang relevan? 
- Bagaimana mengevaluasi performa sistem rekomendasi secara kuantitatif?

### Goals

- Mengembangkan sistem rekomendasi buku yang mampu memberikan saran secara personal berdasarkan histori interaksi dan kemiripan konten.
- Membangun dan membandingkan dua pendekatan sistem rekomendasi:
  * Content-Based Filtering (CBF) untuk memberikan rekomendasi berdasarkan kemiripan konten buku.
  * Collaborative Filtering (CF), khususnya pendekatan Neural Collaborative Filtering (NCF), untuk merekomendasikan buku berdasarkan pola interaksi pengguna.
- Mengevaluasi performa model menggunakan metrik seperti Precision@10, Recall@10, F1-Score, Mean Absolute Error (MAE), dan Root Mean Squared Error (RMSE) untuk menilai kualitas prediksi dan relevansi.

### Solution statements

Untuk mencapai tujuan di atas, proyek ini mengimplementasikan dua pendekatan utama:
- Content-Based Filtering, memanfaatkan informasi konten buku seperti judul, penulis, dan penerbit. Data tersebut direpresentasikan menggunakan teknik TF-IDF Vectorization untuk mengekstrak fitur penting dari teks. Selanjutnya, kemiripan antar buku dihitung menggunakan metode cosine similarity, lalu digunakan model K-Nearest Neighbors (KNN) untuk mencari buku yang mirip dengan buku yang pernah disukai pengguna. Dengan pendekatan ini, sistem dapat memberikan rekomendasi berdasarkan kemiripan karakteristik buku yang pernah dinilai oleh pengguna.
- Collaborative Filtering berbasis deep learning dengan metode Neural Collaborative Filtering (NCF). Pendekatan ini tidak melihat isi buku, melainkan berfokus pada pola interaksi antara pengguna dan buku melalui data rating. Model ini menggunakan embedding layer untuk merepresentasikan pengguna dan item. Melalui jaringan saraf, sistem dapat mempelajari hubungan kompleks di antara keduanya. Hasilnya adalah sistem yang dapat memprediksi kemungkinan ketertarikan seorang pengguna terhadap buku tertentu, bahkan jika buku tersebut belum pernah diulas oleh pengguna serupa. 

## Data Understanding

Dataset yang digunakan dalam proyek ini adalah Book Recommendation Dataset yang tersedia secara publik melalui platform <a href="https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset/data?select=Ratings.csv">Kaggle</a>. Dataset ini terdiri dari tiga file utama, yaitu `Books.csv`, `Users.csv`, dan `Ratings.csv` yang akan dijelaskan secara detail sebagai berikut.

a. Books.csv
Dataset ini berisi metadata dari buku-buku yang tersedia dalam sistem.

| Kolom | Jumlah Baris | Tipe Data | Deskripsi |
|-------|--------------|-----------|-----------|
| `ISBN` | 271360 | object | Nomor identifikasi unik untuk buku (International Standard Book Number). |
| `Book-Title` | 271360 | object | Judul dari buku. |
| `Book-Author` | 271358 | object | Nama penulis buku. |
| `Year-Of-Publication` | 271360 | object | Tahun terbit buku. |
| `Publisher` | 271358 | object | Nama penerbit buku. |
| `Image-URL-S` | 271360 | object | URL gambar ukuran kecil untuk cover buku. |
| `Image-URL-M` | 271360 | object | URL gambar ukuran sedang. |
| `Image-URL-L` | 271357 | object | URL gambar ukuran besar. |

Berikut adalah kondisi data pada file Books.csv:
- Kolom `Image-URL-S`, `Image-URL-M`, dan `Image-URL-L` tidak disertakan dalam proses pemodelan karena tidak memiliki kontribusi terhadap rekomendasi, serta tidak relevan untuk pendekatan berbasis konten maupun interaksi pengguna.
- Terdapat dua nilai kosong pada kolom `Book-Author` dan `Publisher`, sehingga kedua baris tersebut dihapus dari dataset.
- Kolom `Year-Of-Publication` awalnya bertipe tidak konsisten, sehingga dikonversi ke numerik dan difilter hanya tahun antara 1950–2025.

b. Ratings.csv
Dataset ini berisi data interaksi berupa rating yang diberikan pengguna terhadap buku.

| Kolom | Jumlah Baris | Tipe Data | Deskripsi |
|-------|--------------|-----------|-----------|
| `User-ID` | 1149780 | int64 | ID unik pengguna yang memberikan rating. |
| `ISBN` | 1149780 | object | ISBN buku yang diberi rating. |
| `Book-Rating` | 1149780 | int64 | Nilai rating yang diberikan (rentang 0–10, di mana 0 berarti tidak ada rating eksplisit). |

Berikut adalah kondisi data pada file Ratings.csv:
- Kolom `Book-Rating` berisi rating yang sebagian besar bernilai 0, yang dianggap sebagai feedback implisit.
- Pada proyek ini, hanya rating eksplisit (nilai 1–10) yang digunakan. Rating dengan nilai 0 tidak dilibatkan dalam pelatihan model agar sistem fokus pada interaksi yang mencerminkan kepuasan nyata dari pengguna terhadap buku.

c. Users.csv
Dataset ini berisi data demografis pengguna.

| Kolom | Jumlah Baris | Tipe Data | Deskripsi |
|-------|--------------|-----------|-----------|
| `User-ID` | 278858 | int64 | ID unik pengguna. |
| `Location` | 278858 | object | Lokasi pengguna dalam format "kota, negara bagian, negara". |
| `Age` | 168096 | float64 | Usia pengguna. |

Berikut adalah kondisi data pada file Users.csv:
- Kolom `Age` berisi nilai yang tidak valid seperti usia di bawah 5 tahun atau di atas 100 tahun.
- Nilai usia yang tidak berada dalam rentang 5–100 dihapus untuk menjaga kualitas data.

### Exploratory Data Analysis (EDA)

Dataset **ratings** digabungkan dengan data **books** menggunakan kolom `ISBN`, lalu dilanjutkan dengan penggabungan ke data **users** menggunakan `User-ID`. Proses ini bertujuan agar seluruh informasi pengguna, buku, dan rating dapat dianalisis secara menyeluruh dalam satu tabel yang terintegrasi. Dilakukan juga pemilihan kolom-kolom yang relevan untuk kebutuhan dua jenis model yang akan dibangun. Kolom seperti `Book-Title`, `Book-Author`, `Publisher`, `Year-Of-Publication`, `User-ID`, `Book-Rating`, `Location`, dan `Age` dipertahankan karena memiliki nilai informasi penting dalam membangun sistem rekomendasi.

EDA dilanjutkan dengan menganalisis karakteristik data seperti jumlah total interaksi (baris data), jumlah buku unik, penulis unik, pengguna unik, serta variasi lokasi pengguna. Rentang tahun terbit buku dan jumlah data dengan tahun tidak valid juga dievaluasi. Didapat hasilnya sebagai berikut:

```
Jumlah total interaksi (baris data): 1149780
Jumlah buku unik: 241071
Jumlah pengguna unik: 105283
Jumlah penulis unik: 101587
Jumlah lokasi pengguna unik: 26111
Rentang tahun terbit buku: 0 - 2050
Jumlah data dengan tahun terbit tidak valid: 118648
``` 


Pemeriksaan jumlah nilai kosong per kolom dilakukan untuk mengidentifikasi potensi perbaikan data. Didapat hasil seperti berikut:

| Kolom              | Missing Values |
|--------------------|----------------|
| ISBN               | 0              |
| Book-Title         | 118644         |
| Book-Author        | 118646         |
| Year-Of-Publication| 118648         |
| Publisher          | 118646         |
| User-ID            | 0              |
| Book-Rating        | 0              |
| Location           | 0              |
| Age                | 309492         |

Kolom **Book-Title**, **Book-Author**, **Publisher**, dan **Year-Of-Publication** memiliki lebih dari 118.000 missing values, menunjukkan banyak data buku yang tidak lengkap. Kolom **Age** memiliki 309.492 missing values, hampir separuh data, sehingga tidak bisa diandalkan tanpa imputasi.

Berikutnya dilakukan pengecekan data duplikat dengan `book_data.duplicated().sum()`, didapat hasilnya `np.int64(0)`, yang berarti tidak ada data duplikat dalam dataset.



## Data Preparation
Pada bagian ini Anda menerapkan dan menyebutkan teknik data preparation yang dilakukan. Teknik yang digunakan pada notebook dan laporan harus berurutan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan proses data preparation yang dilakukan
- Menjelaskan alasan mengapa diperlukan tahapan data preparation tersebut.

## Modeling
Tahapan ini membahas mengenai model sisten rekomendasi yang Anda buat untuk menyelesaikan permasalahan. Sajikan top-N recommendation sebagai output.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menyajikan dua solusi rekomendasi dengan algoritma yang berbeda.
- Menjelaskan kelebihan dan kekurangan dari solusi/pendekatan yang dipilih.

## Evaluation
Pada bagian ini Anda perlu menyebutkan metrik evaluasi yang digunakan. Kemudian, jelaskan hasil proyek berdasarkan metrik evaluasi tersebut.

Ingatlah, metrik evaluasi yang digunakan harus sesuai dengan konteks data, problem statement, dan solusi yang diinginkan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan formula metrik dan bagaimana metrik tersebut bekerja.

**---Ini adalah bagian akhir laporan---**

_Catatan:_
- _Anda dapat menambahkan gambar, kode, atau tabel ke dalam laporan jika diperlukan. Temukan caranya pada contoh dokumen markdown di situs editor [Dillinger](https://dillinger.io/), [Github Guides: Mastering markdown](https://guides.github.com/features/mastering-markdown/), atau sumber lain di internet. Semangat!_
- Jika terdapat penjelasan yang harus menyertakan code snippet, tuliskan dengan sewajarnya. Tidak perlu menuliskan keseluruhan kode project, cukup bagian yang ingin dijelaskan saja.
