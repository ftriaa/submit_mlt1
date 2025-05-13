# Laporan Proyek Machine Learning - Fitria Anggraini

## Project Overview

Perkembangan teknologi digital telah membawa perubahan signifikan dalam dunia literasi dan industri buku. Akses terhadap informasi dan koleksi buku kini semakin mudah melalui platform digital, sehingga jumlah pilihan yang tersedia bagi pembaca meningkat secara drastis. Namun, melimpahnya pilihan ini menimbulkan tantangan berupa information overload, yaitu kondisi di mana pembaca kesulitan menyaring informasi atau buku yang sesuai. Menurut Hariyati & Heriyanto (2021), fenomena ini turut mengubah perilaku pencarian informasi secara luas dan tidak hanya memengaruhi masyarakat umum, tetapi juga ekosistem literasi secara keseluruhan.

Permasalahan information overload ini dirasakan oleh berbagai pihak dalam ekosistem literasi digital. Bagi pembaca, banyaknya pilihan sering kali menyulitkan mereka dalam menemukan buku yang sesuai dengan minat, kebutuhan, atau preferensi pribadi. Proses pencarian buku yang relevan secara manual menjadi tidak efisien, membingungkan, dan dapat menurunkan minat membaca, terutama bagi pengguna yang tidak memiliki referensi khusus atau waktu yang cukup untuk menelusuri seluruh katalog buku (Alkaff dkk., 2020). Di sisi lain, pengelola perpustakaan, penulis, dan penerbit juga menghadapi tantangan dalam memastikan karya mereka dapat menjangkau audiens yang tepat dan meningkatkan sirkulasi buku (Tian dkk., 2019). Salah satu pihak yang terdampak secara langsung oleh ledakan informasi ini adalah pustakawan. Tuntutan peran pustakawan kini semakin kompleks, tidak hanya sebagai pengelola koleksi, tetapi juga sebagai fasilitator dalam membantu pengguna menemukan informasi valid di tengah banjir data (Hariyati & Heriyanto, 2021).

Sistem rekomendasi buku hadir sebagai solusi untuk mengatasi tantangan tersebut. Sistem ini memanfaatkan data perilaku pengguna (riwayat baca, rating, pencarian) dan fitur intrinsik buku (genre, sinopsis, penulis) untuk menyajikan rekomendasi yang dipersonalisasi, sehingga meningkatkan relevansi dan kepuasan pengguna (Kim dkk., 2023). Dua pendekatan utama dalam sistem rekomendasi adalah content-based filtering dan collaborative filtering. Content-based filtering merekomendasikan buku berdasarkan kemiripan atribut konten, sedangkan collaborative filtering mengandalkan pola interaksi antar pengguna untuk mengidentifikasi kelompok pengguna dengan preferensi serupa (Az Zayyad, 2021; Kim dkk., 2023).

Implementasi sistem rekomendasi buku terbukti dapat meningkatkan kepuasan pengguna, mempercepat proses pencarian buku, serta mendukung pengelolaan katalog perpustakaan atau toko buku digital secara lebih efisien. Selain itu, sistem ini juga memberikan peluang baru bagi penulis dan penerbit untuk menjangkau pembaca yang lebih luas dan meningkatkan penjualan atau sirkulasi buku (Abidatillah, 2024).

Berdasarkan urgensi dan manfaat yang ditawarkan, pengembangan sistem rekomendasi buku menjadi salah satu solusi inovatif yang sangat dibutuhkan dalam ekosistem literasi digital saat ini, baik untuk mendukung kebutuhan pembaca, pengelola perpustakaan, maupun pelaku industri buku secara keseluruhan.

## Business Understanding

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

Dataset yang digunakan dalam proyek ini adalah Book Recommendation Dataset yang tersedia secara publik melalui platform <a href="https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset/data?select=Ratings.csv">Kaggle</a>. Dataset ini terdiri dari tiga file utama, yaitu `Books.csv`, `Users.csv`, dan `Ratings.csv` Ketiga file tersebut saling terhubung melalui kolom kunci seperti `ISBN` dan `User-ID`. Berikut adalah penjelasan rinci masing-masing file beserta kondisi datanya.

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
- Kolom `Year-Of-Publication` awalnya bertipe tidak konsisten, seperti angka dan string, sehingga dikonversi ke numerik dan difilter hanya tahun antara 1950–2025.

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

Sebelum masuk ke analisis pola interaksi pengguna terhadap buku, langkah awal yang dilakukan adalah menggabungkan dataset **ratings**, **books**, dan **users** berdasarkan kolom `ISBN` dan `User-ID`. Proses ini bertujuan agar seluruh informasi pengguna, buku, dan rating dapat dianalisis secara menyeluruh dalam satu tabel yang terintegrasi. Dilakukan juga pemilihan kolom-kolom yang relevan untuk kebutuhan dua jenis model yang akan dibangun. Kolom seperti `Book-Title`, `Book-Author`, `Publisher`, `Year-Of-Publication`, `User-ID`, `Book-Rating`, `Location`, dan `Age` dipertahankan karena memiliki nilai informasi penting dalam membangun sistem rekomendasi.

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

Setelah data digabungkan, diperoleh total 1.149.780 interaksi (baris data) yang merepresentasikan aktivitas pengguna dalam memberikan rating terhadap buku.
Dataset ini mencakup 241.071 buku unik, 105.283 pengguna unik, serta 101.587 penulis unik, menunjukkan cakupan koleksi dan partisipasi yang cukup luas.
Dari sisi lokasi, terdapat 26.111 lokasi pengguna unik, meski nantinya akan terlihat bahwa dominasi interaksi berasal dari wilayah tertentu.

Namun, data juga menunjukkan adanya rentang tahun terbit yang ekstrem, dari 0 hingga 2050, yang tentu saja mengindikasikan adanya entri tidak valid. Sebanyak 118.648 data teridentifikasi memiliki tahun terbit tidak valid, sehingga perlu diperhatikan dalam proses pembersihan atau imputasi data.

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

**Catatan**: Kolom **Book-Title**, **Book-Author**, **Publisher**, dan **Year-Of-Publication** memiliki lebih dari 118.000 missing values, menunjukkan banyak data buku yang tidak lengkap. Kolom **Age** memiliki 309.492 missing values, hampir separuh data, sehingga tidak bisa diandalkan tanpa imputasi.

Berikutnya dilakukan pengecekan data duplikat dengan `book_data.duplicated().sum()`, didapat hasilnya `np.int64(0)`, yang berarti tidak ada data duplikat dalam dataset.

Dengan pemahaman awal terkait struktur dan kualitas data, kini analisis dilanjutkan dengan eksplorasi visual untuk menggali lebih dalam pola distribusi data, karakteristik pengguna, dan interaksi mereka terhadap buku. Tahap ini penting untuk mengidentifikasi potensi bias, persebaran data, serta preferensi pengguna yang nantinya akan mempengaruhi hasil pemodelan sistem rekomendasi.

1. Distribusi Tahun Terbit, Nilai Rating, dan Usia Pengguna

![distribusi](https://raw.githubusercontent.com/ftriaa/submit_mlt1/main/asset/distribusi.png)

- Mayoritas buku dalam dataset diterbitkan antara tahun 1995 hingga 2005, dengan puncak sekitar tahun 2002–2004. Hal ini menunjukkan bahwa dataset lebih merepresentasikan buku-buku modern pada masa tersebut, dan mungkin kurang representatif untuk buku klasik atau terbitan terbaru.
- Sebagian besar pengguna memberikan rating tinggi, terutama pada nilai 7 - 10. Ini menunjukkan bahwa pengguna cenderung memberikan penilaian positif terhadap buku yang mereka baca, atau mungkin hanya memberikan rating pada buku yang mereka sukai. Pola ini bisa menimbulkan bias positif dalam evaluasi kualitas buku.
- Distribusi umur menunjukkan bahwa sebagian besar pengguna berada di rentang usia 20 hingga 40 tahun. Artinya, sistem ini paling banyak digunakan oleh generasi muda dan dewasa awal, yang kemungkinan besar memiliki preferensi tertentu terhadap genre atau jenis buku, sehingga dapat mempengaruhi pola rating dan konsumsi buku.
- Karena rating buku bersifat diskrit dan cenderung berat ke nilai tinggi, pemodelan sistem rekomendasi sebaiknya mempertimbangkan distribusi ini agar tidak terlalu bias terhadap buku-buku populer dengan rating tinggi.

2. Jumlah Buku yang Diterbitkan per Tahun

![jmlbukudithn](https://raw.githubusercontent.com/ftriaa/submit_mlt1/main/asset/jmlbukudithn.png)

Jumlah buku yang diterbitkan meningkat secara signifikan sejak tahun 1970-an, dengan puncaknya terjadi pada akhir 1990-an hingga awal 2000-an. Tahun 2002 mencatat jumlah buku terbanyak yang diterbitkan. Setelah itu, terjadi penurunan tajam dalam jumlah buku yang terdaftar. Ini menunjukkan adanya lonjakan besar dalam publikasi buku selama dekade terakhir abad ke-20 dan awal abad ke-21.

3. Jumlah Rating Buku Berdasarkan Kelompok Usia

![jmlnilaibuku](https://raw.githubusercontent.com/ftriaa/submit_mlt1/main/asset/jmlnilaibuku.png)

Kelompok usia 25–34 merupakan yang paling aktif dalam memberikan penilaian terhadap buku, dengan jumlah data tertinggi dibandingkan kelompok usia lainnya. Disusul oleh kelompok usia 35–44 dan 45–54. Aktivitas penilaian menurun signifikan pada kelompok usia di atas 55 tahun dan yang berusia di bawah 18 tahun, yang menunjukkan bahwa minat atau partisipasi dalam menilai buku lebih besar pada kalangan dewasa muda hingga paruh baya.

4. Buku dengan Rating Terbanyak

![topbuku](https://raw.githubusercontent.com/ftriaa/submit_mlt1/main/asset/topbuku.png)

Buku "Wild Animus" adalah buku yang paling banyak diberi rating, diikuti oleh "The Lovely Bones: A Novel" dan "The Da Vinci Code". Kesepuluh buku ini tampaknya memiliki tingkat popularitas tinggi di kalangan pengguna karena mendapatkan volume rating yang jauh lebih banyak dibandingkan buku lainnya dalam dataset.

5. Penulis dengan Jumlah Rating Terbanyak

![toppenulis](https://raw.githubusercontent.com/ftriaa/submit_mlt1/main/asset/toppenulis.png)

Grafik menunjukkan bahwa Stephen King adalah penulis dengan jumlah rating terbanyak dalam dataset, diikuti oleh Nora Roberts, John Grisham, dan James Patterson. Sepuluh penulis yang ditampilkan memiliki jumlah penilaian yang signifikan dibandingkan penulis lain, yang menunjukkan bahwa buku-buku mereka banyak dinilai oleh pengguna. Hal ini mengindikasikan bahwa penulis-penulis ini termasuk yang paling sering muncul dalam data rating buku pengguna.

6. Penerbit dengan Jumlah Rating Terbanyak

![toppenerbit](https://raw.githubusercontent.com/ftriaa/submit_mlt1/main/asset/toppenerbit.png)

Berdasarkan grafik, Ballantine Books merupakan penerbit dengan jumlah rating terbanyak dalam dataset, disusul oleh Pocket dan Berkley Publishing Group. Sepuluh penerbit yang ditampilkan memiliki jumlah buku yang dinilai paling tinggi dibandingkan penerbit lainnya.

7. Distribusi Lokasi Pengguna

![topkotanegara](https://raw.githubusercontent.com/ftriaa/submit_mlt1/main/asset/topkotanegara.png)

Berdasarkan distribusi lokasi pengguna, kota dengan jumlah pengguna terbanyak adalah Morrow, USA, diikuti oleh Toronto dan Ottawa dari Kanada. Kota-kota lain yang mendominasi sebagian besar data berasal dari wilayah Amerika Serikat dan Kanada, dengan beberapa pengecualian seperti Barcelona, Spanyol dan London, United Kingdom. Ini menunjukkan bahwa interaksi terhadap buku dalam dataset ini banyak berasal dari kota-kota besar di wilayah berbahasa Inggris, khususnya di Amerika Utara.

Dari sisi negara, data menunjukkan bahwa Amerika Serikat (USA) memiliki jumlah pengguna terbanyak secara signifikan, disusul oleh Kanada dan United Kingdom. Negara-negara lainnya seperti Jerman, Spanyol, dan Australia memiliki kontribusi yang jauh lebih kecil. Temuan ini menunjukkan bahwa mayoritas data berasal dari negara-negara berbahasa Inggris, terutama dari USA dan Kanada, yang mendominasi interaksi terhadap buku dalam dataset ini.

## Data Preparation

### 1. Konversi `Year-Of-Publication` ke Numerik dan Filter Rentang Tahun Valid

Kolom `Year-Of-Publication` diubah menjadi tipe data numerik menggunakan fungsi `pd.to_numeric()`, dengan parameter `errors='coerce'` untuk menangani nilai yang tidak valid. Setelah itu, dilakukan penyaringan agar hanya mencakup tahun terbit yang berada dalam rentang 1950 hingga 2025. Langkah ini diperlukan karena rentang tahun yang tidak valid, seperti tahun 0 atau tahun yang jauh di luar kisaran yang wajar, dapat mengganggu analisis dan prediksi.

### 2. Konversi `Age` ke Numerik dan Filter Usia Pengguna

Kolom `Age` juga diubah menjadi tipe data numerik dengan cara yang sama. Penyaringan dilakukan agar hanya mencakup usia pengguna antara 5 hingga 100 tahun. Usia yang berada di luar rentang ini, seperti usia negatif atau lebih dari 100, tidak relevan dan dapat mengganggu analisis lebih lanjut.

### 3. Drop Nilai Kosong pada Kolom Penting

Kolom-kolom penting seperti `Book-Title`, `Book-Author`, `Publisher`, dan `Year-Of-Publication` diperiksa untuk missing values dan dibuang jika ada nilai yang hilang. Jika kolom `Image-URL-M` ada dalam dataset, kolom tersebut juga dipertimbangkan. Kolom-kolom ini mengandung informasi esensial dalam membangun rekomendasi buku, sehingga nilai yang hilang dapat menurunkan kualitas prediksi model.

### 4. Hapus Rating yang Tidak Valid (Rating = 0)

Data dengan rating 0 dianggap sebagai rating non-eksplisit dan dibuang. Rating 0 tidak memberikan informasi yang valid mengenai preferensi pengguna terhadap buku dan dapat merusak kualitas model rekomendasi.

### 5. Reset Indeks

Setelah penghapusan data, indeks DataFrame di-reset untuk mencegah kesalahan dalam indexing dan memastikan dataset terorganisir dengan baik untuk analisis lebih lanjut. Ini adalah langkah pembersihan umum yang menjaga struktur dataset tetap bersih dan mudah dikelola. Lalu tampilkan hasilnya dengan kode berikut

```python
print("Data bersih siap digunakan:", book_data_clean.shape)
```

### 6. Filter Pengguna dengan Minimal 3 Rating
Langkah ini memfilter pengguna yang memberikan setidaknya 3 rating. Pengguna dengan kurang dari 3 rating dihapus untuk memastikan bahwa data pelatihan hanya mencakup pengguna yang memberikan cukup banyak feedback, karena pengguna dengan sedikit rating tidak memberikan cukup informasi untuk membangun model rekomendasi yang akurat, sehingga dihapus agar tidak mengganggu kualitas model.

### 7. Pilih Pengguna Acak Sampai Jumlah Rating Mencapai 30.000
Setelah memfilter pengguna, langkah berikutnya adalah memilih pengguna secara acak sehingga jumlah total rating mendekati 30.000. Pengguna yang dipilih memiliki setidaknya 3 rating. Pembatasan ini bertujuan untuk membuat dataset lebih terkelola dan efisien, serta memungkinkan eksperimen yang lebih cepat dengan data yang representatif. Pembatasan jumlah rating ini juga dipilih untuk menghindari masalah performa di Google Colab, yang memiliki keterbatasan sumber daya komputasi, sehingga pengolahan data dalam skala lebih kecil dapat berjalan lebih lancar tanpa terhenti secara tiba-tiba akibat overloading atau keterbatasan memori.

### 8. Persiapan Fitur Content-Based
Pada langkah ini, fitur content-based dipersiapkan dengan mengubah nilai pada kolom `Book-Title`, `Book-Author`, dan `Publisher` menjadi string kecil (lowercase) dan menghapus spasi tambahan. Kemudian, fitur baru `content_features` digabungkan dari kolom-kolom tersebut. Fitur ini akan digunakan untuk model content-based, yang mengandalkan informasi buku untuk memberikan rekomendasi, seperti judul buku, penulis, penerbit, tahun terbit, dan rating.

### 9. TF-IDF Vectorization

**TF-IDF (Term Frequency - Inverse Document Frequency)** adalah metode pembobotan kata dalam teks yang mengukur pentingnya sebuah kata terhadap dokumen tertentu dalam kumpulan dokumen. TF-IDF bertujuan untuk memberikan bobot lebih tinggi pada kata-kata yang sering muncul dalam dokumen tertentu namun jarang muncul di dokumen lain, sehingga kata tersebut dianggap lebih representatif bagi isi dokumen tersebut (Zhan, 2024).

**Rumus Matematis**

a. **Term Frequency (TF)**
Mengukur seberapa sering kata *t* muncul dalam dokumen *d* relatif terhadap total kata di dokumen tersebut.

```math
TF(t, d) = \frac{f(t, d)}{\sum_{k}f(k, d)}
```

* $f(t, d)$ = jumlah kemunculan kata *t* dalam dokumen *d*.
* $\sum_{k}f(k, d)$ = total jumlah kata dalam dokumen *d*.

b. **Inverse Document Frequency (IDF)**
   Mengukur seberapa jarang kata *t* muncul di seluruh dokumen dalam korpus.

```math
IDF(t) = \log{\left( \frac{N}{1 + n_t} \right)}
```

* $N$ = total jumlah dokumen.
* $n_t$ = jumlah dokumen yang mengandung kata *t*.
* Penambahan 1 pada penyebut digunakan untuk smoothing agar tidak terjadi pembagian dengan nol.

c. **TF-IDF Weighting**

```math
TF\text{-}IDF(t, d) = TF(t, d) \times IDF(t)
```

Dalam implementasi proyek ini, teknik TF-IDF Vectorizer digunakan untuk mengubah fitur teks menjadi representasi numerik yang dapat dihitung kemiripannya. Pada kode, digunakan `TfidfVectorizer` dari library scikit-learn dengan beberapa parameter khusus. Parameter `stop_words='english'` berfungsi untuk menghapus kata-kata umum (stopwords) dalam bahasa Inggris, sehingga hanya kata-kata yang dianggap penting dan bermakna yang akan diproses. Selanjutnya, `max_features=10000` digunakan untuk membatasi jumlah fitur maksimal hingga 10.000 kata atau frasa teratas berdasarkan frekuensi kemunculannya di korpus, agar dimensi vektor tidak terlalu besar dan tetap efisien dalam perhitungan. Parameter `ngram_range=(1, 2)` dipilih agar vektor tidak hanya mempertimbangkan kata tunggal (unigram), tetapi juga pasangan kata berurutan (bigram) yang seringkali lebih bermakna dalam konteks buku, misalnya frasa “Harry Potter” lebih informatif dibandingkan hanya kata “Harry” atau “Potter” saja. Hasil dari proses ini adalah sebuah matriks TF-IDF yang merepresentasikan kemiripan antar buku berdasarkan konten fitur yang telah diekstraksi.

TF-IDF dipilih dalam proyek ini karena sederhana namun efektif untuk merepresentasikan teks, terutama dalam konteks Content-Based Filtering. Metode ini memungkinkan sistem merekomendasikan buku berdasarkan kemiripan informasi konten seperti judul, penulis, dan penerbit, tanpa bergantung pada data interaksi pengguna. TF-IDF cocok digunakan saat fokusnya adalah kesamaan karakteristik deskriptif antar buku. Meski demikian, metode ini tidak memperhitungkan konteks kata dan kurang optimal pada teks dengan distribusi kata yang tidak merata.

### 10. Persiapan Fitur Collaborative Filtering
Untuk model collaborative filtering, kolom `User-ID` dan `Book-Title` dienkode menggunakan `LabelEncoder` untuk mengubah ID pengguna dan ID buku menjadi bentuk numerik. Rating kemudian dinormalisasi ke dalam skala 0–1 agar dapat digunakan oleh algoritma collaborative filtering. Label encoding diperlukan untuk mengonversi data kategorikal menjadi format numerik yang dapat diterima oleh model, dan normalisasi rating memastikan konsistensi dalam skala data yang digunakan.

### 11. Pembagian Data untuk Training dan Testing
Setelah data siap, dataset dibagi menjadi training dan testing set dengan perbandingan 80:20 menggunakan `train_test_split`. Pembagian ini penting untuk mengevaluasi performa model secara objektif, dengan menguji model pada data yang belum pernah dilihat sebelumnya (testing set), sementara model dilatih pada data yang lebih besar (training set). Lalu tampilkan hasilnya dengan kode berikut

```python
print("Train size:", len(train_data))
print("Test size:", len(test_data))
```

## Modeling

### Content-Based Filtering (TF-IDF + KNN)

Dalam proyek sistem rekomendasi buku berbasis konten ini, algoritma **K-Nearest Neighbors (KNN)** digunakan untuk menemukan buku-buku yang paling mirip berdasarkan hasil representasi vektor dari deskripsi buku menggunakan TF-IDF. Secara umum, KNN adalah algoritma berbasis *instance-based learning* yang bekerja dengan mencari **k tetangga terdekat** dari data kueri dalam ruang fitur, berdasarkan jarak atau kemiripan (Trstenjak dkk., 2014).

Kelebihan KNN dalam sistem rekomendasi buku adalah kesederhanaannya serta kemampuannya untuk memberikan hasil rekomendasi yang interpretatif, karena hanya bergantung pada kedekatan antar item dalam ruang vektor. Model ini juga tidak memerlukan proses pelatihan yang rumit dan langsung bisa diterapkan pada data baru tanpa perlu retraining. Selain itu, KNN dengan cosine similarity sangat efektif untuk data teks hasil representasi TF-IDF, karena mampu mengukur kesamaan pola distribusi kata antar dokumen. Namun, KNN memiliki beberapa kelemahan. Salah satunya adalah **waktu komputasi yang tinggi** untuk dataset besar, karena setiap prediksi memerlukan pencarian tetangga satu per satu di seluruh data. Selain itu, KNN cukup **sensitif terhadap fitur yang tidak relevan** atau **noise**, sehingga kualitas preprocessing data seperti pembersihan dan normalisasi menjadi krusial (Jyothiprakash dkk., 2023).

Dalam konteks proyek ini, jarak antar buku dihitung menggunakan **cosine distance**, yang secara matematis dirumuskan sebagai:

```math
\cos(\theta) = \frac{A \cdot B}{\|A\| \times \|B\|}
```

di mana \$A\$ dan \$B\$ adalah vektor representasi dua buku. Semakin kecil sudut antar vektor, semakin besar nilai cosine similarity-nya, yang menunjukkan tingkat kemiripan konten yang lebih tinggi.

Pada implementasinya, dilakukan proses tuning untuk mencari **nilai k (n\_neighbors)** terbaik dengan mencoba beberapa opsi: 5, 10, 20, 30, dan 40. Untuk setiap nilai k, model KNN dibuat menggunakan kelas `NearestNeighbors` dengan parameter `metric='cosine'` agar perhitungan jarak berbasis cosine distance. Berikut cuplikan kode utamanya:

```python
knn = NearestNeighbors(n_neighbors=n_neighbors, algorithm='auto', metric='cosine')
knn.fit(tfidf_matrix)
distances, indices = knn.kneighbors(tfidf_matrix)
score = distances.mean()
```

Di sini, model KNN dilatih menggunakan matriks TF-IDF (`knn.fit(tfidf_matrix)`), lalu digunakan untuk mencari tetangga terdekat dari setiap buku di dalam dataset menggunakan fungsi `kneighbors`. Rata-rata jarak antar buku dihitung (`distances.mean()`) untuk menilai seberapa baik hasil rekomendasi yang diberikan oleh konfigurasi k tersebut. Nilai k dengan rata-rata jarak terendah disimpan sebagai model KNN terbaik.

Sebelum proses rekomendasi dijalankan, perlu dipastikan bahwa input kueri dari pengguna sesuai dengan data judul buku yang tersedia. Di sinilah digunakan teknik **fuzzy string matching**, yang mampu mencocokkan teks meskipun terdapat perbedaan penulisan kecil. Dalam proyek ini, digunakan fungsi `partial_ratio` dari library `fuzzywuzzy` untuk mencocokkan kueri pengguna dengan kolom 'Book-Title' dalam dataset (Ardan dkk., 2023; N dkk., 2023). Berikut ini implementasinya:

```python
query_idx = cb_data[cb_data['Book-Title'].apply(lambda x: fuzz.partial_ratio(x.lower(), query.lower()) > 90)].index
```

Pada baris ini, pengguna akan memasukkan kueri judul buku. Kemudian, fungsi `fuzz.partial_ratio` diterapkan ke setiap judul buku dalam dataset, lalu dibandingkan dengan kueri yang diberikan. Hanya judul buku dengan tingkat kemiripan di atas 90% yang akan dianggap cocok dan diambil indeksnya. Ambang batas 90 ini dipilih agar sistem tetap toleran terhadap kesalahan minor namun cukup ketat untuk menghindari pencocokan yang tidak relevan.

Setelah buku referensi ditemukan melalui fuzzy matching, KNN digunakan untuk mencari tetangga terdekat di ruang vektor TF-IDF. Proses ini dilakukan dengan memanggil fungsi `kneighbors` untuk mencari jarak antar buku.

```python
dists, indices = best_knn.kneighbors(tfidf_matrix[query_idx[0]], n_neighbors=top_n + 1)
scores = 1 - dists[0][1:]
```

Hasilnya berupa daftar indeks buku yang paling mirip, dengan skor similarity yang dihitung dari 1 - cosine distance. Data buku yang direkomendasikan kemudian ditampilkan lengkap dengan informasi penulis, penerbit, tahun terbit, serta skor kemiripannya. Lalu digunakan fungsi `recommend_cbf("PLEADING GUILTY", top_n=10)` untuk mencari 10 buku yang paling mirip dengan judul "PLEADING GUILTY" berdasarkan fitur konten. Hasilnya dapat dilihat seperti berikut:

```
Rekomendasi berdasarkan: 'PLEADING GUILTY':

                                          Book-Title           Book-Author  \
0                                    pleading guilty           scott turow   
1                                    burden of proof           scott turow   
2                         reversible errors: a novel           scott turow   
3  ultimate punishment : a lawyer's reflections o...           scott turow   
4                         that mighty sculptor, time  marguerite yourcenar   
5                                          the fixer       bernard malamud   
6                                         the pickup       nadine gordimer   
7                                         the pickup       nadine gordimer   

                   Publisher  Year-Of-Publication  Similarity  
0       farrar straus giroux               1993.0    0.969040  
1       farrar straus giroux               1990.0    0.761120  
2       farrar straus giroux               2002.0    0.710307  
3  farrar, straus and giroux               2003.0    0.650364  
4       farrar straus giroux               1993.0    0.610967  
5  farrar, straus and giroux               2004.0    0.505764  
6       farrar straus giroux               2001.0    0.496714  
7  farrar, straus and giroux               2001.0    0.496714
```

Didapat bahwa buku pertama yang direkomendasikan, *"Pleading Guilty"* oleh Scott Turow, memiliki tingkat kesamaan sebesar 0.969040, yang menunjukkan bahwa ini adalah buku yang paling mirip dengan kueri pengguna.

Selain itu, daftar rekomendasi mencakup beberapa buku dengan penulis yang sama (Scott Turow), yang menunjukkan bahwa algoritma ini tidak hanya mencari kesamaan berdasarkan judul, tetapi juga mengidentifikasi hubungan antar buku yang memiliki penulis atau tema serupa. Proses ini memungkinkan pengguna untuk menemukan buku yang relevan berdasarkan deskripsi atau metadata lain, seperti penulis atau penerbit, meskipun mereka tidak memiliki interaksi langsung dengan pengguna lain (seperti rating atau review).

### Collaborative Filtering (Neural Collaborative Filtering - NCF)

Metode kedua yang digunakan dalam sistem rekomendasi ini adalah **Collaborative Filtering (CF)** berbasis **Neural Collaborative Filtering (NCF)**. Berbeda dengan Content-Based Filtering yang mengandalkan fitur konten buku, CF memanfaatkan pola interaksi antar pengguna dan item, seperti data rating atau riwayat konsumsi. Tujuan utama NCF adalah memprediksi sejauh mana seorang pengguna akan menyukai sebuah item berdasarkan pola preferensi pengguna lain yang serupa.

Kelebihan **NCF** dalam sistem rekomendasi buku adalah kemampuannya untuk menemukan pola preferensi pengguna tanpa memerlukan detail fitur buku. NCF juga dapat menangkap hubungan implisit antar item dan pengguna yang tidak terlihat langsung dalam data konten, serta unggul dalam mempelajari interaksi kompleks melalui neural network. Namun, metode ini memiliki beberapa kelemahan, salah satunya adalah **cold start problem** yang sulit memberikan rekomendasi untuk pengguna atau item baru. Selain itu, NCF memerlukan data interaksi yang cukup agar pola preferensi dapat terbentuk dengan baik dan membutuhkan sumber daya komputasi lebih dibandingkan dengan algoritma yang lebih sederhana (Ayu dkk., 2025).

Pada proyek ini, **NCF** memetakan pengguna dan buku ke dalam representasi vektor menggunakan **embedding layer** yang lebih kompleks daripada CF klasik (Ayyiyah dkk., 2023). Proses prediksi dilakukan dengan menghitung produk titik (dot product) antar vektor pengguna dan buku, serta menambahkan bias pengguna dan bias buku. Rumus matematis untuk prediksi adalah sebagai berikut:

```math
\text{output} = \sigma \left( \langle \mathbf{u}, \mathbf{v} \rangle + b_u + b_v \right)
```

di mana:

* \$\mathbf{u}\$ adalah vektor embedding pengguna,
* \$\mathbf{v}\$ adalah vektor embedding buku,
* \$b\_u\$ dan \$b\_v\$ adalah bias pengguna dan buku,
* \$\sigma\$ adalah fungsi aktivasi sigmoid.

Model diimplementasikan menggunakan **TensorFlow/Keras** dengan arsitektur yang terdiri dari empat **embedding layers**, yaitu **User Vector**, **User Bias**, **Book Vector**, dan **Book Bias**. Masing-masing layer ini memiliki parameter yang dipelajari untuk memetakan pengguna dan buku ke dalam ruang vektor.

| Layer (Type)                   | Output Shape | Jumlah Parameter | Keterangan                                    |
| ------------------------------ | ------------ | ---------------- | --------------------------------------------- |
| **Embedding (User Vector)**    | (1, 50)      | 87.300           | Vektor angka 50 untuk menggambarkan pengguna. |
| **Embedding\_1 (User Bias)**   | (1, 1)       | 1.746            | Menyimpan kecenderungan rating pengguna.      |
| **Embedding\_2 (Book Vector)** | (1, 50)      | 1.061.150        | Vektor angka 50 untuk menggambarkan buku.     |
| **Embedding\_3 (Book Bias)**   | (1, 1)       | 21.223           | Menyimpan kecenderungan rating buku.          |
| **Total Parameter**            |              | **1.171.419**    | Total parameter yang dipelajari oleh model.   |

Total parameter model ini adalah **1,171,419** dan semuanya dapat dilatih.

Setelah membangun model, proses pelatihan dimulai dengan menggunakan pasangan user-book dan rating yang diberikan. Data diubah menjadi angka dan diproses melalui model untuk mempelajari hubungan antar pengguna dan buku. Fungsi loss yang digunakan adalah **Binary Crossentropy**, karena rating sudah diskalakan dalam rentang \[0, 1]. Optimasi dilakukan dengan **Adam Optimizer** dengan learning rate 0.0001, dan metrik yang digunakan untuk evaluasi adalah **Root Mean Squared Error (RMSE)**.

Pelatihan model dilakukan selama **50 epoch** dengan ukuran batch **64**. Setelah model selesai dilatih, validasi dilakukan menggunakan data uji untuk memastikan kemampuan generalisasi model. Untuk menghasilkan rekomendasi, fungsi `recommend_ncf(278188, top_n=10)` digunakan untuk mendapatkan **10 rekomendasi buku teratas** untuk pengguna dengan ID **278188**.

Berikut adalah hasil rekomendasi untuk pengguna tersebut:

```
Rekomendasi NCF untuk User ID 278188:

• One True Thing (Anna Quindlen)
• Grave Concerns (Mira) (Gwen Hunter)
• Kiss River (Mira) (Diane Chamberlain)
• The Delaney Woman (Jeanette Baker)
• Dreammaker (Harlequin Historicals, No. 486) (Judith Stacy)
• Baby-Sitters Island Adventure (Baby-Sitters Club Super Special, 4) (Ann M. Martin)
• Mary Anne's Bad-Luck Mystery (Baby-Sitters Club (Paperback)) (Ann M. Martin)
• Absolute Power (David Baldacci)
• Dawn on the Coast (Baby-Sitters Club (Paperback)) (Ann M. Martin)
• Karen's Goodbye (Baby-Sitters Little Sister, 19) (Ann M. Martin)
```

Sistem rekomendasi NCF untuk pengguna 278188 didominasi oleh beberapa judul dari penulis yang sama, seperti Ann M. Martin, dengan beberapa buku dari seri Baby-Sitters Club. Rekomendasi ini menunjukkan buku-buku yang diprediksi sesuai dengan preferensi pengguna berdasarkan pola interaksi pengguna sebelumnya dengan buku-buku yang telah diberi rating.

## Evaluation

### Evaluasi Content-Based Filtering (CBF)

Pada sistem rekomendasi berbasis konten (Content-Based Filtering), evaluasi dilakukan menggunakan metrik Precision@k, Recall@k, dan F1-Score@k. Metrik ini dipilih karena tujuan utama dari Content-Based Filtering adalah memberikan daftar rekomendasi item yang relevan bagi pengguna secara spesifik, berdasarkan kemiripan konten dengan preferensi sebelumnya. Berikut adalah formula yang digunakan:

**Precision\@k** mengukur proporsi item relevan di antara k item teratas yang direkomendasikan oleh sistem. 

```math
\text{Precision@k} = \frac{|\text{RelevantItems} \cap \text{RecommendedItems}|}{k}
```

**Recall\@k** mengukur proporsi item relevan yang berhasil direkomendasikan dibandingkan dengan seluruh item relevan yang ada.

```math
\text{Recall@k} = \frac{|\text{RelevantItems} \cap \text{RecommendedItems}|}{|\text{RelevantItems}|}
```

**F1-Score\@k** menghitung harmonic mean dari Precision\@k dan Recall\@k.

```math
\text{F1-Score@k} = 2 \times \frac{\text{Precision@k} \times \text{Recall@k}}{\text{Precision@k} + \text{Recall@k}}
```

**Keterangan simbol**:

* RelevantItems: Item yang benar-benar relevan untuk pengguna.
* RecommendedItems: Item yang direkomendasikan oleh sistem.
* k: Jumlah item dalam Top-k rekomendasi.

**Berikut adalah hasil Evaluasi CBF**

```
📊 Content-Based Filtering Evaluation (Top-10)
• Precision@10: 0.265
• Recall@10: 0.454
• F1-Score@10: 0.291
```

Hasil evaluasi menunjukkan bahwa nilai Precision@10 sebesar 0.265, yang berarti sekitar 26,5% dari 10 rekomendasi teratas yang diberikan sistem terbukti relevan untuk pengguna. Nilai Recall@10 sebesar 0.454 menunjukkan bahwa sistem mampu menjangkau sekitar 45,4% dari seluruh item relevan yang ada. Sedangkan F1-Score@10 sebesar 0.291 mencerminkan keseimbangan antara kedua aspek tersebut. Hasil ini menunjukkan bahwa model sudah mampu menghasilkan rekomendasi yang cukup baik, meskipun masih ada peluang untuk meningkatkan presisi agar daftar rekomendasi semakin relevan.

### Evaluasi Collaborative Filtering (CF)

Untuk model Collaborative Filtering berbasis Neural Collaborative Filtering (NCF), evaluasi dilakukan dengan menggunakan metrik Mean Absolute Error (MAE) dan Root Mean Squared Error (RMSE). Pemilihan metrik ini didasarkan pada tujuan model NCF, yaitu untuk memprediksi nilai rating pengguna terhadap item secara numerik. Oleh karena itu, dibutuhkan metrik yang mampu mengukur seberapa dekat hasil prediksi dengan nilai rating sebenarnya. Berikut adalah formula yang digunakan:

**Mean Absolute Error (MAE)** menghitung rata-rata selisih absolut antara nilai aktual dan prediksi. Semakin kecil MAE, semakin kecil rata-rata kesalahan prediksi.

```math
\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} \left| y_i - \hat{y}_i \right|
```

**Root Mean Squared Error (RMSE)** menghitung akar dari rata-rata kuadrat selisih antara prediksi dan aktual. RMSE memberi penalti lebih besar untuk kesalahan besar karena dikuadratkan.
Berguna untuk memastikan model tidak sering meleset jauh dari nilai aktual.

```math
\text{RMSE} = \sqrt{ \frac{1}{n} \sum_{i=1}^{n} \left( y_i - \hat{y}_i \right)^2 }
```

**Keterangan simbol**:

* $n$: Jumlah sampel data uji.
* $y_i$: Nilai rating aktual pada data ke-i.
* $\hat{y}_i$: Nilai rating hasil prediksi pada data ke-i.

**Berikut adalah hasil Evaluasi CF**

```
📊 Evaluation Metrics:
MAE  : 1.7979
RMSE : 2.1916
```

Pada proyek ini, model NCF menghasilkan MAE sebesar 1.7979 dan RMSE sebesar 2.1916. Nilai MAE menunjukkan bahwa rata-rata kesalahan prediksi rating hanya sekitar 1.8 poin dari skala 0–10. Sementara itu, RMSE yang berada di angka 2.2 menunjukkan bahwa mayoritas prediksi tidak terlalu jauh meleset dari nilai sebenarnya. Nilai-nilai error yang relatif rendah ini menandakan bahwa model cukup akurat dalam mempelajari pola preferensi pengguna dan mampu memberikan prediksi rating yang layak digunakan untuk sistem rekomendasi.

## Conclusion

Proyek ini bertujuan membangun sistem rekomendasi buku untuk membantu pengguna menemukan bacaan yang relevan dengan preferensinya, menggunakan data Book Recommendation Dataset dari Kaggle. Dua pendekatan yang digunakan, yaitu Content-Based Filtering (CBF) yang memanfaatkan kemiripan konten buku, dan Neural Collaborative Filtering (NCF) yang mempelajari pola interaksi pengguna-item. Pemilihan metode ini disesuaikan dengan kebutuhan personalisasi rekomendasi dan tantangan data yang dihadapi.

Hasil evaluasi menunjukkan bahwa CBF mencapai Precision@10 sebesar 26,5% dan Recall@10 sebesar 45,4%, menandakan relevansi rekomendasi yang cukup baik namun masih bisa ditingkatkan dari sisi presisi. Sementara itu, NCF menghasilkan MAE 1.7979 dan RMSE 2.1916, menunjukkan akurasi prediksi rating yang memadai. Secara keseluruhan, kedua model telah memberikan hasil yang sesuai dengan tujuan proyek, namun perbaikan lebih lanjut seperti tuning model atau penggabungan metode (hybrid) masih terbuka sebagai peluang pengembangan.

## Referensi

Abidatillah, C. H. (2024). Implementasi Sistem Rekomendasi Buku Berbasis Konten Menggunakan Vector Space Model Dan Similarity Measure Untuk Pengelolaan Antrean Dan Sirkulasi Di Perpustakaan Berjalan [Universitas Bakrie]. https://repository.bakrie.ac.id/10529/1/00.%20Cover.pdf

Alkaff, M., Khatimi, H., & Eriady, A. (2020). Sistem Rekomendasi Buku Menggunakan Weighted Tree Similarity dan Content Based Filtering. 20(1), 193–202. https://doi.org/10.30812/matrik.v20i1.617

Ardan, I. S., Sulastri, J., & Rakhmawati, A. (2023). Analisis Performansi Entity Matching Dengan Fuzzy Wuzzy Pada Artikel Fairness Ai. Jurnal Teknoinfo, 17(2), 548–556. https://ejurnal.teknokrat.ac.id/index.php/teknoinfo/index

Ayu, P., Mukti, S., & Baizal, Z. K. A. (2025). Enhancing Neural Collaborative Filtering with Metadata for Book Recommender System. IJCCS (Indonesian Journal of Computing and Cybernetics Systems), 19(1), 61–72. https://doi.org/10.22146/ijccs.103611

Ayyiyah, N. K., Kusumaningrum, R., & Rismiyati, R. (2023). Film Recommender System Menggunakan Metode Neural Collaborative Filtering. Jurnal Teknologi Informasi dan Ilmu Komputer (JTIIK), 10(3), 699–708. https://jtiik.ub.ac.id/index.php/jtiik/article/view/6616/pdf

Az Zayyad, rizqi M. (2021). Sistem Rekomendasi Buku Menggunakan Metode Content Based Filtering [Universitas Islam Indonesia]. https://dspace.uii.ac.id/bitstream/handle/123456789/35942/17523144%20Muhammad%20Rizqi%20Az%20Zayyad.pdf?sequence=1

Hariyati, M., & Heriyanto, H. (2021). Kompetensi Pustakawan di Era Industri 4.0 dalam Menghadapi Information Overload. Daluang: Journal of Library and Information Science, 1(1), 1. https://doi.org/10.21580/daluang.v1i1.2021.8005

Jyothiprakash, P., Asif, M. T., Venkata, K., Manoj, L., Subadhra, Y. V., & Lakshmi, V. (2023). CONTENT-BASED RECOMMENDATION SYSTEM WITH K.D. TREE FOR OPEN-SOURCE PROJECTS. International Research Journal of Modernization in Engineering Technology and Science, 5(4), 882–889. https://doi.org/10.56726/IRJMETS35566

Kim, J.-Y. ;, Lim, C.-K., Kim, J.-Y., & Lim, C.-K. (2023). Feature Extracted Deep Neural Collaborative Filtering for E-Book Service Recommendations. Applied Sciences 2023, Vol. 13, Page 6833, 13(11), 6833. https://doi.org/10.3390/APP13116833

N, J. C., Ikele, M. A., Izegbu, I. O., & Olakunle, S. (2023). Content based fuzzy search recommender system based on new user preferences Background Of the study. Global Scientific Journals, 11(9), 1444–1455. www.globalscientificjournal.comwww.globalscientificjournal.com

Tian, Y., Zheng, B., Wang, Y., Zhang, Y., & Wu, Q. (2019). College Library Personalized Recommendation System Based on Hybrid Recommendation Algorithm. Procedia CIRP, 83, 490–494. https://doi.org/10.1016/J.PROCIR.2019.04.126

Trstenjak, B., Mikac, S., & Donko, D. (2014). KNN with TF-IDF Based Framework for Text Categorization. Elsevier Ltd., 69, 1356–1364. https://www.sciencedirect.com/science/article/pii/S1877705814003750?ref=pdf_download&fr=RR-2&rr=93f40c04fac74aca

Zhan, Z. (2024). Comparative Analysis of TF-IDF and Word2Vec in Sentiment Analysis: A Case of Food Reviews. https://doi.org/10.1051/itmconf/20257002013


**---Ini adalah bagian akhir laporan. Terima kasih  sudah membaca.---**
