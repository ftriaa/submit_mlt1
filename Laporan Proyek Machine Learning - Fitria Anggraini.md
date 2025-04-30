# Laporan Proyek Machine Learning - Fitria Anggraini

## Judul Proyek
**"Predictive Analytics - Peramalan Penjualan Produk Ritel Menggunakan Dataset Superstore untuk Mendukung Pengambilan Keputusan Strategis"**
## Domain Proyek

### Latar Belakang

Peramalan penjualan yang akurat sangat penting dalam bisnis ritel untuk mengoptimalkan manajemen inventaris, pengaturan pemasaran, dan perencanaan strategis. Menurut artikel di [RetailCloud tentang Predictive Inventory Analytics][1], analitik prediktif menggunakan data historis, tren pasar, dan variabel eksternal untuk memperkirakan permintaan di masa depan secara lebih akurat. Pendekatan ini membantu perusahaan mengurangi risiko overstock maupun stockout, meningkatkan efisiensi biaya, dan mendukung pengambilan keputusan yang lebih tepat dalam menghadapi dinamika pasar yang cepat berubah.

[Magrini (2023)][2] meninjau berbagai metodologi prediktif seperti analisis deret waktu, regresi, dan algoritma machine learning yang telah membantu bisnis dalam meningkatkan akurasi peramalan penjualan, sehingga berdampak positif pada perencanaan permintaan dan strategi penjualan. Selain itu, [Yadav (2025)][3] membuktikan bahwa algoritma machine learning seperti Random Forest dan XGBoost dapat memberikan prediksi penjualan yang lebih akurat dibandingkan metode tradisional, khususnya pada usaha kecil dan menengah termasuk superstore. Studi oleh [Abuassba et al. (2021)][4] dalam jurnal *Journal of Retailing and Consumer Services* juga menunjukkan bahwa penerapan machine learning dalam forecasting dapat meningkatkan akurasi prediksi hingga 30% dibandingkan metode tradisional.

Maka dari itu, proyek ini berfokus pada pembangunan model prediksi nilai penjualan menggunakan data transaksi Superstore dengan pendekatan machine learning. Diharapkan model ini dapat menjadi fondasi dalam mendukung pengambilan keputusan strategis bagi manajemen ritel.

## Business Understanding

### Problem Statements
Pernyataan masalah ini mengarah pada kebutuhan untuk meramalkan penjualan produk ritel dengan menggunakan dataset Superstore yang ada dengan mengidentifikasi masalah utama yang ingin diselesaikan:

- **Pernyataan Masalah 1**: Algoritma machine learning apa yang paling efektif untuk memprediksi penjualan produk ritel berdasarkan fitur-fitur seperti kategori, wilayah, waktu, dan metode pengiriman?
- **Pernyataan Masalah 2**: Bagaimana cara mengevaluasi hasil prediksi penjualan agar dapat digunakan sebagai dasar dalam pengambilan keputusan strategis seperti pengelolaan persediaan?

### Goals
Tujuan dari pernyataan masalah yang telah diidentifikasi adalah:

- **Tujuan 1**: Membuat model peramalan yang akurat menggunakan dataset Superstore untuk memprediksi penjualan produk berdasarkan fitur yang relevan.
- **Tujuan 2**: Menggunakan hasil peramalan untuk memberikan wawasan kepada pengambil keputusan dalam menentukan jumlah persediaan yang optimal.

### Solution Statements
- **Solusi 1**: Menggunakan beberapa model machine learning seperti **Random Forest**, **XGBoost**, **CatBoost**, dan **LSTM** untuk meramalkan penjualan produk ritel.
- **Solusi 2**: Melakukan **hyperparameter tuning** untuk meningkatkan kinerja model, dan memilih model terbaik berdasarkan evaluasi menggunakan metrik MAE, MSE, RMSE, dan R².

## Data Understanding

Dataset **Sales Forecasting** oleh **Rohit Sahoo** merupakan kumpulan data yang dirancang untuk membantu dalam membangun model prediksi penjualan berdasarkan data historis. Dataset ini sering digunakan dalam proyek analisis data ritel dan peramalan penjualan yang dapat diunduh di [Kaggle][5]. Dataset ini mencakup data penjualan produk ritel selama 4 tahun yang berisi **9800 baris data** dan **18 kolom fitur**, dengan variabel-variabel yang relevan untuk peramalan penjualan, seperti kategori produk, wilayah, metode pengiriman, dan lainnya.

### Variabel-variabel pada Dataset Superstore:
- **Row ID**: ID urut baris data.
- **Order ID**: ID unik untuk setiap transaksi pemesanan.
- **Order Date**: Tanggal pemesanan produk.
- **Ship Date**: Tanggal pengiriman produk.
- **Ship Mode**: Metode pengiriman produk.
- **Customer ID**: ID pelanggan yang melakukan pembelian.
- **Customer Name**: Nama pelanggan.
- **Segment**: Segmen pelanggan.
- **Country/City/State/Region**: Lokasi pelanggan.
- **Product ID**: ID produk yang dibeli.
- **Category/Sub-Category**: Kategori produk.
- **Product Name**: Nama produk.
- **Sales**: Total nilai penjualan produk (target peramalan).
- **Postal Code**: Kode pos alamat pengiriman.

![Screenshot 2025-04-26 090810](https://github.com/user-attachments/assets/bda80630-a160-4061-aa08-cfdbab98a91d)

Berdasarkan output df.info() yang ditampilkan, dataset ini memiliki 9.800 baris data dan terdiri dari 18 kolom. Setiap kolom menunjukkan berbagai informasi terkait transaksi, seperti Row ID, Order ID, Order Date, Ship Date, Customer ID, Product ID, hingga Sales. Hampir seluruh kolom memiliki data lengkap (non-null), kecuali kolom Postal Code yang memiliki sedikit missing value, yaitu sebanyak 11 data kosong dari total 9.800 entri. Tipe data dalam dataset ini terdiri dari tiga jenis: int64 untuk data numerik bulat (contohnya Row ID), float64 untuk data numerik desimal (contohnya Postal Code dan Sales), serta object untuk data bertipe teks atau kategori (seperti Customer Name, City, dan Product Name). Selain itu, ukuran memori yang digunakan oleh dataset ini adalah sekitar 1.3 MB. Setelah itu, dilakukan pengecekan duplikasi data menggunakan superstore.duplicated().any(), yang bertujuan untuk memastikan apakah ada baris data yang sama persis. Didapat hasilnya False, berarti tidak ada data duplikat.


### Exploratory Data Analysis (EDA)
1. **Descriptive Statistics Overview**

| Statistik | Postal Code   | Sales         |
|-----------|---------------|---------------|
| Count     | 9789.000000   | 9789.000000   |
| Mean      | 55273.322403  | 230.116193    |
| Std       | 32041.223413  | 625.302079    |
| Min       | 1040.000000   | 0.444000      |
| 25%       | 23223.000000  | 17.248000     |
| 50%       | 58103.000000  | 54.384000     |
| 75%       | 90008.000000  | 210.392000    |
| Max       | 99301.000000  | 22638.480000  |

**Catatan:** Nilai `Sales` menunjukkan distribusi yang sangat tidak merata (*right-skewed*), di mana rata-rata lebih tinggi dari median. Ini bisa menunjukkan adanya **outlier** (transaksi dengan nilai sangat besar). Transformasi logaritma dapat dipertimbangkan untuk menstabilkan varians data.


2. **Sales Distribution Overview**

![category](https://github.com/user-attachments/assets/bf471ebc-b0c1-4936-96d8-8e71a1b874fc)


Grafik pie ini menggambarkan distribusi penjualan berdasarkan kategori utama. Dari visualisasi terlihat bahwa kategori **Technology** memberikan kontribusi penjualan terbesar sebesar **36,7%**, diikuti oleh **Furniture** dengan **32,1%**, dan **Office Supplies** sebesar **31,2%**. Perbedaan antar kategori tidak terlalu mencolok, namun tetap terlihat bahwa produk-produk teknologi memiliki peran dominan dalam total penjualan. 

![sub_category](https://github.com/user-attachments/assets/6ef16baa-5f12-48e4-a9f7-73a2047e440d)

Grafik ini menampilkan sub-kategori dengan total penjualan tertinggi. Sub-kategori Phones mencatat penjualan sebesar USD 326.488, disusul oleh Chairs dengan USD 322.108. Sub-kategori lain seperti Storage, Tables, dan Binders mengikuti di posisi berikutnya. Mayoritas sub-kategori teratas berasal dari kategori Technology dan Furniture, yang konsisten dengan distribusi penjualan pada grafik pie sebelumnya.

![product](https://github.com/user-attachments/assets/820f1994-7ebd-4452-80cf-ab1b0328dd06)

Grafik batang ini menampilkan 10 produk dengan total penjualan tertinggi secara individu. Produk dengan performa penjualan paling tinggi adalah Canon imageCLASS 2200 Advanced Copier, yang berhasil mencatatkan penjualan lebih dari 60.000 USD. Di bawahnya, terdapat sejumlah produk perkantoran lainnya seperti Binding Machines, Cisco Video System, dan HP Printers. 


3. **Sales by Geography** 

![region](https://github.com/user-attachments/assets/97e55194-b87b-4a53-a70f-ec5caa367335)

Analisis geografis terhadap total penjualan mengungkap bahwa wilayah West menyumbang porsi penjualan terbesar secara regional, yaitu sebesar 31.5%, diikuti oleh East dengan 29.3%. Wilayah Central dan South tertinggal dengan masing-masing 21.9% dan 17.3%.

![state](https://github.com/user-attachments/assets/b2e8fa95-2c10-4761-8c62-baa69894bf57)

Jika dilihat lebih mendetail berdasarkan negara bagian, California mendominasi dengan kontribusi sebesar 28.6% dari total penjualan di antara 10 negara bagian teratas, disusul oleh New York (19.6%) dan Texas (10.8%).

![city](https://github.com/user-attachments/assets/bd8cf3a0-8290-49e2-8ea9-fc0756bd8acb)

Sementara itu, pada tingkat kota, New York City memimpin dengan pangsa penjualan sebesar 25.1%, diikuti oleh Los Angeles (17.2%) dan Seattle (11.5%). Secara keseluruhan, pola ini menunjukkan bahwa penjualan paling kuat terkonsentrasi di wilayah pesisir seperti California dan New York, yang juga merupakan pusat ekonomi besar di Amerika Serikat.


4. **Sales by Customer Segment** 

![segment](https://github.com/user-attachments/assets/928a7349-fa9a-49b2-9723-f4f1ac79d2c1)

Distribusi penjualan berdasarkan segmen pelanggan menunjukkan bahwa segmen Consumer merupakan penyumbang terbesar dengan 50.9% dari total penjualan. Segmen Corporate berada di posisi kedua dengan kontribusi sebesar 30.3%, sementara Home Office menyumbang 18.8%. Temuan ini mengindikasikan bahwa pelanggan individu atau konsumen umum menjadi pasar utama dalam penjualan produk, diikuti oleh segmen bisnis besar dan kantor rumahan.


5. **Top 10 Customers by Sales** 

![customers](https://github.com/user-attachments/assets/6a14e384-3bd4-4597-8781-fecfbd7e404f)

Daftar pelanggan dengan kontribusi penjualan tertinggi menunjukkan bahwa Sean Miller merupakan pelanggan paling bernilai, dengan total pembelian mencapai lebih dari USD 25.000. Diikuti oleh Tamara Chand dengan hampir USD 19.000 penjualan, serta beberapa nama lain seperti Raymond Buch, Tom Ashbrook, dan Adrian Barton, yang masing-masing menyumbang lebih dari USD 14.000. Pola ini menunjukkan bahwa terdapat kelompok pelanggan kunci dengan kontribusi besar terhadap pendapatan. Pelanggan-pelanggan ini bisa menjadi fokus program loyalitas, penawaran eksklusif, atau pendekatan penjualan yang lebih personal untuk mempertahankan dan meningkatkan nilai mereka ke depannya.


6. **Shipping Analysis**  

![shipping](https://github.com/user-attachments/assets/36065b87-7160-4e33-9541-5eb67b6def6e)

Grafik ini menunjukkan distribusi mode pengiriman yang digunakan pelanggan. Dari grafik tersebut, terlihat bahwa Standard Class merupakan metode pengiriman yang paling sering dipilih, dengan jumlah pengiriman mendekati 6.000 kali. Hal ini menunjukkan bahwa pelanggan cenderung memilih opsi pengiriman yang lebih ekonomis dan cukup andal. Second Class berada di urutan kedua, diikuti oleh First Class, sedangkan Same Day merupakan metode pengiriman yang paling jarang digunakan, kemungkinan karena biaya yang lebih tinggi atau keterbatasan layanan.

![duration_shpng](https://github.com/user-attachments/assets/3b62faf7-fd6f-4885-afd6-91db3ff92bcc)

Sementara itu, grafik kedua menunjukkan rata-rata durasi pengiriman untuk setiap mode. Standard Class memiliki rata-rata waktu pengiriman terlama, yaitu sekitar 5 hari. Di sisi lain, Same Day konsisten dengan namanya, memberikan pengiriman dalam waktu kurang dari satu hari. First Class memiliki durasi rata-rata sekitar 2 hari, dan Second Class memerlukan waktu sekitar 3 hari. Pola ini menunjukkan trade-off yang jelas antara kecepatan dan popularitas: meskipun Standard Class memerlukan waktu lebih lama, ia tetap menjadi pilihan utama pelanggan, kemungkinan karena efisiensi biaya.


7. **Sales Trend Over Time** 

![hour](https://github.com/user-attachments/assets/19c83774-249d-4e3c-b24b-dca85ae30978)

Daily Sales Trend
Penjualan harian dari tahun 2015 hingga 2018 menunjukkan fluktuasi tajam. Meskipun terdapat lonjakan pada periode tertentu seperti awal 2015, secara umum tren penjualan harian menunjukkan peningkatan. Ini menandakan pertumbuhan aktivitas bisnis yang dinamis dan kemungkinan dipengaruhi oleh faktor musiman atau promosi tertentu.

![month](https://github.com/user-attachments/assets/556ce0c9-645c-4994-b1a9-23f7f415188f)

Monthly Sales Trend
Penjualan bulanan tampak berfluktuasi namun menunjukkan tren kenaikan secara umum. Lonjakan signifikan sering kali terjadi menjelang akhir tahun, khususnya pada Desember, yang kemungkinan besar disebabkan oleh peningkatan belanja saat musim liburan atau adanya diskon/promosi akhir tahun.

![year](https://github.com/user-attachments/assets/0b8d0541-6592-49ff-bbca-5bece0ad06cf)

Yearly Sales Trend
Penjualan tahunan menunjukkan penurunan dari tahun 2015 ke 2016, namun meningkat tajam pada tahun 2017 dan mencapai puncaknya pada 2018 dengan nilai penjualan sekitar USD 720.000. Ini menunjukkan adanya pertumbuhan yang konsisten dalam performa penjualan tahunan selama tiga tahun terakhir.


8. **Monthly Sales Pattern Year-to-Year** 

![month_y2y](https://github.com/user-attachments/assets/6d1c1e35-87c4-483a-a87b-cd3becdbcd16)

Grafik tersebut menunjukkan tren penjualan bulanan dari 2015 hingga 2018. Terlihat pola musiman yang konsisten, di mana penjualan cenderung rendah di awal tahun (Januari–Februari) dan meningkat tajam menjelang akhir tahun, khususnya di bulan November dan Desember. Lonjakan paling signifikan terjadi di November 2018, dengan penjualan mendekati 110.000. Dari tahun ke tahun, tren pertumbuhan penjualan terlihat jelas, terutama pada bulan-bulan puncak. 


9. **Monthly Observation** 

![month_obsrv](https://github.com/user-attachments/assets/65dfcbd6-6191-45db-a122-bb24f41cc0be)

Grafik tersebut menunjukkan total penjualan per bulan secara agregat. Terlihat bahwa penjualan mengalami peningkatan signifikan di bulan-bulan tertentu, terutama Maret, September, November, dan Desember. November menjadi bulan dengan penjualan tertinggi, mencapai hampir 350.000, disusul oleh Desember. Sebaliknya, bulan Februari mencatatkan penjualan terendah. Selain itu, garis tren menunjukkan fluktuasi yang cukup tajam antar bulan, mencerminkan dinamika permintaan konsumen yang kuat sepanjang tahun.


10. **Numerical Feature Correlation** 

![numeric_crelation](https://github.com/user-attachments/assets/96c610d8-61ca-444f-8005-e5b0ff75d302)

Hasil dari grafik heatmap menunjukkan tidak ada korelasi signifikan antar ketiganya, dengan nilai korelasi sangat rendah (dekat 0). Artinya, fitur-fitur ini cenderung independen satu sama lain dan tidak saling memengaruhi secara linier.


11. **Sales Distribution & Outlier detection** 

![sales_outlier](https://github.com/user-attachments/assets/4f36a195-fa91-436e-8212-ad153332d524)

![sles_distri](https://github.com/user-attachments/assets/7025c5b6-e578-4c44-8e06-86c2395a55e8)

Distribusi penjualan menunjukkan data yang skew ke kanan, di mana sebagian besar transaksi berada di kisaran rendah (sekitar 0–500), sementara hanya sedikit transaksi dengan nilai sangat tinggi yang menjadi outlier. Hal ini mengindikasikan ketidakseimbangan pada data penjualan.


### Data Preparation

**1. Cleaning Data**  
Pada tahap ini, dilakukan beberapa proses untuk merapikan dan menyiapkan dataset sebelum dianalisis lebih lanjut. Pertama, perintah superstore.dropna(axis=0, inplace=True) digunakan untuk menghapus seluruh baris (axis=0) yang memiliki nilai kosong (missing values) di dalam dataset. Dengan menambahkan inplace=True, perubahan ini langsung diterapkan pada dataset tanpa perlu membuat salinan baru. Setelah itu, dilakukan pengecekan duplikasi data menggunakan superstore.duplicated().any(), yang bertujuan untuk memastikan apakah ada baris data yang sama persis. Jika hasilnya False, berarti tidak ada data duplikat; namun jika True, maka terdapat duplikasi yang perlu ditangani. Selanjutnya, kolom-kolom yang tidak relevan untuk analisis, yaitu Row ID, Order ID, Customer ID, dan Product ID, dihapus dari dataset menggunakan superstore.drop(columns=[...]). Penghapusan ini bertujuan untuk menyederhanakan dataset dan mengurangi kolom yang tidak berkontribusi langsung terhadap analisis bisnis. Terakhir, dataset yang sudah dibersihkan disalin ke dalam variabel baru superstore_raw menggunakan superstore.copy(). Hal ini bertujuan untuk menyimpan versi mentah dataset setelah cleaning, sehingga jika dibutuhkan, data asli yang sudah dibersihkan tetap tersedia tanpa harus mengulang proses dari awal.

**2. Transformasi Logaritma pada Penjualan**  
Variabel `Sales` ditransformasikan menggunakan logaritma natural dengan rumus `log(1 + x)`. Hal ini dilakukan karena distribusi penjualan sangat tidak normal dan condong ke kanan (right-skewed), serta mengandung banyak nilai ekstrem. Dengan transformasi log, distribusi data menjadi lebih mendekati normal, mengurangi pengaruh outlier, dan pada akhirnya membantu meningkatkan performa model prediksi.

**3. Pembersihan Outlier**  
Outlier pada kolom penjualan diidentifikasi menggunakan metode Interquartile Range (IQR), dengan menghitung batas bawah dan atas, lalu menghapus data yang berada di luar rentang tersebut. Langkah ini penting untuk menghindari distorsi model akibat nilai-nilai ekstrem yang tidak representatif dan bisa menyebabkan overfitting atau generalisasi yang buruk.

**4. Rekayasa Fitur (Feature Engineering)**  
Dari kolom tanggal, dibuat fitur-fitur baru seperti durasi pengiriman (selisih hari antara order dan pengiriman), tahun, bulan, hari, hari dalam minggu, kuartal, serta indikator musim liburan. Fitur-fitur ini memungkinkan model menangkap pola temporal seperti musiman dan perilaku pembelian harian, serta mempertimbangkan pengaruh logistik terhadap penjualan.

**5. Encoding Variabel Kategorikal**  
Untuk mempersiapkan data bagi algoritma tabular, variabel kategorikal seperti `Category`, `Sub-Category`, `Region`, `Segment`, dan `Ship Mode` dikonversi ke bentuk numerik menggunakan label encoding. Hal ini perlu dilakukan karena sebagian besar algoritma machine learning tidak dapat mengolah data dalam bentuk teks secara langsung.

**6. Persiapan Data untuk Model LSTM**  
Untuk model LSTM, hanya fitur numerik dan waktu yang dipilih, kemudian dinormalisasi menggunakan MinMaxScaler. Data disusun ulang menjadi urutan jendela waktu (sequence) yang sesuai dengan kebutuhan model time series. Data kemudian dibagi ke dalam subset pelatihan, validasi, dan pengujian menjadi 70% : 15% : 15% agar model dapat belajar pola urut waktu dengan lebih efektif.

**7. Persiapan Data untuk Model Tabular**  
Untuk model seperti Random Forest dan XGBoost, data disiapkan dengan menghapus kolom non-relevan seperti nama produk dan pelanggan, yang tidak memiliki pengaruh langsung terhadap target prediksi. Selanjutnya, data dibagi menjadi training, validation, dan testing set dengan pembagian sebesar 70% : 15% : 15% untuk menghindari data leakage dan memastikan hasil evaluasi model yang objektif dan andal.


## Modeling

Pada tahap ini, diterapkan beberapa algoritma machine learning untuk membangun model prediktif penjualan ritel. Model yang digunakan meliputi Random Forest, XGBoost, CatBoost, dan LSTM. Masing-masing dievaluasi untuk menentukan model terbaik yang digunakan dalam mendukung pengambilan keputusan.

### 1. Random Forest  
Random Forest merupakan algoritma ensemble berbasis pohon keputusan yang menggabungkan banyak pohon acak untuk menghasilkan prediksi yang stabil dan akurat. Algoritma ini cocok untuk data numerik maupun kategorikal serta tahan terhadap overfitting. [Scikit-learn (2023)][6] menyediakan dokumentasi lengkap terkait parameter seperti `n_estimators`, `max_depth`, dan `random_state`. Kelemahannya adalah interpretasi model yang kompleks serta waktu pelatihan yang lebih lama pada data besar.

### 2. XGBoost  
XGBoost adalah algoritma boosting yang sangat efisien dan telah terbukti unggul dalam berbagai kompetisi machine learning. Keunggulannya antara lain kemampuan menangani missing value dan regularisasi untuk menghindari overfitting. [Chen & Guestrin (2016)][7] mengembangkan XGBoost dengan fokus pada kecepatan dan performa. Parameter yang digunakan antara lain `learning_rate=0.1`, `max_depth=6`, dan `n_estimators=100`.

### 3. CatBoost  
CatBoost, dikembangkan oleh Yandex, unggul dalam menangani variabel kategorikal tanpa perlu proses encoding manual. Menurut [Prokhorenkova et al. (2018)][8], CatBoost mengurangi bias prediktif dan mempercepat pelatihan dengan pendekatan gradient boosting simetris. Meskipun waktu pelatihan lebih lama dibanding XGBoost, akurasinya tinggi bahkan pada data kompleks.

### 4. LSTM (Long Short-Term Memory)  
LSTM adalah jenis Recurrent Neural Network (RNN) yang dirancang untuk mempelajari pola urutan jangka panjang. Model ini ideal untuk memproses data time-series seperti penjualan harian atau bulanan. [Hochreiter & Schmidhuber (1997)][9] memperkenalkan LSTM sebagai solusi atas kelemahan RNN standar yang tidak mampu menyimpan memori jangka panjang. Dalam proyek ini, digunakan arsitektur dengan 50 unit hidden, dropout 0.2, dan optimizer Adam. Untuk tuning hyperparameter, pendekatan grid search juga digunakan sesuai panduan dari [Brownlee (2017)][10].


## Evaluation

Untuk mengevaluasi performa model regresi dalam memprediksi nilai penjualan, digunakan empat metrik utama: **Mean Absolute Error (MAE)**, **Mean Squared Error (MSE)**, **Root Mean Squared Error (RMSE)**, dan **R-squared (R²)**. Pemilihan metrik-metrik ini merujuk pada praktik umum dalam evaluasi model regresi [Chai & Draxler (2014)][11], [Brownlee (2020)][12], dan disesuaikan dengan konteks data yang bersifat numerik kontinu.

**1. Mean Absolute Error (MAE)**  
MAE menghitung rata-rata absolut selisih antara nilai sebenarnya (yi) dan nilai prediksi (fi). Berbeda dengan MSE/RMSE, MAE tidak mengkuadratkan error, sehingga lebih tahan terhadap outlier.

![rms_mae](https://github.com/user-attachments/assets/41ba4dff-4c7d-443c-a493-2254352b3285)

**2. Mean Squared Error (MSE)**  
MSE mengukur rata-rata dari kuadrat selisih antara nilai sebenarnya (Xi) dan nilai prediksi (xi). Semakin kecil nilai MSE, semakin baik model dalam melakukan prediksi. Semakin kecil nilai MSE, semakin baik prediksi model.

![images (1)](https://github.com/user-attachments/assets/e7afd59e-be2b-4844-8800-39290d80ea81)

**3. Root Mean Squared Error (RMSE)**  
RMSE adalah akar kuadrat dari MSE, mengembalikan metrik ke satuan asli dari target (tidak dalam kuadrat). Ini membuat RMSE lebih mudah diinterpretasikan dalam konteks dunia nyata. Semakin kecil RMSE, semakin baik model dalam memprediksi nilai target.

![20190714113817886](https://github.com/user-attachments/assets/27681863-5c72-4406-baa4-388e6847320a)

**4. R-squared (R²)**  
R² menunjukkan seberapa besar variasi data target yang bisa dijelaskan oleh model. Nilai R² mendekati 1 berarti model sangat baik dalam menjelaskan variansi data. Nilai mendekati 1 menunjukkan bahwa model dapat menjelaskan variabilitas dengan baik.

![images](https://github.com/user-attachments/assets/d6cbeace-fb51-4d22-a9da-f28e4348c8d9)


### Hasil evaluasi dari keempat model yang digunakan ditampilkan dalam tabel berikut:

| Model         | MAE       | MSE        | RMSE      | R²        |
|---------------|-----------|------------|-----------|-----------|
| Random Forest | 0.061773  | 0.021770   | 0.147546  | 0.999998  |
| XGBoost       | 0.609670  | 1.642140   | 1.281460  | 0.999886  |
| CatBoost      | 0.763892  | 2.636579   | 1.623755  | 0.999817  |
| LSTM          | 84.227980 | 11915.4068 | 109.1577  | -0.006134 |

Tabel di atas menunjukkan bahwa model **Random Forest** memberikan hasil paling optimal, dengan nilai MAE, MSE, dan RMSE yang paling rendah serta nilai R² yang hampir sempurna (0.999998), yang menandakan bahwa model ini mampu memprediksi penjualan dengan sangat akurat dan konsisten terhadap data aktual. Model **XGBoost** dan **CatBoost** juga memberikan hasil yang sangat baik, meskipun sedikit di bawah Random Forest dalam hal akurasi. Sebaliknya, model **LSTM** menunjukkan performa yang buruk dengan skor R² negatif, menandakan model gagal memahami pola data.

---

## 📌 Kesimpulan dan Saran

Berdasarkan proyek ini, model **Random Forest** menunjukkan performa terbaik dalam memprediksi penjualan produk ritel dengan skor evaluasi yang paling baik dibandingkan XGBoost, CatBoost, dan LSTM. Hal ini terlihat dari nilai MAE, MSE, dan RMSE yang paling kecil serta R² yang paling tinggi.
Adapun model **LSTM**, meskipun berbasis deep learning, cenderung memberikan prediksi yang lebih konstan dan kurang mampu menangkap fluktuasi musiman penjualan, yang terlihat jelas pada grafik perbandingan aktual dan prediksi.

**Saran:**
- Model tree-based seperti Random Forest cocok digunakan untuk prediksi penjualan jangka pendek dan menengah.
- Untuk menangkap pola musiman atau tren jangka panjang, perlu eksplorasi lebih lanjut terhadap model time series berbasis arsitektur sequence (misalnya: LSTM dengan tuning lanjutan, Prophet, atau ARIMA).

---

## 🚀 Rencana Pengembangan Lebih Lanjut

Beberapa ide pengembangan yang dapat dilakukan ke depan:
- Mengimplementasikan **forecasting ke masa depan** untuk memperkirakan penjualan bulan berikutnya.
- Menambahkan fitur waktu seperti hari libur, hari kerja, atau promosi untuk meningkatkan akurasi.
- Menerapkan teknik feature engineering dan hyperparameter tuning yang lebih optimal.
- Membuat **dashboard interaktif** untuk membantu pengambilan keputusan strategis oleh manajemen.

---

## 📊 Visualisasi Hasil Prediksi vs Aktual

Berikut adalah visualisasi perbandingan antara nilai penjualan aktual dan hasil prediksi dari keempat model:

![predict_month](https://github.com/user-attachments/assets/d43fde12-7d24-4420-828b-a030cbaf92a4)

Visualisasi di atas menunjukkan bahwa prediksi dari **Random Forest, XGBoost, dan CatBoost** hampir mengikuti pola penjualan aktual, sedangkan prediksi **LSTM** terlihat cenderung flat dan kurang adaptif terhadap dinamika data historis.


_Terima kasih telah membaca laporan ini!_


[1]: https://retailcloud.com/predictive-inventory-analytics/

[2]: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5006903

[3]: https://norma.ncirl.ie/7291/1/shaileshsubhashchandyadav.pdf

[4]: https://www.mdpi.com/2076-3417/13/19/11112#B2-applsci-13-11112

[5]: https://www.kaggle.com/datasets/rohitsahoo/sales-forecasting/data

[6]: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html  

[7]: https://xgboost.readthedocs.io/en/stable/  

[8]: https://catboost.ai/docs/  

[9]: https://keras.io/api/layers/recurrent_layers/lstm/  

[10]: https://machinelearningmastery.com/grid-search-hyperparameters-deep-learning-models-python-keras/

[11]: https://doi.org/10.5194/gmd-7-1247-2014 "Chai, T., & Draxler, R. R. (2014). Root mean square error (RMSE) or mean absolute error (MAE)? – Arguments against avoiding RMSE in the literature. *Geoscientific Model Development*, 7(3), 1247–1250."

[12]: https://machinelearningmastery.com/regression-metrics-for-machine-learning/ "Brownlee, J. (2020). Regression Metrics for Machine Learning. *Machine Learning Mastery*."


