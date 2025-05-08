# Proyek Akhir: Menyelesaikan Permasalahan Perusahaan Edutech

## Business Understanding

Jaya Jaya Maju adalah perusahaan multinasional yang telah beroperasi sejak awal tahun 2000. Dengan jumlah karyawan yang kini melampaui seribu orang dan tersebar di berbagai wilayah, perusahaan ini terus berkembang dalam skala operasional dan bisnis. Namun, seiring dengan pertumbuhan tersebut, muncul tantangan baru dalam hal pengelolaan sumber daya manusia.

Salah satu tantangan utama yang dihadapi adalah tingginya angka attrition rate, yaitu persentase karyawan yang keluar dari perusahaan setiap tahunnya. Saat ini, angka tersebut telah melebihi 10%, yang tentunya menimbulkan kekhawatiran dari sisi manajemen, terutama di departemen HR. Tingginya attrition tidak hanya berdampak pada stabilitas tim, tetapi juga menambah beban biaya rekrutmen dan pelatihan karyawan baru.

### Permasalahan Bisnis

1. Faktor-faktor apa saja yang berkontribusi terhadap tingginya tingkat attrition di perusahaan?
2. Bagaimana cara mengidentifikasi karyawan yang berisiko tinggi untuk resign?
3. Bagaimana menyajikan hasil analisis ini agar mudah dipahami dan dimanfaatkan oleh tim HR?

### Cakupan Proyek

Proyek ini bertujuan untuk memahami dan mengurangi tingkat attrition (pengunduran diri) karyawan dengan pendekatan berbasis data. Ruang lingkup proyek meliputi:

a. Eksplorasi dan Pembersihan Data

Melakukan proses data preparation dan data cleaning terhadap dataset *employee\_data.csv* agar siap untuk dianalisis.

b. Analisis Data dan Visualisasi (EDA)

Mengevaluasi faktor-faktor yang memengaruhi attrition seperti pendapatan, status pernikahan, lembur, jarak ke kantor, serta tingkat kepuasan kerja dan lingkungan. Visualisasi dilakukan untuk mengungkap pola dan tren yang relevan.

c. Pembuatan Dashboard Bisnis

Menyusun dashboard interaktif yang menyajikan informasi visual tentang faktor-faktor utama penyebab attrition secara ringkas dan mudah dipahami oleh pengambil keputusan.

d. Pengembangan Model Prediktif

Menggunakan machine learning untuk membangun model yang dapat memprediksi status karyawan (stay/leave) berdasarkan variabel-variabel penyebab attrition.

e. Kesimpulan dan Rekomendasi

Menyusun temuan utama dari hasil analisis serta memberikan saran strategis yang dapat membantu perusahaan dalam menekan tingkat attrition.


### Persiapan

Sumber data: Dataset yang digunakan ialah [Dataset Jaya Jaya Maju](https://github.com/dicodingacademy/dicoding_dataset/blob/main/employee/employee_data.csv) yang berisi data karyawan perusahaan. Dataset awal memiliki 1.470 baris, namun setelah melalui proses pembersihan data (data cleaning), jumlah baris yang digunakan menjadi 1.058 baris yang bersih dan siap untuk dianalisis. Dataset ini terdiri dari 35 kolom, yang mencakup berbagai informasi mengenai karyawan. Berikut adalah penjelasan tentang kolom-kolom yang terdapat dalam dataset:
1. **EmployeeId**: ID unik untuk setiap karyawan yang digunakan untuk identifikasi.
2. **Age**: Usia karyawan yang digunakan untuk analisis kelompok usia.
3. **Attrition**: Status pengunduran diri karyawan (1 untuk resign, 0 untuk tetap bekerja).
4. **BusinessTravel**: Frekuensi perjalanan bisnis karyawan (Travel\_Frequently, Travel\_Rarely, Non-Travel).
5. **DailyRate**: Pembayaran harian yang diterima karyawan.
6. **Department**: Departemen tempat karyawan bekerja (misalnya, Human Resources, Research & Development).
7. **DistanceFromHome**: Jarak (km) dari tempat tinggal karyawan ke kantor.
8. **Education**: Tingkat pendidikan karyawan.
9. **EducationField**: Bidang pendidikan yang ditekuni karyawan (misalnya, Medical, Other, Life Sciences).
10. **EmployeeCount**: Jumlah karyawan di perusahaan (biasanya bernilai 1).
11. **EnvironmentSatisfaction**: Tingkat kepuasan terhadap lingkungan kerja (skor 1-4).
12. **Gender**: Jenis kelamin karyawan (Male, Female).
13. **HourlyRate**: Upah per jam yang diterima karyawan.
14. **JobInvolvement**: Tingkat keterlibatan karyawan dalam pekerjaannya (skor 1-4).
15. **JobLevel**: Tingkat posisi pekerjaan karyawan dalam organisasi.
16. **JobRole**: Peran pekerjaan yang dijalankan oleh karyawan (misalnya, Healthcare Representative, Research Scientist).
17. **JobSatisfaction**: Tingkat kepuasan kerja karyawan (skor 1-4).
18. **MaritalStatus**: Status pernikahan karyawan (Married, Single, Divorced).
19. **MonthlyIncome**: Gaji bulanan yang diterima karyawan.
20. **MonthlyRate**: Pembayaran yang diterima setiap bulan oleh karyawan.
21. **NumCompaniesWorked**: Jumlah perusahaan tempat karyawan pernah bekerja sebelumnya.
22. **Over18**: Menandakan apakah karyawan berusia lebih dari 18 tahun (Y = Ya, N = Tidak).
23. **OverTime**: Status lembur karyawan (Yes = Lembur, No = Tidak).
24. **PercentSalaryHike**: Persentase kenaikan gaji yang diterima karyawan.
25. **PerformanceRating**: Penilaian kinerja karyawan (skor 1-5).
26. **RelationshipSatisfaction**: Kepuasan hubungan kerja dengan rekan-rekan (skor 1-4).
27. **StandardHours**: Jumlah jam kerja standar per minggu (biasanya 80).
28. **StockOptionLevel**: Level opsi saham yang diberikan (0 = tidak ada, 1 = rendah, 2 = sedang, 3 = tinggi).
29. **TotalWorkingYears**: Total tahun pengalaman kerja karyawan.
30. **TrainingTimesLastYear**: Jumlah pelatihan yang diikuti karyawan pada tahun sebelumnya.
31. **WorkLifeBalance**: Tingkat keseimbangan kehidupan kerja karyawan (skor 1-4).
32. **YearsAtCompany**: Jumlah tahun karyawan telah bekerja di perusahaan.
33. **YearsInCurrentRole**: Jumlah tahun karyawan telah bekerja dalam posisi saat ini.
34. **YearsSinceLastPromotion**: Jumlah tahun sejak karyawan terakhir kali dipromosikan.
35. **YearsWithCurrManager**: Jumlah tahun karyawan telah bekerja dengan manajer saat ini.


Setup environment:

Menyimpan dan Memuat Model Terbaik untuk digunakan kembali pada prediksi mendatang
```
import joblib

# Simpan model dan preprocessor
joblib.dump(model, 'model/nama_model.pkl')
joblib.dump(preprocessor, 'model/preprocessor.pkl')
```

Install semua requirements
```
!pip freeze > requirements.txt
```

## Business Dashboard

![Dashboard](https://raw.githubusercontent.com/ftriaa/submit_mlt1/main/asset/Dashboard.png)

Dashboard ini dibangun menggunakan Tableau dan terdiri dari beberapa bagian utama, termasuk ringkasan metrik karyawan, visualisasi berdasarkan atribut personal dan pekerjaan, serta filter interaktif. Dshboard ini dapat dikses melalui link berikut https://public.tableau.com/views/EmployeeAttritionDashboard_17467257157100/Dashboard1?:language=en-US&publish=yes&:sid=&:redirect=auth&:display_count=n&:origin=viz_share_link 

Di bagian atas dashboard, terdapat metrik kunci seperti total karyawan (1.058 orang), jumlah karyawan yang mengundurkan diri (179 orang), dan tingkat attrition sebesar 16.92%. Visualisasi selanjutnya menunjukkan bahwa karyawan yang sering lembur (OverTime = Yes) memiliki tingkat attrition yang jauh lebih tinggi dibandingkan yang tidak lembur. Selain itu, pie chart attrition status menggambarkan proporsi karyawan yang keluar dan yang tetap bekerja.

Dashboard juga menyajikan analisis attrition berdasarkan Job Role, di mana peran seperti Laboratory Technician, Sales Executive, dan Research Scientist menunjukkan jumlah pengunduran diri tertinggi. Selanjutnya, dari analisis berdasarkan Monthly Income, terlihat bahwa karyawan dengan pendapatan di bawah 5.000 lebih banyak mengundurkan diri, menunjukkan bahwa penghasilan rendah menjadi salah satu pemicu utama attrition.

Faktor lain yang dianalisis adalah status pernikahan, di mana karyawan yang belum menikah menunjukkan tingkat attrition tertinggi. Dari sisi Job Satisfaction, terlihat bahwa meskipun jumlah resign tertinggi ada pada kelompok dengan kepuasan kerja tinggi, kelompok dengan kepuasan kerja rendah juga menunjukkan angka pengunduran diri yang signifikan. Kelompok usia 25–29 tahun dan 30–34 tahun adalah yang paling rentan keluar, dan karyawan dengan masa kerja 0–5 tahun memiliki attrition rate paling tinggi, menandakan bahwa karyawan baru cenderung lebih rentan untuk keluar dari perusahaan.

Selain itu, jarak tempat tinggal ke kantor juga memengaruhi attrition. Karyawan yang tinggal 11–15 km dari kantor memiliki tingkat attrition paling tinggi. Faktor kepuasan terhadap lingkungan kerja juga berperan penting, di mana karyawan dengan tingkat Environment Satisfaction rendah lebih banyak yang resign.

Dashboard ini dilengkapi dengan filter interaktif berdasarkan Gender dan Department, yang memungkinkan pengguna untuk menyesuaikan analisis berdasarkan kebutuhan. Melalui dashboard ini, dapat disimpulkan bahwa attrition banyak dipengaruhi oleh lembur, gaji rendah, usia muda, masa kerja pendek, kepuasan kerja dan lingkungan yang rendah, serta jarak rumah yang jauh dari kantor. Insight ini dapat menjadi dasar dalam pengambilan keputusan strategis untuk menurunkan tingkat pengunduran diri karyawan.

## Conclusion

Analisis menunjukkan bahwa beberapa faktor utama memiliki korelasi tinggi terhadap attrition. Melalui visualisasi pada dashboard, kita dapat mengidentifikasi kelompok karyawan yang paling rentan untuk mengundurkan diri, sehingga perusahaan dapat merancang strategi intervensi yang lebih tepat sasaran. Faktor-faktor tersebut antara lain:

a. OverTime

Karyawan yang sering lembur memiliki peluang resign 2 hingga 3 kali lebih tinggi dibandingkan yang tidak lembur.

b. Monthly Income

Karyawan dengan penghasilan di bawah 5000 USD per bulan lebih rentan untuk resign, terutama pada level junior dan mid-level.

c. Job Role

Posisi seperti Sales Executive dan Laboratory Technician memiliki angka attrition lebih tinggi dibandingkan Research Scientist atau Manager.

d. Environment Satisfaction

Skor kepuasan lingkungan kerja di bawah 3 (dari skala 4) menunjukkan ketidakpuasan yang signifikan dan meningkatkan risiko resign.

e. Distance From Home

Karyawan yang tinggal lebih dari 10 km dari kantor cenderung memiliki risiko resign yang lebih tinggi.

f. Job Satisfaction

Karyawan dengan skor kepuasan kerja rendah (1–2 dari 4) lebih sering mengundurkan diri dibandingkan yang merasa puas (skor 3–4).

g. Age

Karyawan berusia 25–35 tahun cenderung lebih aktif mencari peluang baru dibandingkan usia yang lebih senior.

h. Marital Status

Karyawan dengan status lajang lebih banyak yang resign dibandingkan yang sudah menikah, kemungkinan karena lebih fleksibel dan mobile dalam mengambil peluang baru.

i. Total Working Years

Karyawan dengan pengalaman kerja kurang dari 5 tahun lebih sering berpindah pekerjaan dibandingkan mereka yang lebih senior dan cenderung stabil.


### Rekomendasi Action Items (Optional)

Untuk mengurangi angka attrition, perusahaan Jaya Jaya Maju disarankan melakukan:

a. Atur Pola Lembur

Pantau karyawan dengan jam lembur tinggi. Terapkan aturan maksimal lembur dan beri kompensasi tambahan.

b. Tingkatkan Skema Gaji dan Benefit

Fokus peningkatan gaji untuk karyawan dengan penghasilan di bawah 5000 USD. Tawarkan bonus performance-based untuk posisi yang riskan.

c. Program Retensi untuk Karyawan Usia 25–35 tahun

Tawarkan jalur karier yang jelas. Berikan kesempatan promosi atau rotasi antar departemen.

d. Fasilitas Transportasi atau WFH

Pertimbangkan subsidi transportasi untuk karyawan yang tinggal jauh. Implementasikan opsi hybrid working untuk meningkatkan fleksibilitas.

e. Perbaikan Lingkungan dan Kepuasan Kerja

Adakan program feedback 360 derajat setiap 6 bulan. Tingkatkan engagement melalui event internal, workshop, atau employee appreciation.

