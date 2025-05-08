# Proyek Akhir: Menyelesaikan Permasalahan Perusahaan Edutech

## Business Understanding

Jaya Jaya Maju adalah perusahaan multinasional yang telah beroperasi sejak awal tahun 2000. Dengan jumlah karyawan yang kini melampaui seribu orang dan tersebar di berbagai wilayah, perusahaan ini terus berkembang dalam skala operasional dan bisnis. Namun, seiring dengan pertumbuhan tersebut, muncul tantangan baru dalam hal pengelolaan sumber daya manusia.

Salah satu tantangan utama yang dihadapi adalah tingginya angka attrition rate, yaitu persentase karyawan yang keluar dari perusahaan setiap tahunnya. Saat ini, angka tersebut telah melebihi 10%, yang tentunya menimbulkan kekhawatiran dari sisi manajemen, terutama di departemen HR. Tingginya attrition tidak hanya berdampak pada stabilitas tim, tetapi juga menambah beban biaya rekrutmen dan pelatihan karyawan baru.

### Permasalahan Bisnis

1. Faktor-faktor apa saja yang berkontribusi terhadap tingginya tingkat attrition di perusahaan?
2. Bagaimana cara mengidentifikasi karyawan yang berisiko tinggi untuk resign?
3. Bagaimana menyajikan hasil analisis ini agar mudah dipahami dan dimanfaatkan oleh tim HR?

### Cakupan Proyek

- Melakukan eksplorasi dan analisis data attrition karyawan berdasarkan dataset yang disediakan perusahaan.
- Mengidentifikasi pola dan hubungan antara berbagai variabel (seperti jam lembur, pendapatan, jabatan) terhadap attrition.
- Membangun model prediktif untuk memperkirakan kemungkinan seorang karyawan akan resign.
- Menyajikan hasil analisis dalam bentuk visualisasi dan rangkuman temuan untuk mendukung pengambilan keputusan.

### Persiapan

Sumber data: [Lihat Dataset Jaya Jaya Maju](https://github.com/dicodingacademy/dicoding_dataset/blob/main/employee/employee_data.csv)

Setup environment:

Install semua requirements
```
!pip freeze > requirements.txt
```

## Business Dashboard

Jelaskan tentang business dashboard yang telah dibuat. Jika ada, sertakan juga link untuk mengakses dashboard tersebut.

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

