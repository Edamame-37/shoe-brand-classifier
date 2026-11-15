# Shoe Brand Modeling

Semua workflow modeling menggunakan Jupyter Notebook dan kernel Python global.
Tidak ada virtual environment atau proses training yang dijalankan.

## Urutan penggunaan

1. Buka `setup/setup_dependencies.ipynb` dan jalankan semua cell.
2. Restart kernel jika Jupyter memintanya.
3. Buka `dataset/load_shoe_dataset.ipynb` untuk mengunduh, memeriksa, dan
   mengekspor dataset ke format folder YOLO11.
4. Buka `training/stratified_split.ipynb` untuk membuat split train, validation,
   dan test yang seimbang pada `datasets/shoe_brands_stratified`.
5. Buka `training/preprocessing.ipynb` untuk menerapkan EXIF auto-orientation,
   konversi RGB, resize 224×224, dan quality control ke
   `datasets/shoe_brands_preprocessed`.
6. Buka `training/EDA.ipynb` untuk menganalisis distribusi kelas, dimensi,
   kualitas gambar, file rusak, dan exact duplicate sebelum training.
7. Untuk inference cloud, atur `ROBOFLOW_API_KEY`, buka
   `roboflow/shoe_brand_inference.ipynb`, lalu isi `IMAGE_SOURCE`.
8. Untuk memeriksa kesiapan YOLO11, buka
   `yolo11/setup_yolo11.ipynb`.

Kernel Jupyter saat ini menggunakan Python 3.13. `inference-sdk` memerlukan
Python 3.10 sampai sebelum Python 3.14.

## Roboflow API key

Atur API key pada environment sebelum menjalankan Jupyter. Contoh PowerShell:

```powershell
$env:ROBOFLOW_API_KEY = "your_actual_api_key"
jupyter lab
```

Credential hanya dibaca dari environment variable. Notebook menggunakan
`api_key_transport="header"`, sehingga API key dikirim melalui Authorization
header dan tidak ditempatkan pada query string.

## Struktur dataset YOLO11

Dataset bersumber dari `keremberke/shoe-classification` dengan konfigurasi
`full` di Hugging Face. Dataset terdiri dari 576 gambar training, 166 gambar
validation, dan 83 gambar test. Split `validation` dari Hugging Face dipetakan ke
folder `val` untuk format Ultralytics.

Catatan kualitas data: archive `valid.zip` upstream saat ini berisi 166 gambar
Nike dan tidak berisi Adidas atau Converse. Notebook mempertahankan isi dataset
upstream, menampilkan peringatan kelas kosong, dan tidak memulai training.

```text
modeling/datasets/shoe_brands/
|-- train/
|   |-- adidas/
|   |-- converse/
|   `-- nike/
|-- val/
|   |-- adidas/
|   |-- converse/
|   `-- nike/
`-- test/                  # opsional
    |-- adidas/
    |-- converse/
    `-- nike/
```

Notebook YOLO11 hanya memvalidasi struktur dataset dan memuat checkpoint
`yolo11n-cls.pt`. Notebook tersebut tidak memanggil `model.train()`.

Dataset untuk setup dan training model adalah `shoe_brands_preprocessed`, dengan
193 gambar per kelas untuk train, 55 per kelas untuk validation, dan 27 per
kelas untuk test. Dataset hasil ekspor awal dan stratified tetap dipertahankan
sebagai sumber. Augmentasi tidak disimpan secara statis dan harus diterapkan
secara dinamis hanya pada split train ketika training disetujui.
