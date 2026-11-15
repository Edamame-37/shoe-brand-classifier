# Shoe Brand Modeling

Semua workflow modeling menggunakan Jupyter Notebook dan kernel Python global.
Tidak ada virtual environment atau proses training yang dijalankan.

## Urutan penggunaan

1. Buka `setup/setup_dependencies.ipynb` dan jalankan semua cell.
2. Restart kernel jika Jupyter memintanya.
3. Untuk inference cloud, atur `ROBOFLOW_API_KEY`, buka
   `roboflow/shoe_brand_inference.ipynb`, lalu isi `IMAGE_SOURCE`.
4. Untuk memeriksa kesiapan YOLO11, siapkan dataset dan buka
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
