# Shoe Brand Modeling

This directory contains Roboflow cloud inference for the hosted
`shoe-classification-k9ajr/2` model and a local YOLO11 classification setup.
No training command is executed by these scripts.

## Environment setup

Use Python 3.11 because the Roboflow SDK does not support Python 3.14.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r modeling\requirements.txt
```

Copy the environment template and set the Roboflow API key:

```powershell
Copy-Item modeling\.env.example modeling\.env
```

Edit `modeling/.env` and replace the placeholder value. The key is sent in the
`Authorization` header and is never placed in the query string.

## Roboflow inference

Run inference using a local file:

```powershell
python modeling\roboflow_inference.py C:\path\to\shoe.jpg
```

An image URL is also accepted. Use `--output` to save the response while still
printing it to stdout:

```powershell
python modeling\roboflow_inference.py https://example.com/shoe.jpg `
  --output modeling\outputs\prediction.json
```

## YOLO11 classification setup

Arrange the local dataset using Ultralytics' classification folder format:

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
`-- test/                  # optional
    |-- adidas/
    |-- converse/
    `-- nike/
```

Validate the dataset structure and initialize the pretrained YOLO11
classification checkpoint:

```powershell
python modeling\yolo11_setup.py
```

This downloads/loads `yolo11n-cls.pt` when needed, but does not start training.
