# Road Sign Detector — Etap 1

Detekcja znaków drogowych (**Traffic sign**) przy użyciu YOLOv8 i danych Open Images v7.

---

## Struktura projektu

```
road_sign_detector/
├── run_stage1.py              ← punkt wejścia CLI
├── downloader.py              ← pobieranie danych przez FiftyOne
├── requirements.txt
├── README.md
│
├── config/
│   └── settings.yaml          ← cała konfiguracja projektu
│
├── data/
│   ├── raw/                  ← dane źródłowe FiftyOne (tworzone przez downloader.py)
│   │   ├── train/
│   │   │   ├── data/              ← obrazy .jpg
│   │   │   ├── labels/
│   │   │   │   └── detections.csv
│   │   │   └── metadata/
│   │   └── validation/
│   │       ├── data/
│   │       ├── labels/
│   │       │   └── detections.csv
│   │       └── metadata/
│   │
│   ├── prepared/             ← dane YOLO (tworzone przez --prepare)
│   │   ├── images/
│   │   │   ├── raw/               ← wszystkie obrazy po konwersji
│   │   │   ├── train/
│   │   │   ├── val/
│   │   │   └── test/
│   │   ├── labels/
│   │   │   ├── raw/               ← etykiety YOLO (.txt)
│   │   │   ├── train/
│   │   │   ├── val/
│   │   │   └── test/
│   │   └── dataset.yaml
│
├── runs/                      ← generowane automatycznie przez YOLOv8
│   └── road_signs/
│       ├── weights/
│       │   ├── best.pt        ← najlepszy model
│       │   └── last.pt
│       ├── results.csv
│       ├── confusion_matrix.png
│       └── ...
│
├── logs/
│   ├── training.log
│   ├── metrics.json           ← wyniki walidacji (mAP, precision, recall)
│   └── detections/
│       ├── detections.json
│       └── det_*.jpg
│
└── stage1/
    ├── config.py              ← ładowanie konfiguracji, CUDA, logging
    ├── dataset.py             ← konwersja raw/ → prepared/
    ├── trainer.py             ← trening i walidacja YOLOv8
    └── detector.py            ← inferencja, eksport JSON
```

---

## Szybki start

### 1. Instalacja zależności

```bash
# PyTorch z CUDA 13.0 (GPU)
pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/nightly/cu130

# Na CPU (bez GPU)
pip install torch torchvision torchaudio

# Pozostałe pakiety
pip install -r requirements.txt
```

### 2. Pobierz dane

```bash
python downloader.py
```

Zapisuje dane do `raw/train/` i `raw/validation/`.

### 3. Konwertuj dane → format YOLO

```bash
python run_stage1.py --prepare
```

Czyta `raw/`, tworzy `prepared/` z podziałem train/val/test.

### 4. Trenuj model

```bash
# GPU (domyślnie)
python run_stage1.py --train

# CPU
python run_stage1.py --train --device cpu
```

Model zapisywany do `runs/road_signs/weights/best.pt`.

### 5. Walidacja

```bash
python run_stage1.py --validate
```

Wyniki zapisywane do `logs/metrics.json`.

### 6. Detekcja

```bash
python run_stage1.py --detect --input zdjecie.jpg
python run_stage1.py --detect --input katalog/
```

Wyniki zapisywane do `logs/detections/`.

---

## Wszystkie opcje CLI

| Flaga | Opis |
|-------|------|
| `--check` | Sprawdź zależności i środowisko CUDA |
| `--prepare` | Konwertuj `raw/` → `prepared/` |
| `--train` | Trenuj model YOLOv8 |
| `--validate` | Waliduj wytrenowany model |
| `--detect` | Detekcja na obrazie lub katalogu |
| `--source KATALOG` | Nadpisz `source_dir` z settings.yaml |
| `--device DEVICE` | `cpu`, `0` (GPU 0), `0,1` (multi-GPU) |
| `--input SCIEZKA` | Wejście dla `--detect` |
| `--output KATALOG` | Wyjście dla `--detect` |

---

## Konfiguracja (settings.yaml)

```yaml
data:
  source_dir: "raw"          # dane FiftyOne
  data_dir:   "prepared"     # dane YOLO

model:
  architecture: "yolov8n"         # n=szybki / s / m / l / x=dokładny
  device: "0"                     # "0" = GPU, "cpu" = procesor
  epochs: 50
  amp: true                       # FP16 — tylko GPU
```

---

## Przepływ danych

```
downloader.py
    ↓  raw/train/data/*.jpg
    ↓  raw/train/labels/detections.csv

--prepare  (dataset.py)
    ↓  prepared/images/{train,val,test}/*.jpg
    ↓  prepared/labels/{train,val,test}/*.txt
    ↓  prepared/dataset.yaml

--train  (trainer.py)
    ↓  runs/road_signs/weights/best.pt
    ↓  logs/training.log

--validate
    ↓  logs/metrics.json

--detect  (detector.py)
    ↓  logs/detections/detections.json
    ↓  logs/detections/det_*.jpg

[Etap 2 — OCR]          planowany
[Etap 3 — Klasyfikacja] planowany
[Etap 4 — TTS]          planowany
```