# Road Sign Detector — Etap 1

Detekcja znaków drogowych klasy **Traffic sign** przy użyciu YOLOv8 i danych z Open Images v7.

---

## Struktura projektu

```
road_sign_detector/
├── run_stage1.py              ← punkt wejścia (CLI)
├── requirements.txt
├── README.md
├── config/
│   └── settings.yaml          ← konfiguracja projektu
├── runs/
│   └── detect/
│   │   │   ├── runs/  
│   │   │   │   ├── weights/ ← args.yaml...box currves..train batch..val batch..results.csv  confusion matrix etc.
│   │   │   └── val/ ←...box currves..train batch..val batch..results.csv etc. confusion matrix etc.
├── data/                      ← generowane automatycznie przez --prepare
│   ├── images/
│   │   ├── raw/               ← wszystkie obrazy po konwersji
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   ├── labels/
│   │   ├── raw/               ← etykiety YOLO (.txt)
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   ├── dataset.yaml           ← konfiguracja datasetu dla YOLOv8
│   ├──runs/                  ← wyniki treningów i detekcji
│   │   ├── detections/detections.json
│   │   ├── road_signs/ 
│   │   │  ├──best_model.pt 
│   │   │  ├──metrics.json 
│   │   │  ├──training.log
│   ├── train/          ← obrazy .jpg
│   └──  val/          ← obrazy .jpg
└── stage1/
    ├── config.py
    ├── dataset.py             ← konwersja FiftyOne → YOLO
    ├── trainer.py
    └── detector.py
```

---

## Pobieranie danych (FiftyOne)

Dane należy pobrać **ręcznie przed uruchomieniem projektu** przy użyciu biblioteki FiftyOne.

### Instalacja FiftyOne

```bash
pip install fiftyone
```

### Skrypt pobierający dane

```python
import fiftyone.zoo as foz

# Walidacja — mały zestaw do testów
dataset = foz.load_zoo_dataset(
    "open-images-v7",
    split="validation",
    label_types=["detections"],
    classes=["Traffic sign"],
    max_samples=15,        # zwiększ do 500+ dla treningu produkcyjnego
)

# Train — większy zestaw do treningu
dataset_train = foz.load_zoo_dataset(
    "open-images-v7",
    split="train",
    label_types=["detections"],
    classes=["Traffic sign"],
    max_samples=500,
)
```

### Eksport do struktury Open Images

Po pobraniu FiftyOne przechowuje dane we własnym formacie — należy je wyeksportować:

```python
import fiftyone as fo
import fiftyone.zoo as foz
from pathlib import Path

for split in ["train", "validation"]:
    dataset = foz.load_zoo_dataset(
        "open-images-v7",
        split=split,
        label_types=["detections"],
        classes=["Traffic sign"],
        max_samples=500,
    )
    export_dir = str(Path("Data") / split)
    dataset.export(
        export_dir=export_dir,
        dataset_type=fo.types.OpenImagesV7Dataset,
        label_field="detections",
        classes=["Traffic sign"],
    )
    print(f"Wyeksportowano {split} → {export_dir}")
```

### Wynikowa struktura katalogów

```
Data/
├── train/
│   ├── data/               ← obrazy .jpg (nazwy = ImageID)
│   ├── labels/
│   │   └── detections.csv  ← adnotacje bbox w formacie Open Images
│   └── metadata/
│       ├── classes.csv
│       ├── hierarchy.json
│       └── image_ids.csv
└── validation/
    └── ...                 ← ta sama struktura
```

### Format detections.csv

```
ImageID,Source,LabelName,Confidence,XMin,XMax,YMin,YMax,IsOccluded,...
abc123,fiftyone,Traffic sign,1.0,0.123,0.456,0.234,0.567,0,...
```

Współrzędne `XMin`, `XMax`, `YMin`, `YMax` są **znormalizowane** do `[0, 1]`.
Kod automatycznie konwertuje je do formatu YOLO (`cx cy w h`).

---

## Konfiguracja

Edytuj `config/settings.yaml` — najważniejszy parametr to ścieżka do danych:

```yaml
data:
  source_dir: "Data"   # ścieżka względna lub bezwzględna do katalogu z danymi FiftyOne
```

Dla ścieżki bezwzględnej (Windows):

```yaml
data:
  source_dir: "C:/Users/Yami/PycharmProjects/roadsign_detector_tts/Data"
```

---

## Instalacja zależności

```bash
# Krok 1 — PyTorch z obsługą CUDA 13.0
pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/nightly/cu130

# Krok 2 — pozostałe pakiety
pip install -r requirements.txt
```

---

## Użycie

### 1. Sprawdź środowisko

```bash
python run_stage1.py --check
```

Wyświetla wersje PyTorch / CUDA, dostępne GPU oraz czy `Data/` ma poprawną strukturę.

### 2. Konwertuj dane FiftyOne → YOLO

```bash
python run_stage1.py --prepare
```

Jeśli dane są w niestandardowej lokalizacji:

```bash
python run_stage1.py --prepare --source "C:/Users/Yami/PycharmProjects/roadsign_detector_tts/Data"
```

Tworzy `data/images/`, `data/labels/` z podziałem `train/val/test` oraz `data/dataset.yaml`.

### 3. Trenuj model

```bash
python run_stage1.py --train
```

### 4. Pełny pipeline

```bash
python run_stage1.py --prepare --train
```

### 5. Walidacja

```bash
python run_stage1.py --validate
```

### 6. Detekcja na własnych obrazach

```bash
python run_stage1.py --detect --input zdjecie.jpg
python run_stage1.py --detect --input katalog/ --output wyniki/
```

---

## Architektura modelu

| Parametr | Wartość domyślna | Opis |
|----------|------------------|------|
| Model    | `yolov8n`        | YOLOv8 Nano (szybki, lekki) |
| Epoki    | `50`             | Liczba epok treningu |
| Batch    | `16`             | Auto-dostosowywany do VRAM |
| imgsz    | `640`            | Rozmiar obrazu wejściowego |
| AMP      | `true`           | FP16 Mixed Precision (tylko GPU) |
| Klasa    | `Traffic sign`   | Jedna klasa, indeks YOLO = 0 |

---

## Przepływ danych między etapami

```
[FiftyOne — Open Images v7]
       ↓  Data/<split>/data/*.jpg
       ↓  Data/<split>/labels/detections.csv

[--prepare — dataset.py]   filtracja + konwersja bbox → YOLO
       ↓  data/images/{train,val,test}/*.jpg
       ↓  data/labels/{train,val,test}/*.txt  (format: "0 cx cy w h")
       ↓  data/dataset.yaml

[--train — YOLOv8]
       ↓  data/runs/best_model.pt

[--detect — detector.py]
       ↓  detections.json  {image_id, bbox, confidence, crop}

[Etap 2 — OCR]          planowany
[Etap 3 — Klasyfikacja] planowany
[Etap 4 — TTS]          planowany
```