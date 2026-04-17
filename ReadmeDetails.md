# Road Sign Detector — Dokumentacja szczegółowa

---

## Spis treści

1. [--prepare: konwersja danych](#1---prepare-konwersja-danych)
2. [--train: trening modelu](#2---train-trening-modelu)
3. [settings.yaml i config.py](#3-settingsyaml-i-configpy)
4. [Jak działa detektor](#4-jak-działa-detektor)

---

## 1. `--prepare`: konwersja danych

### Dlaczego jest konieczna

YOLOv8 wymaga bardzo konkretnego formatu danych — innego niż ten, który dostarcza FiftyOne.
FiftyOne zapisuje adnotacje w jednym pliku CSV na cały split, z identyfikatorami MID zamiast nazw klas,
ze współrzędnymi w formacie `XMin/XMax/YMin/YMax`. YOLOv8 oczekuje osobnego pliku `.txt` dla każdego
obrazu, z numerycznym indeksem klasy i współrzędnymi środka `cx cy w h`. Krok `--prepare` jest mostem
między tymi dwoma formatami.

### Co dokładnie się dzieje (krok po kroku)

**Krok 1 — wykrycie struktury źródłowej** (`_find_source_entries`)

Program przeszukuje katalog `raw/` w poszukiwaniu podkatalogów ze strukturą FiftyOne. Akceptuje dwa warianty:

```
raw/train/data/*.jpg          ← podkatalogi per split (typowe)
raw/train/labels/detections.csv

raw/data/*.jpg                ← płaska struktura (jeden split)
raw/labels/detections.csv
```

**Krok 2 — parsowanie detections.csv** (`_parse_detections_csv`)

Plik CSV z Open Images ma specyficzny format — może, ale nie musi, mieć nagłówek.
Program wykrywa to automatycznie sprawdzając czy pierwsza kolumna pierwszego wiersza to `imageid`.

Format bez nagłówka (typowy):
```
000abc123, xclick, /m/01mqdt, 1, 0.12, 0.45, 0.30, 0.80, 0, 1, 0, 0, 0
^          ^       ^           ^  ^     ^     ^     ^
ImageID    Source  LabelName   Conf XMin  XMax  YMin  YMax
           (kol.0) (kol.2)        (kol.4)(kol.5)(kol.6)(kol.7)
```

Kluczowy szczegół: kolumna `LabelName` zawiera **MID** (Machine-generated ID z bazy wiedzy Google),
nie nazwę tekstową. Klasa "Traffic sign" ma MID `/m/01mqdt`. Program filtruje wyłącznie wiersze
z tym identyfikatorem i ignoruje wszystkie inne klasy obecne w pliku.

**Krok 3 — konwersja formatu bbox**

Open Images używa współrzędnych bezwzględnych znormalizowanych `[0, 1]` w formacie
`XMin, XMax, YMin, YMax` (lewý górny + prawy dolny róg). YOLOv8 wymaga środka i rozmiaru.

```
Wejście (Open Images):          Wyjście (YOLO):
XMin=0.12  XMax=0.45            cx = (0.12 + 0.45) / 2 = 0.285
YMin=0.30  YMax=0.80            cy = (0.30 + 0.80) / 2 = 0.550
                                w  =  0.45 - 0.12       = 0.330
                                h  =  0.80 - 0.30       = 0.500
```

Wynikowy plik `.txt` (jedna linia = jeden obiekt):
```
0 0.285000 0.550000 0.330000 0.500000
^  ^        ^        ^        ^
class_id  cx       cy       w        h
```

Indeks klasy jest zawsze `0` — projekt ma tylko jedną klasę.
Wszystkie wartości są przycinane do zakresu `[0, 1]` żeby zapobiec błędom przy bbox wychodzących poza krawędź obrazu.

**Krok 4 — kopiowanie obrazów i etykiet** (`_process_split`)

Dla każdego `ImageID` który ma adnotację klasy Traffic sign program:
- dopasowuje `ImageID` do pliku `.jpg` po nazwie (bez rozszerzenia)
- kopiuje obraz do `prepared/images/raw/`
- zapisuje etykietę YOLO do `prepared/labels/raw/`

Obrazy bez adnotacji Traffic sign są pomijane — nie kopiuje ich, bo przy treningu puste etykiety
(obrazy bez obiektów) muszą być świadomie dodane, a nie przypadkowo wrzucone.

**Krok 5 — podział na train/val/test** (`_split_and_copy`)

Wszystkie zebrane obrazy (z obu splitów FiftyOne: `train/` i `validation/`) są mieszane i losowo
dzielone według proporcji z `settings.yaml`:

```
train_split: 0.8   →  80% obrazów  (trening)
val_split:   0.1   →  10% obrazów  (walidacja w trakcie treningu)
test_split:  0.1   →  10% obrazów  (końcowa ocena)
```

Podział jest losowy przy każdym `--prepare`. Jeśli chcesz deterministyczny podział,
możesz dodać `random.seed(42)` na początku `_split_and_copy`.

**Krok 6 — generowanie dataset.yaml** (`_write_yaml`)

YOLOv8 identyfikuje dataset przez plik YAML. Krytyczny szczegół: pole `path` musi wskazywać
na katalog nadrzędny wobec `images/` i `labels/`, nie na sam katalog `images/`.
YOLOv8 automatycznie konstruuje ścieżkę do etykiet przez podmianę `images → labels` w ścieżce
do obrazów. Jeśli `path` kończy się na `images/`, substytucja szuka `images/labels/` zamiast `labels/`.

```yaml
# POPRAWNIE — path wskazuje na prepared/
path:  C:\projekt\prepared
train: images/train       →  C:\projekt\prepared\images\train  (obrazy)
                          →  C:\projekt\prepared\labels\train  (etykiety, auto)
val:   images/val
test:  images/test
nc:    1
names: [Traffic sign]

# BŁĘDNIE — path wskazuje na prepared/images/
path:  C:\projekt\prepared\images
train: train              →  C:\projekt\prepared\images\train  (obrazy OK)
                          →  C:\projekt\prepared\images\labels\train  (etykiety ŹLE)
```

### Wynik

```
prepared/
├── images/
│   ├── raw/        ← wszystkie obrazy po filtracji
│   ├── train/      ← 80% losowo wybranych
│   ├── val/        ← 10%
│   └── test/       ← 10%
├── labels/
│   ├── raw/        ← wszystkie etykiety .txt
│   ├── train/
│   ├── val/
│   └── test/
└── dataset.yaml
```

---

## 2. `--train`: trening modelu

### Architektura YOLOv8n

Projekt używa `yolov8n` (nano) — najmniejszego wariantu rodziny YOLOv8.
Wybór jest celowy: szybki trening, mało pamięci GPU, wystarczający dla jednej prostej klasy.

```
Wejście: obraz 640×640 px (RGB)
    ↓
Backbone (CSPDarknet)
    Warstwa 1–9: ekstrakcja cech niskiego poziomu (krawędzie, tekstury)
    Warstwa 10–20: cechy wysokiego poziomu (kształty, wzorce)
    ↓
Neck (PANet — Path Aggregation Network)
    Łączy cechy z różnych skali (multi-scale fusion)
    Wykrywa obiekty małe (8×8 px) i duże (640×640 px)
    ↓
Head (Detection Head)
    3 skale detekcji: 80×80, 40×40, 20×20 siatek
    Dla każdej komórki siatki: bbox + objectness + class_prob
    ↓
Wyjście: lista bbox [x, y, w, h, conf, class]
```

Parametry `yolov8n.pt` (pretrenowany na COCO):
- Liczba warstw: ~168
- Parametry: ~3.2 mln
- Rozmiar pliku: ~6 MB
- Szybkość: ~80 FPS na GPU RTX 3080

### Transfer learning

Trening zaczyna od wag `yolov8n.pt` pretrenowanych na datasecie COCO (80 klas, 118k obrazów).
Nie uczymy modelu od zera — to byłoby zbyt powolne i wymagałoby setek tysięcy obrazów.
Zamiast tego "dostrajamy" istniejący model do nowej klasy Traffic sign, co wymaga tylko ~500 obrazów.

```
yolov8n.pt (COCO, 80 klas)
    ↓  Fine-tuning
best.pt (Traffic sign, 1 klasa)
```

### Co się dzieje podczas każdej epoki

Jedna epoka = przejście przez cały zbiór treningowy raz.

```
Dla każdego batcha (domyślnie 16 obrazów):
    1. Forward pass  — obraz przez sieć → przewidywane bbox
    2. Loss          — porównanie z etykietami YOLO (.txt)
       box_loss:  błąd lokalizacji bbox (CIoU)
       cls_loss:  błąd klasyfikacji (BCE)
       dfl_loss:  Distribution Focal Loss (dokładność krawędzi)
    3. Backward pass — gradient propaguje wstecz
    4. Optimizer     — wagi sieci aktualizowane (AdamW)

Po epoce:
    5. Walidacja     — mAP50 i mAP50-95 na prepared/images/val/
    6. Early stopping — jeśli mAP50 nie poprawia się przez `patience` epok → stop
    7. Checkpoint    — zapisuje weights/last.pt, nadpisuje weights/best.pt jeśli mAP50 > dotychczasowe
```

### Parametry treningu z settings.yaml

```
epochs:   50    — maksymalna liczba epok
batch:    16    — obrazów na GPU jednocześnie (auto-dobierany do VRAM)
imgsz:   640    — rozmiar wejściowy (obrazy skalowane do 640×640)
patience: 10    — early stopping po 10 epokach bez poprawy
amp:     true   — Automatic Mixed Precision: obliczenia FP16 zamiast FP32 (2× szybciej)
workers:  8     — wątki ładowania danych
```

### AMP (Automatic Mixed Precision)

Przy `amp: true` model używa 16-bitowych liczb zmiennoprzecinkowych (FP16) do obliczeń,
ale zachowuje FP32 dla gradientów. Efekt: 2× szybszy trening, ~40% mniej VRAM.
Jest automatycznie wyłączane na CPU (CPU nie ma akceleracji FP16).

### Pliki wyjściowe treningu

```
runs/road_signs/
├── weights/
│   ├── best.pt      ← model z najwyższym mAP50 (używany do detekcji)
│   └── last.pt      ← model z ostatniej epoki
├── results.csv      ← metryki per epoka (box_loss, mAP50, mAP50-95, precision, recall)
├── args.yaml        ← wszystkie parametry użyte do treningu (do reprodukcji)
├── confusion_matrix.png   ← macierz pomyłek na zbiorze val
├── BoxP_curve.png   ← Precision w funkcji progu conf
├── BoxR_curve.png   ← Recall w funkcji progu conf
├── BoxF1_curve.png  ← F1 score — wskazuje optymalny próg conf
├── train_batch0.jpg ← wizualizacja pierwszego batcha z etykietami (diagnostyka)
└── val_batch0_pred.jpg   ← predykcje na zbiorze val
```

### Interpretacja metryk

```
mAP50       — mean Average Precision przy IoU=0.50
              > 0.5  dobry model
              > 0.7  bardzo dobry
              < 0.3  model się nie nauczył — sprawdź dane

mAP50-95    — mAP uśrednione dla IoU od 0.50 do 0.95 (co 0.05)
              surowa miara precyzji lokalizacji bbox
              zwykle ~50-60% wartości mAP50

box_loss    — błąd lokalizacji bbox
              powinien maleć z epoką (0.05-0.10 = dobry zakres końcowy)
              ~0.000 = model trenował na pustych etykietach (błąd danych)
              stały bez zmian = model nie uczy się (problem z LR lub danymi)

precision   — ile wykryć faktycznie było znakami (mało fałszywych alarmów)
recall      — ile znaków zostało wykrytych (mało pominięć)
```

### Jak działa _find_best_pt

Po zakończeniu treningu `trainer.py` szuka `best.pt` w trzech miejscach, po kolei:

1. `results.save_dir / "weights" / "best.pt"` — ścieżka zwrócona bezpośrednio przez YOLOv8
2. `runs/road_signs/weights/best.pt` — oczekiwana ścieżka
3. `runs/***/best.pt` — przeszukanie rekursywne (fallback gdy YOLOv8 zmieni strukturę katalogów)

---

## 3. `settings.yaml` i `config.py`

### settings.yaml — co konfiguruje

`settings.yaml` to jedyne miejsce gdzie zmieniasz parametry projektu.
Nie dotykasz kodu Python — tylko ten plik.

```yaml
data:
  source_dir:   "raw"       # gdzie leżą dane FiftyOne (raw/train/, raw/validation/)
  data_dir:     "prepared"  # gdzie generować dane YOLO
  images_dir:   "prepared/images"
  labels_dir:   "prepared/labels"
  dataset_yaml: "prepared/dataset.yaml"
  train_split:  0.8         # proporcje podziału (muszą sumować się do 1.0)
  val_split:    0.1
  test_split:   0.1

model:
  architecture: "yolov8n"   # n=najszybszy, s, m, l, x=najdokładniejszy
  pretrained:   true        # zacznij od wag COCO (zalecane)
  device:       "0"         # "0"=GPU 0, "cpu"=procesor, "0,1"=multi-GPU
  cuda_version: "13.0"      # wymagana wersja CUDA (sprawdzana przy starcie)
  imgsz:        640         # rozmiar wejściowy (640 = standard YOLOv8)
  epochs:       50
  batch:        16          # auto-zmniejszany gdy za mało VRAM
  patience:     10          # early stopping
  workers:      8           # wątki DataLoader
  amp:          true        # FP16 (tylko GPU)
  output_dir:   "runs"      # katalog wyników YOLOv8
  model_path:   "runs/road_signs/weights/best.pt"  # ścieżka do detekcji/walidacji

inference:
  conf:         0.25        # próg confidence (detekcje poniżej są odrzucane)
  iou:          0.45        # próg IoU dla NMS (Non-Maximum Suppression)
  line_width:   2           # grubość bbox na wizualizacji

logging:
  level:    "INFO"
  log_file: "logs/training.log"
```

### Kiedy zmieniać które parametry

```
Chcesz szybszy trening:        architecture: "yolov8n"  (już jest)
Chcesz dokładniejszy model:    architecture: "yolov8s" lub "yolov8m"
Mało VRAM (< 4 GB):            batch: 4, workers: 2
Dużo fałszywych alarmów:       conf: 0.40  (podnieś próg)
Model pomija znaki:             conf: 0.10  (obniż próg)
Zduplikowane detekcje:         iou: 0.30   (ostrzejszy NMS)
```

### config.py — co robi

`config.py` to warstwa inicjalizacyjna uruchamiana raz przy starcie. Nie konfiguruje się go ręcznie.

**`load_config()`** — wczytuje `settings.yaml` i waliduje:
- czy wszystkie wymagane sekcje istnieją (`project`, `data`, `model`, `inference`)
- czy `train_split + val_split + test_split == 1.0` (tolerancja `1e-6`)
- rzuca `ValueError` jeśli coś nie gra

**`setup_logging()`** — konfiguruje dwa handlery jednocześnie:
- `StreamHandler` → konsola (widoczne na bieżąco)
- `FileHandler` → `logs/training.log` (pełna historia)

Format logu: `2026-03-12 22:53:29 | INFO | road_sign_detector.trainer | Trening zakończony`

**`check_cuda()`** — sprawdza środowisko GPU i **modyfikuje config** jeśli potrzeba:

```
GPU dostępne + CUDA >= wymagana:
    → config bez zmian, trening na GPU z AMP

GPU niedostępne:
    → device = "cpu", amp = False, batch = 8, workers = 2

CUDA zbyt stara (< wymagana):
    → sys.exit(1) z instrukcją jak zaktualizować

CUDA nowsza niż wymagana:
    → ostrzeżenie, ale kontynuuje (nowsza CUDA jest wstecznie kompatybilna)
```

**`_recommended_batch()`** — heurystyka doboru batcha do VRAM:

```
≥ 24 GB VRAM  →  batch 64
≥ 16 GB       →  batch 32
≥  8 GB       →  batch 16   (GTX 1070 Ti = 8.6 GB → batch 16)
≥  4 GB       →  batch 8
<  4 GB       →  batch 4
```

Wartość jest skalowana przez `(imgsz / 640)²` — przy `imgsz=1280` batch jest 4× mniejszy.

**`ensure_directories()`** — tworzy strukturę katalogów jeśli nie istnieje.
Bezpieczne do wielokrotnego wywołania (`exist_ok=True`).

---

## 4. Jak działa detektor

### Ogólny przepływ

```
obraz (plik .jpg lub numpy array)
    ↓  cv2.imread()
    ↓  model.predict()   ← YOLOv8 inference
    ↓  parsowanie wyników
Lista obiektów Detection
    ↓  draw()            ← wizualizacja (opcjonalnie)
    ↓  export_json()     ← zapis do JSON (wejście dla Etapu 2 OCR)
```

### Dataclass Detection

Każde wykrycie jest reprezentowane jako obiekt `Detection`:

```python
@dataclass
class Detection:
    image_id:       str    # nazwa pliku bez rozszerzenia, np. "00aa7126f62754dc"
    class_id:       int    # zawsze 0 (jedyna klasa: Traffic sign)
    class_name:     str    # "Traffic sign"
    confidence:     float  # pewność modelu, np. 0.8732
    bbox_xyxy:      Tuple[int, int, int, int]          # piksele: (x1, y1, x2, y2)
    bbox_xywh_norm: Tuple[float, float, float, float]  # znormalizowane: (cx, cy, w, h)
    crop:           np.ndarray | None   # wycięty ROI — wejście dla Etapu 2 (OCR)
```

Dwa formaty bbox są przechowywane jednocześnie:
- `bbox_xyxy` — piksele, używane do rysowania i przycinania
- `bbox_xywh_norm` — znormalizowane `[0,1]`, używane w JSON (niezależne od rozdzielczości)

### detect() — szczegóły inferencji

```python
results = self.model.predict(
    source=img,     # numpy array BGR (format OpenCV)
    conf=0.25,      # odrzuć detekcje z confidence < 0.25
    iou=0.45,       # NMS: odrzuć nakładające się bbox z IoU > 0.45
    device="0",     # GPU
    verbose=False,  # wyłącz logi YOLOv8
)
```

**NMS (Non-Maximum Suppression)** — YOLOv8 generuje wiele nakładających się bbox dla tego samego obiektu.
NMS zachowuje tylko ten z najwyższym confidence, a resztę usuwa. Próg `iou=0.45` oznacza:
jeśli dwa bbox nakładają się w ponad 45%, odrzuć słabszy.

Po detekcji każdy box jest konwertowany do formatu `Detection`:
```
box.xyxy  → (x1, y1, x2, y2) w pikselach, przycięte do granic obrazu
box.conf  → float confidence
box.cls   → int class_id

crop = img[y1:y2, x1:x2].copy()   ← wycięty fragment obrazu dla OCR
```

### conf_override i diagnose()

Normalnie `detect()` używa `conf` z `settings.yaml`. Opcjonalny parametr `conf_override`
pozwala to nadpisać bez edytowania pliku — używane przez `--diagnose` i `--conf`:

```
--detect                      →  conf = 0.25 (z settings.yaml)
--detect --conf 0.05          →  conf = 0.05
--diagnose                    →  detect() z conf_override=0.01 (wszystko co model widzi)
```

`diagnose()` uruchamia detekcję z `conf=0.01` i loguje tabelę:
```
conf      próg?   bbox
0.4832    ✓ OK    (123, 45, 287, 190)
0.1204    ✗ < 0.25  (560, 230, 620, 310)   ← widoczne tylko w --diagnose
0.0341    ✗ < 0.25  (12, 400, 89, 480)
```

Jeśli żadna detekcja nie przekracza progu, sugeruje optymalną wartość `--conf`.
Jeśli model nie widzi nic nawet przy `conf=0.01`, problem leży w treningu (nie w progu).

### export_json() — format wyjściowy dla Etapu 2

```json
{
  "00aa7126f62754dc": [
    {
      "image_id":       "00aa7126f62754dc",
      "class_id":       0,
      "class_name":     "Traffic sign",
      "confidence":     0.8732,
      "bbox_xyxy":      [123, 45, 287, 190],
      "bbox_xywh_norm": [0.253906, 0.296875, 0.256250, 0.447917]
    }
  ]
}
```

Pole `crop` (numpy array) jest celowo pomijane w JSON — tablice numpy nie są serializowalne
i ich rozmiar byłby zbyt duży. Etap 2 (OCR) będzie wycinał ROI samodzielnie na podstawie `bbox_xyxy`.

### detect_batch() — przetwarzanie wielu obrazów

Przetwarza obrazy jeden po jednym (nie równolegle). Dla każdego obrazu wywołuje `detect()`
i zbiera wyniki w słowniku `{image_id: [Detection, ...]}`. Co 10 obrazów loguje postęp.
Błędy przy pojedynczym obrazie (np. uszkodzony plik) są łapane i logowane — nie przerywają całego batcha.
