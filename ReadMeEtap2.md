# Etap 2 — OCR: odczyt tekstu ze znaków drogowych

## Co zostało dodane

Dwa nowe pliki:

```
stage2/
├── __init__.py
└── ocr.py          ← OcrReader: wrapper EasyOCR

run_stage2.py       ← punkt wejścia CLI
```

Jedna zmiana w istniejącym pliku:

```
requirements.txt    ← odkomentowane: easyocr>=1.7.0
```

---

## Jak to działa

Etap 1 (YOLOv8) wykrywa znaki i wycina każdy z nich jako `crop` (numpy BGR).
Etap 2 przekazuje ten `crop` do EasyOCR i zwraca odczytany tekst lub `None`.

```
Detection.crop (numpy BGR)
    ↓  _preprocess()       skalowanie do min. 100px + CLAHE
    ↓  easyocr.readtext()  model OCR
    ↓  filtrowanie         conf >= 0.4, długość <= 20 znaków
str | None
```

Znaki bez tekstu (symboliczne: zakaz wyprzedzania, pierwszeństwo, itp.)
zwracają `None` i są pomijane w wynikach.

---

## Użycie

```bash
# Instalacja
pip install easyocr>=1.7.0

# Pojedynczy obraz
python run_stage2.py --input zdjecie.jpg

# Katalog
python run_stage2.py --input data/prepared/images/test/

# Z własnym katalogiem wyników
python run_stage2.py --input katalog/ --output logs/ocr

# Na CPU (gdy brak GPU lub problemy z pamięcią)
python run_stage2.py --input katalog/ --cpu
```

---

## Format wyjścia

Plik `logs/ocr/ocr_results.json`:

```json
{
  "00aa7126f62754dc": [
    {
      "bbox_xyxy": [123, 45, 287, 190],
      "det_conf":  0.8732,
      "text":      "30"
    }
  ],
  "0003bb040a62c86f": []
}
```

Klucz = nazwa pliku bez rozszerzenia. Pusta lista = znak bez tekstu.

---

## Parametry OCR (`stage2/ocr.py`)

Dwie stałe na górze pliku — jedyne co może wymagać dostosowania:

```python
_CONF_THRESHOLD = 0.4   # wyniki EasyOCR poniżej tej wartości są odrzucane
_MAX_TEXT_LENGTH = 20   # dłuższy tekst to prawdopodobnie szum
```

---

## Znane ograniczenia

**Znaki daleko od kamery** — crop < 40×40 px surowych pikseli daje słabe wyniki OCR.
Preprocessing skaluje do min. 100px, ale informacja raz utracona nie wraca.

**Znaki symboliczne** — EasyOCR może zwrócić przypadkowe litery z wzorów na znaku
(np. "C" z okrągłego znaku zakazu). Filtr confidence 0.4 eliminuje większość szumu,
ale nie wszystko. Jeśli wyniki zawierają dużo fałszywych odczytów, podnieś `_CONF_THRESHOLD`.

**Pierwsze uruchomienie** — EasyOCR pobiera modele (~100 MB) przy pierwszym użyciu.
Kolejne uruchomienia są szybkie (modele są cache'owane lokalnie).