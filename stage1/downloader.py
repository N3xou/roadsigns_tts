"""
downloader.py
Pobieranie danych Traffic sign z Open Images v7 przez FiftyOne.

Uruchomienie:
    python downloader.py

Dane zapisywane są do raw/train/ i raw/validation/
zgodnie ze strukturą oczekiwaną przez --prepare.
"""

import fiftyone.zoo as foz

print("Pobieranie: train (1500 próbek)...")
foz.load_zoo_dataset(
    "open-images-v7",
    split="train",
    label_types=["detections"],
    classes=["Traffic sign"],
    max_samples=1500,
    dataset_dir="data/raw/train",
)

print("Pobieranie: validation (200 próbek)...")
foz.load_zoo_dataset(
    "open-images-v7",
    split="validation",
    label_types=["detections"],
    classes=["Traffic sign"],
    max_samples=200,
    dataset_dir="data/raw/validation",
)

print("Gotowe. Uruchom teraz:")
print("  python run_stage1.py --prepare")