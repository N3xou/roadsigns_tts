"""
dataset.py
Konwersja danych pobranych przez FiftyOne (Open Images v7) do formatu YOLO.

Oczekiwana struktura wejściowa (fiftyone export):
    raw/
    ├── train/
    │   ├── data/               ← obrazy .jpg
    │   ├── labels/
    │   │   └── detections.csv  ← adnotacje bbox
    │   └── metadata/
    │       ├── classes.csv
    │       ├── hierarchy.json
    │       └── image_ids.csv
    └── validation/             ← opcjonalnie, ta sama struktura

Format detections.csv (Open Images / fiftyone):
    ImageID, Source, LabelName, Confidence, XMin, XMax, YMin, YMax,
    IsOccluded, IsTruncated, IsGroupOf, IsDepiction, IsInside

Współrzędne XMin/XMax/YMin/YMax są znormalizowane [0, 1].
Konwersja do YOLO:
    cx = (XMin + XMax) / 2
    cy = (YMin + YMax) / 2
    w  = XMax - XMin
    h  = YMax - YMin
"""

import csv
import logging
import random
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

logger = logging.getLogger("road_sign_detector.dataset")

CLASS_NAME = "Traffic sign"
CLASS_MID  = "/m/01mqdt"
YOLO_INDEX = 0

# Kolumny detections.csv (Open Images — bez nagłówka)
# ImageID, Source, LabelName, Confidence, XMin, XMax, YMin, YMax,
# IsOccluded, IsTruncated, IsGroupOf, IsDepiction, IsInside,
# ...opcjonalne pola xclick...
_CSV_COLS = [
    "ImageID", "Source", "LabelName", "Confidence",
    "XMin", "XMax", "YMin", "YMax",
    "IsOccluded", "IsTruncated", "IsGroupOf", "IsDepiction", "IsInside",
]


class DatasetManager:
    """
    Czyta dane eksportowane przez FiftyOne i buduje strukturę YOLO.

    Przepływ:
    1. Wczytaj detections.csv — filtruj wiersze klasy "Traffic sign"
    2. Dopasuj każdy ImageID do pliku .jpg w katalogu data/
    3. Zapisz etykiety YOLO (.txt) do data/labels/raw/
    4. Skopiuj obrazy do data/images/raw/
    5. Podziel na train / val / test
    6. Wygeneruj dataset.yaml dla YOLOv8
    """

    def __init__(self, config: Dict):
        self.data_cfg     = config["data"]
        self.images_dir   = Path(self.data_cfg["images_dir"])
        self.labels_dir   = Path(self.data_cfg["labels_dir"])
        self.dataset_yaml = Path(self.data_cfg["dataset_yaml"])
        self.source_dir   = Path(self.data_cfg["source_dir"])

        self.train_ratio = self.data_cfg.get("train_split", 0.8)
        self.val_ratio   = self.data_cfg.get("val_split", 0.1)

    # ----------------------------------------------------------
    # Publiczne API
    # ----------------------------------------------------------

    def prepare(self) -> Path:
        """
        Konwertuje dane fiftyone → strukturę YOLO i generuje dataset.yaml.

        Returns:
            Ścieżka do wygenerowanego dataset.yaml.
        """
        logger.info("=== Przygotowanie datasetu z danych FiftyOne ===")
        logger.info("Źródło: %s", self.source_dir)

        # 1. Znajdź wszystkie pliki detections.csv i data/ w source_dir
        entries = self._find_source_entries()
        if not entries:
            raise FileNotFoundError(
                f"Brak danych w katalogu: {self.source_dir}\n"
                "Oczekiwana struktura:\n"
                "  <source_dir>/<split>/data/*.jpg\n"
                "  <source_dir>/<split>/labels/detections.csv"
            )

        # 2. Przetwórz każdy split (train / validation)
        all_image_paths: List[Path] = []
        for split_name, paths in entries.items():
            logger.info("Przetwarzam split: %s", split_name)
            imgs = self._process_split(
                images_src=paths["images_dir"],
                detections_csv=paths["detections_csv"],
            )
            all_image_paths.extend(imgs)
            logger.info("  → %d obrazów z klasą '%s'", len(imgs), CLASS_NAME)

        if not all_image_paths:
            raise ValueError(
                f"Brak adnotacji klasy '{CLASS_NAME}' w detections.csv.\n"
                "Sprawdź czy dane zostały pobrane z odpowiednią klasą."
            )

        logger.info("Łącznie: %d obrazów", len(all_image_paths))

        # 3. Podziel na train / val / test
        self._split_and_copy(all_image_paths)

        # 4. Wygeneruj dataset.yaml
        yaml_path = self._write_yaml()
        logger.info("=== Dataset gotowy: %s ===", yaml_path)
        return yaml_path

    def get_class_names(self) -> List[str]:
        return [CLASS_NAME]

    # ----------------------------------------------------------
    # Wykrywanie struktury źródłowej
    # ----------------------------------------------------------

    def _find_source_entries(self) -> Dict[str, Dict[str, Path]]:
        """
        Przeszukuje source_dir w poszukiwaniu podkatalogów ze strukturą fiftyone.

        Akceptuje:
          - <source_dir>/train/data/  + <source_dir>/train/labels/detections.csv
          - <source_dir>/validation/data/ + ...
          - <source_dir>/data/ + <source_dir>/labels/detections.csv  (płaska struktura)

        Returns:
            {"train": {"images_dir": Path, "detections_csv": Path}, ...}
        """
        entries: Dict[str, Dict[str, Path]] = {}

        # Przypadek 1: płaska struktura (data/ i labels/ bezpośrednio w source_dir)
        flat_csv = self.source_dir / "labels" / "detections.csv"
        flat_img = self.source_dir / "data"
        if flat_csv.exists() and flat_img.is_dir():
            entries["all"] = {"images_dir": flat_img, "detections_csv": flat_csv}
            logger.info("Struktura: płaska (%s)", self.source_dir)
            return entries

        # Przypadek 2: podkatalogi per split (train/, validation/, test/)
        for subdir in sorted(self.source_dir.iterdir()):
            if not subdir.is_dir():
                continue
            csv_path = subdir / "labels" / "detections.csv"
            img_dir  = subdir / "data"
            if csv_path.exists() and img_dir.is_dir():
                entries[subdir.name] = {"images_dir": img_dir, "detections_csv": csv_path}
                logger.info("Znaleziono split: %s", subdir.name)

        if not entries:
            logger.error(
                "Nie znaleziono struktury fiftyone w: %s\n"
                "Oczekiwane: <split>/data/*.jpg i <split>/labels/detections.csv",
                self.source_dir,
            )
        return entries

    # ----------------------------------------------------------
    # Przetwarzanie splitu
    # ----------------------------------------------------------

    def _process_split(
        self,
        images_src: Path,
        detections_csv: Path,
    ) -> List[Path]:
        """
        Dla jednego splitu:
        - wczytuje detections.csv
        - filtruje wiersze z LabelName == "Traffic sign"
        - kopiuje obrazy do data/images/raw/
        - zapisuje etykiety YOLO do data/labels/raw/

        Returns:
            Lista ścieżek skopiowanych obrazów (w raw/).
        """
        raw_img = self.images_dir / "raw"
        raw_lbl = self.labels_dir / "raw"
        raw_img.mkdir(parents=True, exist_ok=True)
        raw_lbl.mkdir(parents=True, exist_ok=True)

        # Wczytaj adnotacje z detections.csv
        annotations = self._parse_detections_csv(detections_csv)
        if not annotations:
            logger.warning("  Brak adnotacji klasy '%s' w: %s", CLASS_NAME, detections_csv)
            return []

        # Dopasuj image_id → plik .jpg w images_src
        image_files = self._index_images(images_src)

        saved: List[Path] = []
        missing = 0

        for image_id, boxes in annotations.items():
            # Szukaj pliku obrazu po image_id (bez rozszerzenia)
            img_src = image_files.get(image_id)
            if img_src is None:
                logger.debug("Brak pliku dla ImageID: %s", image_id)
                missing += 1
                continue

            # Docelowe ścieżki
            img_dst = raw_img / img_src.name
            lbl_dst = raw_lbl / img_src.with_suffix(".txt").name

            # Kopiuj obraz (pomijaj jeśli już istnieje)
            if not img_dst.exists():
                shutil.copy2(img_src, img_dst)

            # Zapisz etykiety YOLO
            self._write_yolo_label(lbl_dst, boxes)
            saved.append(img_dst)

        if missing:
            logger.warning("  Nie znaleziono pliku dla %d ImageID.", missing)

        return saved

    @staticmethod
    def _parse_detections_csv(csv_path: Path) -> Dict[str, List[Dict]]:
        """
        Parsuje detections.csv w formacie Open Images.

        Format: BEZ nagłówka, kolumny pozycyjne:
          0: ImageID
          1: Source
          2: LabelName  ← MID, np. /m/01mqdt
          3: Confidence
          4: XMin  5: XMax  6: YMin  7: YMax
          8+: IsOccluded, IsTruncated, ... (opcjonalne)

        Obsługuje też wariant Z nagłówkiem (wykrywa automatycznie).
        Filtruje wiersze z LabelName == CLASS_MID ("/m/01mqdt").
        """
        annotations: Dict[str, List[Dict]] = {}

        with open(csv_path, encoding="utf-8", newline="") as f:
            content = f.read().lstrip("\ufeff").strip()

        if not content:
            return annotations

        lines = content.splitlines()

        # Wykryj czy pierwszy wiersz to nagłówek
        first_col = lines[0].split(",")[0].strip().strip('"').lower()
        has_header = first_col == "imageid"

        if has_header:
            reader_rows = csv.DictReader(lines)
            if reader_rows.fieldnames:
                reader_rows.fieldnames = [n.strip().strip('"') for n in reader_rows.fieldnames]
            for row in reader_rows:
                row = {k.strip().strip('"'):v.strip().strip('"') for k,v in row.items()}
                if row.get("LabelName", "") != CLASS_MID:
                    continue
                image_id = row.get("ImageID", "")
                if not image_id:
                    continue
                try:
                    box = {
                        "xmin": float(row["XMin"]),
                        "xmax": float(row["XMax"]),
                        "ymin": float(row["YMin"]),
                        "ymax": float(row["YMax"]),
                    }
                except (KeyError, ValueError):
                    continue
                annotations.setdefault(image_id, []).append(box)
        else:
            # Bez nagłówka — indeksy pozycyjne
            for row in csv.reader(lines):
                if len(row) < 8:
                    continue
                if row[2].strip().strip('"') != CLASS_MID:
                    continue
                image_id = row[0].strip().strip('"')
                if not image_id:
                    continue
                try:
                    box = {
                        "xmin": float(row[4]),
                        "xmax": float(row[5]),
                        "ymin": float(row[6]),
                        "ymax": float(row[7]),
                    }
                except (IndexError, ValueError):
                    continue
                annotations.setdefault(image_id, []).append(box)

        logger.info(
            "  detections.csv: %d obrazów z '%s' (%s)",
            len(annotations), CLASS_NAME, CLASS_MID,
        )
        return annotations

    @staticmethod
    def _index_images(images_dir: Path) -> Dict[str, Path]:
        """
        Buduje mapowanie {image_id: Path} dla wszystkich .jpg i .png w katalogu.
        image_id = nazwa pliku bez rozszerzenia (zgodna z ImageID w CSV).
        """
        index: Dict[str, Path] = {}
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            for p in images_dir.glob(ext):
                index[p.stem] = p
        logger.info("  Obrazów w katalogu data/: %d", len(index))
        return index

    @staticmethod
    def _write_yolo_label(path: Path, boxes: List[Dict]) -> None:
        """
        Zapisuje plik .txt w formacie YOLO: <0> <cx> <cy> <w> <h>
        Indeks klasy zawsze 0 (jedyna klasa w projekcie).
        Współrzędne wejściowe: XMin/XMax/YMin/YMax znormalizowane [0, 1].
        """
        lines = []
        for b in boxes:
            cx = (b["xmin"] + b["xmax"]) / 2
            cy = (b["ymin"] + b["ymax"]) / 2
            w  =  b["xmax"] - b["xmin"]
            h  =  b["ymax"] - b["ymin"]
            cx, cy = max(0.0, min(1.0, cx)), max(0.0, min(1.0, cy))
            w,  h  = max(0.0, min(1.0, w)),  max(0.0, min(1.0, h))
            lines.append(f"{YOLO_INDEX} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
        path.write_text("\n".join(lines), encoding="utf-8")

    # ----------------------------------------------------------
    # Struktura YOLO
    # ----------------------------------------------------------

    def _split_and_copy(self, image_paths: List[Path]) -> None:
        """Losowo dzieli obrazy na train/val/test i kopiuje z etykietami."""
        paths = image_paths.copy()
        random.shuffle(paths)
        n       = len(paths)
        n_train = int(n * self.train_ratio)
        n_val   = int(n * self.val_ratio)

        splits = {
            "train": paths[:n_train],
            "val":   paths[n_train: n_train + n_val],
            "test":  paths[n_train + n_val:],
        }

        for split_name, imgs in splits.items():
            img_out = self.images_dir / split_name
            lbl_out = self.labels_dir / split_name
            img_out.mkdir(parents=True, exist_ok=True)
            lbl_out.mkdir(parents=True, exist_ok=True)

            copied_lbl = 0
            for img_path in imgs:
                shutil.copy2(img_path, img_out / img_path.name)
                lbl_src = self.labels_dir / "raw" / img_path.with_suffix(".txt").name
                if lbl_src.exists():
                    shutil.copy2(lbl_src, lbl_out / lbl_src.name)
                    copied_lbl += 1

            pct = len(imgs) / n * 100 if n else 0
            logger.info(
                "Split %-6s: %3d obrazów (%.0f%%), %d etykiet",
                split_name, len(imgs), pct, copied_lbl,
            )

    def _write_yaml(self) -> Path:
        """
        Generuje dataset.yaml dla YOLOv8 z jedną klasą.

        Struktura którą rozumie YOLOv8:
            path:  <absolutna ścieżka do prepared/>
            train: images/train
            val:   images/val
            test:  images/test

        YOLOv8 automatycznie szuka etykiet w:
            <path>/labels/train/  (podmienia 'images' → 'labels' w ścieżce)
        Dlatego 'path' musi wskazywać na prepared/, NIE na prepared/images/.
        """
        self.dataset_yaml.parent.mkdir(parents=True, exist_ok=True)

        # data_dir to katalog nadrzędny images/ i labels/
        # images_dir = prepared/images  →  data_dir = prepared/
        data_dir = self.images_dir.parent.resolve()

        data = {
            "path":  str(data_dir),
            "train": "images/train",
            "val":   "images/val",
            "test":  "images/test",
            "nc":    1,
            "names": [CLASS_NAME],
        }
        with open(self.dataset_yaml, "w", encoding="utf-8") as f:
            yaml.dump(data, f, allow_unicode=True, default_flow_style=False)
        logger.info(
            "dataset.yaml → %s  (path=%s, nc=1, klasa: '%s')",
            self.dataset_yaml, data_dir, CLASS_NAME,
        )
        return self.dataset_yaml