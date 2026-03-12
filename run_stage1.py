"""
run_stage1.py
==============================================================
ETAP 1 — Detekcja znaków drogowych z YOLOv8 + Open Images
Klasa: Traffic sign
==============================================================

Użycie:
  python run_stage1.py --check                     # sprawdź zależności i CUDA
  python run_stage1.py --prepare                   # konwertuj dane FiftyOne → YOLO
  python run_stage1.py --prepare --source data/    # nadpisz ścieżkę źródłową
  python run_stage1.py --train                     # trenuj model
  python run_stage1.py --prepare --train           # pełny pipeline
  python run_stage1.py --validate                  # walidacja modelu
  python run_stage1.py --detect --input obraz.jpg  # detekcja na obrazie
  python run_stage1.py --detect --input katalog/   # detekcja na katalogu
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from stage1.config   import load_config, setup_logging, ensure_directories, check_cuda
from stage1.dataset  import DatasetManager
from stage1.trainer  import Trainer
from stage1.detector import Detector


# ---------------------------------------------------------------
# Kroki pipeline
# ---------------------------------------------------------------

def step_check(config: dict, logger: logging.Logger) -> None:
    """Sprawdza zależności i środowisko CUDA."""
    logger.info("=== Sprawdzanie środowiska ===")

    deps = {
        "torch":       "pip install torch --index-url https://download.pytorch.org/whl/nightly/cu130",
        "ultralytics": "pip install ultralytics",
        "cv2":         "pip install opencv-python-headless",
        "yaml":        "pip install PyYAML",
        "numpy":       "pip install numpy",
    }
    for lib, cmd in deps.items():
        try:
            __import__(lib)
            logger.info("  ✓ %s", lib)
        except ImportError:
            logger.warning("  ✗ %s  →  %s", lib, cmd)

    try:
        import torch
        logger.info("")
        logger.info("CUDA:")
        logger.info("  PyTorch       : %s", torch.__version__)
        logger.info("  CUDA dostępna : %s", torch.cuda.is_available())
        if torch.cuda.is_available():
            logger.info("  Wersja CUDA   : %s", torch.version.cuda)
            for i in range(torch.cuda.device_count()):
                p = torch.cuda.get_device_properties(i)
                logger.info("  GPU %d          : %s (%.1f GB)", i, p.name, p.total_memory / 1e9)
    except ImportError:
        logger.warning("  PyTorch niezainstalowany.")

    logger.info("")
    logger.info("Konfiguracja:")
    logger.info("  Klasa        : Traffic sign")
    logger.info("  Źródło danych: %s", config["data"]["source_dir"])
    logger.info("  Architektura : %s", config["model"]["architecture"])
    logger.info("  Epoki        : %d", config["model"]["epochs"])
    logger.info("  Urządzenie   : %s", config["model"]["device"])
    logger.info("  AMP (FP16)   : %s", config["model"].get("amp", True))

    # Sprawdź czy source_dir istnieje
    src = Path(config["data"]["source_dir"])
    if src.exists():
        logger.info("")
        logger.info("Dane FiftyOne (%s):", src)
        for subdir in sorted(src.iterdir()):
            if subdir.is_dir():
                imgs = list((subdir / "data").glob("*.jpg")) if (subdir / "data").exists() else []
                csv  = (subdir / "labels" / "detections.csv").exists()
                logger.info("  %s/  →  %d obrazów, detections.csv: %s",
                            subdir.name, len(imgs), "✓" if csv else "✗")
    else:
        logger.warning("  Katalog źródłowy nie istnieje: %s", src)


def step_prepare(config: dict, args, logger: logging.Logger) -> Path:
    """Konwertuje dane FiftyOne (Open Images) do struktury YOLO."""
    logger.info("=== Przygotowanie datasetu ===")

    if args.source:
        config["data"]["source_dir"] = args.source
        logger.info("Źródło (CLI): %s", args.source)

    ensure_directories(config)

    manager = DatasetManager(config)
    yaml_path = manager.prepare()

    logger.info("Dataset gotowy: %s", yaml_path)
    return yaml_path


def step_train(config: dict, logger: logging.Logger) -> Path:
    """Trenuje model YOLOv8."""
    logger.info("=== Trening modelu ===")
    yaml_path = config["data"]["dataset_yaml"]

    if not Path(yaml_path).exists():
        logger.error("Brak dataset.yaml: %s\nUruchom najpierw: --prepare", yaml_path)
        sys.exit(1)

    trainer = Trainer(config)
    model_path = trainer.train(yaml_path)
    return model_path


def step_validate(config: dict, logger: logging.Logger) -> None:
    """Waliduje wytrenowany model."""
    logger.info("=== Walidacja modelu ===")
    trainer = Trainer(config)
    trainer.validate()


def step_detect(
    config: dict,
    input_path: str,
    output_dir: str,
    logger: logging.Logger,
) -> None:
    """Uruchamia detekcję na obrazie lub katalogu."""
    logger.info("=== Detekcja: %s ===", input_path)

    detector = Detector(config)
    detector.load_model()

    inp = Path(input_path)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if inp.is_dir():
        image_paths = sorted(inp.glob("*.jpg")) + sorted(inp.glob("*.png"))
        logger.info("Znaleziono %d obrazów.", len(image_paths))
    elif inp.is_file():
        image_paths = [inp]
    else:
        logger.error("Nie znaleziono: %s", input_path)
        sys.exit(1)

    import cv2
    all_detections = {}

    for img_path in image_paths:
        dets = detector.detect(img_path)
        all_detections[img_path.stem] = dets

        if dets:
            img = cv2.imread(str(img_path))
            detector.draw(img, dets, save_path=out / f"det_{img_path.name}")
        logger.info("%-30s: %d wykryć", img_path.name, len(dets))

    detector.export_json(all_detections, out / "detections.json")

    total = sum(len(v) for v in all_detections.values())
    logger.info("Gotowe: %d wykryć na %d obrazach → %s", total, len(image_paths), out)


# ---------------------------------------------------------------
# Wejście
# ---------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Road Sign Detector — Etap 1",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--config",   default="config/settings.yaml")
    parser.add_argument("--check",    action="store_true", help="Sprawdź zależności i CUDA")
    parser.add_argument("--prepare",  action="store_true", help="Konwertuj dane FiftyOne → YOLO")
    parser.add_argument("--train",    action="store_true", help="Trenuj YOLOv8")
    parser.add_argument("--validate", action="store_true", help="Waliduj model")
    parser.add_argument("--detect",   action="store_true", help="Wykryj znaki na obrazach")
    parser.add_argument("--input",    default="data/images/test",
                        help="Obraz lub katalog (dla --detect)")
    parser.add_argument("--output",   default="data/runs/detections",
                        help="Katalog wyników detekcji")
    parser.add_argument("--source",   default=None, metavar="KATALOG",
                        help="Ścieżka do danych FiftyOne (nadpisuje settings.yaml). "
                             "Przykład: --source C:/Users/Yami/.../data")

    args = parser.parse_args()

    if not any([args.check, args.prepare, args.train, args.validate, args.detect]):
        parser.print_help()
        sys.exit(0)

    config = load_config(args.config)
    logger = setup_logging(config)
    logger.info("Road Sign Detector v%s — Etap 1", config["project"]["version"])

    config = check_cuda(config)

    if args.check:   step_check(config, logger)
    if args.prepare: step_prepare(config, args, logger)
    if args.train:   step_train(config, logger)
    if args.validate: step_validate(config, logger)
    if args.detect:  step_detect(config, args.input, args.output, logger)

    logger.info("=== Zakończono ===")


if __name__ == "__main__":
    main()