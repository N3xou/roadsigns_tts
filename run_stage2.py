"""
run_stage2.py
==============================================================
ETAP 2 — Odczyt tekstu ze znaków drogowych (OCR)
EasyOCR na cropach z Etapu 1 (YOLOv8)
==============================================================

Użycie:
  python run_stage2.py --input zdjecie.jpg
  python run_stage2.py --input katalog/
  python run_stage2.py --input katalog/ --output logs/ocr
  python run_stage2.py --input zdjecie.jpg --cpu
"""
import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from stage1.config   import load_config, setup_logging, check_cuda
from stage1.detector import Detector
from stage2.ocr      import OcrReader


def run(config: dict, input_path: Path, output_dir: Path, use_gpu: bool) -> None:
    logger = logging.getLogger("road_sign_detector")

    detector = Detector(config)
    detector.load_model()
    reader = OcrReader(gpu=use_gpu)

    images = [input_path] if input_path.is_file() else (
        sorted(input_path.glob("*.jpg")) + sorted(input_path.glob("*.png"))
    )

    if not images:
        logger.error("Brak obrazów w: %s", input_path)
        sys.exit(1)

    logger.info("Przetwarzanie %d obrazów...", len(images))
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for img_path in images:
        detections = detector.detect(img_path)
        ocr_results = []

        for det in detections:
            text = reader.read(det.crop)
            if text is None:
                logger.debug("%s: bbox=%s — brak tekstu", img_path.name, det.bbox_xyxy)
                continue

            logger.info("%-30s bbox=%-25s text=%r", img_path.name, str(det.bbox_xyxy), text)
            ocr_results.append({
                "bbox_xyxy":  list(det.bbox_xyxy),
                "det_conf":   round(det.confidence, 4),
                "text":       text,
            })

        results[img_path.stem] = ocr_results

    out_file = output_dir / "ocr_results.json"
    out_file.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Wyniki → %s", out_file)

    total = sum(len(v) for v in results.values())
    signs_with_text = sum(1 for v in results.values() if v)
    logger.info("Łącznie: %d/%d obrazów z tekstem, %d odczytów",
                signs_with_text, len(images), total)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Road Sign Detector — Etap 2 (OCR)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--input",  required=True, help="Obraz .jpg/.png lub katalog")
    parser.add_argument("--output", default="logs/ocr", help="Katalog wyników (domyślnie: logs/ocr)")
    parser.add_argument("--config", default="config/settings.yaml")
    parser.add_argument("--cpu",    action="store_true", help="Wymuś CPU dla EasyOCR")
    args = parser.parse_args()

    config = load_config(args.config)
    setup_logging(config)

    if args.cpu:
        config["model"]["device"] = "cpu"
        config["model"]["amp"]    = False

    config = check_cuda(config)

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Błąd: nie znaleziono: {input_path}", file=sys.stderr)
        sys.exit(1)

    use_gpu = config["model"]["device"] != "cpu"
    run(config, input_path, Path(args.output), use_gpu)


if __name__ == "__main__":
    main()