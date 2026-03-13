"""
trainer.py
Trening i walidacja modelu YOLOv8 na danych Open Images.
Obsługa CUDA 13.0 z Automatic Mixed Precision (AMP/FP16).
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger("road_sign_detector.trainer")


class Trainer:
    """
    Opakowuje trening YOLOv8 z parametrami z pliku konfiguracyjnego.
    Obsługuje CUDA 13.0 z AMP (FP16) dla przyspieszenia treningu na GPU.

    Wymaga: pip install ultralytics
            torch nightly z CUDA 13.0 (patrz requirements.txt)
    """

    def __init__(self, config: Dict):
        self.config    = config
        self.model_cfg = config["model"]
        self.data_cfg  = config["data"]
        self.model     = None

    def train(self, dataset_yaml: Optional[str] = None) -> Path:
        """
        Uruchamia trening YOLOv8.

        Args:
            dataset_yaml: Ścieżka do dataset.yaml (domyślnie z konfiguracji)

        Returns:
            Ścieżka do najlepszego modelu (best.pt)
        """
        from ultralytics import YOLO

        yaml_path  = dataset_yaml or self.data_cfg["dataset_yaml"]
        arch       = self.model_cfg["architecture"]
        model_name = f"{arch}.pt" if self.model_cfg.get("pretrained", True) else f"{arch}.yaml"
        device     = self.model_cfg["device"]
        use_amp    = self.model_cfg.get("amp", True) and device != "cpu"

        # Użyj ścieżki bezwzględnej — YOLOv8 ignoruje relative project i dokłada runs/detect/
        project_dir = Path(self.model_cfg["output_dir"]).resolve()

        logger.info("=== Trening YOLOv8 ===")
        logger.info("Model: %s | Epoki: %d | Urządzenie: %s | AMP: %s",
                    model_name, self.model_cfg["epochs"], device, use_amp)
        logger.info("Wyniki → %s/road_signs/", project_dir)

        self._log_gpu_info()

        self.model = YOLO(model_name)

        start = time.time()
        results = self.model.train(
            data=str(Path(yaml_path).resolve()),
            epochs=self.model_cfg["epochs"],
            imgsz=self.model_cfg["imgsz"],
            batch=self.model_cfg["batch"],
            device=device,
            patience=self.model_cfg["patience"],
            workers=self.model_cfg["workers"],
            amp=use_amp,
            project=str(project_dir),
            name="road_signs",
            exist_ok=True,
            save=True,
            save_period=10,
            verbose=True,
        )

        elapsed = (time.time() - start) / 60
        logger.info("Trening zakończony w %.1f min.", elapsed)

        # Odczytaj faktyczną ścieżkę bezpośrednio z wyników YOLOv8
        best_pt = self._find_best_pt(results, project_dir)

        if best_pt and best_pt.exists():
            logger.info("Najlepszy model → %s", best_pt)
            return best_pt

        logger.warning("best.pt nie znaleziony w: %s", project_dir)
        return project_dir / "road_signs" / "weights" / "best.pt"

    @staticmethod
    def _find_best_pt(results, project_dir: Path) -> Optional[Path]:
        """
        Próbuje znaleźć best.pt w kolejności priorytetów:
        1. results.save_dir (bezpośrednio z YOLOv8 — najbardziej wiarygodne)
        2. project_dir/road_signs/weights/best.pt
        3. Rekursywne przeszukanie project_dir
        """
        # 1. Z obiektu results (YOLOv8 >= 8.0)
        try:
            save_dir = Path(results.save_dir)
            candidate = save_dir / "weights" / "best.pt"
            if candidate.exists():
                logger.info("save_dir z results: %s", save_dir)
                return candidate
        except AttributeError:
            pass

        # 2. Oczekiwana ścieżka
        candidate = project_dir / "road_signs" / "weights" / "best.pt"
        if candidate.exists():
            return candidate

        # 3. Przeszukaj katalog projektu rekursywnie
        found = list(project_dir.rglob("best.pt"))
        if found:
            logger.info("Znaleziono best.pt: %s", found[0])
            return found[0]

        return None

    def validate(self, model_path: Optional[str] = None) -> Dict:
        """
        Waliduje model na zbiorze walidacyjnym.

        Returns:
            Słownik z metrykami mAP50, mAP50-95, precision, recall
        """
        from ultralytics import YOLO

        path   = model_path or self.model_cfg.get("model_path") or self.model_cfg.get("model_save_path")
        device = self.model_cfg["device"]
        model  = YOLO(path)

        logger.info("Walidacja: %s | device: %s", path, device)
        metrics = model.val(
            data=self.data_cfg["dataset_yaml"],
            device=device,
            imgsz=self.model_cfg["imgsz"],
        )

        results = {
            "mAP50":     round(float(metrics.box.map50), 4),
            "mAP50_95":  round(float(metrics.box.map),   4),
            "precision": round(float(metrics.box.mp),     4),
            "recall":    round(float(metrics.box.mr),     4),
        }

        logger.info("Wyniki walidacji:")
        for k, v in results.items():
            logger.info("  %-12s: %.4f", k, v)

        out_path = Path("logs") / "metrics.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, indent=2))
        logger.info("Metryki zapisane: %s", out_path)

        return results

    # ----------------------------------------------------------
    # Pomocnicze
    # ----------------------------------------------------------

    @staticmethod
    def _log_gpu_info() -> None:
        """Loguje informacje o GPU przed treningiem."""
        try:
            import torch
            if not torch.cuda.is_available():
                logger.info("GPU niedostępne — trening na CPU.")
                return
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                mem   = props.total_memory / 1e9
                logger.info(
                    "GPU %d: %s | VRAM: %.1f GB | CUDA: %s | SM: %d.%d",
                    i, props.name, mem, torch.version.cuda,
                    props.major, props.minor,
                )
        except Exception as e:
            logger.debug("Nie można odczytać informacji o GPU: %s", e)