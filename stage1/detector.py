"""
detector.py
Detekcja znaków drogowych na obrazach i wideo.
Wyjście (lista Detection) stanowi wejście dla Etapu 2 (OCR).
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from ultralytics import YOLO
import cv2
import numpy as np

logger = logging.getLogger("road_sign_detector.detector")


@dataclass
class Detection:
    """
    Wynik detekcji pojedynczego znaku.
    Pole `crop` (wycięty ROI) jest wejściem dla Etapu 2 — OCR.
    """
    image_id:        str
    class_id:        int
    class_name:      str
    confidence:      float
    bbox_xyxy:       Tuple[int, int, int, int]
    bbox_xywh_norm:  Tuple[float, float, float, float]
    crop:            Optional[np.ndarray] = field(default=None, repr=False)

    @property
    def area(self) -> int:
        x1, y1, x2, y2 = self.bbox_xyxy
        return (x2 - x1) * (y2 - y1)

    def to_dict(self) -> Dict:
        """Serializacja do słownika (bez tablicy numpy)."""
        return {
            "image_id":       self.image_id,
            "class_id":       self.class_id,
            "class_name":     self.class_name,
            "confidence":     round(self.confidence, 4),
            "bbox_xyxy":      list(self.bbox_xyxy),
            "bbox_xywh_norm": [round(v, 6) for v in self.bbox_xywh_norm],
        }


class Detector:
    """
    Detektor znaków drogowych oparty na YOLOv8.
    Wyniki (lista Detection z polem crop) są przekazywane do kolejnych etapów.
    """

    def __init__(self, config: Dict, model_path: Optional[str] = None):
        self.config     = config
        self.model_cfg  = config["model"]
        self.inf_cfg    = config["inference"]
        self.model_path = model_path or self.model_cfg.get("model_path")
        self.model      = None
        self.class_names: List[str] = []

    def load_model(self) -> None:
        """Wczytuje model YOLOv8 z dysku."""

        if not Path(self.model_path).exists():
            raise FileNotFoundError(
                f"Model nie znaleziony: {self.model_path}\n"
                "Uruchom trening: python run_stage1.py --train"
            )
        self.model = YOLO(self.model_path)
        self.class_names = list(self.model.names.values())
        logger.info("Model wczytany: %s | Klasy: %s", self.model_path, self.class_names)

    def detect(
        self,
        image: Union[str, Path, np.ndarray],
        image_id: Optional[str] = None,
    ) -> List[Detection]:
        """
        Wykrywa znaki na pojedynczym obrazie.

        Args:
            image:    Ścieżka do pliku lub tablica numpy BGR
            image_id: Opcjonalny identyfikator (domyślnie: nazwa pliku)

        Returns:
            Lista obiektów Detection
        """
        self._ensure_loaded()

        if isinstance(image, (str, Path)):
            p = Path(image)
            img = cv2.imread(str(p))
            if img is None:
                raise ValueError(f"Nie można wczytać obrazu: {p}")
            image_id = image_id or p.stem
        else:
            img = image
            image_id = image_id or "frame"

        h, w = img.shape[:2]

        results = self.model.predict(
            source=img,
            conf=self.inf_cfg.get("conf", 0.25),
            iou=self.inf_cfg.get("iou", 0.45),
            device=self.model_cfg["device"],
            verbose=False,
        )

        detections: List[Detection] = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                cls_id        = int(box.cls[0])
                conf          = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                cx = ((x1 + x2) / 2) / w
                cy = ((y1 + y2) / 2) / h
                bw = (x2 - x1) / w
                bh = (y2 - y1) / h

                crop = img[y1:y2, x1:x2].copy() if x2 > x1 and y2 > y1 else None

                detections.append(Detection(
                    image_id=image_id,
                    class_id=cls_id,
                    class_name=self.class_names[cls_id] if cls_id < len(self.class_names) else str(cls_id),
                    confidence=conf,
                    bbox_xyxy=(x1, y1, x2, y2),
                    bbox_xywh_norm=(cx, cy, bw, bh),
                    crop=crop,
                ))

        logger.debug("'%s': %d wykryć.", image_id, len(detections))
        return detections

    def detect_batch(self, image_paths: List[Union[str, Path]]) -> Dict[str, List[Detection]]:
        """Przetwarza wiele obrazów. Zwraca {image_id: [Detection, ...]}."""
        self._ensure_loaded()
        all_results: Dict[str, List[Detection]] = {}
        total = len(image_paths)

        for i, path in enumerate(image_paths, 1):
            try:
                dets = self.detect(path)
                all_results[Path(path).stem] = dets
                if i % 10 == 0 or i == total:
                    logger.info("Batch: %d/%d", i, total)
            except Exception as e:
                logger.warning("Błąd: %s — %s", path, e)

        detected = sum(len(v) for v in all_results.values())
        logger.info("Batch: %d obrazów, %d wykryć łącznie.", len(all_results), detected)
        return all_results

    def draw(
        self,
        image: np.ndarray,
        detections: List[Detection],
        save_path: Optional[Union[str, Path]] = None,
    ) -> np.ndarray:
        """Nanosi bounding boxy i etykiety na obraz. Opcjonalnie zapisuje."""
        COLORS = [(0, 255, 0), (255, 128, 0), (0, 128, 255), (255, 0, 255)]
        lw  = self.inf_cfg.get("line_width", 2)
        vis = image.copy()

        for det in detections:
            x1, y1, x2, y2 = det.bbox_xyxy
            color = COLORS[det.class_id % len(COLORS)]
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, lw)

            label = det.class_name
            if self.inf_cfg.get("show_conf", True):
                label += f" {det.confidence:.2f}"

            font = cv2.FONT_HERSHEY_SIMPLEX
            (tw, th), _ = cv2.getTextSize(label, font, 0.5, 1)
            cv2.rectangle(vis, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
            cv2.putText(vis, label, (x1 + 2, y1 - 4), font, 0.5,
                        (255, 255, 255), 1, cv2.LINE_AA)

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(save_path), vis)
            logger.info("Wizualizacja → %s", save_path)

        return vis

    def export_json(
        self,
        detections: Dict[str, List[Detection]],
        output_path: Union[str, Path],
    ) -> None:
        """Eksportuje wyniki do JSON (wejście dla Etapu 2)."""
        data = {img_id: [d.to_dict() for d in dets] for img_id, dets in detections.items()}
        Path(output_path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("Wyniki JSON → %s", output_path)

    def _ensure_loaded(self) -> None:
        if self.model is None:
            self.load_model()