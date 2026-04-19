"""
ocr.py
Odczyt tekstu z wyciętych znaków drogowych (crop z Detection).

Użycie:
    reader = OcrReader(gpu=True)
    text = reader.read(detection.crop)   # "30" | "STOP" | None
"""

import logging
import cv2
import numpy as np

logger = logging.getLogger("road_sign_detector.ocr")

# Próg confidence EasyOCR — wyniki poniżej są ignorowane
_CONF_THRESHOLD = 0.4

# Znaki drogowe mają krótki tekst; wszystko dłuższe to szum (
_MAX_TEXT_LENGTH = 100


def _preprocess(crop: np.ndarray) -> np.ndarray:
    """
    Skaluje crop do min. 100px i poprawia kontrast.
    EasyOCR radzi sobie słabo na obrazach < 80px wysokości.
    """
    h, w = crop.shape[:2]
    if min(h, w) < 100:
        scale = 100 / min(h, w)
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    gray = clahe.apply(gray)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


class OcrReader:
    """
    Wrapper EasyOCR dla jednej klasy znaków drogowych.
    Inicjalizacja jest kosztowna (~2s) — twórz raz, używaj wielokrotnie.
    """

    def __init__(self, gpu: bool = True):
        import easyocr
        logger.info("Inicjalizacja EasyOCR (gpu=%s)...", gpu)
        self._reader = easyocr.Reader(["pl", "en"], gpu=gpu)
        logger.info("EasyOCR gotowy.")

    def read(self, crop: np.ndarray | None) -> str | None:
        """
        Odczytuje tekst z pojedynczego cropu.

        Returns:
            Odczytany tekst (np. "30", "STOP") lub None jeśli brak tekstu.
        """
        if crop is None or crop.size == 0:
            return None

        processed = _preprocess(crop)

        raw = self._reader.readtext(processed, detail=1, paragraph=False)

        parts = []
        for _bbox, text, conf in raw:
            text = text.strip()
            if conf < _CONF_THRESHOLD:
                continue
            if not text or len(text) > _MAX_TEXT_LENGTH:
                continue
            parts.append(text)

        if not parts:
            return None

        result = " ".join(parts)
        logger.debug("OCR: %r", result)
        return result