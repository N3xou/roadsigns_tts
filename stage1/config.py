"""
config.py
Wczytywanie konfiguracji projektu, inicjalizacja loggera
oraz wykrywanie i walidacja środowiska CUDA.
"""

import logging
import sys
from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(config_path: str = "config/settings.yaml") -> Dict[str, Any]:
    """
    Wczytuje plik YAML z konfiguracją.

    Raises:
        FileNotFoundError: jeśli plik nie istnieje
        ValueError: jeśli brakuje wymaganych kluczy lub suma splitów ≠ 1.0
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Nie znaleziono pliku konfiguracyjnego: {path}")

    with open(path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    _validate(config)
    return config


def _validate(config: Dict[str, Any]) -> None:
    for key in ("project", "data", "model", "inference"):
        if key not in config:
            raise ValueError(f"Brak wymaganego klucza w konfiguracji: '{key}'")

    d = config["data"]
    total = d.get("train_split", 0) + d.get("val_split", 0) + d.get("test_split", 0)
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Suma train/val/test musi wynosić 1.0 (wynosi: {total:.2f})")

def setup_logging(config: Dict[str, Any]) -> logging.Logger:
    """Konfiguruje logger projektu — zapis do pliku i konsoli."""
    log_cfg  = config.get("logging", {})
    level    = getattr(logging, log_cfg.get("level", "INFO").upper(), logging.INFO)
    log_file = log_cfg.get("log_file", "data/runs/training.log")

    Path(log_file).parent.mkdir(parents=True, exist_ok=True)

    fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    logging.basicConfig(
        level=level,
        format=fmt,
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file, encoding="utf-8"),
        ],
    )

    logger = logging.getLogger("road_sign_detector")
    logger.info("Logger zainicjalizowany (poziom: %s)", log_cfg.get("level", "INFO"))
    return logger


def check_cuda(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Wykrywa dostępność CUDA i weryfikuje zgodność z wymaganą wersją.

    Jeśli GPU jest niedostępne lub wersja CUDA nie pasuje, automatycznie
    przełącza device na 'cpu' i zwraca zaktualizowany config.

    Returns:
        Zaktualizowany config z poprawnym device.
    """
    logger = logging.getLogger("road_sign_detector.config")
    required_cuda = config["model"].get("cuda_version", "13.0")

    try:
        import torch
    except ImportError:
        logger.error("PyTorch nie jest zainstalowany. Zainstaluj zgodnie z README.")
        sys.exit(1)

    if not torch.cuda.is_available():
        logger.warning("CUDA niedostępna — przełączam na CPU.")
        config["model"]["device"] = "cpu"
        config["model"]["amp"]    = False
        config["model"]["batch"]  = 8
        config["model"]["workers"] = 2
        return config

    # Wykryj zainstalowaną wersję CUDA
    installed_cuda = torch.version.cuda or "nieznana"
    device_name    = torch.cuda.get_device_name(0)
    gpu_mem_gb     = torch.cuda.get_device_properties(0).total_memory / 1e9

    logger.info("GPU wykryte: %s (VRAM: %.1f GB)", device_name, gpu_mem_gb)
    logger.info("Wersja CUDA: zainstalowana=%s | wymagana=%s", installed_cuda, required_cuda)

    # Porównaj wersje (major.minor)
    def _ver(v: str):
        parts = v.split(".")
        return (int(parts[0]), int(parts[1]) if len(parts) > 1 else 0)

    inst_v = _ver(installed_cuda)
    req_v  = _ver(required_cuda)

    if inst_v < req_v:
        logger.error(
            "Zainstalowana CUDA %s jest starsza niż wymagana %s.\n"
            "Zainstaluj sterowniki NVIDIA wspierające CUDA %s\n"
            "lub PyTorch nightly: pip install torch --index-url "
            "https://download.pytorch.org/whl/nightly/cu%s",
            installed_cuda, required_cuda, required_cuda,
            required_cuda.replace(".", ""),
        )
        sys.exit(1)

    if inst_v > req_v:
        logger.warning(
            "Zainstalowana CUDA %s jest nowsza niż wymagana %s — kontynuuję.",
            installed_cuda, required_cuda,
        )

    # Dostosuj batch do dostępnej pamięci GPU
    batch = _recommended_batch(gpu_mem_gb, config["model"]["imgsz"])
    if batch != config["model"]["batch"]:
        logger.info(
            "Dostosowano batch: %d → %d (VRAM: %.1f GB)",
            config["model"]["batch"], batch, gpu_mem_gb,
        )
        config["model"]["batch"] = batch

    logger.info(
        "CUDA OK — device=%s | AMP=%s | batch=%d",
        config["model"]["device"],
        config["model"].get("amp", True),
        config["model"]["batch"],
    )
    return config


def _recommended_batch(vram_gb: float, imgsz: int) -> int:
    """Heurystyka doboru rozmiaru batcha na podstawie VRAM i rozdzielczości."""
    # Przybliżone zużycie pamięci: ~2 MB/obraz przy imgsz=640
    scale = (imgsz / 640) ** 2
    if vram_gb >= 24:
        return max(1, int(64 / scale))
    elif vram_gb >= 16:
        return max(1, int(32 / scale))
    elif vram_gb >= 8:
        return max(1, int(16 / scale))
    elif vram_gb >= 4:
        return max(1, int(8 / scale))
    else:
        return max(1, int(4 / scale))


def ensure_directories(config: Dict[str, Any]) -> None:
    """Tworzy wymagane katalogi projektu."""
    dirs = [
        config["data"]["data_dir"],
        config["data"]["images_dir"],
        config["data"]["labels_dir"],
        config["model"]["output_dir"],
        "data/images/raw",
        "data/images/train",
        "data/images/val",
        "data/images/test",
        "data/labels/raw",
        "data/labels/train",
        "data/labels/val",
        "data/labels/test",
    ]
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)