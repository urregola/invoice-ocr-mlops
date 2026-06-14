import io
import re
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import pytesseract
from PIL import Image


def read_image_bytes(file_bytes: bytes) -> np.ndarray:
    """Convierte bytes de imagen a RGB numpy array."""
    image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    return np.array(image)


def ocr_extraer_texto(img_rgb: np.ndarray) -> str:
    """Extrae texto con Tesseract usando modo de bloque uniforme."""
    return pytesseract.image_to_string(img_rgb, config="--psm 6")


def limpiar_lineas(texto: str) -> List[str]:
    return [line.strip() for line in texto.split("\n") if line.strip()]


def extraer_items(lineas: List[str]) -> List[Dict[str, Optional[str]]]:
    items = []
    for linea in lineas:
        match = re.search(r"(\d+[\.,]\d{2})", linea)
        if match:
            precio = match.group(1).replace(",", ".")
            producto = linea.replace(match.group(1), "").strip(" -:;\t")
            items.append({"producto": producto or "NO_DETECTADO", "precio": precio})
    return items


def extraer_fecha(lineas: List[str]) -> Optional[str]:
    for linea in lineas:
        match = re.search(r"(\d{2}[/-]\d{2}[/-]\d{2,4})", linea)
        if match:
            return match.group(1)
    return None


def extraer_total(lineas: List[str]) -> Optional[str]:
    for linea in lineas:
        if "total" in linea.lower():
            match = re.search(r"(\d+[\.,]\d{2})", linea)
            if match:
                return match.group(1).replace(",", ".")
    return None


def detectar_anomalias_simple(img_rgb: np.ndarray, patch_size: int = 64, threshold: float = 35.0) -> List[Dict[str, int]]:
    """Detecta regiones visualmente distintas con un método ligero basado en intensidad.

    Esta versión es rápida para API y tests. En el notebook se conserva la versión con
    embeddings MobileNetV2.
    """
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    global_mean = float(np.mean(gray))
    anomalies: List[Dict[str, int]] = []
    for y in range(0, max(h - patch_size, 1), patch_size):
        for x in range(0, max(w - patch_size, 1), patch_size):
            patch = gray[y : y + patch_size, x : x + patch_size]
            if patch.size == 0:
                continue
            score = abs(float(np.mean(patch)) - global_mean)
            if score > threshold:
                anomalies.append({"x": x, "y": y, "width": patch_size, "height": patch_size, "score": round(score, 3)})
    return anomalies


@dataclass
class AnalysisResult:
    fecha: Optional[str]
    total: Optional[str]
    items: List[Dict[str, Optional[str]]]
    lineas_detectadas: List[str]
    numero_anomalias: int
    anomalias: List[Dict[str, int]]


def analizar_albaran(file_bytes: bytes) -> Dict[str, Any]:
    img = read_image_bytes(file_bytes)
    texto = ocr_extraer_texto(img)
    lineas = limpiar_lineas(texto)
    anomalias = detectar_anomalias_simple(img)
    resultado = AnalysisResult(
        fecha=extraer_fecha(lineas),
        total=extraer_total(lineas),
        items=extraer_items(lineas),
        lineas_detectadas=lineas[:30],
        numero_anomalias=len(anomalias),
        anomalias=anomalias[:50],
    )
    return asdict(resultado)
