# Análisis automático de albaranes con MLOps

**Autora:** Laura Urrego Manzano  
**Proyecto base:** Análisis automático de albaranes mediante Computer Vision y Deep Learning.

Este repositorio adapta el proyecto final de Deep Learning a una entrega de MLOps. El sistema analiza imágenes de albaranes/facturas mediante OCR, extracción de campos e identificación ligera de zonas visualmente anómalas.

## Funcionalidades

- API REST con FastAPI.
- Endpoint `/health` para comprobar el estado del servicio.
- Endpoint `/predict` para subir una imagen y obtener:
  - líneas de texto detectadas,
  - fecha,
  - total,
  - productos/precios detectados,
  - número de anomalías visuales.
- Script de experimento con Weights & Biases.
- Tests básicos con pytest.
- Dockerfile para ejecutar el servicio en local o desplegarlo.
- Workflow de CI con GitHub Actions.

## Estructura

```text
.
├── api/
│   └── main.py
├── src/
│   ├── pipeline.py
│   └── train.py
├── tests/
│   └── test_pipeline.py
├── notebooks/
│   └── Proyecto_Final_Deep_Learning_vision.ipynb
├── docs/
│   └── Informe_Proyecto_Final_v21_Laura_Urrego.pdf
├── data/
│   ├── raw/
│   └── processed/
├── Dockerfile
├── requirements.txt
├── render.yaml
└── README.md
```

## Instalación local

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

También es necesario tener instalado Tesseract OCR. En Ubuntu/Debian:

```bash
sudo apt-get update
sudo apt-get install -y tesseract-ocr tesseract-ocr-spa
```

## Lanzar la API en local

```bash
uvicorn api.main:app --reload
```

La documentación interactiva queda disponible en:

```text
http://127.0.0.1:8000/docs
```

Ejemplo de petición:

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -F "file=@data/raw/ejemplo.jpg"
```

## Ejecutar tests

```bash
pytest -q
```

## Ejecutar con Docker

```bash
docker build -t albaranes-mlops .
docker run -p 8000:8000 albaranes-mlops
```

## Experimentos con Weights & Biases

1. Iniciar sesión:

```bash
wandb login
```

2. Colocar imágenes en `data/raw/`.

3. Ejecutar el experimento:

```bash
python -m src.train --data data/raw --project albaranes-mlops --max-images 20
```

El script registra métricas como número de productos detectados y número de anomalías por imagen. Después se debe crear un **W&B Report** desde la interfaz de W&B incluyendo gráficos, tabla de resultados y análisis crítico.

## Despliegue sugerido en Render

1. Subir este proyecto a un repositorio público de GitHub.
2. Crear un nuevo servicio web en Render.
3. Seleccionar el repositorio.
4. Elegir despliegue con Docker.
5. Usar `/health` como health check.
6. Copiar el endpoint público generado por Render.

## Entrega

En el cuerpo de la tarea deben indicarse estos tres enlaces:

```text
GitHub: <pegar enlace al repositorio público>
Weights & Biases Report: <pegar enlace al report>
Endpoint en producción: <pegar endpoint público de Render/Railway/Cloud Run>
```

Para comprimir el proyecto antes de entregar:

```bash
zip -r UrregoLaura.zip . -x "*.git*" "*venv*" "*.venv*" "*__pycache__*" "*wandb*"
```
