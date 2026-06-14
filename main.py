from fastapi import FastAPI, File, UploadFile, HTTPException
from src.pipeline import analizar_albaran

app = FastAPI(
    title="API de análisis automático de albaranes",
    description="Servicio MLOps para OCR, extracción de campos y detección ligera de anomalías visuales.",
    version="1.0.0",
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Sube una imagen válida: jpg, png o similar.")
    content = await file.read()
    return analizar_albaran(content)
