from PIL import Image, ImageDraw
import io

from src.pipeline import extraer_fecha, extraer_items, extraer_total, analizar_albaran


def test_extractors():
    lineas = ["Fecha 25/12/2018", "Producto A 9.00", "Round Total (RM): 9.00"]
    assert extraer_fecha(lineas) == "25/12/2018"
    assert extraer_total(lineas) == "9.00"
    assert extraer_items(lineas)[0]["precio"] == "9.00"


def test_analizar_albaran_returns_schema():
    img = Image.new("RGB", (300, 200), "white")
    draw = ImageDraw.Draw(img)
    draw.text((20, 20), "Total 9.00", fill="black")
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    result = analizar_albaran(buffer.getvalue())
    assert "items" in result
    assert "numero_anomalias" in result
    assert "lineas_detectadas" in result
