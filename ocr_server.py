from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import base64, io, os
from PIL import Image
import pytesseract

# Si Tesseract no está en PATH (Windows):
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

app = FastAPI(title="OCR Adapter (CPU) — Tesseract")

class OCRIn(BaseModel):
    image_b64: str

class OCROut(BaseModel):
    text: str
    confidence: float | None = None  # Tesseract no entrega confianza simple

@app.post("/ocr", response_model=OCROut)
def ocr_endpoint(body: OCRIn):
    try:
        img_bytes = base64.b64decode(body.image_b64)
        img = Image.open(io.BytesIO(img_bytes))
        # Tip: preprocesado simple puede mejorar resultados en escaneados:
        # img = img.convert("L")  # grayscale
        text = pytesseract.image_to_string(img)
        return OCROut(text=(text or "").strip(), confidence=None)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
