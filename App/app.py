import pickle
import io
import os
import re
import json
import logging
import random
import numpy as np
from pathlib import Path
from PIL import Image
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from typing import cast
from torchvision.transforms import Compose
import open_clip
import faiss
from google import genai
from google.genai import types as genai_types
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request
import uvicorn
from ultralytics import YOLOWorld

# ─────────────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────────────
FAISS_FILE   = "./ikea_faiss.index"
LABELS_FILE  = "./ikea_faiss_labels.pkl"
CATALOG_FILE = "./catalogo_ikea.json"

TOP_K = 3
N_TTA = 7

# ── API Google Gemini (modelo principal) ──────────────
GEMINI_API_KEY    = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-3.5-flash-lite")
GEMINI_TIMEOUT_MS = 30_000   # 30 s — evita peticiones colgadas ante latencia
GEMINI_MAX_IMAGE  = 1024     # Píxeles máximos al enviar la imagen a la API

logger = logging.getLogger("ikea-scanner")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

# Cliente oficial de Google Gemini (lazy: None si falta la clave o el SDK falla)
_genai_client = None
if GEMINI_API_KEY:
    try:
        _genai_client = genai.Client(
            api_key=GEMINI_API_KEY,
            http_options=genai_types.HttpOptions(timeout=GEMINI_TIMEOUT_MS),
        )
        logger.info("✅ Cliente Google Gemini inicializado (modelo %s)", GEMINI_MODEL_NAME)
    except Exception as e:
        logger.error(
            "⚠️ No se pudo inicializar el cliente de Gemini (%s: %s). Activando fallback local...",
            type(e).__name__, e,
        )
else:
    logger.info("ℹ️ GEMINI_API_KEY no configurada. Activando fallback local (YOLO + OpenCLIP + FAISS)...")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"✅ Dispositivo: {DEVICE.upper()}")

# ─────────────────────────────────────────────────────
# 1. CARGA DE MODELOS
# ─────────────────────────────────────────────────────
print("⏳ Cargando OpenCLIP ViT-B/32...")
result          = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
clip_model      = cast(open_clip.CLIP, result[0])
clip_preprocess = cast(Compose, result[2])
clip_model      = clip_model.to(DEVICE)
clip_model.eval()
for param in clip_model.parameters():
    param.requires_grad = False
print("✅ OpenCLIP ViT-B/32 cargado (dim=512)")

yolo_model = YOLOWorld("yolov8s-world.pt")

# Defines exactamente qué quieres detectar — en inglés da mejores resultados
yolo_model.set_classes([
    # ── ASIENTOS ────────────────────────────────────────
    "chair", "dining chair", "office chair", "armchair", "rocking chair",
    "folding chair", "bar stool", "stool", "bench", "sofa", "couch",
    "sectional sofa", "sofa bed", "loveseat", "chaise longue", "pouf",
    "footstool", "ottoman",

    # ── CAMAS Y DORMITORIO ───────────────────────────────
    "bed", "bed frame", "bunk bed", "loft bed", "day bed", "sofa bed",
    "headboard", "mattress", "crib", "baby cot", "children bed", "cushion", "pillow",

    # ── MESAS ───────────────────────────────────────────
    "dining table", "kitchen table", "coffee table", "side table",
    "end table", "console table", "desk", "computer desk", "writing desk",
    "standing desk", "folding table", "nesting tables", "bedside table",
    "nightstand", "dressing table", "vanity table", "picnic table",
    "outdoor table", "garden table",

    # ── ALMACENAMIENTO Y ARMARIOS ────────────────────────
    "wardrobe", "closet", "armoire", "dresser", "chest of drawers",
    "drawer unit", "cabinet", "sideboard", "buffet", "TV cabinet",
    "TV unit", "media console", "storage unit", "storage box",
    "storage bench", "ottoman with storage", "shoe cabinet", "shoe rack",
    "hallway cabinet", "filing cabinet", "trofast", "kallax",

    # ── ESTANTERÍAS ─────────────────────────────────────
    "bookcase", "bookshelf", "shelving unit", "wall shelf", "floating shelf",
    "display shelf", "corner shelf", "ladder shelf", "open shelving",
    "wall unit", "modular shelving",

    # ── ILUMINACIÓN ─────────────────────────────────────
    "floor lamp", "desk lamp", "table lamp", "ceiling lamp",
    "pendant lamp", "chandelier", "wall lamp", "wall light",
    "LED strip light", "spotlight", "reading lamp", "arc lamp",
    "bedside lamp", "outdoor lamp", "lantern",

    # ── COCINA Y COMEDOR ─────────────────────────────────
    "kitchen cabinet", "kitchen shelf", "kitchen trolley", "kitchen cart",
    "kitchen island", "bar cabinet", "wine rack", "dish rack",
    "kitchen organizer", "spice rack",

    # ── TEXTILES ─────────────────────────────────────────
    "rug", "carpet", "curtain", "blinds", "roller blind", "cushion",
    "throw pillow", "blanket", "bedspread", "duvet", "pillow",
    "bath mat", "towel rack",

    # ── BAÑO ─────────────────────────────────────────────
    "bathroom cabinet", "bathroom shelf", "bathroom mirror",
    "bathroom organizer", "towel rail", "toilet brush holder",
    "soap dispenser", "shower curtain",

    # ── ESCRITORIO Y OFICINA ─────────────────────────────
    "monitor stand", "desk organizer", "magazine rack",
    "whiteboard", "pin board", "noticeboard",

    # ── INFANTIL ─────────────────────────────────────────
    "changing table", "baby changing unit", "toy storage",
    "toy chest", "kids wardrobe", "kids shelf", "highchair",
    "baby chair", "play table", "kids desk",

    # ── EXTERIOR ─────────────────────────────────────────
    "garden chair", "garden sofa", "outdoor sofa", "garden bench",
    "deck chair", "sun lounger", "parasol", "garden storage",
    "outdoor storage box", "planter", "plant pot", "plant stand",

    # ── DECORACIÓN ───────────────────────────────────────
    "mirror", "wall mirror", "picture frame", "photo frame",
    "wall art", "painting", "clock", "wall clock",
    "vase", "candle holder", "candlestick", "decorative bowl",
    "figurine", "plant pot", "indoor plant", "artificial plant",
    "room divider", "screen divider", "coat rack", "hat stand",
    "umbrella stand", "tray", "basket", "decorative basket",

    # ── ELECTRÓNICA Y ACCESORIOS ─────────────────────────
    "television", "TV", "monitor", "laptop",
    "power strip", "cable management",
])

print("⏳ Cargando índice FAISS...")
if not Path(FAISS_FILE).exists() or not Path(LABELS_FILE).exists():
    raise FileNotFoundError(
        "❌ No se encontró el índice FAISS. Ejecuta build_faiss.py primero."
    )

faiss_index = faiss.read_index(FAISS_FILE)
with open(LABELS_FILE, "rb") as f:
    index_labels: list[str] = pickle.load(f)
print(f"✅ FAISS listo: {faiss_index.ntotal} vectores | {len(set(index_labels))} productos únicos")

catalogo_real: dict = {}
if Path(CATALOG_FILE).exists():
    with open(CATALOG_FILE, "r", encoding="utf-8") as f:
        catalogo_real = json.load(f)
    print(f"✅ Catálogo JSON cargado: {len(catalogo_real)} productos")
else:
    print("⚠️  No se encontró el JSON del catálogo.")


# ─────────────────────────────────────────────────────
# 2. HELPERS INTERNOS
# ─────────────────────────────────────────────────────

def _augment_query(img: Image.Image) -> Image.Image:
    img = T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05)(img)

    # PIL.Image.rotate en lugar de TF.rotate — acepta Image directamente
    if random.random() > 0.5:
        angle = random.uniform(-6, 6)
        img = img.rotate(angle, expand=False, fillcolor=(255, 255, 255))

    if random.random() > 0.5:
        img = T.GaussianBlur(kernel_size=3, sigma=(0.3, 1.2))(img)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=random.randint(60, 85))
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def _get_embedding(img: Image.Image) -> torch.Tensor:
    """Devuelve el embedding L2-normalizado. Shape: [1, 512]"""
    tensor = cast(torch.Tensor, clip_preprocess(img)).unsqueeze(0).to(DEVICE)
    with torch.inference_mode():
        emb = clip_model.encode_image(tensor)
        return emb / emb.norm(dim=-1, keepdim=True).clamp(min=1e-8)


# ─────────────────────────────────────────────────────
# 2.5 API GOOGLE GEMINI (MODELO PRINCIPAL)
# ─────────────────────────────────────────────────────

GEMINI_SYSTEM_PROMPT = """
Eres un experto en el catálogo de IKEA España. Analiza la imagen del mueble adjunta
y, usando la herramienta de Búsqueda de Google, localiza el producto EXACTO en la
web oficial de IKEA (www.ikea.com/es/es) para confirmar sus datos reales: nombre del
producto, subtítulo (descripción corta), precio en euros, URL de la página del
producto, URL de la imagen oficial, peso y dimensiones.

Devuelve ÚNICAMENTE un único objeto JSON válido y ejecutable. No incluyas
introducciones, explicaciones, comentarios ni bloques de código Markdown (```)
fuera del JSON. El objeto debe tener EXACTAMENTE esta estructura:

{
  "best_match": { ...producto... },
  "alternatives": [ { ...producto... }, ... ]
}

Cada producto (best_match y cada elemento de alternatives) debe tener EXACTAMENTE
estas 10 claves, en el mismo formato que usa el catálogo local:

1. "id": string con el formato de clave del catálogo "CATEGORÍA | NOMBRE DEL PRODUCTO".
          Ejemplo: "Sofá cama 3 plazas | VRETSTORP". La categoría va en español.
2. "confidence": número entre 0 y 1 con 4 decimales. Ejemplo: 0.8742
3. "confidence_pct": string con el porcentaje en formato "87.4%"
4. "nombre": string con el nombre corto del producto. Ejemplo: "VRETSTORP"
5. "subtitulo": string con la descripción corta real (color, medidas, materiales).
6. "precio": número (precio en euros) o el string "No disponible".
7. "imagen": string con la URL oficial de la imagen del producto en ikea.com.
8. "url": string con la URL oficial de la página del producto en ikea.com.
9. "peso": string con el peso real ("11.30 kg") o "" si se desconoce.
10. "ubicacion": objeto con la forma {"pasillo": string, "estanteria": número o string}.

Reglas:
- "alternatives" debe contener otros productos IKEA plausibles ordenados de más a menos
  probable (puede ser una lista vacía si solo hay un candidato claro).
- Sé conservador con "confidence" según lo seguro que estés del match.
- Si no puedes identificar ningún producto IKEA, devuelve:
  {"best_match": {"id": "No identificado", "confidence": 0.0, "confidence_pct": "0.0%",
   "nombre": "No identificado", "subtitulo": "", "precio": "No disponible", "imagen": "",
   "url": "#", "peso": "", "ubicacion": {"pasillo": "-", "estanteria": "-"}},
   "alternatives": []}
"""


def _extract_json(text: str | None) -> dict | None:
    """Extrae el primer objeto JSON válido de la respuesta del modelo,
    tolerando bloques de código Markdown o texto sobrante.
    Devuelve None (sin lanzar excepción) si el texto es None, vacío o no JSON."""
    if text is None:
        logger.error("⚠️ API Gemini: 'text' es None en la respuesta")
        return None
    t = text.strip()
    if not t:
        logger.error("⚠️ API Gemini: respuesta de texto vacía")
        return None
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?\s*", "", t)
        t = re.sub(r"\s*```$", "", t)
    start, end = t.find("{"), t.rfind("}")
    if start == -1 or end == -1 or end <= start:
        logger.error("⚠️ API Gemini: la respuesta no contiene un objeto JSON. Contenido: %s", t[:300])
        return None
    try:
        return json.loads(t[start:end + 1])
    except (json.JSONDecodeError, ValueError) as e:
        logger.error("⚠️ API Gemini: JSON corrupto en la respuesta (%s). Contenido: %s", e, t[:300])
        return None


def _normalizar_producto(p: dict, label_default: str = "No identificado") -> dict:
    """Garantiza que un producto tenga todos los campos del contrato del frontend."""
    info = p if isinstance(p, dict) else {}
    conf = 0.0
    try:
        conf = round(float(info.get("confidence", 0.0)), 4)
    except (TypeError, ValueError):
        conf = 0.0
    pct = info.get("confidence_pct")
    if not isinstance(pct, str):
        pct = f"{round(conf * 100, 1)}%"
    ubic = info.get("ubicacion")
    if not isinstance(ubic, dict):
        ubic = {"pasillo": "-", "estanteria": "-"}
    return {
        "id":             info.get("id", label_default),
        "confidence":     conf,
        "confidence_pct": pct,
        "nombre":         info.get("nombre", label_default),
        "subtitulo":      info.get("subtitulo", ""),
        "precio":         info.get("precio", "No disponible"),
        "imagen":         info.get("imagen", ""),
        "url":            info.get("url", "#"),
        "peso":           info.get("peso", ""),
        "ubicacion":      ubic,
    }


def _validar_respuesta_api(payload: dict) -> dict:
    """Valida y normaliza la respuesta de la API. Lanza ValueError si el JSON
    no cumple el contrato esperado (mismo formato que la búsqueda local)."""
    if not isinstance(payload, dict):
        raise ValueError("La API no devolvió un objeto JSON")
    best = payload.get("best_match")
    alts = payload.get("alternatives")
    if not isinstance(best, dict):
        raise ValueError("best_match ausente o con formato inválido")
    if not isinstance(alts, list):
        raise ValueError("alternatives ausente o con formato inválido")
    return {
        "best_match":   _normalizar_producto(best),
        "alternatives": [_normalizar_producto(a) for a in alts],
    }


def _extract_gemini_text(response) -> str | None:
    """Extrae el texto de la respuesta del SDK de Gemini de forma segura.
    Devuelve None si la respuesta no contiene texto (content bloqueado, vacío...)."""
    if response is None:
        return None
    try:
        text = getattr(response, "text", None)
        if isinstance(text, str) and text.strip():
            return text
    except (AttributeError, ValueError):
        pass
    try:
        for candidate in getattr(response, "candidates", []) or []:
            for part in getattr(candidate.content, "parts", []) or []:
                text = getattr(part, "text", None)
                if isinstance(text, str) and text.strip():
                    return text
    except (AttributeError, TypeError):
        pass
    return None


def _gemini_identify(image_bytes: bytes, content_type: str) -> dict | None:
    """Intento principal: identifica el producto con la API oficial de Google
    Gemini (modelo configurable, por defecto gemini-3.5-flash-lite).

    Estrategia en dos intentos:
      1. Con la herramienta de Búsqueda de Google (Search Grounding) para
         localizar el mueble exacto en la web de IKEA.
      2. Sin herramienta de búsqueda (tools=None) si el intento 1 falla por
         cuota 429 / RESOURCE_EXHAUSTED, timeout, error HTTP o JSON inválido,
         usando la visión nativa del modelo.

    El formato JSON se impone exclusivamente vía GEMINI_SYSTEM_PROMPT y se
    normaliza con _extract_json. Devuelve el JSON parseado o None si ambos
    intentos fallan o devuelven un JSON inválido, de modo que el fallback local
    (FAISS) tome el control sin excepciones no capturadas."""
    if _genai_client is None:
        logger.error("⚠️ Cliente Google Gemini no disponible. Activando fallback local...")
        return None

    # Redimensionar y re-comprimir como JPEG para reducir el payload
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img.thumbnail((GEMINI_MAX_IMAGE, GEMINI_MAX_IMAGE))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    mime = content_type if content_type and content_type.startswith("image/") else "image/jpeg"

    intentos = [
        ("con Búsqueda de Google", [genai_types.Tool(google_search=genai_types.GoogleSearch())]),
        ("sin búsqueda web (visión nativa)", None),
    ]

    for etiqueta, tools in intentos:
        try:
            response = _genai_client.models.generate_content(
                model=GEMINI_MODEL_NAME,
                contents=[
                    genai_types.Part.from_bytes(data=buf.getvalue(), mime_type=mime),
                    "Identifica el mueble de IKEA de la imagen y devuelve únicamente el JSON solicitado.",
                ],
                config=genai_types.GenerateContentConfig(
                    system_instruction=GEMINI_SYSTEM_PROMPT,
                    tools=tools,
                    temperature=0.1,
                    max_output_tokens=2048,
                ),
            )
        except Exception as e:
            if _es_error_cuota_429(e):
                logger.warning(
                    "⚠️ Gemini %s: cuota de búsqueda web agotada (429 RESOURCE_EXHAUSTED: %s). Reintentando %s...",
                    etiqueta, e, intentos[1][0],
                )
            else:
                logger.error("⚠️ Gemini %s falló (%s: %s).", etiqueta, type(e).__name__, e)
            continue

        text = _extract_gemini_text(response)
        payload = _extract_json(text) if text is not None else None
        if payload is None:
            logger.error("⚠️ Gemini %s no devolvió un JSON válido.", etiqueta)
            continue

        logger.info("✅ Reconocimiento completado vía Google Gemini (%s, intento %s)", GEMINI_MODEL_NAME, etiqueta)
        return payload

    return None


def _es_error_cuota_429(e: Exception) -> bool:
    """True si la excepción corresponde a cuota agotada (HTTP 429 o estado
    RESOURCE_EXHAUSTED / RATE_LIMIT del SDK de Gemini)."""
    code = getattr(e, "code", None)
    if code == 429:
        return True
    status = str(getattr(e, "status", "") or "").upper()
    msg = str(getattr(e, "message", "") or e).lower()
    return "RESOURCE_EXHAUSTED" in status or "RATE_LIMIT" in status or "quota" in msg


# ─────────────────────────────────────────────────────
# 3. MOTOR DE BÚSQUEDA PRINCIPAL
# ─────────────────────────────────────────────────────

def search(image: Image.Image, top_k: int = TOP_K) -> list[dict]:

    # --- PASO 1: DETECCIÓN Y RECORTE CON YOLO-World ---
    img_rgb = image.convert("RGB")
    resultados_yolo = yolo_model(img_rgb, verbose=False)[0]
    imagen_a_procesar = img_rgb  # Fallback: imagen completa si YOLO no detecta nada

    if len(resultados_yolo.boxes) > 0:
        # Con YOLO-World no filtramos por clase — todo lo detectado ya es mueble
        # Simplemente cogemos la caja más grande
        caja_mas_grande = max(
            resultados_yolo.boxes,
            key=lambda b: (b.xyxy[0][2] - b.xyxy[0][0]) * (b.xyxy[0][3] - b.xyxy[0][1])
        )

        x1, y1, x2, y2 = caja_mas_grande.xyxy[0].tolist()
        imagen_a_procesar = img_rgb.crop((
            max(0, x1 - 10), max(0, y1 - 10), x2 + 10, y2 + 10
        ))

    # --- PASO 2: TTA — Centroide de N embeddings del query ---
    query_embs = [_get_embedding(imagen_a_procesar)]  # Original siempre incluido

    for _ in range(N_TTA - 1):
        aug = _augment_query(imagen_a_procesar.copy())
        query_embs.append(_get_embedding(aug))

    query_centroid = torch.stack(query_embs).mean(dim=0)
    query_centroid = query_centroid / query_centroid.norm(dim=-1, keepdim=True)

    # --- PASO 3: BÚSQUEDA FAISS ---
    query_np = query_centroid.cpu().numpy().astype(np.float32)

    # Pedimos top_k * 20 candidatos porque varios vectores pueden ser del mismo producto
    scores, indices = faiss_index.search(query_np, k=top_k * 20)
    scores  = scores[0]
    indices = indices[0]

    # --- PASO 4: AGRUPAR POR PRODUCTO (máximo, no promedio) ---
    # Si un producto tiene 50 imágenes en el catálogo, nos quedamos
    # con su imagen más parecida al query — no el promedio de las 50.
    best_per_product: dict[str, float] = {}
    for score, idx in zip(scores, indices):
        if idx == -1:  # FAISS devuelve -1 si no hay suficientes vecinos
            continue
        label = index_labels[idx]
        if label not in best_per_product or score > best_per_product[label]:
            best_per_product[label] = float(score)

    ranked = sorted(best_per_product.items(), key=lambda x: x[1], reverse=True)

    # --- PASO 5: FORMATEO DE RESULTADOS ---
    results = []
    for label, score in ranked[:top_k]:
        cat, pv = label.split(" | ") if " | " in label else ("", label)
        pname, *var = pv.split(" — ")
        info = catalogo_real.get(label, {})
        results.append({
            "id":             label,
            "confidence":     round(score, 4),
            "confidence_pct": f"{round(score * 100, 1)}%",
            "nombre":         info.get("nombre", label),
            "subtitulo":      info.get("subtitulo", ""),
            "precio":         info.get("precio", "No disponible"),
            "imagen":         info.get("imagen", ""),
            "url":            info.get("url", "#"),
            "peso":           info.get("peso", ""),
            "ubicacion":      info.get("ubicacion", {"pasillo": "-", "estanteria": "-"})
        })

    return results

# App

app = FastAPI(title="IKEA Scanner API", version="4.0.0")

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

@app.get("/")
def init():
    return {"/app":"full app","/model":"model testing","/health":"server status","/info":"general info","/catalog":"product catalog JSON"}

@app.get("/catalog")
def get_catalog():
    """Devuelve el catálogo de productos en formato JSON"""
    if not catalogo_real:
        return JSONResponse(
            status_code=404,
            content={"error": "Catálogo no disponible"}
        )
    return JSONResponse(content=catalogo_real)

@app.get("/app", response_class=HTMLResponse)
def ui(request: Request):
    return templates.TemplateResponse(request=request, name="app.html")

@app.get("/model", response_class=HTMLResponse)
def ui_secundaria(request: Request):
    return templates.TemplateResponse(request=request, name="model.html")

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.get("/info")
def info():
    return {"status":"online","model":"DINOv2-small","device":DEVICE,"products":len(set(index_labels))}

@app.get("/share")
def share():
    return FileResponse("./IKEA_App_QR.png")

@app.post("/identify")
async def identify(file: UploadFile = File(...)):
    if not (file.content_type or "").startswith("image/"):
        raise HTTPException(status_code=400, detail="El archivo debe ser una imagen")
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error leyendo imagen: {e}")
    return _recognize(image_bytes, image, file.content_type)


def _recognize(image_bytes: bytes, image: Image.Image, content_type: str) -> dict:
    """Orquestador con resiliencia: intenta primero la API oficial de Google
    Gemini (modelo principal) y, ante cualquier fallo (error HTTP, cuota de 500
    RPD agotada, timeout o JSON inválido), conmuta automáticamente al pipeline
    local (YOLO-World + OpenCLIP + FAISS). La respuesta final tiene siempre el
    mismo contrato JSON para el frontend."""
    if _genai_client is not None:
        try:
            raw = _gemini_identify(image_bytes, content_type)
            if raw is None:
                logger.warning("⚠️ Google Gemini no devolvió un JSON válido. Activando fallback local (YOLO-World + OpenCLIP + FAISS)...")
            else:
                payload = _validar_respuesta_api(raw)
                logger.info("✅ Reconocimiento completado vía Google Gemini (%s)", GEMINI_MODEL_NAME)
                return payload
        except Exception as e:
            logger.warning(
                "⚠️ Google Gemini falló (%s: %s). Activando fallback local (YOLO-World + OpenCLIP + FAISS)...",
                type(e).__name__, e,
            )
    else:
        logger.info("ℹ️ Cliente Google Gemini no disponible. Activando fallback local (YOLO-World + OpenCLIP + FAISS)...")

    results = search(image, top_k=TOP_K)
    return {"best_match": results[0], "alternatives": results[1:]}

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=7860, reload=False)
