import pickle
import io
import json
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
        image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error leyendo imagen: {e}")
    results = search(image, top_k=TOP_K)
    return {"best_match": results[0], "alternatives": results[1:]}

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=7860, reload=False)
