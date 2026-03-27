"""
app.py — API IKEA Scanner para Hugging Face Spaces.
Modelo: DINOv2-small | UI web integrada + escaneo por cámara
"""

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

# ==========================================
# 1. CARGA DE MODELOS
# ==========================================
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


# ==========================================
# 2. HELPERS INTERNOS
# ==========================================

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


# ==========================================
# 3. MOTOR DE BÚSQUEDA PRINCIPAL
# ==========================================

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


# ─────────────────────────────────────────────────────
app = FastAPI(title="IKEA Scanner API", version="4.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

HTML_APP = r"""
<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0">
<title>IKEA App</title>
<!-- jsQR para escaneo QR real -->
<script src="https://cdn.jsdelivr.net/npm/jsqr@1.4.0/dist/jsQR.min.js"></script>
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans:wght@400;500;600;700;900&display=swap');

:root {
  --blue: #0058a3;
  --blue-dark: #003460;
  --yellow: #ffdb00;
  --green: #1f8423;
  --purple: #52207d;
  --red: #ef4444;
  --orange: #ef7744;
  --bg: #e8e8e3;
  --card: #f9fafb;
  --gray: #a4a4a4;
  --text: #1a1a1a;
  --border: #d1d5db;
}

* { box-sizing: border-box; margin: 0; padding: 0; -webkit-tap-highlight-color: transparent; }

body {
  background: linear-gradient(135deg, #0d1b2a 0%, #1a1a3e 50%, #0d1b2a 100%);
  display: flex;
  justify-content: center;
  align-items: center;
  min-height: 100vh;
  font-family: 'Noto Sans', sans-serif;
  overflow: hidden;
}

/* ── PHONE SHELL ── */
.phone {
  width: 402px;
  height: 874px;
  border-radius: 44px;
  overflow: hidden;
  position: relative;
  box-shadow: 0 48px 96px rgba(0,0,0,0.7), 0 0 0 1px #444,
              inset 0 0 0 1px #666, 0 0 60px rgba(0,88,163,0.15);
  background: #111;
}

.screens { width: 100%; height: 100%; position: relative; overflow: hidden; }

.screen {
  position: absolute; top: 0; left: 0;
  width: 100%; height: 100%;
  background: var(--bg);
  display: flex; flex-direction: column;
  transform: translateX(100%);
  transition: transform 0.32s cubic-bezier(0.4,0,0.2,1);
  overflow: hidden;
}
.screen.active    { transform: translateX(0); }
.screen.exit-left { transform: translateX(-100%); }

/* ── STATUS BAR ── */
.status-bar {
  height: 44px;
  background: var(--blue);
  display: flex; align-items: center;
  justify-content: space-between;
  padding: 0 20px;
  flex-shrink: 0;
  position: sticky;
  top: 0;
  z-index: 99;
  /* NO border-radius aquí – la cabecera continúa abajo */
}
.status-time  { color: white; font-weight: 700; font-size: 14px; }
.status-icons { color: white; font-size: 12px; display: flex; gap: 6px; align-items: center; }

/* ── HEADER ── (FIX: sin border-radius inferior, unido al status-bar) */
.header {
  background: var(--blue);
  padding: 0 16px 14px;
  flex-shrink: 0;
  display: flex;
  align-items: center;
  gap: 10px;
  top: 44px
  position: sticky;
  z-index: 100;
  overflow: visible;
}
/* Separador visual entre header y contenido */
.header + .content,
.header + div.aisle-filters,
.header + .content-wrapper {
  border-top: 0px solid rgba(0,0,0,0.12);
}

.logo {
  background: var(--yellow);
  color: var(--blue);
  font-weight: 900; font-size: 13px;
  padding: 5px 9px; border-radius: 7px;
  letter-spacing: 1px; flex-shrink: 0;
}

.logo:active {
    transform: scale(0.95);
    opacity: 0.8;
}

.header-title { color: white; font-weight: 700; font-size: 15px; flex: 1; }
.header-icons { display: flex; align-items: center; gap: 7px; margin-left: auto; overflow: visible; }

.cart-icon-wrap { position: relative; cursor: pointer; overflow: visible !important; }
.cart-badge {
  position: absolute; top: -6px !important; right: -6px;
  background: var(--yellow); color: var(--blue);
  border-radius: 50%; width: 16px; height: 16px;
  font-size: 9px; font-weight: 900;
  display: flex; align-items: center; justify-content: center;
  overflow: visible !important;
}

.header-icon {
  width: 30px; height: 30px;
  background: rgba(255,255,255,0.18);
  border-radius: 50%;
  display: flex; align-items: center; justify-content: center;
  font-size: 14px; cursor: pointer; border: none;
  transition: background 0.15s, transform 0.12s;
}
.header-icon:active { background: rgba(255,255,255,0.38); transform: scale(0.88); }

/* ── CONTENT ── */
.content { flex: 1; overflow-y: auto; overflow-x: hidden; -webkit-overflow-scrolling: touch; }
.content::-webkit-scrollbar { width: 0; }

/* ── BOTTOM NAV ── */
.bottom-nav {
  height: 72px; background: var(--card);
  border-top: 1px solid var(--border);
  display: flex; align-items: center; padding: 0 4px;
  flex-shrink: 0; box-shadow: 0 -4px 16px rgba(0,0,0,0.08);
  position: sticky;
  bottom: 0;
  z-index: 100;
}
.nav-item {
  flex: 1; display: flex; flex-direction: column; align-items: center;
  gap: 3px; cursor: pointer; padding: 6px 2px; border-radius: 12px;
  transition: all 0.18s; border: none; background: none; position: relative;
}
.nav-item:active { transform: scale(0.88); }
.nav-emoji { font-size: 21px; line-height: 1; }
.nav-label { font-size: 9.5px; font-weight: 500; color: var(--gray); }
.nav-item.active .nav-label { color: var(--blue); font-weight: 700; }
.nav-item.active::after {
  content: ''; position: absolute; bottom: 2px;
  width: 24px; height: 3px; background: var(--blue); border-radius: 2px;
}

/* ── HERO / BANNER ── */
.hero-banner {
  background: linear-gradient(140deg, var(--blue) 0%, var(--blue-dark) 100%);
  padding: 14px 18px 20px; color: white; flex-shrink: 0;
}
.hero-location { font-size: 9.5px; font-weight: 900; letter-spacing: 2px; color: var(--yellow); margin-bottom: 6px; }
.hero-title { font-size: 20px; font-weight: 900; line-height: 1.2; margin-bottom: 12px; }
.hero-chips { display: flex; gap: 7px; flex-wrap: wrap; }
.hero-chip {
  background: rgba(255,219,0,0.18); border: 1px solid rgba(255,219,0,0.45);
  color: var(--yellow); font-size: 9.5px; font-weight: 900;
  padding: 3px 9px; border-radius: 20px;
}

/* ── ACTION GRID ── */
.action-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; padding: 14px; }
.action-card {
  background: var(--card); border-radius: 18px; padding: 18px 14px 14px;
  cursor: pointer; box-shadow: 0 3px 10px rgba(0,0,0,0.09);
  transition: all 0.18s; border: none; text-align: left;
  display: flex; flex-direction: column; gap: 6px;
}
.action-card:active { transform: scale(0.94); }
.action-emoji { font-size: 30px; display: block; }
.action-name  { font-size: 14px; font-weight: 700; }
.action-desc  { font-size: 10.5px; color: var(--gray); line-height: 1.3; }

.gestures-card { background: var(--card); border-radius: 18px; padding: 14px; margin: 0 14px 14px; box-shadow: 0 3px 10px rgba(0,0,0,0.08); }
.gestures-title { font-size: 10px; font-weight: 900; letter-spacing: 1.5px; color: var(--gray); margin-bottom: 10px; }
.gesture-grid   { display: grid; grid-template-columns: 1fr 1fr; gap: 7px; }
.gesture-item   { padding: 9px 11px; border-radius: 10px; font-size: 10.5px; font-weight: 700; display: flex; align-items: center; gap: 5px; }

/* ── SECTION TITLE ── */
.section-title { font-size: 10px; font-weight: 900; letter-spacing: 1.5px; color: var(--gray); padding: 14px 18px 7px; }

/* ── PRODUCT CARD ── */
.product-card {
  background: var(--card); border-radius: 16px; margin: 0 14px 9px;
  padding: 13px; display: flex; gap: 11px; align-items: flex-start;
  box-shadow: 0 2px 8px rgba(0,0,0,0.07); cursor: pointer; transition: transform 0.14s;
}
.product-card:active { transform: scale(0.98); }
.product-img {
  width: 54px; height: 54px; border-radius: 12px;
  background: linear-gradient(135deg, #dde, #cce);
  display: flex; align-items: center; justify-content: center;
  font-size: 26px; flex-shrink: 0; overflow: hidden;
}
.product-info  { flex: 1; min-width: 0; }
.product-name  { font-size: 13.5px; font-weight: 700; }
.product-desc  { font-size: 10.5px; color: var(--gray); margin-top: 2px; line-height: 1.3; }
.product-price { font-size: 14px; font-weight: 900; color: var(--text); white-space: nowrap; }
.product-location {
  display: inline-flex; align-items: center; gap: 3px;
  background: rgba(0,88,163,0.08); color: var(--blue);
  font-size: 9.5px; font-weight: 700; padding: 2px 7px; border-radius: 6px; margin-top: 5px;
}
.product-actions { display: flex; gap: 5px; margin-top: 9px; padding-top: 9px; border-top: 1px solid var(--border); }

/* ── BUTTONS ── */
.btn {
  flex: 1; padding: 8px 9px; border-radius: 10px;
  font-size: 11px; font-weight: 700; border: none; cursor: pointer;
  transition: all 0.18s; display: flex; align-items: center; justify-content: center; gap: 3px;
}
.btn:active { transform: scale(0.94); }
.btn-primary { background: var(--blue); color: white; }
.btn-success { background: var(--green); color: white; }
.btn-danger  { background: rgba(239,68,68,0.1); color: var(--red); }
.btn-fav     { background: rgba(255,219,0,0.18); color: #9a6500; }
.btn-fav.active { background: var(--yellow); }
.btn-ghost   { background: var(--bg); color: var(--text); }
.btn-outline { background: transparent; color: var(--blue); border: 1.5px solid var(--blue); }

/* Pay / big button */
.pay-btn {
  display: block; margin: 10px 14px; padding: 15px;
  background: var(--blue); color: white; border: none; border-radius: 14px;
  font-size: 15px; font-weight: 900; cursor: pointer; width: calc(100% - 28px);
  transition: all 0.18s; box-shadow: 0 4px 14px rgba(0,88,163,0.32);
  text-align: center;
}
.pay-btn:active { transform: scale(0.97); }
.pay-btn.secondary { background: rgba(0,88,163,0.08); color: var(--blue); box-shadow: none; border: 1.5px solid rgba(0,88,163,0.2); }
.pay-btn.danger    { background: rgba(239,68,68,0.08); color: var(--red); box-shadow: none; border: 1px solid rgba(239,68,68,0.2); }

/* Qty controls */
.qty-controls { display: flex; align-items: center; gap: 7px; background: var(--bg); border-radius: 10px; padding: 3px 9px; }
.qty-btn {
  width: 26px; height: 26px; border-radius: 50%; border: none;
  background: var(--blue); color: white; font-size: 16px; font-weight: 900;
  cursor: pointer; display: flex; align-items: center; justify-content: center;
  transition: all 0.14s; flex-shrink: 0;
}
.qty-btn:active { transform: scale(0.85); }
.qty-num { font-size: 15px; font-weight: 700; min-width: 18px; text-align: center; }

/* Aisle filters */
.aisle-filters { display: flex; gap: 7px; padding: 9px 14px; overflow-x: auto; flex-shrink: 0; }
.aisle-filters::-webkit-scrollbar { height: 0; }
.aisle-btn { padding: 6px 14px; border-radius: 20px; font-size: 11.5px; font-weight: 700; border: none; cursor: pointer; white-space: nowrap; transition: all 0.18s; }
.aisle-btn.active { background: var(--blue); color: white; }
.aisle-btn:not(.active) { background: var(--bg); color: var(--text); }

/* Toggle */
.toggle { width: 44px; height: 26px; border-radius: 13px; background: var(--border); position: relative; cursor: pointer; transition: background 0.3s; flex-shrink: 0; border: none; }
.toggle.on { background: var(--blue); }
.toggle::after { content: ''; width: 20px; height: 20px; border-radius: 50%; background: white; position: absolute; top: 3px; left: 3px; transition: transform 0.3s; box-shadow: 0 2px 4px rgba(0,0,0,0.2); }
.toggle.on::after { transform: translateX(18px); }

/* Settings */
.settings-title { font-size: 10px; font-weight: 900; letter-spacing: 1.5px; color: var(--gray); margin-bottom: 7px; padding-left: 3px; }
.settings-card  { background: var(--card); border-radius: 16px; overflow: hidden; box-shadow: 0 2px 8px rgba(0,0,0,0.07); }
.settings-item  { display: flex; align-items: center; gap: 11px; padding: 13px 14px; border-bottom: 1px solid var(--border); cursor: pointer; transition: background 0.14s; }
.settings-item:last-child { border-bottom: none; }
.settings-item:active { background: rgba(0,88,163,0.04); }
.settings-emoji { width: 34px; height: 34px; background: var(--bg); border-radius: 9px; display: flex; align-items: center; justify-content: center; font-size: 17px; flex-shrink: 0; }
.settings-info  { flex: 1; }
.settings-name  { font-size: 12.5px; font-weight: 700; }
.settings-desc  { font-size: 10.5px; color: var(--gray); margin-top: 1px; }

/* ────────────────────────────────── */
/*  LOGIN SCREEN                      */
/* ────────────────────────────────── */
.login-screen {
  background: var(--card);
  display: flex;
  flex-direction: column;
  height: 100%;
}
.login-hero {
  background: linear-gradient(140deg, var(--blue) 0%, var(--blue-dark) 100%);
  padding: 44px 20px 30px;
  display: flex; flex-direction: column; align-items: center;
  flex-shrink: 0;
}
.login-logo {
  background: var(--yellow); color: var(--blue);
  font-size: 32px; font-weight: 900; padding: 12px 26px;
  border-radius: 14px; letter-spacing: 4px; margin-bottom: 10px;
}
.login-tagline { color: rgba(255,255,255,0.65); font-size: 12px; margin-top: 4px; }

.login-body { padding: 28px 20px 20px; flex: 1; overflow-y: auto; }
.login-heading { font-size: 22px; font-weight: 900; margin-bottom: 6px; }
.login-sub     { font-size: 13px; color: var(--gray); line-height: 1.5; margin-bottom: 24px; }

.form-group { margin-bottom: 14px; }
.form-label { font-size: 9.5px; font-weight: 900; letter-spacing: 1.5px; color: var(--gray); margin-bottom: 7px; display: block; }
.form-input {
  width: 100%; padding: 13px 15px;
  border: 2px solid var(--border); border-radius: 12px;
  font-size: 14px; font-family: 'Noto Sans', sans-serif;
  background: var(--bg); transition: border-color 0.18s; outline: none;
}
.form-input:focus { border-color: var(--blue); background: white; }

/* ── Login buttons stacked, full-width ── */
.login-actions {
  display: flex; flex-direction: column; gap: 10px; margin-top: 8px;
}
.login-actions .pay-btn { margin: 0; width: 100%; }
.login-forgot { text-align: right; font-size: 11.5px; color: var(--blue); font-weight: 700; cursor: pointer; margin-top: 4px; }
.login-switch { text-align: center; font-size: 12px; color: var(--gray); margin-top: 16px; }
.login-switch span { color: var(--blue); font-weight: 700; cursor: pointer; }

/* ────────────────────────────────── */
/*  CAMERA / SCANNER SCREENS          */
/* ────────────────────────────────── */
.camera-wrap {
  position: relative; 
  height: 350px; 
  background: #111;
  overflow: hidden; 
  flex-shrink: 0;
  transition: height 0.4s cubic-bezier(0.25, 0.8, 0.25, 1); 
  cursor: pointer; 
}

.camera-wrap.expanded {
  height: 350px; 
  cursor: default;
}
.camera-wrap video,
.camera-wrap canvas.cam-canvas {
  position: absolute; top: 0; left: 0;
  width: 100%; height: 100%; object-fit: cover; display: block;
}
.cam-canvas { display: none; }

/* QR scanner overlay */
.qr-overlay {
  position: absolute; inset: 0; pointer-events: none;
  display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 12px;
}
.qr-frame {
  width: 180px; height: 180px;
  border: 2px solid rgba(255,255,255,0.25); border-radius: 14px;
  position: relative; animation: qr-pulse 2s ease-in-out infinite;
}
@keyframes qr-pulse { 0%,100%{border-color:rgba(255,255,255,0.25);} 50%{border-color:rgba(0,88,163,0.85);box-shadow:0 0 20px rgba(0,88,163,0.5);} }
.qr-corner { position: absolute; width: 18px; height: 18px; border-color: white; border-style: solid; }
.qr-corner.tl { top:-1px; left:-1px; border-width:3px 0 0 3px; border-radius:4px 0 0 0; }
.qr-corner.tr { top:-1px; right:-1px; border-width:3px 3px 0 0; border-radius:0 4px 0 0; }
.qr-corner.bl { bottom:-1px; left:-1px; border-width:0 0 3px 3px; border-radius:0 0 0 4px; }
.qr-corner.br { bottom:-1px; right:-1px; border-width:0 3px 3px 0; border-radius:0 0 4px 0; }
.qr-scan-line {
  position: absolute; left: 8px; right: 8px; height: 2px;
  background: linear-gradient(90deg, transparent, #0058a3, transparent);
  animation: qr-scan 2s ease-in-out infinite;
}
@keyframes qr-scan { 0%{top:8px;} 100%{top:170px;} }
.qr-hint { color: rgba(255,255,255,0.7); font-size: 12px; font-weight: 600; }

.cam-status {
  position: absolute; top: 12px; left: 50%; transform: translateX(-50%);
  background: rgba(0,0,0,0.6); backdrop-filter: blur(6px);
  color: white; font-size: 11px; font-weight: 700;
  padding: 5px 13px; border-radius: 20px; white-space: nowrap; transition: all 0.3s;
}
.cam-status.detecting { color: var(--yellow); }
.cam-status.found     { color: #4ade80; }

.flash-btn {
  position: absolute; top: 12px; right: 12px;
  width: 38px; height: 38px; border-radius: 50%;
  background: rgba(255,255,255,0.18); border: none;
  color: white; font-size: 17px; cursor: pointer;
  display: flex; align-items: center; justify-content: center;
  transition: background 0.18s;
}
.flash-btn:active { background: rgba(255,219,0,0.5); }

.cam-no-access {
  position: absolute; inset: 0; background: #111;
  display: flex; flex-direction: column; align-items: center; justify-content: center;
  gap: 8px; color: rgba(255,255,255,0.55); font-size: 12px; text-align: center; padding: 20px;
}
.cam-no-access .icon { font-size: 44px; margin-bottom: 4px; }

/* AR extras */
.ar-controls-bar {
  position: absolute; bottom: 0; left: 0; right: 0;
  padding: 10px 14px 12px;
  background: linear-gradient(to top, rgba(0,0,0,0.75) 0%, transparent 100%);
  display: flex; align-items: center; gap: 9px;
}
#ar-capture-btn {
  flex: 1; background: var(--yellow); color: #111;
  border: none; border-radius: 13px; padding: 12px;
  font-size: 13px; font-weight: 900; cursor: pointer; transition: all 0.18s;
}
#ar-capture-btn:disabled { background: #888; color: #bbb; cursor: not-allowed; }
#ar-flip-btn {
  width: 42px; height: 42px; border-radius: 50%;
  background: rgba(255,255,255,0.18); border: 1px solid rgba(255,255,255,0.28);
  color: white; font-size: 17px; cursor: pointer;
  display: flex; align-items: center; justify-content: center; transition: all 0.18s;
}
#ar-snap-preview {
  position: absolute; top: 11px; right: 58px;
  width: 52px; height: 52px; border-radius: 9px;
  border: 2px solid white; object-fit: cover; display: none;
  box-shadow: 0 2px 8px rgba(0,0,0,0.4);
}
.ar-vf-overlay { position: absolute; inset: 0; pointer-events: none; }
.ar-vf-overlay::before {
  content: ''; position: absolute; inset: 0;
  background: radial-gradient(ellipse 55% 50% at 50% 50%, transparent 45%, rgba(0,0,0,0.42) 100%);
}
.ar-vf-frame {
  position: absolute; 
  top: 50%; 
  left: 50%; 
  transform: translate(-50%, -50%);  
  width: 52%; 
  aspect-ratio: 1; 
}
.ar-vf-frame .vc { position: absolute; width: 20px; height: 20px; border-color: var(--yellow); border-style: solid; }
.ar-vf-frame .vc.tl { top:0;left:0;border-width:3px 0 0 3px;border-radius:4px 0 0 0; }
.ar-vf-frame .vc.tr { top:0;right:0;border-width:3px 3px 0 0;border-radius:0 4px 0 0; }
.ar-vf-frame .vc.bl { bottom:0;left:0;border-width:0 0 3px 3px;border-radius:0 0 0 4px; }
.ar-vf-frame .vc.br { bottom:0;right:0;border-width:0 3px 3px 0;border-radius:0 0 4px 0; }

.ar-loader {
  display: none; position: absolute; inset: 0;
  background: rgba(0,0,0,0.55); backdrop-filter: blur(3px);
  flex-direction: column; align-items: center; justify-content: center;
  gap: 10px; color: white; font-size: 13px; font-weight: 700;
}
.ar-loader.on { display: flex; }
.ar-spinner { width: 36px; height: 36px; border: 4px solid rgba(255,255,255,0.2); border-top-color: var(--yellow); border-radius: 50%; animation: spin 0.7s linear infinite; }
@keyframes spin { to { transform: rotate(360deg); } }

.ar-conf-pill { display: inline-flex; align-items: center; gap: 3px; background: rgba(31,132,35,0.1); color: var(--green); font-size: 9.5px; font-weight: 700; padding: 2px 7px; border-radius: 6px; margin-top: 3px; }
.ar-conf-pill.low { background: rgba(239,119,68,0.1); color: var(--orange); }

/* ────────────────────────────────── */
/*  MAPA                              */
/* ────────────────────────────────── */
.map-container { background: var(--card); margin: 10px 14px; border-radius: 18px; overflow: hidden; box-shadow: 0 3px 10px rgba(0,0,0,0.09); }
.store-map { 
    position: relative; 
    height: 320px; /* Ajusta según necesites */
    background: #f8fafc; 
    border-bottom: 1px solid var(--border); 
    overflow: hidden;
}
.map-section { position: absolute; border-radius: 8px; display: flex; align-items: center; justify-content: center; font-size: 10.5px; font-weight: 700; cursor: pointer; transition: all 0.18s; }
.map-section:active { transform: scale(0.95); }
.map-section.highlighted { box-shadow: 0 0 0 2px var(--blue), 0 0 0 4px rgba(0,88,163,0.2); transform: scale(1.03); }
.corridor { position: absolute; border-radius: 6px; display: flex; align-items: flex-end; justify-content: center; padding-bottom: 3px; font-size: 9.5px; font-weight: 900; color: var(--blue); }
.you-dot { position: absolute; width: 13px; height: 13px; border-radius: 50%; background: var(--blue); border: 2px solid white; box-shadow: 0 0 0 4px rgba(0,88,163,0.3); animation: pulse-dot 2s ease-in-out infinite; z-index: 3; }
@keyframes pulse-dot { 0%,100%{box-shadow:0 0 0 4px rgba(0,88,163,0.3);} 50%{box-shadow:0 0 0 8px rgba(0,88,163,0.14);} }
.map-legend { padding: 10px 14px; display: flex; gap: 14px; }
.legend-item { display: flex; align-items: center; gap: 5px; font-size: 10.5px; font-weight: 600; }
.legend-dot { width: 9px; height: 9px; border-radius: 3px; }
.store-map { 
    position: relative; 
    height: 320px; /* Ajusta la altura si lo necesitas */
    background: #f0f4f8; 
    border-bottom: 1px solid var(--border); 
}

/* Pines numéricos de la ruta */
.route-dot { 
    position: absolute; 
    width: 22px; 
    height: 22px; 
    border-radius: 50%; 
    background: var(--blue, #0058a3); 
    color: white; 
    font-size: 11px; 
    font-weight: 900; 
    display: flex; 
    align-items: center; 
    justify-content: center; 
    border: 2px solid white; 
    box-shadow: 0 2px 5px rgba(0,0,0,0.3); 
    z-index: 10; 
    /* Centrado exacto */
    transform: translate(-50%, -50%);
    cursor: pointer;
    transition: transform 0.2s, background 0.3s;
}

.route-dot:active { transform: translate(-50%, -50%) scale(0.9); }

/* Pin cuando el producto ya está recogido */
.route-dot.status-done {
    background: var(--green, #1f8423);
    border-color: #1a6d1d;
}

/* El SVG que dibujará las líneas rectilíneas */
.map-route-svg {
    position: absolute; 
    top: 0; left: 0; 
    width: 100%; height: 100%; 
    pointer-events: none; /* No interfiere con los clics en los pines */
    z-index: 5;
}

.route-item-card {
    background: var(--card);
    border-radius: 13px;
    margin: 0 14px 8px;
    padding: 12px;
    display: flex;
    align-items: center;
    gap: 12px;
    box-shadow: 0 2px 6px rgba(0,0,0,0.06);
    transition: background-color 0.3s;
}

.route-step-number {
    width: 28px; height: 28px;
    border-radius: 50%;
    background: var(--blue, #0058a3);
    color: white;
    font-size: 12px; font-weight: 900;
    display: flex; align-items: center; justify-content: center;
    flex-shrink: 0;
}

.route-step-number.status-done { background: var(--green, #1f8423); }

/* Ajuste de la tarjeta de la lista de abajo */
.route-item { 
    background: var(--card); 
    border-radius: 13px; 
    margin: 0 14px 7px; 
    padding: 11px 13px; 
    display: flex; 
    align-items: center; 
    gap: 11px; 
    box-shadow: 0 2px 6px rgba(0,0,0,0.06); 
    transition: background-color 0.3s;
}
.route-step { 
    width: 28px; height: 28px; 
    border-radius: 50%; 
    background: var(--blue, #0058a3); 
    color: white; 
    font-size: 11px; font-weight: 900; 
    display: flex; align-items: center; justify-content: center; 
    flex-shrink: 0; 
}
.route-step.done { background: var(--green, #1f8423); }

/* ────────────────────────────────── */
/*  CESTA / FAVORITOS / PAGAR         */
/* ────────────────────────────────── */
.pay-summary { background: var(--card); border-radius: 18px; margin: 10px 14px; overflow: hidden; box-shadow: 0 3px 10px rgba(0,0,0,0.08); }
.pay-header  { background: var(--blue); padding: 12px 14px; color: white; font-size: 10px; font-weight: 900; letter-spacing: 1.5px; }
.pay-item    { display: flex; align-items: center; gap: 11px; padding: 11px 14px; border-bottom: 1px solid var(--border); cursor: pointer; }
.pay-item:active { background: rgba(0,88,163,0.03); }
.pay-item-img  { width: 38px; height: 38px; border-radius: 8px; background: #dde; display: flex; align-items: center; justify-content: center; font-size: 19px; flex-shrink: 0; }
.pay-item-info { flex: 1; }
.pay-item-name { font-size: 12.5px; font-weight: 700; }
.pay-item-loc  { font-size: 9.5px; color: var(--blue); font-weight: 600; }
.pay-item-price { font-size: 13px; font-weight: 900; }
.pay-total-row  { display: flex; justify-content: space-between; padding: 7px 14px; font-size: 12.5px; }
.pay-total-row.final { font-weight: 900; font-size: 14px; padding: 11px 14px; border-top: 2px solid var(--border); background: rgba(0,88,163,0.04); }
.payment-option { background: var(--card); border-radius: 13px; margin: 0 14px 7px; padding: 13px 14px; display: flex; align-items: center; gap: 13px; cursor: pointer; border: 2px solid var(--border); transition: all 0.18s; }
.payment-option.selected { border-color: var(--blue); background: rgba(0,88,163,0.04); }
.payment-option:active { transform: scale(0.98); }
.payment-icon { width: 34px; height: 34px; border-radius: 50%; background: var(--bg); display: flex; align-items: center; justify-content: center; font-size: 17px; flex-shrink: 0; }
.payment-name   { font-size: 12.5px; font-weight: 700; }
.payment-detail { font-size: 10.5px; color: var(--gray); }
.radio-dot { width: 20px; height: 20px; border-radius: 50%; border: 2px solid var(--border); display: flex; align-items: center; justify-content: center; flex-shrink: 0; }
.payment-option.selected .radio-dot { border-color: var(--blue); background: var(--blue); }
.radio-dot::after { content: ''; width: 8px; height: 8px; border-radius: 50%; }
.payment-option.selected .radio-dot::after { background: white; }

/* ────────────────────────────────── */
/*  FAVORITOS                         */
/* ────────────────────────────────── */
.fav-product-card { background: var(--card); border-radius: 16px; margin: 0 14px 9px; padding: 14px; box-shadow: 0 2px 8px rgba(0,0,0,0.07); }
.fav-card-top { display: flex; gap: 11px; align-items: flex-start; margin-bottom: 11px; }

/* ────────────────────────────────── */
/*  HISTORIAL                         */
/* ────────────────────────────────── */
.order-card    { background: var(--card); border-radius: 16px; margin: 0 14px 9px; overflow: hidden; box-shadow: 0 2px 8px rgba(0,0,0,0.07); }
.order-header  { padding: 11px 14px; background: rgba(0,88,163,0.05); display: flex; align-items: center; gap: 11px; border-bottom: 1px solid var(--border); }
.order-date-badge { width: 48px; height: 48px; border-radius: 11px; background: var(--blue); color: white; display: flex; flex-direction: column; align-items: center; justify-content: center; font-size: 9.5px; font-weight: 900; flex-shrink: 0; }
.order-date-day  { font-size: 17px; font-weight: 900; line-height: 1; }
.order-title   { font-size: 13.5px; font-weight: 700; }
.order-subtitle { font-size: 10.5px; color: var(--gray); margin-top: 2px; }
.order-price   { font-size: 14px; font-weight: 900; color: var(--blue); }
.order-item    { padding: 9px 14px; display: flex; gap: 9px; align-items: center; border-bottom: 1px solid var(--border); }
.order-item:last-child { border-bottom: none; }
.order-item-img { width: 42px; height: 42px; border-radius: 9px; background: #dde; display: flex; align-items: center; justify-content: center; font-size: 20px; flex-shrink: 0; }

/* ────────────────────────────────── */
/*  PERFIL                            */
/* ────────────────────────────────── */
.perfil-hero {
  background: linear-gradient(140deg, var(--blue) 0%, var(--blue-dark) 100%);
  padding: 18px 18px 24px; display: flex; flex-direction: column; align-items: center; flex-shrink: 0;
}
.perfil-avatar-wrap { position: relative; margin-bottom: 11px; }
.perfil-avatar-big {
  width: 84px; height: 84px; border-radius: 50%;
  background: var(--yellow); display: flex; align-items: center; justify-content: center;
  font-size: 42px; border: 4px solid rgba(255,255,255,0.28); cursor: pointer;
}
.perfil-avatar-edit {
  position: absolute; bottom: 0; right: 0; width: 26px; height: 26px;
  border-radius: 50%; background: white; display: flex; align-items: center;
  justify-content: center; font-size: 12px; cursor: pointer; box-shadow: 0 2px 6px rgba(0,0,0,0.2);
}
.perfil-name  { font-size: 20px; font-weight: 900; color: white; }
.perfil-email { font-size: 11.5px; color: rgba(255,255,255,0.7); margin-top: 3px; }
.perfil-badge { display: inline-flex; align-items: center; gap: 4px; background: rgba(255,219,0,0.18); border: 1px solid rgba(255,219,0,0.5); color: var(--yellow); font-size: 10.5px; font-weight: 700; padding: 4px 11px; border-radius: 20px; margin-top: 9px; }

.perfil-stats { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 9px; padding: 14px; }
.perfil-stat-card { background: var(--card); border-radius: 16px; padding: 14px 7px; text-align: center; box-shadow: 0 2px 8px rgba(0,0,0,0.07); }
.perfil-stat-emoji { font-size: 24px; display: block; margin-bottom: 5px; }
.perfil-stat-val   { font-size: 18px; font-weight: 900; color: var(--blue); }
.perfil-stat-lbl   { font-size: 9px; color: var(--gray); font-weight: 700; letter-spacing: 0.5px; margin-top: 2px; }

.family-card {
  margin: 0 14px 13px; border-radius: 18px; padding: 18px;
  background: linear-gradient(130deg, #004ea3, #002f6c 60%, #001d44);
  color: white; box-shadow: 0 4px 20px rgba(0,88,163,0.4); position: relative; overflow: hidden;
}
.family-card::before { content:''; position:absolute; top:-28px; right:-28px; width:110px; height:110px; border-radius:50%; background:rgba(255,219,0,0.08); }
.family-logo-row  { display: flex; align-items: center; gap: 9px; margin-bottom: 18px; }
.family-logo      { background: var(--yellow); color: var(--blue); font-size: 11px; font-weight: 900; padding: 3px 7px; border-radius: 5px; }
.family-brand     { font-size: 11px; font-weight: 700; opacity: 0.65; letter-spacing: 1px; }
.family-card-number { font-size: 15px; font-weight: 700; letter-spacing: 3px; opacity: 0.88; font-family: monospace; margin-bottom: 14px; }
.family-bottom    { display: flex; justify-content: space-between; align-items: flex-end; }
.family-name      { font-size: 13px; font-weight: 900; }
.family-points-val { font-size: 18px; font-weight: 900; color: var(--yellow); }
.family-points-lbl { font-size: 8.5px; opacity: 0.6; font-weight: 700; letter-spacing: 0.5px; }
.family-level     { display: inline-flex; align-items: center; gap: 4px; background: rgba(255,219,0,0.18); border: 1px solid rgba(255,219,0,0.4); color: var(--yellow); font-size: 9px; font-weight: 700; padding: 2px 7px; border-radius: 9px; margin-top: 3px; }

.perfil-section { margin: 0 14px 13px; }
.perfil-section-title { font-size: 9.5px; font-weight: 900; letter-spacing: 1.5px; color: var(--gray); margin-bottom: 7px; padding-left: 3px; }
.perfil-card  { background: var(--card); border-radius: 16px; overflow: hidden; box-shadow: 0 2px 8px rgba(0,0,0,0.07); }
.perfil-field { display: flex; align-items: center; gap: 11px; padding: 12px 14px; border-bottom: 1px solid var(--border); cursor: pointer; transition: background 0.14s; }
.perfil-field:last-child { border-bottom: none; }
.perfil-field:active { background: rgba(0,88,163,0.04); }
.perfil-field-icon  { font-size: 17px; width: 30px; text-align: center; flex-shrink: 0; }
.perfil-field-label { font-size: 9.5px; color: var(--gray); font-weight: 700; letter-spacing: 0.5px; }
.perfil-field-val   { font-size: 12.5px; font-weight: 700; margin-top: 1px; }

/* Edit overlay */
.edit-overlay {
  display: none; position: absolute; inset: 0; background: rgba(0,0,0,0.55);
  z-index: 200; align-items: flex-end; justify-content: center;
}
.edit-overlay.open { display: flex; animation: slideUp 0.28s ease-out; }
@keyframes slideUp { from{transform:translateY(60px);opacity:0;} to{transform:translateY(0);opacity:1;} }
.edit-sheet {
  background: var(--card); border-radius: 22px 22px 0 0;
  padding: 20px 20px 30px; width: 100%; max-height: 75%;
  overflow-y: auto;
}
.edit-sheet-handle { width: 40px; height: 4px; background: var(--border); border-radius: 2px; margin: 0 auto 16px; }
.edit-sheet-title  { font-size: 17px; font-weight: 900; margin-bottom: 18px; }
.edit-sheet .form-group { margin-bottom: 12px; }
.edit-sheet-actions { display: flex; gap: 10px; margin-top: 18px; }
.edit-sheet-actions button { flex: 1; }

/* ────────────────────────────────── */
/*  PRODUCTO DETAIL                   */
/* ────────────────────────────────── */
.producto-hero {
  position: relative; height: 240px; background: linear-gradient(135deg, #e8e8f4, #d0d8f0);
  flex-shrink: 0; display: flex; align-items: center; justify-content: center;
  overflow: hidden;
}
.producto-hero-emoji { font-size: 110px; opacity: 0.9; }
.producto-badge {
  position: absolute; top: 14px; right: 14px;
  background: var(--yellow); color: var(--blue-dark);
  font-size: 11px; font-weight: 900; padding: 5px 11px; border-radius: 20px;
}
.producto-body { padding: 18px; }
.producto-name-row { display: flex; align-items: flex-start; justify-content: space-between; gap: 8px; margin-bottom: 6px; }
.producto-name  { font-size: 24px; font-weight: 900; }
.producto-desc  { font-size: 13px; color: var(--gray); line-height: 1.5; margin-bottom: 14px; }
.producto-price { font-size: 28px; font-weight: 900; color: var(--blue); }
.producto-price-sub { font-size: 11px; color: var(--gray); margin-top: 1px; }

.info-pills { display: flex; gap: 7px; flex-wrap: wrap; margin-bottom: 16px; }
.info-pill  {
  display: inline-flex; align-items: center; gap: 5px;
  background: var(--card); border-radius: 10px; padding: 7px 11px;
  font-size: 11px; font-weight: 700; box-shadow: 0 2px 6px rgba(0,0,0,0.07);
}
.info-pill .pill-label { font-size: 9px; color: var(--gray); display: block; }
.info-pill .pill-val   { font-weight: 900; font-size: 12px; }

.producto-actions { display: flex; gap: 9px; margin-bottom: 14px; }
.producto-actions .btn { padding: 13px; font-size: 13px; }

.related-title { font-size: 10px; font-weight: 900; letter-spacing: 1.5px; color: var(--gray); margin-bottom: 10px; }
.related-scroll { display: flex; gap: 10px; overflow-x: auto; padding-bottom: 4px; }
.related-scroll::-webkit-scrollbar { height: 0; }
.related-card {
  background: var(--card); border-radius: 14px; padding: 12px;
  min-width: 130px; flex-shrink: 0; cursor: pointer; box-shadow: 0 2px 6px rgba(0,0,0,0.08);
  transition: transform 0.15s;
}
.related-card:active { transform: scale(0.95); }
.related-emoji { font-size: 32px; display: block; margin-bottom: 7px; }
.related-name  { font-size: 12px; font-weight: 700; }
.related-price { font-size: 11px; color: var(--blue); font-weight: 900; margin-top: 2px; }

/* QR result */
.qr-result {
  background: rgba(31,132,35,0.08); border: 1px solid rgba(31,132,35,0.25);
  border-radius: 13px; padding: 12px 14px; display: flex; align-items: center;
  gap: 10px; margin: 11px 14px;
}
.qr-result-icon { font-size: 22px; }
.qr-result-name { font-size: 13.5px; font-weight: 700; color: var(--green); }
.qr-result-detail { font-size: 10.5px; color: var(--gray); margin-top: 1px; }

/* ────────────────────────────────── */
/*  PROFILE CARD (config)             */
/* ────────────────────────────────── */
.profile-card {
  background: linear-gradient(135deg, var(--blue), var(--blue-dark));
  margin: 10px 14px; border-radius: 20px; padding: 18px; color: white;
  display: flex; align-items: center; gap: 14px;
  box-shadow: 0 4px 18px rgba(0,88,163,0.32); cursor: pointer; transition: transform 0.15s;
}
.profile-card:active { transform: scale(0.98); }
.profile-avatar { width: 66px; height: 66px; border-radius: 50%; background: var(--yellow); display: flex; align-items: center; justify-content: center; font-size: 30px; flex-shrink: 0; border: 3px solid rgba(255,255,255,0.25); }
.profile-name   { font-size: 17px; font-weight: 900; }
.profile-email  { font-size: 11.5px; opacity: 0.7; margin-top: 2px; }
.profile-badge  { display: inline-flex; align-items: center; gap: 4px; background: rgba(255,219,0,0.18); border: 1px solid rgba(255,219,0,0.4); color: var(--yellow); font-size: 9.5px; font-weight: 700; padding: 3px 8px; border-radius: 9px; margin-top: 5px; }
.profile-chevron { font-size: 17px; opacity: 0.55; margin-left: auto; }

/* ────────────────────────────────── */
/*  SUCCESS                           */
/* ────────────────────────────────── */
.success-screen { display: flex; flex-direction: column; align-items: center; justify-content: center; height: 100%; padding: 36px; text-align: center; gap: 18px; background: linear-gradient(135deg, #f0f8f0, var(--card)); }
.success-icon { width: 96px; height: 96px; border-radius: 50%; background: rgba(31,132,35,0.1); border: 3px solid var(--green); display: flex; align-items: center; justify-content: center; font-size: 46px; animation: success-pop 0.5s cubic-bezier(0.175,0.885,0.32,1.275); }
@keyframes success-pop { 0%{transform:scale(0);} 100%{transform:scale(1);} }
.success-title  { font-size: 22px; font-weight: 900; color: var(--green); }
.success-subtitle { font-size: 13px; color: var(--gray); line-height: 1.5; }

/* ────────────────────────────────── */
/*  ANIMATIONS & MISC                 */
/* ────────────────────────────────── */
@keyframes fadeInUp { from{opacity:0;transform:translateY(18px);} to{opacity:1;transform:translateY(0);} }
.fade-in-up { animation: fadeInUp 0.36s ease-out forwards; }
.s1{animation-delay:.05s;opacity:0;} .s2{animation-delay:.10s;opacity:0;}
.s3{animation-delay:.15s;opacity:0;} .s4{animation-delay:.20s;opacity:0;}
.s5{animation-delay:.25s;opacity:0;}

@keyframes ripple { to{transform:scale(3);opacity:0;} }
.ripple-btn { position: relative; overflow: hidden; }
.ripple-btn .ripple-effect { position: absolute; border-radius: 50%; width: 40px; height: 40px; margin-top:-20px; margin-left:-20px; background: rgba(255,255,255,0.28); animation: ripple 0.5s linear; pointer-events: none; }

.badge-animate { animation: badge-bounce 0.28s ease-in-out; }
@keyframes badge-bounce { 0%,100%{transform:scale(1);} 50%{transform:scale(1.3);} }

.toast {
  position: fixed; bottom: 88px; left: 50%;
  transform: translateX(-50%) translateY(18px);
  background: #1a1a1a; color: white; padding: 11px 18px;
  border-radius: 22px; font-size: 12.5px; font-weight: 600;
  opacity: 0; transition: all 0.28s; pointer-events: none; z-index: 1000;
  white-space: nowrap; max-width: 340px;
}
.toast.show { opacity: 1; transform: translateX(-50%) translateY(0); }

@media (max-width: 768px) {
  html, body {
    background: var(--bg) !important;
    display: block !important;
    height: 100dvh !important;
    overflow: hidden !important; /* Esto es lo que impide que la barra del navegador se esconda */
    overscroll-behavior: none !important; /* Evita el efecto rebote al llegar al final */
  }

  .phone {
    width: 100vw !important;
    height: 100vh !important;
    height: 100dvh !important;
    border-radius: 0 !important;
    box-shadow: none !important;
    border: none !important;
  }

  .screens {
    height: 100dvh !important;
  }

  .screens {
    height: 100dvh !important;
  }
  
  .status-bar {
    display: none !important;
  }
  
  .header {
    top: 0 !important; /* Sin status-bar, el header sube arriba del todo */
    padding-top: max(14px, env(safe-area-inset-top)) !important;
  }
  
  .bottom-nav {
    position: fixed !important;
    bottom: 0 !important;
    left: 0 !important;
    right: 0 !important;
    height: 75px !important;
    vertical-align: middle;
  }
  
  .content {
    overflow-y: auto !important;
    -webkit-overflow-scrolling: touch !important;
    padding-bottom: calc(72px + env(safe-area-inset-bottom)) !important;
  }
  
  .camera-wrap {
    position: relative; 
    height: 300px; 
    background: #111;
    overflow: hidden; 
    flex-shrink: 0;
    transition: height 0.4s cubic-bezier(0.25, 0.8, 0.25, 1); 
    cursor: pointer; 
  }

  .camera-wrap.expanded {
    height: calc(100dvh - 130px) !important;
  }
  
}

</style>
</head>
<body>
<div class="phone">
<div class="screens" id="screens">

<!-- ═══════════════════════════════════════ LOGIN ═══ -->
<div class="screen active" id="screen-login">
  <div class="login-screen">
    <div class="login-hero">
      <div class="login-logo">IKEA</div>
      <div class="login-tagline">IKEA Bilbao · Planta 1</div>
    </div>
    <div class="login-body">
      <div class="login-heading">¡Hola de nuevo! 👋</div>
      <div class="login-sub">Inicia sesión para acceder a tu lista, ver tus favoritos y pagar más rápido.</div>
      <div class="form-group">
        <label class="form-label">CORREO ELECTRÓNICO</label>
        <input class="form-input" id="login-email" type="email" placeholder="ejemplo@correo.com" value="usuario@ikea.es">
      </div>
      <div class="form-group">
        <label class="form-label">CONTRASEÑA</label>
        <input class="form-input" id="login-pass" type="password" placeholder="••••••••••" value="••••••••">
      </div>
      <div class="login-forgot" onclick="showToast('📧 Revisa tu correo para recuperar contraseña')">¿Has olvidado tu contraseña?</div>
      <div class="login-actions">
        <button class="pay-btn ripple-btn" onclick="doLogin()">Iniciar sesión</button>
        <button class="pay-btn secondary ripple-btn" onclick="goTo('registro')">Crear cuenta de IKEA</button>
      </div>
      <div class="login-switch">¿Todavía no tienes cuenta? <span onclick="goTo('registro')">Regístrate</span></div>
    </div>
  </div>
</div>

<!-- ═══════════════════════════════════════ REGISTRO ═══ -->
<div class="screen" id="screen-registro">
  <div class="login-screen">
    <div class="login-hero">
      <div class="login-logo">IKEA</div>
      <div class="login-tagline">IKEA Family</div>
    </div>
    <div class="login-body">
      <div class="login-heading">Únete a IKEA Family 💛</div>
      <div class="login-sub">Crea tu cuenta gratis y disfruta de descuentos exclusivos, café gratis y mucho más.</div>
      <div class="form-group">
        <label class="form-label">NOMBRE</label>
        <input class="form-input" id="reg-name" type="text" placeholder="Tu nombre completo">
      </div>
      <div class="form-group">
        <label class="form-label">CORREO ELECTRÓNICO</label>
        <input class="form-input" id="reg-email" type="email" placeholder="ejemplo@correo.com">
      </div>
      <div class="form-group">
        <label class="form-label">CONTRASEÑA</label>
        <input class="form-input" id="reg-pass" type="password" placeholder="Mínimo 8 caracteres">
      </div>
      <div class="login-actions">
        <button class="pay-btn ripple-btn" onclick="doRegister()">Crear cuenta</button>
        <button class="pay-btn secondary ripple-btn" onclick="goTo('login')">Ya tengo una cuenta</button>
      </div>
    </div>
  </div>
</div>

<!-- ═══════════════════════════════════════ INICIO ═══ -->
<div class="screen" id="screen-inicio">
  <div class="status-bar"><span class="status-time">12:00</span><div class="status-icons">📶 🔋</div></div>
  <div class="header">
    <div class="logo" onclick="navTo('inicio')" style="cursor: pointer;">IKEA</div>
    <span class="header-title">Inicio</span>
    <div class="header-icons">
      <div class="cart-icon-wrap" onclick="goTo('cesta')">
        <div class="header-icon">🛒</div>
        <div class="cart-badge" id="cart-badge">0</div>
      </div>
      <button class="header-icon" onclick="goTo('config')">⚙️</button>
      <button class="header-icon" onclick="goTo('perfil')">👤</button>
    </div>
  </div>
  <div class="content">
    <div class="hero-banner">
      <div class="hero-location">IKEA BILBAO · PLANTA 1</div>
      <div class="hero-title">Compra sin carrito,<br>solo con tu móvil 📱</div>
      <div class="hero-chips">
        <span class="hero-chip">📷 Escaneo</span>
        <span class="hero-chip">🔮 AR</span>
        <span class="hero-chip">🗺️ Mapa</span>
        <span class="hero-chip">🎙️ Voz</span>
      </div>
    </div>
    <div class="action-grid">
      <button class="action-card fade-in-up s1 ripple-btn" onclick="goTo('buscar')">
        <span class="action-emoji">🔍</span>
        <span class="action-name">Búsqueda</span>
        <span class="action-desc">Manual · Por voz</span>
      </button>
      <button class="action-card fade-in-up s2 ripple-btn" onclick="goTo('escaner')">
        <span class="action-emoji">📷</span>
        <span class="action-name" style="color:var(--blue)">Escanear</span>
        <span class="action-desc">QR · Código de barras</span>
      </button>
      <button class="action-card fade-in-up s3 ripple-btn" onclick="goTo('ar')">
        <span class="action-emoji">🔮</span>
        <span class="action-name" style="color:var(--purple)">Vista AR</span>
        <span class="action-desc">Reconocer muebles</span>
      </button>
      <button class="action-card fade-in-up s4 ripple-btn" onclick="goTo('mapa')">
        <span class="action-emoji">🗺️</span>
        <span class="action-name" style="color:var(--green)">Mapa</span>
        <span class="action-desc">Ruta óptima por pasillos</span>
      </button>
    </div>
    <div id="panel-gestos">
      <div class="gestures-card" onclick="requestSensorPermissions()" style="margin: 15px; padding: 10px; background: rgba(255,219,0,0.1); border-left: 4px solid var(--yellow, #FFDB00); border-radius: 8px; cursor: pointer; text-align: left; box-shadow: 0 2px 5px rgba(0,0,0,0.05); min-height: 135px; display: flex; flex-direction: column; justify-content: center;">
          <div style="font-size: 16px; margin-bottom: 20px; color: #111;">📳 <strong>Activar Gestos Inteligentes</strong></div>
          <div style="font-size: 13px; color: #444;">Toca aquí para habilitar los sensores. Agita el móvil para borrar o inclínalo para favoritos.</div>
      </div>
    </div>
    <div class="section-title" style="margin-top:16px;">PRODUCTOS POPULARES</div>
    <div class="related-scroll" id="home-products" style="padding:0 14px;">
      <div style="text-align:center;padding:30px;color:var(--gray);">Cargando productos...</div>
    </div>
    <div style="height:14px;"></div>
  </div>
  <div class="bottom-nav">
    <button class="nav-item active" onclick="navTo('inicio')"><span class="nav-emoji">🏠</span><span class="nav-label">Inicio</span></button>
    <button class="nav-item" onclick="navTo('shopping')"><span class="nav-emoji">🛍️</span><span class="nav-label">Shopping</span></button>
    <button class="nav-item" onclick="navTo('cesta')"><span class="nav-emoji">🛒</span><span class="nav-label">Cesta</span></button>
    <button class="nav-item" onclick="navTo('favoritos')"><span class="nav-emoji">⭐</span><span class="nav-label">Favoritos</span></button>
    <button class="nav-item" onclick="navTo('mapa')"><span class="nav-emoji">🗺️</span><span class="nav-label">Mapa</span></button>
  </div>
</div>

<!-- ═══════════════════════════════════════ BUSCAR ═══ -->
<div class="screen" id="screen-buscar">
  <div class="status-bar"><span class="status-time">12:00</span><div class="status-icons">📶 🔋</div></div>
  <div class="header">
    <button class="header-icon" onclick="goBack()">←</button>
    <span class="header-title">Búsqueda</span>
    <div class="header-icons">
      <div class="cart-icon-wrap" onclick="goTo('cesta')"><div class="header-icon">🛒</div><div class="cart-badge" id="cart-badge-buscar">5</div></div>
    </div>
  </div>
  <div class="content">
    <div style="padding:11px 14px;">
      <div style="background:var(--card);border-radius:13px;padding:11px 14px;display:flex;align-items:center;gap:9px;box-shadow:0 2px 6px rgba(0,0,0,0.07);border:2px solid var(--blue);">
        <span style="font-size:19px;">🔍</span>
        <input style="flex:1;border:none;background:none;font-size:13.5px;font-family:'Noto Sans',sans-serif;outline:none;" placeholder="Buscar en catálogo IKEA..." oninput="filterProducts(this.value)" id="search-input">
        <button style="background:var(--bg);border:none;padding:5px 10px;border-radius:7px;font-size:19px;cursor:pointer;" onclick="startVoiceSearch()">🎙️</button>
      </div>
    </div>
    <div class="section-title">RESULTADOS</div>
    <div id="search-results"></div>
    <div style="height:14px;"></div>
  </div>
  <div class="bottom-nav">
    <button class="nav-item" onclick="navTo('inicio')"><span class="nav-emoji">🏠</span><span class="nav-label">Inicio</span></button>
    <button class="nav-item active" onclick="navTo('shopping')"><span class="nav-emoji">🛍️</span><span class="nav-label">Shopping</span></button>
    <button class="nav-item" onclick="navTo('cesta')"><span class="nav-emoji">🛒</span><span class="nav-label">Cesta</span></button>
    <button class="nav-item" onclick="navTo('favoritos')"><span class="nav-emoji">⭐</span><span class="nav-label">Favoritos</span></button>
    <button class="nav-item" onclick="navTo('mapa')"><span class="nav-emoji">🗺️</span><span class="nav-label">Mapa</span></button>
  </div>
</div>

<!-- ═══════════════════════════════════════ ESCÁNER QR ═══ -->
<div class="screen" id="screen-escaner">
  <div class="status-bar"><span class="status-time">12:00</span><div class="status-icons">📶 🔋</div></div>
  <div class="header">
    <button class="header-icon" onclick="goBack()">←</button>
    <span class="header-title">Escaneo QR</span>
    <div class="header-icons">
      <div class="cart-icon-wrap" onclick="goTo('cesta')"><div class="header-icon">🛒</div><div class="cart-badge">5</div></div>
    </div>
  </div>
  <div class="content">
    <!-- Cámara real para QR -->
    <div class="camera-wrap" id="qr-camera-wrap" onclick="expandCamera('qr', event)">    <video id="qr-video" autoplay playsinline muted></video>
      <canvas id="qr-canvas" class="cam-canvas"></canvas>
      <div class="qr-overlay">
        <div class="qr-frame">
          <div class="qr-corner tl"></div>
          <div class="qr-corner tr"></div>
          <div class="qr-corner bl"></div>
          <div class="qr-corner br"></div>
          <div class="qr-scan-line"></div>
        </div>
        <p class="qr-hint">Centra el código QR en el recuadro</p>
      </div>
      <div class="cam-status" id="qr-status">📷 Buscando código QR...</div>
      <button class="flash-btn" id="qr-flash-btn" onclick="toggleQRFlash()">💡</button>
      <div class="cam-no-access" id="qr-no-cam" style="display:none;">
        <div class="icon">📷</div>
        <div>Cámara no disponible</div>
        <div style="font-size:10.5px;color:rgba(255,255,255,0.4);margin-top:3px;">Acepta el permiso y recarga la página</div>
      </div>
    </div>
    <div id="qr-detected" style="display:none;">
      <div class="qr-result">
        <span class="qr-result-icon">✅</span>
        <div>
          <div class="qr-result-name" id="qr-product-name">-</div>
          <div class="qr-result-detail" id="qr-product-detail">-</div>
        </div>
      </div>
      <div style="display:flex;gap:8px;padding:0 14px 12px;">
        <button class="btn btn-primary" style="flex:2;" id="qr-add-btn" onclick="qrAddToCart()">+ Añadir a cesta</button>
        <button class="btn btn-fav" id="qr-fav-btn" onclick="qrToggleFav()">☆</button>
        <button class="btn btn-outline" onclick="qrViewProduct()">Ver</button>
      </div>
    </div>
    <div style="padding:14px;display:flex;flex-direction:column;gap:9px;">
      <div style="display:flex;align-items:center;gap:9px;">
        <div style="flex:1;height:1px;background:var(--border);"></div>
        <span style="font-size:10.5px;font-weight:700;color:var(--gray);">O BUSCA MANUALMENTE</span>
        <div style="flex:1;height:1px;background:var(--border);"></div>
      </div>
      <div style="padding:10px 14px;"><div style="background:var(--card);border-radius:13px;padding:11px 14px;display:flex;align-items:center;gap:9px;box-shadow:0 2px 6px rgba(0,0,0,0.07);"><span>🔍</span><input style="flex:1;border:none;background:none;font-size:13.5px;font-family:'Noto Sans',sans-serif;outline:none;" placeholder="Buscar en catálogo..." onclick="openSearchAndFocus()"><span style="cursor:pointer;" onclick="startVoiceSearch()">🎙️</span></div></div>
    </div>
  </div>
  <div class="bottom-nav">
    <button class="nav-item active" onclick="navTo('inicio')"><span class="nav-emoji">🏠</span><span class="nav-label">Inicio</span></button>
    <button class="nav-item" onclick="navTo('shopping')"><span class="nav-emoji">🛍️</span><span class="nav-label">Shopping</span></button>
    <button class="nav-item" onclick="navTo('cesta')"><span class="nav-emoji">🛒</span><span class="nav-label">Cesta</span></button>
    <button class="nav-item" onclick="navTo('favoritos')"><span class="nav-emoji">⭐</span><span class="nav-label">Favoritos</span></button>
    <button class="nav-item" onclick="navTo('mapa')"><span class="nav-emoji">🗺️</span><span class="nav-label">Mapa</span></button>
  </div>
</div>

<!-- ═══════════════════════════════════════ VISTA AR ═══ -->
<div class="screen" id="screen-ar">
  <div class="status-bar"><span class="status-time">12:00</span><div class="status-icons">📶 🔋</div></div>
  <div class="header">
    <button class="header-icon" onclick="goBack()">←</button>
    <span class="header-title">Vista AR · IA</span>
    <div class="header-icons">
      <div class="cart-icon-wrap" onclick="goTo('cesta')"><div class="header-icon">🛒</div><div class="cart-badge">5</div></div>
    </div>
  </div>
    <div class="content">
    <div class="camera-wrap" id="ar-camera-wrap" onclick="expandCamera('ar', event)">    <video id="ar-video" autoplay playsinline muted></video>
      <canvas id="ar-canvas" class="cam-canvas"></canvas>
      <div class="ar-vf-overlay">
        <div class="ar-vf-frame">
          <div class="vc tl"></div><div class="vc tr"></div>
          <div class="vc bl"></div><div class="vc br"></div>
        </div>
      </div>
      <div class="cam-status" id="ar-status">📷 Centra el mueble</div>
      <img id="ar-snap-preview" src="" alt="">
      <div class="ar-loader" id="ar-loader"><div class="ar-spinner"></div>Identificando con IA…</div>
      <div class="cam-no-access" id="ar-no-cam" style="display:none;">
        <div class="icon">📷</div>
        <div>Cámara no disponible</div>
        <div style="font-size:10.5px;color:rgba(255,255,255,0.4);margin-top:3px;">Acepta el permiso y recarga</div>
      </div>
      <div class="ar-controls-bar">
        <button id="ar-capture-btn" class="ripple-btn" onclick="arCapture()">🔍 Identificar mueble</button>
        <button id="ar-flip-btn" onclick="arFlipCamera()">🔄</button>
      </div>
    </div>
    <div style="padding:11px 14px 0;">
      <p class="section-title" style="padding:0 0 7px;">DETECTADO POR IA</p>
      <div style="background:var(--card);border-radius:16px;overflow:hidden;box-shadow:0 2px 8px rgba(0,0,0,0.07);">
        <div id="ar-detected-list">
          <div style="padding:22px 14px;text-align:center;color:var(--gray);">
            <div style="font-size:38px;margin-bottom:7px;">🔮</div>
            <div style="font-size:12.5px;font-weight:700;">Apunta a un mueble<br>y pulsa <strong>Identificar</strong></div>
          </div>
        </div>
      </div>
    </div>
    <div style="padding:10px 14px 0;">
      <p class="section-title" style="padding:0 0 7px;">PRODUCTOS POPULARES</p>
      <div id="ar-suggested-products" class="related-scroll" style="padding:0;">
        <div style="text-align:center;padding:20px;color:var(--gray);">Cargando...</div>
      </div>
    </div>
    <div style="padding:10px 14px;"><div style="background:var(--card);border-radius:13px;padding:11px 14px;display:flex;align-items:center;gap:9px;box-shadow:0 2px 6px rgba(0,0,0,0.07);"><span>🔍</span><input style="flex:1;border:none;background:none;font-size:13.5px;font-family:'Noto Sans',sans-serif;outline:none;" placeholder="Buscar en catálogo..." onclick="openSearchAndFocus()"><span style="cursor:pointer;" onclick="startVoiceSearch()">🎙️</span></div></div>
    <div style="height:14px;"></div>
  </div>
  <div class="bottom-nav">
    <button class="nav-item active" onclick="navTo('inicio')"><span class="nav-emoji">🏠</span><span class="nav-label">Inicio</span></button>
    <button class="nav-item" onclick="navTo('shopping')"><span class="nav-emoji">🛍️</span><span class="nav-label">Shopping</span></button>
    <button class="nav-item" onclick="navTo('cesta')"><span class="nav-emoji">🛒</span><span class="nav-label">Cesta</span></button>
    <button class="nav-item" onclick="navTo('favoritos')"><span class="nav-emoji">⭐</span><span class="nav-label">Favoritos</span></button>
    <button class="nav-item" onclick="navTo('mapa')"><span class="nav-emoji">🗺️</span><span class="nav-label">Mapa</span></button>
  </div>
</div>

<!-- ═══════════════════════════════════════ MAPA ═══ -->
<div class="screen" id="screen-mapa">
  <div class="status-bar"><span class="status-time">12:00</span><div class="status-icons">📶 🔋</div></div>
  <div class="header">
    <div class="logo" onclick="navTo('inicio')" style="cursor: pointer;">IKEA</div>
    <span class="header-title">Mapa</span>
    <div class="header-icons">
      <div class="cart-icon-wrap" onclick="goTo('cesta')"><div class="header-icon">🛒</div><div class="cart-badge" id="cart-badge-mapa">5</div></div>
      <button class="header-icon" onclick="goTo('config')">⚙️</button>
    </div>
  </div>
  <div class="aisle-filters">
    <button class="aisle-btn active" onclick="setAisle(this,'todos')">Todos</button>
    <button class="aisle-btn" onclick="setAisle(this,'A')">Pasillo A</button>
    <button class="aisle-btn" onclick="setAisle(this,'B')">Pasillo B</button>
    <button class="aisle-btn" onclick="setAisle(this,'C')">Pasillo C</button>
    <button class="aisle-btn" onclick="setAisle(this,'D')">Pasillo D</button>
  </div>
  <div class="content">
    <div class="map-container">
      <div class="store-map" id="store-map-area">
        
        <svg id="map-route-svg" class="map-route-svg"></svg>
        
        <div class="map-section" style="background:rgba(0,88,163,0.15);border:1.5px solid rgba(0,88,163,0.4);color:var(--blue);left:2%;top:2%;width:23%;height:24%;">Salón</div>
        <div class="map-section" style="background:rgba(0,88,163,0.15);border:1.5px solid rgba(0,88,163,0.4);color:var(--blue);left:2%;top:28%;width:23%;height:23%;">Dormitorio</div>
        <div class="map-section" style="background:rgba(0,88,163,0.15);border:1.5px solid rgba(0,88,163,0.4);color:var(--blue);left:2%;top:53%;width:23%;height:21%;">Cocina</div>
        <div class="map-section" style="background:rgba(0,88,163,0.15);border:1.5px solid rgba(0,88,163,0.4);color:var(--blue);left:2%;top:76%;width:23%;height:19%;">Oficina</div>
        
        <div class="map-section" style="background:rgba(255,204,0,0.2);border:1.5px solid rgba(255,204,0,0.6);color:#8a6d00;left:26%;top:2%;width:23%;height:29%;">Almacenaje</div>
        <div class="map-section" style="background:rgba(255,204,0,0.2);border:1.5px solid rgba(255,204,0,0.6);color:#8a6d00;left:26%;top:33%;width:23%;height:21%;">Iluminación</div>
        <div class="map-section" style="background:rgba(255,204,0,0.2);border:1.5px solid rgba(255,204,0,0.6);color:#8a6d00;left:26%;top:56%;width:23%;height:21%;">Textiles</div>
        <div class="map-section" style="background:rgba(255,204,0,0.2);border:1.5px solid rgba(255,204,0,0.6);color:#8a6d00;left:26%;top:79%;width:23%;height:16%;">Baño</div>

        <div class="corridor" style="background:rgba(150,160,170,0.15);border:1px dashed rgba(150,160,170,0.4);color:#666;left:50%;top:2%;width:11%;height:93%;">A</div>
        <div class="corridor" style="background:rgba(150,160,170,0.15);border:1px dashed rgba(150,160,170,0.4);color:#666;left:62%;top:2%;width:11%;height:93%;">B</div>
        <div class="corridor" style="background:rgba(150,160,170,0.15);border:1px dashed rgba(150,160,170,0.4);color:#666;left:74%;top:2%;width:11%;height:93%;">C</div>
        <div class="corridor" style="background:rgba(150,160,170,0.15);border:1px dashed rgba(150,160,170,0.4);color:#666;left:86%;top:2%;width:11%;height:93%;">D</div>

        <div class="you-dot" style="left:10%;bottom:10%;"></div>
        <div style="position:absolute;left:4%;bottom:3%;font-size:9.5px;font-weight:700;color:var(--blue);">📍 Tú</div>
        <div style="position:absolute;bottom:2%;left:50%;transform:translateX(-50%);font-size:9.5px;font-weight:700;color:var(--gray);">Entrada / Caja</div>
      </div>
      
      <div class="map-legend">
        <div class="legend-item"><div class="legend-dot" style="background:var(--blue);"></div>Exposición</div>
        <div class="legend-item"><div class="legend-dot" style="background:#ffcc00;border:1px solid #d4a900;"></div>Autoservicio</div>
        <div class="legend-item"><div class="legend-dot" style="background:var(--blue);border-radius:50%;"></div>Tu ruta</div>
      </div>
    </div>
    <div style="padding:0 14px 7px;">
      <div style="background:var(--card);border-radius:13px;padding:9px 13px;display:flex;gap:9px;justify-content:space-around;box-shadow:0 2px 6px rgba(0,0,0,0.07);">
        <div style="text-align:center;"><div style="font-size:17px;font-weight:900;color:var(--blue);" id="route-prod-count">0</div><div style="font-size:9.5px;color:var(--gray);">Productos</div></div>
        <div style="text-align:center;"><div style="font-size:17px;font-weight:900;color:var(--green);">~<span id="route-time">0</span> min</div><div style="font-size:9.5px;color:var(--gray);">Estimado</div></div>
        <div style="text-align:center;"><div style="font-size:17px;font-weight:900;" id="route-total">0,00 €</div><div style="font-size:9.5px;color:var(--gray);">Total</div></div>
      </div>
    </div>
    <div id="route-list"></div>
    <div style="height:14px;"></div>
  </div>
  <div class="bottom-nav">
    <button class="nav-item" onclick="navTo('inicio')"><span class="nav-emoji">🏠</span><span class="nav-label">Inicio</span></button>
    <button class="nav-item" onclick="navTo('shopping')"><span class="nav-emoji">🛍️</span><span class="nav-label">Shopping</span></button>
    <button class="nav-item" onclick="navTo('cesta')"><span class="nav-emoji">🛒</span><span class="nav-label">Cesta</span></button>
    <button class="nav-item" onclick="navTo('favoritos')"><span class="nav-emoji">⭐</span><span class="nav-label">Favoritos</span></button>
    <button class="nav-item active"><span class="nav-emoji">🗺️</span><span class="nav-label">Mapa</span></button>
  </div>
</div>

<!-- ═══════════════════════════════════════ CESTA ═══ -->
<div class="screen" id="screen-cesta">
  <div class="status-bar"><span class="status-time">12:00</span><div class="status-icons">📶 🔋</div></div>
  <div class="header">
    <div class="logo" onclick="navTo('inicio')" style="cursor: pointer;">IKEA</div>
    <span class="header-title">Cesta</span>
    <div class="header-icons"><div class="cart-icon-wrap"><div class="header-icon">🛒</div><div class="cart-badge" id="cart-badge-cesta">0</div></div></div>
  </div>
  <div class="content">
    <div style="background:rgba(255,219,0,0.1);border-left:4px solid var(--yellow);margin:11px 14px;padding:9px 13px;border-radius:0 9px 9px 0;font-size:11.5px;font-weight:600;">
      💡 <strong>Gestos:</strong> 
      <br> 📳️ Agita para borrar último
      <br> ↗️ Inclina para favorito
    </div>
    <div style="display:flex;gap:7px;padding:3px 14px 9px;overflow-x:auto;">
      <button class="aisle-btn active" onclick="sortCart(this,'az')">A-Z</button>
      <button class="aisle-btn" onclick="sortCart(this,'price')">💰 Precio</button>
      <button class="aisle-btn" onclick="sortCart(this,'weight')">⚖️ Peso</button>
      <button class="aisle-btn" onclick="sortCart(this,'route')">📍 Ruta</button>
      <button class="aisle-btn" onclick="startVoiceSort()">🎙 Voz</button>
    </div>
    <div id="cart-list"></div>
    <div style="background:var(--card);border-radius:16px;margin:0 14px 7px;overflow:hidden;box-shadow:0 2px 8px rgba(0,0,0,0.07);">
      <div style="padding:9px 14px;display:flex;justify-content:space-between;font-size:12.5px;"><span style="color:var(--gray);">Subtotal</span><span id="subtotal-val" style="font-weight:700;">0,00 €</span></div>
      <div style="padding:9px 14px;display:flex;justify-content:space-between;font-size:12.5px;border-top:1px solid var(--border);"><span style="color:var(--gray);">Bolsa IKEA</span><span id="bag-val" style="font-weight:700;">1,00 €</span></div>
      <div style="padding:13px 14px;display:flex;justify-content:space-between;font-size:15px;font-weight:900;border-top:2px solid var(--border);background:rgba(0,88,163,0.04);"><span>Total</span><span id="total-val" style="color:var(--blue);">0,00 €</span></div>
    </div>
    <button class="pay-btn ripple-btn" onclick="goTo('pagar')">Ir a pagar →</button>
    <div style="height:14px;"></div>
  </div>
  <div class="bottom-nav">
    <button class="nav-item" onclick="navTo('inicio')"><span class="nav-emoji">🏠</span><span class="nav-label">Inicio</span></button>
    <button class="nav-item" onclick="navTo('shopping')"><span class="nav-emoji">🛍️</span><span class="nav-label">Shopping</span></button>
    <button class="nav-item active"><span class="nav-emoji">🛒</span><span class="nav-label">Cesta</span></button>
    <button class="nav-item" onclick="navTo('favoritos')"><span class="nav-emoji">⭐</span><span class="nav-label">Favoritos</span></button>
    <button class="nav-item" onclick="navTo('mapa')"><span class="nav-emoji">🗺️</span><span class="nav-label">Mapa</span></button>
  </div>
</div>

<!-- ═══════════════════════════════════════ FAVORITOS ═══ -->
<div class="screen" id="screen-favoritos">
  <div class="status-bar"><span class="status-time">12:00</span><div class="status-icons">📶 🔋</div></div>
  <div class="header">
    <div class="logo" onclick="navTo('inicio')" style="cursor: pointer;">IKEA</div>
    <span class="header-title">Favoritos</span>
    <div class="header-icons"><div class="cart-icon-wrap" onclick="goTo('cesta')"><div class="header-icon">🛒</div><div class="cart-badge" id="cart-badge-fav">5</div></div></div>
  </div>
  <div class="content"><div id="fav-list"></div><div style="height:14px;"></div></div>
  <div class="bottom-nav">
    <button class="nav-item" onclick="navTo('inicio')"><span class="nav-emoji">🏠</span><span class="nav-label">Inicio</span></button>
    <button class="nav-item" onclick="navTo('shopping')"><span class="nav-emoji">🛍️</span><span class="nav-label">Shopping</span></button>
    <button class="nav-item" onclick="navTo('cesta')"><span class="nav-emoji">🛒</span><span class="nav-label">Cesta</span></button>
    <button class="nav-item active"><span class="nav-emoji">⭐</span><span class="nav-label">Favoritos</span></button>
    <button class="nav-item" onclick="navTo('mapa')"><span class="nav-emoji">🗺️</span><span class="nav-label">Mapa</span></button>
  </div>
</div>

<!-- ═══════════════════════════════════════ SHOPPING ═══ -->
<div class="screen" id="screen-shopping">
  <div class="status-bar"><span class="status-time">12:00</span><div class="status-icons">📶 🔋</div></div>
  <div class="header">
    <div class="logo" onclick="navTo('inicio')" style="cursor: pointer;">IKEA</div>
    <span class="header-title">Shopping</span>
    <div class="header-icons"><div class="cart-icon-wrap" onclick="goTo('cesta')"><div class="header-icon">🛒</div><div class="cart-badge">5</div></div></div>
  </div>
  <div class="content">
    <div style="padding:14px;display:grid;grid-template-columns:1fr 1fr;gap:10px;">
      <button class="action-card ripple-btn" onclick="goTo('escaner')"><span class="action-emoji">📷</span><span class="action-name" style="color:var(--blue)">Escanear QR</span><span class="action-desc">Código de barras</span></button>
      <button class="action-card ripple-btn" onclick="goTo('ar')"><span class="action-emoji">🔮</span><span class="action-name" style="color:var(--purple)">Vista AR</span><span class="action-desc">Reconocer muebles</span></button>
    </div>
    <div style="padding:0 14px 11px;"><div style="background:var(--card);border-radius:13px;padding:11px 14px;display:flex;align-items:center;gap:9px;box-shadow:0 2px 6px rgba(0,0,0,0.07);"><span>🔍</span><input style="flex:1;border:none;background:none;font-size:13.5px;font-family:'Noto Sans',sans-serif;outline:none;" placeholder="Buscar en catálogo IKEA..." onclick="openSearchAndFocus()"><span style="cursor:pointer;" onclick="startVoiceSearch()">🎙️</span></div></div>
    <div class="section-title">TODOS LOS PRODUCTOS</div>
    <div id="shopping-list"></div>
    <div style="height:14px;"></div>
  </div>
  <div class="bottom-nav">
    <button class="nav-item" onclick="navTo('inicio')"><span class="nav-emoji">🏠</span><span class="nav-label">Inicio</span></button>
    <button class="nav-item active"><span class="nav-emoji">🛍️</span><span class="nav-label">Shopping</span></button>
    <button class="nav-item" onclick="navTo('cesta')"><span class="nav-emoji">🛒</span><span class="nav-label">Cesta</span></button>
    <button class="nav-item" onclick="navTo('favoritos')"><span class="nav-emoji">⭐</span><span class="nav-label">Favoritos</span></button>
    <button class="nav-item" onclick="navTo('mapa')"><span class="nav-emoji">🗺️</span><span class="nav-label">Mapa</span></button>
  </div>
</div>

<!-- ═══════════════════════════════════════ PRODUCTO DETALLE ═══ -->
<div class="screen" id="screen-producto">
  <div class="status-bar"><span class="status-time">12:00</span><div class="status-icons">📶 🔋</div></div>
  <div class="header">
    <button class="header-icon" onclick="goBack()">←</button>
    <span class="header-title" id="prod-header-title">Producto</span>
    <div class="header-icons">
      <button class="header-icon" id="prod-fav-btn" onclick="toggleProductFav()">☆</button>
      <div class="cart-icon-wrap" onclick="goTo('cesta')"><div class="header-icon">🛒</div><div class="cart-badge">5</div></div>
    </div>
  </div>
  <div class="content">
    <div class="producto-hero">
      <div class="producto-hero-emoji" id="prod-hero-emoji">📦</div>
      <img id="prod-hero-image" src="" alt="Producto" style="display:none;width:100%;height:180px;object-fit:contain;border-radius:20px;background:#f0f0f0;">
      <div class="producto-badge" id="prod-badge">NUEVO</div>
    </div>
    <div class="producto-body">
      <div class="producto-name-row">
        <div>
          <div class="producto-name" id="prod-name">-</div>
          <div class="producto-desc" id="prod-desc">-</div>
        </div>
        <div style="text-align:right;">
          <div class="producto-price" id="prod-price">- €</div>
          <div class="producto-price-sub">IVA incluido</div>
        </div>
      </div>
      <div class="info-pills" id="prod-pills">
        <!-- Filled by JS -->
      </div>
      <div class="producto-actions" id="prod-actions">
        <button class="btn btn-primary" style="flex:1;" onclick="addProductToCart()">+ Añadir a la cesta</button>
      </div>
      <div class="related-title">TAMBIÉN TE PUEDE GUSTAR</div>
      <div class="related-scroll" id="prod-related">
        <!-- Filled by JS -->
      </div>
      <div style="margin-top:14px;">
        <div class="related-title">DESCRIPCIÓN</div>
        <p id="prod-long-desc" style="font-size:12.5px;color:var(--gray);line-height:1.6;background:var(--card);border-radius:13px;padding:13px;box-shadow:0 2px 6px rgba(0,0,0,0.07);"></p>
      </div>
    </div>
    <div style="height:20px;"></div>
  </div>
    <div class="bottom-nav">
    <button class="nav-item" onclick="navTo('inicio')"><span class="nav-emoji">🏠</span><span class="nav-label">Inicio</span></button>
    <button class="nav-item active"><span class="nav-emoji">🛍️</span><span class="nav-label">Shopping</span></button>
    <button class="nav-item" onclick="navTo('cesta')"><span class="nav-emoji">🛒</span><span class="nav-label">Cesta</span></button>
    <button class="nav-item" onclick="navTo('favoritos')"><span class="nav-emoji">⭐</span><span class="nav-label">Favoritos</span></button>
    <button class="nav-item" onclick="navTo('mapa')"><span class="nav-emoji">🗺️</span><span class="nav-label">Mapa</span></button>
  </div>
</div>

<!-- ═══════════════════════════════════════ PAGAR ═══ -->
<div class="screen" id="screen-pagar">
  <div class="status-bar"><span class="status-time">12:00</span><div class="status-icons">📶 🔋</div></div>
  <div class="header">
    <button class="header-icon" onclick="goBack()">←</button>
    <span class="header-title">Pagar</span>
    <div class="header-icons"><div class="cart-icon-wrap"><div class="header-icon">🛒</div><div class="cart-badge">5</div></div></div>
  </div>
  <div class="content">
    <div class="pay-summary">
      <div class="pay-header">RESUMEN DEL PEDIDO</div>
      <div id="pay-summary-list"></div>
      <div class="pay-total-row"><span style="color:var(--gray)">Subtotal</span><span id="pay-subtotal">0,00 €</span></div>
      <div class="pay-total-row"><span style="color:var(--gray)">Bolsa IKEA</span><span id="pay-bag">1,00 €</span></div>
      <div class="pay-total-row final"><span>Total</span><span style="color:var(--blue)" id="pay-total">0,00 €</span></div>
    </div>
    <div class="section-title">MÉTODO DE PAGO</div>
    <div class="payment-option selected" onclick="selectPayment(this)">
      <div class="payment-icon">💳</div>
      <div class="payment-info"><div class="payment-name">Tarjeta de crédito</div><div class="payment-detail">**** **** **** 4242</div></div>
      <div class="radio-dot"></div>
    </div>
    <div class="payment-option" onclick="selectPayment(this)">
      <div class="payment-icon">🏧</div>
      <div class="payment-info"><div class="payment-name">Tarjeta de débito</div><div class="payment-detail">**** **** **** 8080</div></div>
      <div class="radio-dot"></div>
    </div>
    <div class="payment-option" onclick="selectPayment(this)">
      <div class="payment-icon">📱</div>
      <div class="payment-info"><div class="payment-name">Apple Pay / Google Pay</div><div class="payment-detail">Pago rápido con huella</div></div>
      <div class="radio-dot"></div>
    </div>
    <button class="pay-btn ripple-btn" id="pay-confirm-btn" onclick="confirmPayment()">Confirmar y pagar · 0,00 €</button>
    <div style="height:14px;"></div>
  </div>
  <div class="bottom-nav">
    <button class="nav-item" onclick="navTo('inicio')"><span class="nav-emoji">🏠</span><span class="nav-label">Inicio</span></button>
    <button class="nav-item active"><span class="nav-emoji">🛍️</span><span class="nav-label">Shopping</span></button>
    <button class="nav-item" onclick="navTo('cesta')"><span class="nav-emoji">🛒</span><span class="nav-label">Cesta</span></button>
    <button class="nav-item" onclick="navTo('favoritos')"><span class="nav-emoji">⭐</span><span class="nav-label">Favoritos</span></button>
    <button class="nav-item" onclick="navTo('mapa')"><span class="nav-emoji">🗺️</span><span class="nav-label">Mapa</span></button>
  </div>
</div>

<!-- ═══════════════════════════════════════ HISTORIAL ═══ -->
<div class="screen" id="screen-historial">
  <div class="status-bar"><span class="status-time">12:00</span><div class="status-icons">📶 🔋</div></div>
  <div class="header">
    <button class="header-icon" onclick="goBack()">←</button>
    <span class="header-title">Historial</span>
    <div class="header-icons"><div class="cart-icon-wrap" onclick="goTo('cesta')"><div class="header-icon">🛒</div><div class="cart-badge" id="hist-cart-badge">0</div></div></div>
  </div>
  <div class="content">
    <div style="background:linear-gradient(135deg,var(--blue),var(--blue-dark));margin:10px 14px;border-radius:20px;padding:14px;color:white;">
      <div style="font-size:9px;letter-spacing:1.5px;font-weight:900;opacity:0.7;margin-bottom:10px;">RESUMEN DE COMPRAS</div>
      <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:9px;">
        <div style="background:rgba(255,255,255,0.1);border-radius:13px;padding:11px 7px;text-align:center;"><span style="font-size:20px;">🏬</span><div style="font-size:19px;font-weight:900;margin-top:3px;" id="hist-visits">0</div><div style="font-size:8.5px;opacity:0.7;">Visitas</div></div>
        <div style="background:rgba(255,255,255,0.1);border-radius:13px;padding:11px 7px;text-align:center;"><span style="font-size:20px;">📦</span><div style="font-size:19px;font-weight:900;margin-top:3px;" id="hist-items">0</div><div style="font-size:8.5px;opacity:0.7;">Artículos</div></div>
        <div style="background:rgba(255,255,255,0.1);border-radius:13px;padding:11px 7px;text-align:center;"><span style="font-size:20px;">💰</span><div style="font-size:19px;font-weight:900;margin-top:3px;" id="hist-spent">0€</div><div style="font-size:8.5px;opacity:0.7;">Gastado</div></div>
      </div>
    </div>
    <div class="section-title">PEDIDOS ANTERIORES</div>
    <div id="hist-orders-list">
      <div style="text-align:center;padding:40px;color:var(--gray);"><div style="font-size:48px;margin-bottom:12px;">📋</div><div style="font-weight:700;">Sin pedidos todavía</div><div style="font-size:12px;margin-top:4px;">Tus compras aparecerán aquí</div></div>
    </div>
    <div style="height:14px;"></div>
  </div>
    <div class="bottom-nav">
    <button class="nav-item" onclick="navTo('inicio')"><span class="nav-emoji">🏠</span><span class="nav-label">Inicio</span></button>
    <button class="nav-item active"><span class="nav-emoji">🛍️</span><span class="nav-label">Shopping</span></button>
    <button class="nav-item" onclick="navTo('cesta')"><span class="nav-emoji">🛒</span><span class="nav-label">Cesta</span></button>
    <button class="nav-item" onclick="navTo('favoritos')"><span class="nav-emoji">⭐</span><span class="nav-label">Favoritos</span></button>
    <button class="nav-item" onclick="navTo('mapa')"><span class="nav-emoji">🗺️</span><span class="nav-label">Mapa</span></button>
  </div>
</div>

<!-- ═══════════════════════════════════════ CONFIG ═══ -->
<div class="screen" id="screen-config">
  <div class="status-bar"><span class="status-time">12:00</span><div class="status-icons">📶 🔋</div></div>
  <div class="header">
    <div class="logo" onclick="navTo('inicio')" style="cursor: pointer;">IKEA</div>
    <span class="header-title">Configuración</span>
    <div class="header-icons"><div class="cart-icon-wrap" onclick="goTo('cesta')"><div class="header-icon">🛒</div><div class="cart-badge">5</div></div></div>
  </div>
  <div class="content">
    <div class="profile-card" onclick="goTo('perfil')">
      <div class="profile-avatar">👤</div>
      <div style="flex:1;">
        <div class="profile-name" id="cfg-name">Nombre Usuario</div>
        <div class="profile-email" id="cfg-email">usuario@ikea.es</div>
        <div class="profile-badge">⭐ IKEA Family</div>
      </div>
      <span class="profile-chevron">›</span>
    </div>
    <div style="padding:3px 14px 14px;">
      <div class="settings-title">PREFERENCIAS DE LA APP</div>
      <div class="settings-card">
        <div class="settings-item">
          <div class="settings-emoji">🌐</div>
          <div class="settings-info"><div class="settings-name">Idioma</div></div>
          <select style="border:1px solid var(--border);border-radius:7px;padding:4px 7px;font-size:11.5px;background:var(--bg);">
            <option>Español</option><option>English</option><option>Français</option>
          </select>
        </div>
        <div class="settings-item">
          <div class="settings-emoji">📏</div>
          <div class="settings-info"><div class="settings-name">Medidas</div></div>
          <select style="border:1px solid var(--border);border-radius:7px;padding:4px 7px;font-size:11.5px;background:var(--bg);">
            <option>kg / cm</option><option>lb / in</option>
          </select>
        </div>
      </div>
    </div>
    <div style="padding:0 14px 14px;">
      <div class="settings-title">NOTIFICACIONES</div>
      <div class="settings-card">
        <div class="settings-item"><div class="settings-emoji">📦</div><div class="settings-info"><div class="settings-name">Alertas de stock</div><div class="settings-desc">Avisa si un artículo se agota</div></div><button class="toggle on" onclick="this.classList.toggle('on')"></button></div>
        <div class="settings-item"><div class="settings-emoji">🏷️</div><div class="settings-info"><div class="settings-name">Alertas de ofertas</div><div class="settings-desc">Artículos de tu lista en oferta</div></div><button class="toggle" onclick="this.classList.toggle('on')"></button></div>
        <div class="settings-item"><div class="settings-emoji">🗺️</div><div class="settings-info"><div class="settings-name">Guía de ruta</div><div class="settings-desc">Navegación paso a paso</div></div><button class="toggle on" onclick="this.classList.toggle('on')"></button></div>
      </div>
    </div>
    <div style="padding:0 14px 14px;">
      <div class="settings-title">GESTOS E INTERACCIONES</div>
      <div class="settings-card">
        <div class="settings-item"><div class="settings-emoji">🎙️</div><div class="settings-info"><div class="settings-name">Comandos de voz</div></div><button class="toggle on" onclick="this.classList.toggle('on')"></button></div>
        <div class="settings-item"><div class="settings-emoji">↗️</div><div class="settings-info"><div class="settings-name">Inclinar para favoritos</div></div><button class="toggle on" onclick="this.classList.toggle('on')"></button></div>
        <div class="settings-item"><div class="settings-emoji">📳️</div><div class="settings-info"><div class="settings-name">Agitar para borrar</div></div><button class="toggle on" onclick="this.classList.toggle('on')"></button></div>
      </div>
    </div>
    <div style="padding:0 14px 14px;">
      <div class="settings-title">CUENTA</div>
      <div class="settings-card">
        <div class="settings-item" onclick="goTo('perfil')"><div class="settings-emoji">👤</div><div class="settings-info"><div class="settings-name">Gestionar perfil</div><div class="settings-desc">Ver y editar datos personales</div></div><span style="color:var(--gray);">›</span></div>
        <div class="settings-item" onclick="goTo('historial')"><div class="settings-emoji">🧾</div><div class="settings-info"><div class="settings-name">Historial de compras</div></div><span style="color:var(--gray);">›</span></div>
      </div>
    </div>
    <button class="pay-btn danger" onclick="goTo('login')">Cerrar sesión</button>
    <div style="height:14px;"></div>
  </div>
  <div class="bottom-nav">
    <button class="nav-item" onclick="navTo('inicio')"><span class="nav-emoji">🏠</span><span class="nav-label">Inicio</span></button>
    <button class="nav-item" onclick="navTo('shopping')"><span class="nav-emoji">🛍️</span><span class="nav-label">Shopping</span></button>
    <button class="nav-item" onclick="navTo('cesta')"><span class="nav-emoji">🛒</span><span class="nav-label">Cesta</span></button>
    <button class="nav-item" onclick="navTo('favoritos')"><span class="nav-emoji">⭐</span><span class="nav-label">Favoritos</span></button>
    <button class="nav-item" onclick="navTo('mapa')"><span class="nav-emoji">🗺️</span><span class="nav-label">Mapa</span></button>
  </div>
</div>

<!-- ═══════════════════════════════════════ PERFIL ═══ -->
<div class="screen" id="screen-perfil">
  <div class="status-bar"><span class="status-time">12:00</span><div class="status-icons">📶 🔋</div></div>
  <div class="header">
    <button class="header-icon" onclick="goBack()">←</button>
    <span class="header-title">Mi Perfil</span>
    <div class="header-icons"><button class="header-icon" onclick="openEditSheet()">✏️</button></div>
  </div>
  <div class="content">
    <div class="perfil-hero">
      <div class="perfil-avatar-wrap">
        <div class="perfil-avatar-big" onclick="showToast('📸 Toca para cambiar foto')">👤</div>
        <div class="perfil-avatar-edit" onclick="showToast('📸 Cambiar foto')">📷</div>
      </div>
      <div class="perfil-name" id="perfil-name-display">Nombre Usuario</div>
      <div class="perfil-email" id="perfil-email-display">usuario@ikea.es</div>
      <div class="perfil-badge">⭐ IKEA Family · Miembro desde 2021</div>
    </div>
    <div class="perfil-stats">
      <div class="perfil-stat-card"><span class="perfil-stat-emoji">🏬</span><div class="perfil-stat-val">0</div><div class="perfil-stat-lbl">VISITAS</div></div>
      <div class="perfil-stat-card"><span class="perfil-stat-emoji">⭐</span><div class="perfil-stat-val" id="perfil-points">0</div><div class="perfil-stat-lbl">PUNTOS</div></div>
      <div class="perfil-stat-card"><span class="perfil-stat-emoji">💚</span><div class="perfil-stat-val">0 €</div><div class="perfil-stat-lbl">AHORRADO</div></div>
    </div>
    <div class="family-card">
      <div class="family-logo-row">
        <div class="family-logo">IKEA</div>
        <div class="family-brand">FAMILY CARD</div>
      </div>
      <div class="family-card-number">•••• •••• •••• 7890</div>
      <div class="family-bottom">
        <div>
          <div class="family-name" id="family-name-display">Nombre Usuario</div>
          <div class="family-level">🥈 Nivel Plata</div>
        </div>
        <div style="text-align:right;">
          <div class="family-points-val" id="family-points-display">4.830</div>
          <div class="family-points-lbl">PUNTOS</div>
        </div>
      </div>
    </div>
    <div class="perfil-section">
      <div class="perfil-section-title">DATOS PERSONALES</div>
      <div class="perfil-card">
        <div class="perfil-field" onclick="openEditSheet('name')"><div class="perfil-field-icon">👤</div><div class="perfil-field-info"><div class="perfil-field-label">NOMBRE</div><div class="perfil-field-val" id="pf-name">Nombre Usuario</div></div><span style="color:var(--gray);">›</span></div>
        <div class="perfil-field" onclick="openEditSheet('email')"><div class="perfil-field-icon">✉️</div><div class="perfil-field-info"><div class="perfil-field-label">CORREO</div><div class="perfil-field-val" id="pf-email">usuario@ikea.es</div></div><span style="color:var(--gray);">›</span></div>
        <div class="perfil-field" onclick="openEditSheet('phone')"><div class="perfil-field-icon">📱</div><div class="perfil-field-info"><div class="perfil-field-label">TELÉFONO</div><div class="perfil-field-val" id="pf-phone">+34 600 000 000</div></div><span style="color:var(--gray);">›</span></div>
        <div class="perfil-field" onclick="openEditSheet('bday')"><div class="perfil-field-icon">🎂</div><div class="perfil-field-info"><div class="perfil-field-label">FECHA DE NACIMIENTO</div><div class="perfil-field-val" id="pf-bday">15 / 03 / 1990</div></div><span style="color:var(--gray);">›</span></div>
      </div>
    </div>
    <div class="perfil-section">
      <div class="perfil-section-title">PREFERENCIAS</div>
      <div class="perfil-card">
        <div class="perfil-field" onclick="showToast('🏪 Cambiar tienda favorita')"><div class="perfil-field-icon">🏪</div><div class="perfil-field-info"><div class="perfil-field-label">TIENDA FAVORITA</div><div class="perfil-field-val">IKEA Bilbao</div></div><span style="color:var(--gray);">›</span></div>
        <div class="perfil-field" onclick="showToast('💳 Cambiar método predeterminado')"><div class="perfil-field-icon">💳</div><div class="perfil-field-info"><div class="perfil-field-label">PAGO PREDETERMINADO</div><div class="perfil-field-val">Visa •••• 4242</div></div><span style="color:var(--gray);">›</span></div>
      </div>
    </div>
    <div class="perfil-section">
      <div class="perfil-section-title">ACTIVIDAD</div>
      <div class="perfil-card">
        <div class="perfil-field" onclick="goTo('historial')"><div class="perfil-field-icon">🧾</div><div class="perfil-field-info"><div class="perfil-field-label">HISTORIAL</div><div class="perfil-field-val">Ver mis compras anteriores</div></div><span style="color:var(--gray);">›</span></div>
        <div class="perfil-field" onclick="navTo('favoritos')"><div class="perfil-field-icon">⭐</div><div class="perfil-field-info"><div class="perfil-field-label">FAVORITOS</div><div class="perfil-field-val">4 artículos guardados</div></div><span style="color:var(--gray);">›</span></div>
      </div>
    </div>
    <button class="pay-btn danger" onclick="onclick="goTo('login')">🗑️ Eliminar cuenta</button>
    <div style="height:14px;"></div>
  </div>
  <!-- Edit Sheet overlay -->
  <div class="edit-overlay" id="edit-overlay" onclick="closeEditSheetOnBg(event)">
    <div class="edit-sheet" id="edit-sheet">
      <div class="edit-sheet-handle"></div>
      <div class="edit-sheet-title" id="edit-sheet-title">Editar perfil</div>
      <div id="edit-sheet-body"><!-- filled by JS --></div>
      <div class="edit-sheet-actions">
        <button class="btn btn-ghost" onclick="closeEditSheet()">Cancelar</button>
        <button class="btn btn-primary" onclick="saveEditSheet()">Guardar cambios</button>
      </div>
    </div>
  </div>
    <div class="bottom-nav">
    <button class="nav-item" onclick="navTo('inicio')"><span class="nav-emoji">🏠</span><span class="nav-label">Inicio</span></button>
    <button class="nav-item active"><span class="nav-emoji">🛍️</span><span class="nav-label">Shopping</span></button>
    <button class="nav-item" onclick="navTo('cesta')"><span class="nav-emoji">🛒</span><span class="nav-label">Cesta</span></button>
    <button class="nav-item" onclick="navTo('favoritos')"><span class="nav-emoji">⭐</span><span class="nav-label">Favoritos</span></button>
    <button class="nav-item" onclick="navTo('mapa')"><span class="nav-emoji">🗺️</span><span class="nav-label">Mapa</span></button>
  </div>
</div>

<!-- ═══════════════════════════════════════ SUCCESS ═══ -->
<div class="screen" id="screen-success">
  <div class="status-bar"><span class="status-time">12:00</span><div class="status-icons">📶 🔋</div></div>
  <div class="success-screen">
    <div class="success-icon">✅</div>
    <div class="success-title">¡Pago Completado!</div>
    <div class="success-subtitle">Tu pedido ha sido procesado correctamente. Dirígete a caja para recoger tus artículos.</div>
    <div style="background:rgba(31,132,35,0.08);border:1px solid rgba(31,132,35,0.2);border-radius:14px;padding:14px 18px;width:100%;text-align:center;">
      <div style="font-size:10px;letter-spacing:1.5px;font-weight:900;color:var(--gray);margin-bottom:3px;">NÚMERO DE PEDIDO</div>
      <div style="font-size:22px;font-weight:900;color:var(--green);" id="success-order-id">#IK-0000</div>
    </div>
    <div style="font-size:12.5px;color:var(--gray);" id="success-order-summary">0,00 € · 0 artículos · IKEA Bilbao</div>
    <button class="pay-btn ripple-btn" onclick="clearCart();goTo('inicio');renderHistorial();renderProfile();" style="width:100%;">Volver al inicio 🏠</button>
  </div>
</div>

</div><!-- /screens -->
<div class="toast" id="toast"></div>
</div><!-- /phone -->

<script>
// ═══════════════════════════════ DATA ═══════════════════════════════

let IKEA_DB = {};        // { "Puerta | VOXTORP": { ...datos completos } }
let allProducts = [];   // Array plano para iteración
let catalogLoaded = false;

// Cargar catálogo desde el servidor
async function loadCatalog() {
  try {
    const res = await fetch('/catalog');
    if (res.ok) {
      const data = await res.json();
      IKEA_DB = {};
      allProducts = [];

      let idCounter = 1;
      Object.entries(data).forEach(([key, product]) => {
        // La key es el identificador único del JSON (ej: "Puerta | VOXTORP")
        const priceNum = typeof product.precio === 'number' ? product.precio :
                        (parseFloat(String(product.precio).replace(/[^\d,]/g, '').replace(',', '.')) || 0);
        const loc = product.ubicacion || { pasillo: '-', estanteria: '-' };

        // Guardar con la key completa como identificador único
        IKEA_DB[key] = {
          key: key,
          nombre: product.nombre || key.split('|')[0].trim(),
          subtitulo: product.subtitulo || '',
          descripcion: product.descripcion || product.subtitulo || '',
          categoria: product.categoria || '',
          price: priceNum,
          priceStr: priceNum > 0 ? priceNum.toFixed(2).replace('.', ',') + ' €' : 'Consultar',
          location: `Pasillo ${loc.pasillo}·${loc.estanteria}`,
          pasillo: String(loc.pasillo || '-'),
          estanteria: String(loc.estanteria || '-'),
          peso: product.peso || '-',
          image: product.imagen || '',
          url: product.url || '#',
          // Campos legacy para compatibilidad
          name: product.nombre || key.split('|')[0].trim(),
          desc: product.subtitulo || '',
          longDesc: product.descripcion || product.subtitulo || '',
          emoji: product.emoji || getCategoryEmoji(product.categoria || ''),
          weight: product.peso || '-',
          location: `Pasillo ${loc.pasillo}·${loc.estanteria}`,
          stock: product.stock || 10
        };
      });

      // Crear array plano con IDs únicos
      allProducts = Object.keys(IKEA_DB).map((key, i) => ({
        id: i + 1,
        key: key,
        ...IKEA_DB[key]
      }));

      catalogLoaded = true;
      console.log('✅ Catálogo cargado:', allProducts.length, 'productos');
      console.log('📦 Primer producto:', allProducts[0]?.key);
    }
  } catch (e) {
    console.error('❌ Error cargando catálogo:', e);
    showToast('⚠️ Error cargando catálogo');
  }

  // Inicializar UI con los datos disponibles
  initUIWithCatalog();
}

// Helper para obtener emoji según categoría
function getCategoryEmoji(categoria) {
  const emojiMap = {
    'Puerta': '🚪', 'Puerta armario': '🚪', 'Silla': '🪑', 'Sofá': '🛋️',
    'Sofá cama': '🛋️', 'Mesa': '🪑', 'Lámpara': '💡', 'Marco': '🖼️',
    'Fregadero': '🚰', 'Armario': '🗄️', 'Cama': '🛏️', 'Estantería': '📚',
    'Cómoda': '🗄️', 'Escritorio': '🖥️', 'Silla de oficina': '🪑',
    'Funda': '🧵', 'Alfombra': '🧵', 'Decoración': '🏠'
  };
  return emojiMap[categoria] || '📦';
}

// Función para buscar producto por key, nombre, o partial match
function findProduct(searchTerm) {
  if (!searchTerm || !allProducts.length) return null;

  // Buscar por key exacta
  if (IKEA_DB[searchTerm]) return IKEA_DB[searchTerm];

  // Buscar por key parcial
  const lowerSearch = searchTerm.toLowerCase();
  const byKey = Object.keys(IKEA_DB).find(k => k.toLowerCase().includes(lowerSearch));
  if (byKey) return IKEA_DB[byKey];

  // Buscar por nombre
  const byName = Object.values(IKEA_DB).find(p =>
    p.nombre && p.nombre.toLowerCase() === lowerSearch
  );
  if (byName) return byName;

  // Buscar por nombre parcial
  const byNamePartial = Object.values(IKEA_DB).find(p =>
    p.nombre && p.nombre.toLowerCase().includes(lowerSearch)
  );
  if (byNamePartial) return byNamePartial;

  return null;
}

// Helper para formatear precios
function formatPrice(price) {
  if (typeof price === 'string') return price;
  if (typeof price !== 'number') return 'Consultar';
  return price > 0 ? price.toFixed(2).replace('.', ',') + ' €' : 'Consultar';
}

// Helper para obtener imagen de producto
function getProductImage(product) {
  return product.image || product.imagen || '';
}

// Helper para obtener nombre para mostrar
function getProductDisplayName(product) {
  return product.key || product.nombre || 'Producto';
}

// Inicializar UI después de cargar catálogo
function initUIWithCatalog() {
  if (typeof renderCart === 'function') renderCart();
  if (typeof renderFavs === 'function') renderFavs();
  if (typeof renderShopping === 'function') renderShopping();
  if (typeof renderSearch === 'function') renderSearch('');
  if (typeof renderPaySummary === 'function') renderPaySummary();
  if (typeof renderRoute === 'function') renderRoute();
  if (typeof updateAllBadges === 'function') updateAllBadges();
  if (typeof renderProfile === 'function') renderProfile();
  if (typeof renderHomeProducts === 'function') renderHomeProducts();
}

// ═══════════════════════════════ DATOS DE USUARIO ═══════════════════════════════

let cartItems = [];    // Se llena dinámicamente desde el JSON
let favItems = [];     // Se llena dinámicamente desde el JSON
let routeItems = [];   // Se genera dinámicamente desde el carrito
let orderHistory = []; // Historial de pedidos completados

let userProfile = { name:'Nombre Usuario', email:'usuario@ikea.es', phone:'+34 600 000 000', bday:'15 / 03 / 1990' };
let currentProductId = null;
let currentProductKey = null;  // Key única del JSON para el producto actual
let currentScreen = 'login';
let screenHistory = [];

// ═══════════════════════════════ NAVIGATION ═══════════════════════════════

function goTo(screen) {
  const prev = document.getElementById('screen-' + currentScreen);
  const next = document.getElementById('screen-' + screen);
  if (!next || screen === currentScreen) return;
  
  screenHistory.push(currentScreen);
  
  next.style.transition = 'none'; 
  next.classList.remove('exit-left', 'active'); 

  void next.offsetWidth; 

  next.style.transition = ''; 

  if (prev) {
      prev.classList.remove('active');
      prev.classList.add('exit-left');
  }

  next.classList.add('active');
  
  onScreenLeave(currentScreen);
  currentScreen = screen;
  onScreenEnter(screen);
  
  if (screen === 'mapa') {
    setTimeout(() => {
      renderRoute();
    }, 50);
  }
  
}

function goBack() {
  if (!screenHistory.length) return;
  const prevScreen = screenHistory.pop();
  const prevEl = document.getElementById('screen-' + prevScreen);
  const currEl = document.getElementById('screen-' + currentScreen);

  if (currEl) {
      currEl.classList.remove('active', 'exit-left');
  }
  
  if (prevEl) {

      prevEl.style.transition = 'none';
      prevEl.classList.add('exit-left');
      prevEl.classList.remove('active');
      
      void prevEl.offsetWidth;
      
      prevEl.style.transition = '';

      prevEl.classList.remove('exit-left');
      prevEl.classList.add('active');
  }
  
  onScreenLeave(currentScreen);
  currentScreen = prevScreen;
  onScreenEnter(prevScreen);
}
function navTo(screen) { screenHistory = []; goTo(screen); }

function onScreenLeave(s) {
  if (s === 'ar')     stopCamera('ar');
  if (s === 'escaner') stopCamera('qr');
}
function onScreenEnter(s) {
  updateAllBadges();
  if (s === 'inicio')      renderHomeProducts();
  if (s === 'cesta')        renderCart();
  if (s === 'favoritos')    renderFavs();
  if (s === 'shopping')    renderShopping();
  if (s === 'buscar')       renderSearch('');
  if (s === 'pagar')        renderPaySummary();
  if (s === 'mapa')         renderRoute();
  if (s === 'ar')           { startCamera('ar'); renderARSuggestedProducts(); }
  if (s === 'escaner')      startCamera('qr');
  if (s === 'perfil')       renderProfile();
  if (s === 'historial')    renderHistorial();
}

// ═══════════════════════════════ HOME PRODUCTS ═══════════════════════════════

function renderHomeProducts() {
  const container = document.getElementById('home-products');
  if (!container) return;

  if (!catalogLoaded || !allProducts.length) {
    container.innerHTML = '<div style="text-align:center;padding:30px;color:var(--gray);">Cargando productos...</div>';
    return;
  }

  // Seleccionar productos aleatorios (máximo 8)
  const shuffled = [...allProducts].sort(() => Math.random() - 0.5);
  const randomProducts = shuffled.slice(0, 8);

  container.innerHTML = randomProducts.map(p => {
    const dbData = IKEA_DB[p.key] || p;
    const displayName = dbData.nombre || dbData.key || 'Producto';
    const priceStr = dbData.priceStr || formatPrice(dbData.price);
    const descStr = dbData.subtitulo || dbData.desc || '';
    const emojiStr = dbData.emoji || '📦';
    const imageStr = dbData.image || '';
    const locationStr = dbData.location || 'Consultar';

    return `
      <div class="product-card" style="min-width:140px;flex-direction:column;gap:6px;" onclick="openProduct('${p.key}')">
        ${imageStr ? `<img src="${imageStr}" alt="${displayName}" style="width:60px;height:60px;object-fit:contain;border-radius:8px;" onerror="this.style.display='none';this.nextElementSibling.style.display='flex';"><div class="product-img" style="display:none;width:60px;height:60px;">${emojiStr}</div>` : `<div class="product-img" style="width:60px;height:60px;">${emojiStr}</div>`}
        <div style="font-size:11px;font-weight:700;color:var(--text);text-align:center;">${displayName}</div>
        <div style="font-size:10px;color:var(--gray);text-align:center;">${descStr.substring(0, 25)}${descStr.length > 25 ? '...' : ''}</div>
        <div style="font-size:12px;font-weight:900;color:var(--blue);text-align:center;">${priceStr}</div>
        <div style="font-size:9px;color:var(--green);text-align:center;">📍 ${locationStr}</div>
      </div>
    `;
  }).join('');
}

// ═══════════════════════════════ AR SUGGESTED PRODUCTS ═══════════════════════════════

function renderARSuggestedProducts() {
  const container = document.getElementById('ar-suggested-products');
  if (!container) return;

  if (!catalogLoaded || !allProducts.length) {
    container.innerHTML = '<div style="text-align:center;padding:20px;color:var(--gray);">Cargando...</div>';
    return;
  }

  // Seleccionar 4 productos aleatorios
  const shuffled = [...allProducts].sort(() => Math.random() - 0.5);
  const randomProducts = shuffled.slice(0, 4);

  container.innerHTML = randomProducts.map(p => {
    const dbData = IKEA_DB[p.key] || p;
    const displayName = dbData.nombre || dbData.key || 'Producto';
    const priceStr = dbData.priceStr || formatPrice(dbData.price);
    const descStr = dbData.subtitulo || dbData.desc || '';
    const emojiStr = dbData.emoji || '📦';
    const imageStr = dbData.image || '';
    const locationStr = dbData.location || 'Consultar';

    return `
      <div class="product-card" style="min-width:160px;flex-direction:column;gap:4px;" onclick="openProduct('${p.key}')">
        ${imageStr ? `<img src="${imageStr}" alt="${displayName}" style="width:50px;height:50px;object-fit:contain;border-radius:8px;" onerror="this.style.display='none';this.nextElementSibling.style.display='flex';"><div class="product-img" style="display:none;width:50px;height:50px;">${emojiStr}</div>` : `<div class="product-img" style="width:50px;height:50px;">${emojiStr}</div>`}
        <div style="font-size:10px;font-weight:700;color:var(--text);">${displayName}</div>
        <div style="font-size:9px;color:var(--gray);">${descStr.substring(0, 20)}${descStr.length > 20 ? '...' : ''}</div>
        <div style="font-size:11px;font-weight:900;color:var(--blue);">${priceStr}</div>
      </div>
    `;
  }).join('');
}

// ═══════════════════════════════ AUTH ═══════════════════════════════

function doLogin() {
  const email = document.getElementById('login-email').value.trim();
  const pass  = document.getElementById('login-pass').value.trim();
  if (!email || !pass) { showToast('⚠️ Rellena todos los campos'); return; }
  userProfile.email = email;
  userProfile.name  = email.split('@')[0];
  goTo('inicio');
}
function doRegister() {
  const name  = document.getElementById('reg-name').value.trim();
  const email = document.getElementById('reg-email').value.trim();
  const pass  = document.getElementById('reg-pass').value.trim();
  if (!name||!email||!pass) { showToast('⚠️ Rellena todos los campos'); return; }
  if (pass.length < 8)      { showToast('⚠️ Contraseña mínimo 8 caracteres'); return; }
  userProfile = { name, email, phone:'+34 600 000 000', bday:'-- / -- / ----' };
  showToast('✅ ¡Cuenta creada!');
  goTo('inicio');
}

// ═══════════════════════════════ PRODUCT DETAIL ═══════════════════════════════

function openProduct(idOrName) {
  let p = null;
  let productKey = null;

  if (typeof idOrName === 'number') {
    // Buscar por ID
    p = allProducts.find(prod => prod.id === idOrName);
    if (p) productKey = p.key;
  } else {
    // Buscar por key o nombre usando la nueva función
    p = findProduct(idOrName);
    if (p) productKey = p.key;
  }

  if (!p) {
    showToast('Producto no encontrado');
    console.error('Producto no encontrado:', idOrName);
    return;
  }

  currentProductId = p.id;
  currentProductKey = productKey;

  // Obtener datos completos del producto
  const dbData = p;

  // Mostrar imagen o emoji
  const heroEmoji = document.getElementById('prod-hero-emoji');
  const heroImage = document.getElementById('prod-hero-image');
  if (dbData.image) {
    heroEmoji.style.display = 'none';
    heroImage.src = dbData.image;
    heroImage.style.display = 'block';
    heroImage.onerror = function() {
      this.style.display = 'none';
      heroEmoji.style.display = 'flex';
      heroEmoji.textContent = dbData.emoji || '📦';
    };
  } else {
    heroEmoji.style.display = 'flex';
    heroEmoji.textContent = dbData.emoji || '📦';
    if (heroImage) heroImage.style.display = 'none';
  }

  document.getElementById('prod-header-title').textContent = dbData.nombre || dbData.key || 'Producto';
  document.getElementById('prod-name').textContent = dbData.nombre || dbData.key || '';
  document.getElementById('prod-desc').textContent = dbData.subtitulo || dbData.desc || '';
  document.getElementById('prod-price').textContent = dbData.priceStr || formatPrice(dbData.price);
  document.getElementById('prod-long-desc').textContent = dbData.descripcion || dbData.longDesc || dbData.subtitulo || '';

  const stock = dbData.stock || 10;
  document.getElementById('prod-badge').textContent = stock <= 3 ? '⚠️ POCAS UNIDADES' : 'EN TIENDA';
  document.getElementById('prod-badge').style.background = stock <= 3 ? 'var(--orange)' : 'var(--yellow)';

  // Pills
  document.getElementById('prod-pills').innerHTML = `
    <div class="info-pill"><span>📍</span><div><span class="pill-label">PASILLO</span><span class="pill-val">${dbData.location}</span></div></div>
    <div class="info-pill"><span>⚖️</span><div><span class="pill-label">PESO</span><span class="pill-val">${dbData.peso || dbData.weight || '-'}</span></div></div>
    <div class="info-pill"><span>📦</span><div><span class="pill-label">CATEGORÍA</span><span class="pill-val">${dbData.categoria || '-'}</span></div></div>
  `;

  // Fav button
  const isFav = favItems.some(f => f.key === productKey);
  document.getElementById('prod-fav-btn').textContent = isFav ? '⭐' : '☆';

  // Add to cart button
  const inCart = cartItems.some(c => c.key === productKey);
  const addBtn = document.getElementById('prod-actions').querySelector('.btn-primary');
  if (addBtn) {
    addBtn.textContent = inCart ? '✅ En la cesta' : '+ Añadir a la cesta';
    addBtn.className = 'btn ' + (inCart ? 'btn-success' : 'btn-primary');
  }

  // Related products
  const related = allProducts.filter(x => x.key !== productKey).slice(0, 4);
  document.getElementById('prod-related').innerHTML = related.map(r => `
    <div class="related-card" onclick="openProduct('${r.key}')">
      ${r.image ? `<img src="${r.image}" alt="${r.nombre}" style="width:40px;height:40px;object-fit:contain;border-radius:8px;">` : `<span class="related-emoji">${r.emoji || '📦'}</span>`}
      <div class="related-name">${r.nombre || r.key}</div>
      <div class="related-price">${r.priceStr || formatPrice(r.price)}</div>
    </div>
  `).join('');

  goTo('producto');
  
  refreshProductButtonState();
  
}

function addProductToCart() {
  if (!currentProductKey) return;
  const p = IKEA_DB[currentProductKey];
  if (!p) return;

  addToCart(currentProductKey, p.price, p.location, p.emoji || '📦', p.image);
  showToast(`✅ ${p.nombre || currentProductKey} añadido a la cesta`);
  const btn = document.getElementById('prod-actions').querySelector('button');
  if (btn) { btn.textContent = '✅ En la cesta'; btn.className = 'btn btn-success'; }
}

function toggleProductFav() {
  
  if (!currentProductKey) return;
  const p = IKEA_DB[currentProductKey];
  if (!p) return;

  const idx = favItems.findIndex(f => f.key === currentProductKey);
  const btn = document.getElementById('prod-fav-btn');

  if (idx >= 0) {
    favItems.splice(idx, 1);
    btn.textContent = '☆';
    showToast('✖️ Eliminado de Favoritos');
  } else {
    favItems.push({
      id: Date.now(),
      key: currentProductKey,
      name: p.nombre || currentProductKey,
      price: p.price,
      location: p.location,
      peso: p.peso,
      emoji: p.emoji,
      image: p.image,
      inCart: cartItems.some(c => c.key === currentProductKey)
    });
    btn.textContent = '⭐';
    showToast('⭐ Guardado en Favoritos');
  }
}

// ═══════════════════════════════ CART ═══════════════════════════════

function renderCart() {
  const list = document.getElementById('cart-list');
  if (!list) return;
  if (!cartItems.length) {
    list.innerHTML = '<div style="text-align:center;padding:48px 20px;color:var(--gray);"><div style="font-size:48px;margin-bottom:12px;">🛒</div><div style="font-weight:700;">Tu cesta está vacía</div></div>';
    updateTotal();
    return;
  }
  list.innerHTML = cartItems.map(item => {
    // Obtener datos del catálogo o usar datos del item
    const dbData = IKEA_DB[item.key] || {};
    const displayName = item.name || dbData.nombre || item.key || 'Producto';
    const priceStr = item.priceStr || dbData.priceStr || formatPrice(item.price);
    const descStr = item.desc || dbData.subtitulo || dbData.desc || '';
    const emojiStr = item.emoji || dbData.emoji || '📦';
    const imageStr = item.image || dbData.image || '';
    const weightStr = item.peso || dbData.peso || item.weight || dbData.weight || '-';

    return `
      <div class="product-card" id="cart-item-${item.id}" style="flex-direction:column;gap:0;" onclick="openProduct('${item.key}')">
        <div style="display:flex;gap:11px;align-items:flex-start;">
          ${imageStr ? `<img src="${imageStr}" alt="${displayName}" style="width:50px;height:50px;object-fit:contain;border-radius:8px;flex-shrink:0;" onerror="this.style.display='none';this.nextElementSibling.style.display='flex';"><div class="product-img" style="display:none;">${emojiStr}</div>` : `<div class="product-img">${emojiStr}</div>`}
          <div style="flex:1;">
            <div class="product-name">${displayName}</div>
            <div class="product-desc">${descStr}</div>
            <div style="display:flex;gap:5px;margin-top:5px;flex-wrap:wrap;">
              <span class="product-location">📍 ${item.location}</span>
              <span class="product-location" style="background:rgba(82,32,125,0.1);color:var(--purple);">⚖️ ${weightStr}</span>
            </div>
          </div>
          <div style="text-align:right;flex-shrink:0;"><div class="product-price">${priceStr}</div></div>
        </div>
        <div style="display:flex;align-items:center;justify-content:space-between;gap:10px;margin-top:10px;padding-top:10px;border-top:1px solid var(--border);" onclick="event.stopPropagation()">
          <div style="display:flex;gap:8px;align-items:center;">
            <button style="width:36px;height:36px;border-radius:10px;border:none;background:rgba(239,68,68,0.1);color:var(--red);font-size:16px;cursor:pointer;display:flex;align-items:center;justify-content:center;flex-shrink:0;transition:all 0.15s;" onclick="removeFromCart(${item.id})" title="Eliminar">🗑️</button>
            <button style="width:36px;height:36px;border-radius:10px;border:none;background:${item.inFav ? 'var(--yellow)' : 'rgba(255,219,0,0.18)'};color:#9a6500;font-size:16px;cursor:pointer;display:flex;align-items:center;justify-content:center;flex-shrink:0;transition:all 0.15s;" onclick="toggleFavFromCart(${item.id})" title="Favorito">${item.inFav ? '⭐' : '☆'}</button>
            <button style="width:36px;height:36px;border-radius:10px;border:none;background:var(--bg);color:var(--blue);font-size:16px;cursor:pointer;display:flex;align-items:center;justify-content:center;flex-shrink:0;transition:all 0.15s;" onclick="goTo('mapa');showToast('📍 ${displayName}')" title="Ver en mapa">📍</button>
          </div>
          <div class="qty-controls">
            <button class="qty-btn" onclick="changeQty(${item.id},-1)">−</button>
            <span class="qty-num" id="qty-${item.id}">${item.qty}</span>
            <button class="qty-btn" onclick="changeQty(${item.id},1)">+</button>
          </div>
        </div>
      </div>
    `;
  }).join('');
  updateTotal();
  updateRouteFromCart();
}

function addToCart(key, price, location, emoji, image) {
  const existing = cartItems.find(i => i.key === key);
  if (existing) {
    existing.qty++;
  } else {
    const dbData = IKEA_DB[key] || {};
    cartItems.push({
      id: Date.now(),
      key: key,
      name: dbData.nombre || key.split('|')[0].trim(),
      price: Number(price) || dbData.price || 0,
      priceStr: dbData.priceStr || formatPrice(price),
      location: location || dbData.location || 'Consultar',
      peso: dbData.peso || dbData.weight || '-',
      emoji: emoji || dbData.emoji || '📦',
      image: image || dbData.image || '',
      desc: dbData.subtitulo || dbData.desc || '',
      qty: 1,
      inFav: false
    });
  }
  updateAllBadges();
  renderCart();
}

function removeFromCart(id) {
  const el = document.getElementById('cart-item-'+id);
  if (el) { el.style.opacity='0'; el.style.transform='translateX(-100%)'; el.style.transition='all 0.28s'; }
  setTimeout(() => { cartItems = cartItems.filter(i=>i.id!==id); renderCart(); }, 300);
  updateAllBadges();
  updateRouteFromCart();
}
function changeQty(id, delta) {
  const item = cartItems.find(i=>i.id===id); if(!item) return;
  item.qty = Math.max(1, item.qty+delta);
  document.getElementById('qty-'+id).textContent = item.qty;
  updateTotal();
}
function toggleFavFromCart(id) {
  const item = cartItems.find(i=>i.id===id); if(!item) return;
  item.inFav = !item.inFav;

  // Sincronizar con favItems
  const favIdx = favItems.findIndex(f => f.key === item.key);
  if (item.inFav && favIdx < 0) {
    favItems.push({
      id: Date.now(),
      key: item.key,
      name: item.name,
      price: item.price,
      location: item.location,
      peso: item.peso,
      emoji: item.emoji,
      image: item.image,
      inCart: true
    });
  } else if (!item.inFav && favIdx >= 0) {
    favItems.splice(favIdx, 1);
  }

  renderCart();
  showToast(item.inFav ? '⭐ Guardado en Favoritos' : '✖️ Eliminado de Favoritos');
}
function updateTotal() {
  const subtotal = cartItems.reduce((s,i)=>s+i.price*i.qty, 0);
  const total = subtotal + 1;
  const fmt = n => n.toFixed(2).replace('.',',')+' €';
  ['subtotal-val','pay-subtotal'].forEach(id => { const el=document.getElementById(id); if(el) el.textContent=fmt(subtotal); });
  ['total-val','pay-total'].forEach(id => { const el=document.getElementById(id); if(el) el.textContent=fmt(total); });
  const pb = document.getElementById('pay-confirm-btn');
  if (pb) pb.textContent = 'Confirmar y pagar · '+fmt(total);
}
function updateAllBadges() {
  const count = cartItems.reduce((s,i)=>s+i.qty, 0);
  document.querySelectorAll('.cart-badge').forEach(b => {
    b.textContent=count; b.classList.add('badge-animate');
    setTimeout(()=>b.classList.remove('badge-animate'),300);
  });
}
function clearCart() { cartItems=[]; updateAllBadges(); renderCart(); updateRouteFromCart(); }
function sortCart(btn, mode) {
    // 1. Manejo del estilo visual de los botones
    document.querySelectorAll('#screen-cesta .aisle-btn').forEach(b => b.classList.remove('active'));
    
    if (btn) {
        // Si hemos hecho clic en un botón físico, lo marcamos
        btn.classList.add('active');
    } else {
        // Si viene por voz, buscamos el botón correspondiente en el HTML y lo marcamos
        const targetBtn = document.querySelector(`#screen-cesta .aisle-btn[onclick*="${mode}"]`);
        if (targetBtn) targetBtn.classList.add('active');
    }

    // 2. Lógica de ordenado a prueba de fallos
    if (mode === 'price') {
        // De mayor a menor precio
        cartItems.sort((a, b) => (b.price || 0) - (a.price || 0));
    } else if (mode === 'weight') {
        // De mayor a menor peso (si prefieres de menor a mayor, cambia el orden de a y b)
        cartItems.sort((a, b) => (b.weight || 0) - (a.weight || 0));
    } else if (mode === 'az') {
        // El || '' evita que crashee si un producto no tiene nombre
        cartItems.sort((a, b) => (a.name || '').localeCompare(b.name || ''));
    } else if (mode === 'route') {
        // El || '' evita el crasheo que tenías con las ubicaciones vacías
        cartItems.sort((a, b) => (a.location || '').localeCompare(b.location || ''));
    }
    
    // 3. Renderizamos la cesta actualizada
    if (typeof renderCart === 'function') renderCart();
}

// Generar ruta desde el carrito
function updateRouteFromCart() {
  routeItems = cartItems.map((item, idx) => {
    const dbData = IKEA_DB[item.key] || {};
    return {
      step: idx + 1,
      key: item.key,
      name: item.name || dbData.nombre || item.key,
      location: item.location,
      done: false,
      emoji: item.emoji || dbData.emoji || '📦'
    };
  });
}

// ═══════════════════════════════ FAVS ═══════════════════════════════

function renderFavs() {
  const list = document.getElementById('fav-list');
  if (!list) return;
  if (!catalogLoaded) {
    list.innerHTML = '<div style="text-align:center;padding:48px 20px;color:var(--gray);"><div style="font-size:48px;margin-bottom:12px;">⭐</div><div style="font-weight:700;">Cargando favoritos...</div></div>';
    return;
  }
  if (!favItems.length) {
    list.innerHTML = '<div style="text-align:center;padding:48px 20px;color:var(--gray);"><div style="font-size:48px;margin-bottom:12px;">⭐</div><div style="font-weight:700;">Sin favoritos todavía</div></div>';
    return;
  }
  list.innerHTML = favItems.map(item => {
    const dbData = IKEA_DB[item.key] || {};
    const displayName = item.name || dbData.nombre || item.key || 'Producto';
    const priceStr = item.priceStr || dbData.priceStr || formatPrice(item.price);
    const descStr = dbData.subtitulo || dbData.desc || item.desc || '';
    const emojiStr = item.emoji || dbData.emoji || '📦';
    const imageStr = item.image || dbData.image || '';

    return `
      <div class="fav-product-card">
        <div class="fav-card-top" onclick="openProduct('${item.key}')" style="cursor:pointer;">
          ${imageStr ? `<img src="${imageStr}" alt="${displayName}" style="width:44px;height:44px;object-fit:contain;border-radius:8px;flex-shrink:0;" onerror="this.style.display='none';this.nextElementSibling.style.display='flex';"><div class="product-img" style="display:none;">${emojiStr}</div>` : `<div class="product-img">${emojiStr}</div>`}
          <div style="flex:1;">
            <div class="product-name">${displayName}</div>
            <div class="product-desc">${descStr}</div>
            <span class="product-location">📍 ${item.location || dbData.location || 'Consultar'}</span>
          </div>
          <div style="text-align:right;"><div class="product-price">${priceStr}</div></div>
        </div>
        <div style="display:flex;gap:7px;">
          <button class="btn ${item.inCart ? 'btn-success' : 'btn-primary'}" style="flex:2;"
            onclick="${item.inCart ? '' : `addToCart('${item.key}',${item.price || dbData.price || 0},'${item.location || dbData.location || 'Consultar'}','${emojiStr}','${imageStr}');this.textContent='✅ En cesta';this.className='btn btn-success';`}">
            ${item.inCart ? '✅ En cesta' : '+ Añadir a la cesta'}
          </button>
          <button class="btn btn-danger" onclick="removeFav(${item.id})">🗑️</button>
        </div>
      </div>
    `;
  }).join('');
}
function removeFav(id) { favItems=favItems.filter(i=>i.id!==id); renderFavs(); showToast('✖️ Eliminado de Favoritos'); }

// ═══════════════════════════════ SEARCH ═══════════════════════════════

function renderSearch(query) {
  const list = document.getElementById('search-results');
  if (!list) return;
  if (!catalogLoaded || !allProducts.length) {
    list.innerHTML = '<div style="text-align:center;padding:40px;color:var(--gray);">Cargando catálogo...</div>';
    return;
  }

  let results = [];

  if (!query || query.trim() === '') {
    // Si no hay búsqueda: desordenamos una copia del array y cogemos solo 20
    const shuffled = [...allProducts].sort(() => 0.5 - Math.random());
    results = shuffled.slice(0, 20);
  } else {
    // Si hay búsqueda: filtramos sobre los +6000 productos
    const lowerQuery = query.toLowerCase();
    results = allProducts.filter(p =>
      (p.name && p.name.toLowerCase().includes(lowerQuery)) ||
      (p.desc && p.desc.toLowerCase().includes(lowerQuery)) ||
      (IKEA_DB[p.name]?.desc && IKEA_DB[p.name].desc.toLowerCase().includes(lowerQuery))
    );

    results = results.slice(0, 20); 
  }

  list.innerHTML = results.map(p => {
    const dbData = IKEA_DB[p.key] || p;
    const priceStr = dbData.priceStr || formatPrice(dbData.price);
    const locationStr = dbData.location || 'Consultar';
    const weightStr = dbData.peso || dbData.weight || '-';
    const descStr = dbData.subtitulo || dbData.desc || '';
    const emojiStr = dbData.emoji || '📦';
    const imageStr = dbData.image || '';
    const displayName = dbData.nombre || dbData.key || 'Producto';

    return `
      <div class="product-card" style="flex-direction:column;gap:7px;" onclick="openProduct('${p.key}')">
        <div style="display:flex;gap:11px;align-items:center;">
          ${imageStr ? `<img src="${imageStr}" alt="${displayName}" style="width:50px;height:50px;object-fit:contain;border-radius:8px;flex-shrink:0;" onerror="this.style.display='none';this.nextElementSibling.style.display='flex';"><div class="product-img" style="display:none;">${emojiStr}</div>` : `<div class="product-img">${emojiStr}</div>`}
          <div style="flex:1;">
            <div class="product-name">${displayName}</div>
            <div class="product-desc">${descStr}</div>
            <span class="product-location">📍 ${locationStr} · ⚖️ ${weightStr}</span>
          </div>
          <div class="product-price">${priceStr}</div>
        </div>
        <div style="display:flex;gap:5px;" onclick="event.stopPropagation()">
          <button class="btn btn-primary" style="flex:2;" onclick="event.stopPropagation();addToCart('${p.key}',${dbData.price || 0},'${locationStr}','${emojiStr}','${imageStr}');showToast('✅ ${displayName} añadido')">+ Añadir</button>
          <button class="btn btn-outline" onclick="event.stopPropagation();openProduct('${p.key}')">Ver</button>
        </div>
      </div>
    `;
  }).join('') || `<div style="text-align:center;padding:40px;color:var(--gray);">Sin resultados para "${query}"</div>`;
}
function filterProducts(q) { renderSearch(q); }

function renderShopping() {
  const list = document.getElementById('shopping-list');
  if (!list) return;
  if (!catalogLoaded || !allProducts.length) {
    list.innerHTML = '<div style="text-align:center;padding:40px;color:var(--gray);">Cargando catálogo...</div>';
    return;
  }

  // 1. Clocar el array, desordenarlo y quedarnos solo con 20 productos para el escaparate
  const shuffled = [...allProducts].sort(() => 0.5 - Math.random());
  const randomProducts = shuffled.slice(0, 20);

  // 2. Mapear SOLO esos 20 productos aleatorios
  list.innerHTML = randomProducts.map(p => {
    const dbData = IKEA_DB[p.key] || p;
    const priceStr = dbData.priceStr || formatPrice(dbData.price);
    const locationStr = dbData.location || 'Consultar';
    const descStr = dbData.subtitulo || dbData.desc || '';
    const emojiStr = dbData.emoji || '📦';
    const imageStr = dbData.image || '';
    const displayName = dbData.nombre || dbData.key || 'Producto';

    return `
      <div class="product-card" onclick="openProduct('${p.key}')">
        ${imageStr ? `<img src="${imageStr}" alt="${displayName}" style="width:60px;height:60px;object-fit:contain;border-radius:8px;flex-shrink:0;" onerror="this.style.display='none';this.nextElementSibling.style.display='flex';" onload="this.style.display='block';this.nextElementSibling.style.display='none';"><div class="product-img" style="display:none;">${emojiStr}</div>` : `<div class="product-img">${emojiStr}</div>`}
        <div class="product-info">
          <div class="product-name">${displayName}</div>
          <div class="product-desc">${descStr}</div>
          <span class="product-location">📍 ${locationStr}</span>
          <div class="product-actions" onclick="event.stopPropagation()">
            <button class="btn btn-primary" onclick="addToCart('${p.key}',${dbData.price || 0},'${locationStr}','${emojiStr}','${imageStr}');showToast('✅ ${displayName} añadido')">+ Añadir</button>
          </div>
        </div>
        <div style="text-align:right;flex-shrink:0;"><div class="product-price">${priceStr}</div></div>
      </div>
    `;
  }).join('');
}

// ═══════════════════════════════ PAY ═══════════════════════════════

function renderPaySummary() {
  const list = document.getElementById('pay-summary-list');
  if (!list) return;
  list.innerHTML = cartItems.map(item => {
    const dbData = IKEA_DB[item.key] || item;
    const displayName = item.name || dbData.nombre || item.key || 'Producto';
    const emojiStr = item.emoji || dbData.emoji || '📦';
    const imageStr = item.image || dbData.image || '';
    return `
      <div class="pay-item" onclick="openProduct('${item.key}')">
        ${imageStr ? `<img src="${imageStr}" alt="${displayName}" style="width:40px;height:40px;object-fit:contain;border-radius:6px;margin-right:10px;" onerror="this.style.display='none';">` : `<div class="pay-item-img">${emojiStr}</div>`}
        <div class="pay-item-info"><div class="pay-item-name">${displayName} x${item.qty}</div><div class="pay-item-loc">📍 ${item.location}</div></div>
        <div class="pay-item-price">${formatPrice(item.price * item.qty)}</div>
      </div>
    `;
  }).join('');
  updateTotal();
}
function selectPayment(el) { document.querySelectorAll('.payment-option').forEach(o=>o.classList.remove('selected')); el.classList.add('selected'); }
function confirmPayment() {
  const btn = document.getElementById('pay-confirm-btn');
  btn.textContent = '⏳ Procesando...';
  btn.disabled = true;

  setTimeout(() => {
    const MONTHS = ['ENE','FEB','MAR','ABR','MAY','JUN','JUL','AGO','SEP','OCT','NOV','DIC'];
    const now      = new Date();
    const subtotal = cartItems.reduce((s, i) => s + i.price * i.qty, 0);
    const bagFee   = cartItems.length > 0 ? 1 : 0;
    const total    = subtotal + bagFee;
    const nItems   = cartItems.reduce((s, i) => s + i.qty, 0);
    const orderId  = '#IK-' + now.getFullYear() + '-' + String(orderHistory.length + 1001).padStart(4, '0');

    if (!cartItems.length) {
      btn.textContent = 'Confirmar y pagar · 0,00 €';
      btn.disabled = false;
      showToast('⚠️ La cesta está vacía');
      return;
    }

    orderHistory.push({
      id: orderId,
      day: now.getDate(),
      month: MONTHS[now.getMonth()],
      year: now.getFullYear(),
      items: cartItems.map(i => ({ ...i })),
      subtotal,
      bagFee,
      total,
      nItems
    });

    const oid  = document.getElementById('success-order-id');
    if (oid) oid.textContent = orderId;
    const osum = document.getElementById('success-order-summary');
    if (osum) osum.textContent = `${formatPrice(total)} · ${nItems} artículo${nItems !== 1 ? 's' : ''} · IKEA Bilbao`;

    btn.disabled = false;
    renderHistorial();
    renderProfile();
    goTo('success');
  }, 1800);
}

// ═══════════════════════════════ HISTORIAL ═══════════════════════════════

function renderHistorial() {
  // --- Estadísticas globales ---
  const totalVisits = orderHistory.length;
  const totalItems  = orderHistory.reduce((s, o) => s + o.nItems, 0);
  const totalSpent  = orderHistory.reduce((s, o) => s + o.total, 0);

  const elVisits = document.getElementById('hist-visits');
  const elItems  = document.getElementById('hist-items');
  const elSpent  = document.getElementById('hist-spent');
  if (elVisits) elVisits.textContent = totalVisits;
  if (elItems)  elItems.textContent  = totalItems;
  if (elSpent)  elSpent.textContent  = Math.round(totalSpent) + '€';

  // --- Lista de pedidos ---
  const listEl = document.getElementById('hist-orders-list');
  if (!listEl) return;

  if (!orderHistory.length) {
    listEl.innerHTML = `
      <div style="text-align:center;padding:40px;color:var(--gray);">
        <div style="font-size:48px;margin-bottom:12px;">📋</div>
        <div style="font-weight:700;">Sin pedidos todavía</div>
        <div style="font-size:12px;margin-top:4px;">Tus compras aparecerán aquí</div>
      </div>`;
    return;
  }

  // Mostrar los pedidos del más reciente al más antiguo
  listEl.innerHTML = [...orderHistory].reverse().map(order => {
    const itemsHtml = order.items.map(item => {
      const d = IKEA_DB[item.key] || {};
      const name     = item.name || d.nombre || item.key || 'Producto';
      const imageStr = item.image || d.image || '';
      const emojiStr = item.emoji || d.emoji || '📦';
      return `
        <div style="padding:9px 14px;display:flex;gap:10px;align-items:center;border-bottom:1px solid var(--border);">
          ${imageStr
            ? `<img src="${imageStr}" alt="${name}" style="width:42px;height:42px;object-fit:contain;border-radius:9px;flex-shrink:0;" onerror="this.style.display='none';">`
            : `<div style="width:42px;height:42px;border-radius:9px;background:#dde;display:flex;align-items:center;justify-content:center;font-size:20px;flex-shrink:0;">${emojiStr}</div>`}
          <div style="flex:1;min-width:0;">
            <div style="font-size:12.5px;font-weight:700;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">${name}${item.qty > 1 ? ' x' + item.qty : ''}</div>
            <div style="font-size:10.5px;color:var(--gray);">📍 ${item.location || 'Consultar'}</div>
          </div>
          <div style="font-size:13px;font-weight:900;flex-shrink:0;">${formatPrice(item.price * item.qty)}</div>
        </div>`;
    }).join('');

    return `
      <div style="background:var(--card);border-radius:16px;margin:0 14px 10px;overflow:hidden;box-shadow:0 2px 8px rgba(0,0,0,0.07);">
        <!-- Cabecera del pedido -->
        <div style="padding:11px 14px;background:rgba(0,88,163,0.05);display:flex;align-items:center;gap:11px;border-bottom:1px solid var(--border);">
          <div style="width:48px;height:48px;border-radius:11px;background:var(--blue);color:white;display:flex;flex-direction:column;align-items:center;justify-content:center;font-size:9.5px;font-weight:900;flex-shrink:0;">
            <span style="font-size:17px;font-weight:900;line-height:1;">${order.day}</span>
            <span>${order.month}</span>
          </div>
          <div style="flex:1;">
            <div style="font-size:13.5px;font-weight:700;">Pedido ${order.id}</div>
            <div style="font-size:10.5px;color:var(--gray);margin-top:2px;">${order.nItems} artículo${order.nItems !== 1 ? 's' : ''}</div>
          </div>
          <div style="font-size:14px;font-weight:900;color:var(--blue);">${formatPrice(order.total)}</div>
        </div>
        <!-- Líneas de productos -->
        ${itemsHtml}
        <!-- Totales -->
        <div style="padding:8px 14px;display:flex;justify-content:space-between;font-size:12px;color:var(--gray);border-top:1px solid var(--border);">
          <span>Subtotal</span><span style="font-weight:700;color:var(--text);">${formatPrice(order.subtotal)}</span>
        </div>
        <div style="padding:8px 14px 10px;display:flex;justify-content:space-between;font-size:12px;color:var(--gray);">
          <span>Bolsa IKEA</span><span style="font-weight:700;color:var(--text);">1,00 €</span>
        </div>
      </div>`;
  }).join('');
}
let computedRouteData = [];
let mapActiveFilter = 'todos';

function renderRoute() {
    computeShortestRoute();
    renderMapSectionHighlights();
    renderMapPinsAndPath();
    renderOrderedRouteList();
}

// Filtros del mapa
function setAisle(btn, filter) {
    mapActiveFilter = filter;
    document.querySelectorAll('#screen-mapa .aisle-btn').forEach(b => b.classList.remove('active'));
    if (btn) btn.classList.add('active');
    renderRoute();
}

function getFilteredRouteData() {
    if (mapActiveFilter === 'todos') return computedRouteData;
    return computedRouteData.filter(item => item.aisle === mapActiveFilter);
}

function renderMapSectionHighlights() {
    const sections = document.querySelectorAll('#screen-mapa .map-section, #screen-mapa .corridor');
    sections.forEach(section => section.classList.remove('highlighted'));

    if (mapActiveFilter === 'todos') return;

    document.querySelectorAll('#screen-mapa .corridor').forEach(corridor => {
        if ((corridor.textContent || '').trim() === mapActiveFilter) {
            corridor.classList.add('highlighted');
        }
    });
}

function computeShortestRoute() {
    const items = (typeof cartItems !== 'undefined') ? cartItems : [];

    if (!items || items.length === 0) {
        computedRouteData = [];
        updateRouteStats();
        return;
    }

    // 1. Preparar los productos con sus coordenadas finales
    const xCoords = { A: 55.5, B: 67.5, C: 79.5, D: 91.5 };
    let pending = items.map(item => {
        const dbData = IKEA_DB[item.key] || {};
        const shelf = parseInt(dbData.estanteria) || 1;
        return {
            key: item.key,
            name: dbData.nombre || item.name || item.key.split('|')[0],
            price: item.price || dbData.price || 0,
            aisle: (dbData.pasillo || 'A').toUpperCase(),
            shelf: shelf,
            recogido: localStorage.getItem(`recogido_${item.key}`) === 'true',
            coords: {
                xPerc: xCoords[(dbData.pasillo || 'A').toUpperCase()] || 55.5,
                // AJUSTE: Inicio 8% + Rango 80% (Termina en 88%)
                yPerc: 8 + ((shelf - 1) * (80 / 19))
            }
        };
    });

    // 2. ALGORITMO POR DISTANCIA
    // Empezamos en la posición de la entrada (Almacén)
    
    let currentPos = { x: 50, y: 15 }; 
    let orderedRoute = [];

    while (pending.length > 0) {
        let nearestIndex = -1;
        let minDistance = Infinity;

        pending.forEach((prod, index) => {
            // Calculamos distancia Manhattan (pasillos + profundidad)
            const dist = Math.abs(currentPos.x - prod.coords.xPerc) + 
                         Math.abs(currentPos.y - prod.coords.yPerc);

            if (dist < minDistance) {
                minDistance = dist;
                nearestIndex = index;
            }
        });

        // Extraemos el más cercano y actualizamos posición actual
        const nextProd = pending.splice(nearestIndex, 1)[0];
        orderedRoute.push(nextProd);
        currentPos = { x: nextProd.coords.xPerc, y: nextProd.coords.yPerc };
    }

    computedRouteData = orderedRoute;
    updateRouteStats();
}

function updateRouteStats() {
    const visibleRoute = getFilteredRouteData();
    const totalProducts = visibleRoute.length;
    const totalPrice = visibleRoute.reduce((sum, p) => sum + (p.price || 0), 0);

    // Ej: 1.5 min por producto + base
    const estimatedTime = totalProducts > 0 ? Math.round(totalProducts * 1.5 + 2) : 0;

    document.getElementById('route-prod-count').textContent = totalProducts;
    document.getElementById('route-time').textContent = estimatedTime;
    document.getElementById('route-total').textContent =
        totalPrice.toFixed(2).replace('.', ',') + ' €';
}

function renderMapPinsAndPath() {
    const mapArea = document.getElementById('store-map-area');
    const svgArea = document.getElementById('map-route-svg');

    if (!mapArea || !svgArea) return;

    mapArea.querySelectorAll('.dynamic-route-dot').forEach(el => el.remove());
    svgArea.innerHTML = '';

    const visibleRoute = getFilteredRouteData();

    if (!visibleRoute.length) {
        return;
    }

    const startPos = { xPerc: 10, yPerc: 90 };
    const endPos = { xPerc: 50, yPerc: 98 };

    let routePoints = [`${startPos.xPerc}%,${startPos.yPerc}%`];
    let currentPos = startPos;

    visibleRoute.forEach((r, i) => {
        const pin = document.createElement('div');
        pin.className = `route-dot dynamic-route-dot ${r.recogido ? 'status-done' : ''}`;
        pin.textContent = r.recogido ? '✓' : (i + 1);
        pin.style.left = `${r.coords.xPerc}%`;
        pin.style.top = `${r.coords.yPerc}%`;
        pin.title = `${r.name} · Pasillo ${r.aisle} · Estantería ${r.shelf}`;
        mapArea.appendChild(pin);

        routePoints.push(`${r.coords.xPerc}%,${currentPos.yPerc}%`);
        routePoints.push(`${r.coords.xPerc}%,${r.coords.yPerc}%`);
        currentPos = r.coords;
    });

    routePoints.push(`${endPos.xPerc}%,${currentPos.yPerc}%`);
    routePoints.push(`${endPos.xPerc}%,${endPos.yPerc}%`);

    const productPolyline = document.createElementNS("http://www.w3.org/2000/svg", "polyline");
    productPolyline.setAttribute("points", routePoints.join(' '));
    productPolyline.setAttribute("fill", "none");
    productPolyline.setAttribute("stroke", "var(--blue, #0058a3)");
    productPolyline.setAttribute("stroke-width", "2.5");
    productPolyline.setAttribute("stroke-dasharray", "6 4");
    svgArea.appendChild(productPolyline);

    const personPathPoints = [];
    currentPos = startPos;
    personPathPoints.push(`${startPos.xPerc}%,${startPos.yPerc}%`);

    visibleRoute.forEach(r => {
        personPathPoints.push(`${currentPos.xPerc}%,${r.coords.yPerc}%`);
        personPathPoints.push(`${r.coords.xPerc}%,${r.coords.yPerc}%`);
        currentPos = r.coords;
    });

    personPathPoints.push(`${currentPos.xPerc}%,${endPos.yPerc}%`);
    personPathPoints.push(`${endPos.xPerc}%,${endPos.yPerc}%`);

    const personPolyline = document.createElementNS("http://www.w3.org/2000/svg", "polyline");
    personPolyline.setAttribute("points", personPathPoints.join(' '));
    personPolyline.setAttribute("fill", "none");
    personPolyline.setAttribute("stroke", "var(--green, #1f8423)");
    personPolyline.setAttribute("stroke-width", "3");
    personPolyline.setAttribute("stroke-dasharray", "10 6");
    personPolyline.setAttribute("stroke-linecap", "round");
    personPolyline.setAttribute("opacity", "0.9");
    svgArea.appendChild(personPolyline);
}

function renderOrderedRouteList() {
    const list = document.getElementById('route-list');
    if (!list) return;

    const visibleRoute = getFilteredRouteData();

    if (!computedRouteData.length) {
        list.innerHTML = '<div style="padding:20px;text-align:center;">Cesta vacía</div>';
        return;
    }

    if (!visibleRoute.length) {
        list.innerHTML = `<div style="padding:20px;text-align:center;color:var(--gray);">No hay productos en ${mapActiveFilter === 'todos' ? 'la ruta' : 'el pasillo ' + mapActiveFilter}</div>`;
        return;
    }

    list.innerHTML = visibleRoute.map((r, i) => {

        const cleanName = r.name.includes('|')
            ? r.name.split('|')[0].trim()
            : r.name;

        return `
        <div class="route-item-card">
            <div class="route-step-number ${r.recogido ? 'status-done' : ''}">
                ${r.recogido ? '✓' : (i + 1)}
            </div>

            <div style="flex:1;">
                <div style="font-weight:700;">${cleanName}</div>
                <div style="font-size:10px;color:var(--blue);">
                    📍 Pasillo ${r.aisle} · Estantería ${r.shelf}
                </div>
            </div>

            <div style="font-size:11px;font-weight:700;">
                ${(r.price || 0).toFixed(2).replace('.', ',')} €
            </div>
        </div>
        `;
    }).join('');
}

// AUX
function updateRouteHeader(count, time) {
    document.getElementById('route-prod-count').textContent = count;
    document.getElementById('route-time').textContent = time;
}

// ═══════════════════════════════ PERFIL ═══════════════════════════════

function renderProfile() {
  const totalVisits = orderHistory.length;
  const totalItems = orderHistory.reduce((sum, order) => sum + (order.nItems || 0), 0);
  const totalSpent = orderHistory.reduce((sum, order) => sum + (order.total || 0), 0);
  const totalPoints = Math.round(totalSpent * 10);

  document.getElementById('perfil-name-display').textContent  = userProfile.name;
  document.getElementById('perfil-email-display').textContent = userProfile.email;
  document.getElementById('family-name-display').textContent  = userProfile.name;
  document.getElementById('pf-name').textContent  = userProfile.name;
  document.getElementById('pf-email').textContent = userProfile.email;
  document.getElementById('pf-phone').textContent = userProfile.phone;
  document.getElementById('pf-bday').textContent  = userProfile.bday;
  document.getElementById('cfg-name').textContent  = userProfile.name;
  document.getElementById('cfg-email').textContent = userProfile.email;

  const perfilStats = document.querySelector('#screen-perfil .perfil-stats');
  if (perfilStats) {
    const statsHtml = [
      `<div class="perfil-stat-card"><span class="perfil-stat-emoji">🏬</span><div class="perfil-stat-val">${totalVisits}</div><div class="perfil-stat-lbl">VISITAS</div></div>`,
      `<div class="perfil-stat-card"><span class="perfil-stat-emoji">⭐</span><div class="perfil-stat-val" id="perfil-points">${totalPoints.toLocaleString('es-ES')}</div><div class="perfil-stat-lbl">PUNTOS</div></div>`,
      `<div class="perfil-stat-card"><span class="perfil-stat-emoji">💚</span><div class="perfil-stat-val">${formatPrice(totalSpent)}</div><div class="perfil-stat-lbl">AHORRADO</div></div>`
    ];

    if (orderHistory.length > 0) {
      statsHtml.push(`<div class="perfil-stat-card"><span class="perfil-stat-emoji">💳</span><div class="perfil-stat-val">${formatPrice(totalSpent)}</div><div class="perfil-stat-lbl">GASTADO</div></div>`);
    }

    perfilStats.innerHTML = statsHtml.join('');
  }

  const familyPoints = document.getElementById('family-points-display');
  if (familyPoints) {
    familyPoints.textContent = totalPoints.toLocaleString('es-ES');
  }

  const activityCard = document.querySelector('#screen-perfil .perfil-section:last-of-type .perfil-card');
  if (activityCard) {
    activityCard.innerHTML = `
      <div class="perfil-field" onclick="goTo('historial')"><div class="perfil-field-icon">🧾</div><div class="perfil-field-info"><div class="perfil-field-label">HISTORIAL</div><div class="perfil-field-val">${orderHistory.length > 0 ? `Ver ${orderHistory.length} compra${orderHistory.length !== 1 ? 's' : ''} realizada${orderHistory.length !== 1 ? 's' : ''}` : 'Sin compras en esta sesión'}</div></div><span style="color:var(--gray);">›</span></div>
      <div class="perfil-field" onclick="navTo('favoritos')"><div class="perfil-field-icon">⭐</div><div class="perfil-field-info"><div class="perfil-field-label">FAVORITOS</div><div class="perfil-field-val">${favItems.length} artículo${favItems.length !== 1 ? 's' : ''} guardado${favItems.length !== 1 ? 's' : ''}</div></div><span style="color:var(--gray);">›</span></div>
    `;
  }
}

let editField = null;
const editConfigs = {
  name:  { title:'Editar nombre',         field:'name',  label:'NOMBRE',               type:'text',  placeholder:'Tu nombre completo' },
  email: { title:'Editar correo',         field:'email', label:'CORREO ELECTRÓNICO',   type:'email', placeholder:'ejemplo@correo.com' },
  phone: { title:'Editar teléfono',       field:'phone', label:'TELÉFONO',             type:'tel',   placeholder:'+34 600 000 000' },
  bday:  { title:'Fecha de nacimiento',   field:'bday',  label:'FECHA (DD / MM / AAAA)',type:'text',  placeholder:'15 / 03 / 1990' },
};

function openEditSheet(field) {
  const cfg = field ? editConfigs[field] : null;
  editField = field || null;
  const title = cfg ? cfg.title : 'Editar perfil';
  const body  = cfg ? `
    <div class="form-group">
      <label class="form-label">${cfg.label}</label>
      <input class="form-input" id="edit-field-input" type="${cfg.type}" placeholder="${cfg.placeholder}" value="${userProfile[cfg.field]||''}">
    </div>
  ` : Object.entries(editConfigs).map(([k,c]) => `
    <div class="form-group">
      <label class="form-label">${c.label}</label>
      <input class="form-input edit-all-input" data-field="${k}" type="${c.type}" placeholder="${c.placeholder}" value="${userProfile[k]||''}">
    </div>
  `).join('');
  document.getElementById('edit-sheet-title').textContent = title;
  document.getElementById('edit-sheet-body').innerHTML    = body;
  document.getElementById('edit-overlay').classList.add('open');
  setTimeout(() => { const inp = document.getElementById('edit-field-input'); if(inp) inp.focus(); }, 300);
}

function closeEditSheet() { document.getElementById('edit-overlay').classList.remove('open'); }
function closeEditSheetOnBg(e) { if(e.target.id==='edit-overlay') closeEditSheet(); }

function saveEditSheet() {
  if (editField) {
    const inp = document.getElementById('edit-field-input');
    if (inp && inp.value.trim()) { userProfile[editField] = inp.value.trim(); showToast('✅ Cambios guardados'); }
  } else {
    document.querySelectorAll('.edit-all-input').forEach(inp => {
      if (inp.value.trim()) userProfile[inp.dataset.field] = inp.value.trim();
    });
    showToast('✅ Perfil actualizado');
  }
  closeEditSheet();
  renderProfile();
}

// ═══════════════════════════════ CAMERA ═══════════════════════════════

let streams = { ar: null, qr: null };
let arFacing = 'environment';
let qrDetected = false;
let qrScanning = false;
const AR_API = window.location.origin + '/identify';
// Canvas compartido para captura de frames
let captureCanvas = null;
let captureCtx = null;

async function startCamera(type) {
  const video = document.getElementById(type + '-video');
  const noCam = document.getElementById(type + '-no-cam');
  const statusEl = document.getElementById(type + '-status');

  // Detener cámara anterior
  stopCamera(type);

  // Crear canvas para captura si no existe
  if (!captureCanvas) {
    captureCanvas = document.createElement('canvas');
    captureCtx = captureCanvas.getContext('2d', { willReadFrequently: true });
  }

  if (type === 'ar') {
    const btn = document.getElementById('ar-capture-btn');
    if (btn) btn.disabled = true;
  }

  document.getElementById(type + '-camera-wrap').classList.add('expanded');

  try {
    const facing = type === 'ar' ? arFacing : 'environment';
    const mediaStream = await navigator.mediaDevices.getUserMedia({
      video: {
        facingMode: facing,
        width: { ideal: 1280 },
        height: { ideal: 720 }
      },
      audio: false
    });

    streams[type] = mediaStream;
    video.srcObject = mediaStream;
    await video.play();

    // Esperar a que el video esté listo
    await new Promise(resolve => {
      if (video.readyState >= 2) resolve();
      else video.onloadedmetadata = resolve;
    });

    noCam.style.display = 'none';

    if (type === 'ar') {
      const btn = document.getElementById('ar-capture-btn');
      if (btn) btn.disabled = false;
      if (statusEl) {
        statusEl.textContent = '📷 Centra el mueble';
        statusEl.className = 'cam-status';
      }
    }

    if (type === 'qr') {
      qrDetected = false;
      qrScanning = true;
      const qrDetectedEl = document.getElementById('qr-detected');
      if (qrDetectedEl) qrDetectedEl.style.display = 'none';
      if (statusEl) {
        statusEl.textContent = '📷 Buscando código QR...';
        statusEl.className = 'cam-status';
      }
      startQRScanning();
    }

  } catch (err) {
    console.error('Error accessing camera:', err);
    if (noCam) noCam.style.display = 'flex';
    if (statusEl) {
      statusEl.textContent = '⚠️ Sin cámara';
      statusEl.className = 'cam-status';
    }
  }
}

function stopCamera(type) {
  if (type === 'qr') {
    qrScanning = false;
    stopQRScanning();
  }

  if (streams[type]) {
    streams[type].getTracks().forEach(track => track.stop());
    streams[type] = null;
  }

  const video = document.getElementById(type + '-video');
  if (video) {
    video.srcObject = null;
    video.pause();
  }
}

// ─── QR SCAN ───

let qrAnimationFrameId = null;
let qrLastScanTime = 0;
const QR_SCAN_INTERVAL = 100; // ms entre scans

function startQRScanning() {
  if (qrAnimationFrameId) {
    cancelAnimationFrame(qrAnimationFrameId);
  }
  qrLastScanTime = 0;
  qrScanning = true;
  scanQRLoop();
}

function stopQRScanning() {
  qrScanning = false;
  if (qrAnimationFrameId) {
    cancelAnimationFrame(qrAnimationFrameId);
    qrAnimationFrameId = null;
  }
}

function scanQRLoop(timestamp) {
  if (!qrScanning || qrDetected || !streams['qr']) {
    qrAnimationFrameId = null;
    return;
  }

  if (timestamp - qrLastScanTime >= QR_SCAN_INTERVAL) {
    qrLastScanTime = timestamp;
    scanQRFrame();
  }

  qrAnimationFrameId = requestAnimationFrame(scanQRLoop);
}

function scanQRFrame() {
  if (qrDetected || !qrScanning) return;

  const video = document.getElementById('qr-video');
  if (!video || !video.srcObject || video.readyState < 2) return;

  try {
    let w = video.videoWidth;
    let h = video.videoHeight;

    if (!w || !h) {
      w = video.clientWidth || 320;
      h = video.clientHeight || 240;
    }

    if (!w || !h || w === 0 || h === 0) return;

    // Solo redimensionar si es necesario
    if (captureCanvas.width !== w || captureCanvas.height !== h) {
      captureCanvas.width = w;
      captureCanvas.height = h;
    }

    captureCtx.drawImage(video, 0, 0, w, h);

    const imageData = captureCtx.getImageData(0, 0, w, h);

    if (typeof jsQR !== 'undefined') {
      const code = jsQR(imageData.data, w, h, {
        inversionAttempts: 'dontInvert'
      });

      if (code && code.data) {
        qrDetected = true;
        handleQRDetected(code.data);
        return;
      }
    }
  } catch (e) {
    // Ignorar errores de contexto - son normales durante transiciones
  }
}

function handleQRDetected(data) {
  qrDetected = true;
  stopQRScanning();

  document.getElementById('qr-camera-wrap').classList.remove('expanded');

  const statusEl = document.getElementById('qr-status');
  if (statusEl) {
    statusEl.textContent = '✅ ¡Detectado!';
    statusEl.className = 'cam-status found';
  }

  const qrDetectedEl = document.getElementById('qr-detected');
  if (qrDetectedEl) {
    qrDetectedEl.style.display = 'block';
    qrDetectedEl.style.animation = 'fadeInUp 0.36s ease-out';
  }

  const cleanData = data.trim();

  // Búsqueda directa por URL — el QR de IKEA lleva exactamente la URL del producto
  const product = Object.values(IKEA_DB).find(p => p.url === cleanData)
               ?? Object.values(IKEA_DB).find(p => p.url?.includes(cleanData))
               ?? Object.values(IKEA_DB).find(p => cleanData.includes(p.url))
               ?? findProduct(cleanData); // Fallback por nombre/key por si acaso

  const qrProductNameEl   = document.getElementById('qr-product-name');
  const qrProductDetailEl = document.getElementById('qr-product-detail');

  if (product) {
    if (qrProductNameEl) {
      qrProductNameEl.textContent = [product.nombre, product.subtitulo]
        .filter(Boolean).join(' – ');
    }
    if (qrProductDetailEl) {
      qrProductDetailEl.textContent = [
        product.location,
        product.priceStr || formatPrice(product.price),
        product.peso || '-'
      ].filter(s => s && s !== '-').join(' · ');
    }

    window._qrProduct = product;
    showToast(`✅ ${product.nombre}`);

  } else {
    if (qrProductNameEl)   qrProductNameEl.textContent   = `Código: ${cleanData}`;
    if (qrProductDetailEl) qrProductDetailEl.textContent = 'Producto no encontrado';

    window._qrProduct = null;
    showToast('⚠️ Producto no encontrado en el catálogo');
  }
}

function qrAddToCart() {
  const p = window._qrProduct;
  if (!p) {
    showToast('⚠️ Primero detecta un producto');
    return;
  }
  addToCart(p.key, p.price, p.location, p.emoji, p.image);
  showToast(`✅ ${p.nombre || p.key} añadido a la cesta`);
}

function qrViewProduct() {
  const p = window._qrProduct;
  if (p) {
    openProduct(p.key);
  } else {
    showToast('⚠️ Primero detecta un producto');
  }
}

function qrToggleFav() {
  const p = window._qrProduct;
  if (!p) {
    showToast('⚠️ Primero detecta un producto');
    return;
  }
  const idx = favItems.findIndex(f => f.key === p.key);
  const btn = document.getElementById('qr-fav-btn');
  if (idx >= 0) {
    favItems.splice(idx, 1);
    if (btn) btn.textContent = '☆';
    showToast('✖️ Eliminado de Favoritos');
  } else {
    favItems.push({
      id: Date.now(),
      key: p.key,
      name: p.nombre || p.key,
      price: p.price,
      location: p.location,
      peso: p.peso,
      emoji: p.emoji,
      image: p.image
    });
    if (btn) btn.textContent = '⭐';
    showToast('⭐ Guardado en Favoritos');
  }
}

function resetQRScanner() {
  qrDetected = false;
  const qrDetectedEl = document.getElementById('qr-detected');
  if (qrDetectedEl) qrDetectedEl.style.display = 'none';
  const statusEl = document.getElementById('qr-status');
  if (statusEl) {
    statusEl.textContent = '📷 Buscando código QR...';
    statusEl.className = 'cam-status';
  }
  window._qrProduct = null;
  startQRScanning();
}

function toggleQRFlash() {
  const btn = document.getElementById('qr-flash-btn');
  const isOn = btn.style.background.includes('219');
  btn.style.background = isOn ? 'rgba(255,255,255,0.18)' : 'rgba(255,219,0,0.4)';
  showToast('💡 Flash ' + (isOn ? 'desactivado' : 'activado'));
}

// ─── AR IA ───

async function arFlipCamera() {
  arFacing = arFacing === 'environment' ? 'user' : 'environment';
  await startCamera('ar');
  showToast('🔄 Cámara cambiada');
}

async function arCapture() {
  const video = document.getElementById('ar-video');
  const captBtn = document.getElementById('ar-capture-btn');
  const statusEl = document.getElementById('ar-status');
  const loader = document.getElementById('ar-loader');

  if (!streams['ar'] || !video.srcObject) {
    showToast('⚠️ No hay cámara activa');
    return;
  }

  // Deshabilitar botón inmediatamente
  if (captBtn) captBtn.disabled = true;

  try {
    // Esperar un frame para asegurar que el video tiene dimensiones
    await new Promise(resolve => requestAnimationFrame(resolve));

    // Capturar frame sin pausar el video
    let w = video.videoWidth;
    let h = video.videoHeight;

    // Si las dimensiones no están disponibles, usar las del video element
    if (!w || !h) {
      w = video.clientWidth || 640;
      h = video.clientHeight || 480;
    }

    // Solo redimensionar canvas si las dimensiones cambiaron significativamente
    if (captureCanvas.width !== w || captureCanvas.height !== h) {
      captureCanvas.width = w;
      captureCanvas.height = h;
    }

    // Dibujar frame actual
    captureCtx.drawImage(video, 0, 0, w, h);

    if (statusEl) {
      statusEl.textContent = '🔍 Identificando…';
      statusEl.className = 'cam-status detecting';
    }
    if (loader) loader.classList.add('on');

    // Crear blob de forma asíncrona
    const blob = await new Promise(resolve => {
      captureCanvas.toBlob(resolve, 'image/jpeg', 0.85);
    });

    if (!blob) {
      throw new Error('Error al crear imagen');
    }

    const formData = new FormData();
    formData.append('file', new File([blob], 'ar_capture.jpg', { type: 'image/jpeg' }));

    const response = await fetch(AR_API, {
      method: 'POST',
      body: formData
    });

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }

    const data = await response.json();
    renderARResults(data);

    document.getElementById('ar-camera-wrap').classList.remove('expanded');

    if (statusEl) {
      statusEl.textContent = '✅ Detectado';
      statusEl.className = 'cam-status found';
    }

  } catch (e) {
    console.error('Error en AR:', e);
    if (statusEl) {
      statusEl.textContent = '⚠️ API no disponible';
      statusEl.className = 'cam-status';
    }

    const list = document.getElementById('ar-detected-list');
    if (list) {
      list.innerHTML = `
        <div style="padding:16px;text-align:center;color:var(--gray);">
          <div style="font-size:28px;margin-bottom:7px;">🔌</div>
          <div style="font-size:12.5px;font-weight:700;">Sin conexión con la API</div>
          <div style="font-size:10.5px;margin-top:3px;">Asegúrate de que app.py está corriendo.</div>
        </div>`;
    }
  } finally {
    // Reactivar botón y ocultar loader
    if (captBtn) captBtn.disabled = false;
    if (loader) loader.classList.remove('on');
    // NO modificar el video stream - la cámara sigue funcionando
  }
}

function renderARResults(data) {
  const list = document.getElementById('ar-detected-list');
  if (!list) return;

  const all = [data.best_match, ...(data.alternatives || [])].filter(Boolean);
  window._arResults = all;
  if (!all.length) {
    list.innerHTML = '<div style="padding:18px;text-align:center;color:var(--gray);">Sin resultados</div>';
    return;
  }

  list.innerHTML = all.map((item, i) => {
    // La respuesta de la API viene con product_name que puede ser la key completa
    let searchTerm = item.product_name || item.id || '';

    // Buscar en el catálogo usando la nueva función
    const product = findProduct(searchTerm);

    if (product) {
      // Producto encontrado en el catálogo
      const displayName = product.nombre || product.key || 'Producto';
      const priceStr = product.priceStr || formatPrice(product.price);
      const locationStr = product.location || 'Consultar';
      const emojiStr = product.emoji || '📦';
      const descStr = product.subtitulo || product.desc || '';
      const imageStr = product.image || '';
      const conf = Math.round((item.confidence || item.confidence_pct || 0) * 100);

      return `
        <div style="padding:11px 13px;display:flex;gap:9px;align-items:center;border-bottom:${i < all.length - 1 ? '1px solid var(--border)' : 'none'};">
          ${imageStr ? `<img src="${imageStr}" alt="${displayName}" style="width:50px;height:50px;object-fit:contain;border-radius:8px;flex-shrink:0;" onerror="this.style.display='none';this.nextElementSibling.style.display='flex';"><div class="product-img" style="display:none;${i === 0 ? 'border:2px solid var(--blue);' : ''}">${emojiStr}</div>` : `<div class="product-img" style="${i === 0 ? 'border:2px solid var(--blue);' : ''}">${emojiStr}</div>`}
          <div style="flex:1;cursor:pointer;" onclick="openProduct('${product.key}')">
            <div class="product-name">${displayName}</div>
            <div class="product-desc">${descStr}</div>
            <span class="product-location">📍 ${locationStr}</span>
            <div><span class="ar-conf-pill ${conf >= 65 ? '' : 'low'}">⚡ ${conf}% confianza</span></div>
          </div>
          <div style="text-align:right;">
            <div class="product-price">${priceStr}</div>
            <button class="btn btn-primary" style="margin-top:4px;padding:5px 9px;font-size:10.5px;"
              onclick="addToCart('${product.key}',${product.price || 0},'${locationStr}','${emojiStr}','${imageStr}');showToast('✅ ${displayName} añadido')">+</button>
          </div>
        </div>
      `;
    } else {
      // Producto NO encontrado - mostrar con la información de la API
      const displayName = searchTerm || 'Desconocido';
      const conf = Math.round((item.confidence || item.confidence_pct || 0) * 100);
      const priceStr = item.precio ? formatPrice(item.precio) : 'Consultar';

      return `
        <div style="padding:11px 13px;display:flex;gap:9px;align-items:center;border-bottom:${i < all.length - 1 ? '1px solid var(--border)' : 'none'};background:rgba(255,165,0,0.1);">
          <div class="product-img" style="${i === 0 ? 'border:2px solid var(--orange);' : ''}">❓</div>
          <div style="flex:1;">
            <div class="product-name">${displayName}</div>
            <div class="product-desc" style="color:var(--gray);">No encontrado en el catálogo</div>
            <div><span class="ar-conf-pill ${conf >= 65 ? '' : 'low'}">⚡ ${conf}% confianza</span></div>
          </div>
          <div style="text-align:right;">
            <div class="product-price">${priceStr}</div>
          </div>
        </div>
      `;
    }
  }).join('');
}

// Expande la cámara al hacer clic
function expandCamera(type, event) {
    // Si hacemos clic en el flash o el botón de AR, no hacemos nada
    if (event && (event.target.closest('button') || event.target.closest('.ar-controls-bar'))) return;

    const wrap = document.getElementById(type + '-camera-wrap');
    
    // Solo actuamos si está encogida
    if (!wrap.classList.contains('expanded')) {
        wrap.classList.add('expanded');
        
        // UX Top: Si expandes la de QR, asumimos que quieres volver a escanear
        if (type === 'qr' && qrDetected) {
            resetQRScanner();
        }
    }
}

// ═══════════════════════════════ PAYMENT ═══════════════════════════════

function confirmPaymentLegacyDisabled() {
  const btn=document.getElementById('pay-confirm-btn'); btn.textContent='⏳ Procesando...'; btn.disabled=true;
  setTimeout(()=>goTo('success'), 1800);
}

// ═══════════════════════════════ TOAST ═══════════════════════════════

let toastTimeout;
function showToast(msg) {
  const t=document.getElementById('toast');
  t.textContent=msg; t.classList.add('show');
  clearTimeout(toastTimeout);
  toastTimeout=setTimeout(()=>t.classList.remove('show'), 2500);
}

// ═══════════════════════════════ RIPPLE ═══════════════════════════════

document.addEventListener('click', e => {
  const btn = e.target.closest('.ripple-btn'); if(!btn) return;
  const r = document.createElement('div'); r.className='ripple-effect';
  const rect = btn.getBoundingClientRect();
  r.style.left=(e.clientX-rect.left)+'px'; r.style.top=(e.clientY-rect.top)+'px';
  btn.appendChild(r); setTimeout(()=>r.remove(), 600);
});

// ═══════════════════════════════ INIT ═══════════════════════════════

// Cargar catálogo primero, luego inicializar UI
document.addEventListener('DOMContentLoaded', () => {
  loadCatalog();
});

// Inicialización legacy (para cuando el catálogo ya está cargado localmente)
// Esto se ejecuta después de loadCatalog()
window.addEventListener('load', () => {
  // Si por alguna razón el catálogo no se cargó, usar datos locales
  setTimeout(() => {
    if (!catalogLoaded) {
      console.log('⚠️ Usando catálogo local');
      initLocalCatalog();
      initUIWithCatalog();
    }
  }, 1000);
});

// ═══════════════════════════════ RELOJ ═══════════════════════════════

function actualizarReloj() {
    const ahora = new Date();
    let horas = ahora.getHours();
    let minutos = ahora.getMinutes();
    
    // Añadir un cero a la izquierda si los minutos son menores de 10
    minutos = minutos < 10 ? '0' + minutos : minutos;
    
    const horaTexto = `${horas}:${minutos}`;
    
    // Busca TODOS los elementos que tengan la clase 'status-time' y les cambia el texto
    document.querySelectorAll('.status-time').forEach(reloj => {
        reloj.textContent = horaTexto;
    });
}

// Ejecutar nada más cargar la app
actualizarReloj();
// Actualizar cada 60 segundos
setInterval(actualizarReloj, 60000);

// ═══════════════════════════════ GESTOS ═══════════════════════════════

let gestosActivos = false;

function getActiveScreenId() {
    const activeScreen = document.querySelector('.screen.active');
    return activeScreen ? activeScreen.id : null;
}

const PANTALLAS_GESTO_BORRAR   = ['screen-inicio', 'screen-buscar', 'screen-shopping', 'screen-cesta', 'screen-favoritos', 'screen-producto', 'screen-ar', 'screen-escaner'];
const PANTALLAS_GESTO_FAVORITO = ['screen-producto', 'screen-ar', 'screen-escaner'];

function updateAllCartBadges() {
    const total = typeof cartItems !== 'undefined' ? cartItems.reduce((s, i) => s + i.qty, 0) : 0;
    document.querySelectorAll('.cart-badge').forEach(badge => {
        badge.textContent = total;
        badge.classList.add('badge-animate');
        setTimeout(() => badge.classList.remove('badge-animate'), 300);
    });
}

function updateAllCartBadges() {
    // Calcula el total real sumando las cantidades de cada item
    const total = typeof cartItems !== 'undefined' ? cartItems.reduce((s, i) => s + (i.qty || 1), 0) : 0;
    document.querySelectorAll('.cart-badge').forEach(badge => {
        badge.textContent = total;
        badge.classList.add('badge-animate');
        setTimeout(() => badge.classList.remove('badge-animate'), 300);
    });
}

function refreshProductButtonState() {
  
    if (getActiveScreenId() !== 'screen-producto' || !currentProductKey) return;
    
    const actionsContainer = document.getElementById('prod-actions');
    if (!actionsContainer) return;
    
    const addBtn = actionsContainer.querySelector('button.btn');
    if (!addBtn) return;

    const isInCart = cartItems.some(item => item.key === currentProductKey);

    if (isInCart) {
        addBtn.textContent = '✅ En la cesta';
        addBtn.className = 'btn btn-success ripple-btn'; 
    } else {
        addBtn.textContent = '+ Añadir a la cesta';
        addBtn.className = 'btn btn-primary ripple-btn'; // Tu clase original azul
    }
}

function refreshCurrentScreen(screenId) {
  
    refreshProductButtonState();

    switch (screenId) {
        case 'screen-cesta':
            if (typeof renderCart === 'function') renderCart();
            break;
        case 'screen-favoritos':
            if (typeof renderFavs === 'function') renderFavs();
            break;
        case 'screen-buscar':
            if (typeof renderSearch === 'function') renderSearch();
            break;
    }
    updateAllCartBadges();
    
}

function undoLastCartItem() {
    const screenId = getActiveScreenId();
    if (!PANTALLAS_GESTO_BORRAR.includes(screenId)) return;

    if (!cartItems || cartItems.length === 0) {
        showToast('ℹ️ Tu cesta ya está vacía');
        return;
    }

    if (screenId === 'screen-producto') {
        if (!currentProductKey) return;
        const index = cartItems.findIndex(item => item.key === currentProductKey);

        if (index !== -1) {
            const removed = cartItems.splice(index, 1)[0];
            showToast(`🗑️ "${removed.name || removed.key}" eliminado de la cesta`);
        } else {
            showToast('ℹ️ Este producto no está en tu cesta');
        }
        refreshCurrentScreen(screenId);
        return;
    }

    const removed = cartItems.pop();
    showToast(`🗑️ "${removed.name || removed.key}" eliminado de la cesta`);
    refreshCurrentScreen(screenId);
}

function favoriteCurrentItem() {
    const screenId = getActiveScreenId();
    if (!PANTALLAS_GESTO_FAVORITO.includes(screenId)) return;

    let product = null;

    if (screenId === 'screen-producto') {
        product = IKEA_DB[currentProductKey];
    } else if (screenId === 'screen-ar') {
        if (window._arResults && window._arResults.length > 0) {
            const topResult = window._arResults[0];
            const searchTerm = topResult.product_name || topResult.name || topResult.id || '';
            product = typeof findProduct === 'function' ? findProduct(searchTerm) : null;
        }
        if (!product) {
            showToast('ℹ️ No hay producto detectado por la IA');
            return;
        }
    } else if (screenId === 'screen-escaner') {
        product = window._qrProduct || null;
        if (!product) {
            showToast('ℹ️ No se ha escaneado ningún producto');
            return;
        }
    }

    if (!product) return;

    const exists = favItems.find(f => f.key === product.key);
    if (exists) {
        showToast('ℹ️ Ya está en favoritos');
        return;
    }

    favItems.push({
        id: Date.now(),
        key: product.key,
        name: product.nombre || product.key,
        price: product.price,
        location: product.location,
        peso: product.peso || product.weight || '-',
        emoji: product.emoji || '📦',
        image: product.image || '',
        inCart: cartItems.some(c => c.key === product.key)
    });

    showToast(`⭐ "${product.nombre || product.key}" guardado en Favoritos`);

    if (screenId === 'screen-producto') {
        const favBtn = document.getElementById('prod-fav-btn');
        if (favBtn) favBtn.textContent = '⭐';
    }

    if (typeof renderFavs === 'function') renderFavs();
}

class ShakeDetector {
    constructor(options) {
        this.threshold = options.threshold || 15;
        this.timeout = options.timeout || 1000;
        this.onShake = options.onShake;
        
        this.lastTime = Date.now();
        this.lastX = null;
        this.lastY = null;
        this.lastZ = null;
        this.lastShake = Date.now();
        
        this.handler = this.devicemotion.bind(this);
    }

    start() { window.addEventListener('devicemotion', this.handler, { passive: true }); }
    stop() { window.removeEventListener('devicemotion', this.handler); }

    devicemotion(e) {
        const current = e.accelerationIncludingGravity || e.acceleration;
        if (!current) return;
        
        const now = Date.now();
        // Comprobamos cada 100ms (filtra el ruido microscópico)
        if (now - this.lastTime > 100) { 
            const x = current.x, y = current.y, z = current.z;
            
            if (this.lastX === null) {
                this.lastX = x; this.lastY = y; this.lastZ = z; 
                return;
            }
            
            // Calculamos el latigazo en cada eje
            const deltaX = Math.abs(this.lastX - x);
            const deltaY = Math.abs(this.lastY - y);
            const deltaZ = Math.abs(this.lastZ - z);
            
            // Magia pura: Un agitado real mueve al menos DOS ejes a la vez. 
            // Un frenazo caminando o el giro del Favorito suele mover solo uno fuerte.
            if (((deltaX > this.threshold) && (deltaY > this.threshold)) || 
                ((deltaX > this.threshold) && (deltaZ > this.threshold)) || 
                ((deltaY > this.threshold) && (deltaZ > this.threshold))) {
                
                if (now - this.lastShake > this.timeout) {
                    if (typeof this.onShake === 'function') this.onShake();
                    this.lastShake = now;
                }
            }
            
            this.lastTime = now;
            this.lastX = x; this.lastY = y; this.lastZ = z;
        }
    }
}

// --- Lógica de Agitar ---
let miShakeEvent = null;

function initShakeLibrary() {
    if (miShakeEvent) return; 
    
    // Instanciamos nuestro propio motor. 12 es un buen equilibrio.
    miShakeEvent = new ShakeDetector({
        threshold: 12, 
        timeout: 1000, 
        onShake: undoLastCartItem // Llamamos directamente a tu función de borrar
    });
    
    miShakeEvent.start();
}

// --- Lógica de Flick (Giro rápido y suave a la derecha) ---
let tiltCooldownTime = 0;
let gammaHistory = []; 

function handleOrientation(event) {
    const screenId = getActiveScreenId();
    if (!PANTALLAS_GESTO_FAVORITO.includes(screenId)) return;

    const now = Date.now();
    if (now - tiltCooldownTime < 2000) return; // Cooldown post-favorito

    const gamma = event.gamma; // Rotación izquierda/derecha (-90 a 90)
    const beta = event.beta;   // Inclinación adelante/atrás

    if (gamma === null || beta === null) return;

    // Condición de seguridad: El usuario debe estar sosteniendo el móvil frente a él
    // Si está totalmente boca abajo, ignoramos.
    if (beta > 60) {
        gammaHistory = []; // Limpiamos historial si la postura no es natural
        return; 
    }

    // Mantenemos un historial de los últimos 250ms para analizar la fluidez del movimiento
    gammaHistory.push({ time: now, val: gamma });
    gammaHistory = gammaHistory.filter(h => now - h.time < 200);

    if (gammaHistory.length > 2) {
        const oldest = gammaHistory[0];
        const recentDelta = gamma - oldest.val; // Cuántos grados ha girado
        const timeDelta = now - oldest.time;    // En cuánto tiempo

        // Detectar un giro RÁPIDO (>35 grados en menos de 250ms) hacia la DERECHA
        // recentDelta > 35 asegura que es un movimiento intencionado
        // La división (recentDelta / timeDelta) asegura que haya velocidad (flick) y no sea un giro lento
        if (recentDelta > 35 && timeDelta > 50 && (recentDelta / timeDelta) > 0.18) {
            tiltCooldownTime = now;
            gammaHistory = []; 
            favoriteCurrentItem();
        }
    }
}

function requestSensorPermissions() {
    if (gestosActivos) {
        showToast('✅ Los sensores ya están activos');
        return;
    }

    if (location.protocol !== 'https:' && location.hostname !== 'localhost' && location.hostname !== '127.0.0.1') {
        showToast('⚠️ Los gestos requieren conexión HTTPS');
        return;
    }

    const activarSensores = () => {
      
        initShakeLibrary();
        window.addEventListener('deviceorientation', handleOrientation, { passive: true });
        gestosActivos = true;
        activarInterfazGestos();
        
    };

    // iOS 13+ requiere permiso por interacción del usuario
    if (typeof DeviceMotionEvent !== 'undefined' && typeof DeviceMotionEvent.requestPermission === 'function') {
        DeviceMotionEvent.requestPermission()
            .then(permissionState => {
                if (permissionState === 'granted') activarSensores();
                else showToast('⚠️ Permiso de sensores denegado');
            })
            .catch(console.error);
    } else {
        activarSensores();
    }
}

function activarInterfazGestos() {
    const panel = document.getElementById('panel-gestos');
    if (panel) {
        panel.innerHTML = `
            <div class="gestures-card fade-in-up s5" style="margin: 15px; min-height: 135px;">
              <div class="gestures-title" style="font-size:12px;font-weight:bold;margin-bottom:10px;color:#666;">GESTOS ACTIVOS</div>
              <div class="gesture-grid" style="display:grid;grid-template-columns:1fr 1fr;gap:8px;">
                <div class="gesture-item" style="background:rgba(82,32,125,0.1);border:1px solid rgba(82,32,125,0.3);padding:10px;border-radius:6px;display:flex;align-items:center;justify-content:center;gap:5px;">
                  <span>🎙️</span><span style="color:var(--purple);font-size:10px;">Voz → Buscar</span>
                </div>
                <div class="gesture-item" style="background:rgba(0,88,163,0.1);border:1px solid rgba(0,88,163,0.3);padding:10px;border-radius:6px;display:flex;align-items:center;justify-content:center;gap:5px;">
                  <span>📷</span><span style="color:var(--blue);font-size:10px;">Cámara → Añadir</span>
                </div>
                <div class="gesture-item" style="background:rgba(239,119,68,0.1);border:1px solid rgba(239,119,68,0.3);padding:10px;border-radius:6px;display:flex;align-items:center;justify-content:center;gap:5px;">
                  <span>↪️</span><span style="color:var(--orange);font-size:10px;">Giro Der → Fav</span>
                </div>
                <div class="gesture-item" style="background:rgba(239,68,68,0.1);border:1px solid rgba(239,68,68,0.3);padding:10px;border-radius:6px;display:flex;align-items:center;justify-content:center;gap:5px;">
                  <span>📳</span><span style="color:var(--red);font-size:10px;">Agitar → Borrar</span>
                </div>
              </div>
            </div>
        `;
    }
    showToast('✅ ¡Sensores activados!');
}

// Helper para inicializar el micro
function getSpeechRecognition() {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) {
        showToast('⚠️ Tu navegador no soporta dictado por voz');
        return null;
    }
    const recognition = new SpeechRecognition();
    recognition.lang = 'es-ES'; // Idioma español
    recognition.interimResults = false;
    return recognition;
}

function startVoiceSearch() {
    const recognition = getSpeechRecognition();
    if (!recognition) return;

    recognition.onstart = () => showToast('🎙️ Escuchando... Di lo que buscas');
    
    recognition.onresult = (event) => {
        // Pillamos lo que has dicho y le quitamos el punto final si lo pone
        let transcript = event.results[0][0].transcript;
        if (transcript.endsWith('.')) transcript = transcript.slice(0, -1);
        
        showToast(`✅ Buscando: "${transcript}"`);
        
        // 1. Te llevamos a la pantalla de búsqueda
        goTo('buscar');
        
        setTimeout(() => {
            // 2. Metemos el texto en la barra principal
            const searchInput = document.getElementById('search-input');
            if (searchInput) {
                searchInput.value = transcript;
                
                // 3. Forzamos a que salte la búsqueda (el oninput no salta solo por JS)
                if (typeof filterProducts === 'function') {
                    filterProducts(transcript);
                }
            }
        }, 100); // Mismo margen para asegurar que la pantalla ha cargado
    };
    
    recognition.start();
}

function startVoiceSort() {
    const recognition = getSpeechRecognition();
    if (!recognition) return;

    recognition.onstart = () => showToast('🎙️ Di: Precio, Peso, Ruta o Nombre...');
    
    recognition.onresult = (event) => {
        // Pasamos todo a minúsculas y quitamos espacios extra para que no falle
        const command = event.results[0][0].transcript.toLowerCase().trim();
        
        if (command.includes('precio') || command.includes('dinero') || command.includes('caro')) {
            showToast('🗣️ Ordenando por: Precio');
            sortCart(null, 'price');
        } 
        else if (command.includes('peso') || command.includes('gramos') || command.includes('kilos')) {
            showToast('🗣️ Ordenando por: Peso');
            sortCart(null, 'weight');
        } 
        else if (command.includes('ruta') || command.includes('pasillo') || command.includes('ubicación')) {
            showToast('🗣️ Ordenando por: Ruta');
            sortCart(null, 'route');
        } 
        else if (command.includes('a z') || command.includes('alfabético') || command.includes('nombre')) {
            showToast('🗣️ Ordenando por: Nombre');
            sortCart(null, 'az');
        } 
        else {
            showToast(`⚠️ Comando no reconocido: "${command}"`);
        }
    };
    
    recognition.start();
}

function openSearchAndFocus() {
    goTo('buscar');
    
    // Le damos 100ms de margen para que la pantalla termine de hacerse visible
    setTimeout(() => {
        const searchInput = document.getElementById('search-input');
        if (searchInput) {
            searchInput.focus();
        }
    }, 100);
}

</script>
</body>
</html>
"""

HTML_MODEL = """
<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>IKEA Scanner</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }

    body {
      font-family: 'Segoe UI', sans-serif;
      background: #f5f5f0;
      min-height: 100vh;
      display: flex;
      flex-direction: column;
      align-items: center;
    }

    header {
      width: 100%;
      background: #0058a3;
      color: white;
      padding: 18px 24px;
      display: flex;
      align-items: center;
      gap: 14px;
    }
    header .logo {
      font-size: 28px;
      font-weight: 900;
      background: #ffdb00;
      color: #0058a3;
      padding: 4px 12px;
      border-radius: 4px;
    }
    header h1 { font-size: 20px; font-weight: 600; }

    main {
      width: 100%;
      max-width: 640px;
      padding: 32px 16px;
      display: flex;
      flex-direction: column;
      gap: 20px;
    }

    /* ── Selector de modo ── */
    .mode-tabs {
      display: flex;
      background: white;
      border-radius: 12px;
      padding: 4px;
      gap: 4px;
      box-shadow: 0 2px 8px rgba(0,0,0,0.07);
    }
    .mode-tab {
      flex: 1;
      padding: 11px;
      border: none;
      border-radius: 9px;
      font-size: 15px;
      font-weight: 600;
      cursor: pointer;
      background: transparent;
      color: #666;
      transition: all 0.2s;
    }
    .mode-tab.active {
      background: #0058a3;
      color: white;
    }

    /* ── Subir archivo ── */
    .upload-area {
      display: block;
      background: white;
      border: 2px dashed #0058a3;
      border-radius: 16px;
      padding: 40px 24px;
      text-align: center;
      cursor: pointer;
      transition: background 0.2s;
    }
    .upload-area:hover { background: #e8f0fb; }
    .upload-area input { display: none; }
    .upload-area .icon { font-size: 48px; margin-bottom: 12px; }
    .upload-area p { color: #555; font-size: 15px; }
    .upload-area strong { color: #0058a3; }

    #preview-wrap {
      display: none;
      background: white;
      border-radius: 16px;
      overflow: hidden;
      box-shadow: 0 2px 12px rgba(0,0,0,0.08);
    }
    #preview-wrap img {
      width: 100%;
      max-height: 320px;
      object-fit: contain;
      background: #fafafa;
    }

    button#scan-btn {
      background: #0058a3;
      color: white;
      border: none;
      border-radius: 12px;
      padding: 16px;
      font-size: 17px;
      font-weight: 700;
      cursor: pointer;
      width: 100%;
      transition: background 0.2s;
      display: none;
    }
    button#scan-btn:hover { background: #004a8c; }
    button#scan-btn:disabled { background: #aaa; cursor: not-allowed; }

    /* ── Cámara ── */
    #camera-section { display: none; flex-direction: column; gap: 12px; }

    #camera-wrap {
      background: black;
      border-radius: 16px;
      overflow: hidden;
      box-shadow: 0 2px 12px rgba(0,0,0,0.15);
      position: relative;
    }
    #camera-video {
      width: 100%;
      max-height: 340px;
      object-fit: cover;
      display: block;
    }
    /* Marco guía centrado */
    #camera-guide {
      position: absolute;
      inset: 0;
      display: flex;
      align-items: center;
      justify-content: center;
      pointer-events: none;
    }
    #camera-guide .frame {
      width: 58%;
      aspect-ratio: 1;
      border: 2px dashed rgba(255,219,0,0.85);
      border-radius: 10px;
      box-shadow: 0 0 0 9999px rgba(0,0,0,0.25);
    }
    #cam-hint {
      text-align: center;
      font-size: 13px;
      color: #888;
    }
    #cam-controls {
      display: flex;
      gap: 10px;
    }
    #capture-btn {
      flex: 1;
      background: #ffdb00;
      color: #333;
      border: none;
      border-radius: 12px;
      padding: 15px;
      font-size: 16px;
      font-weight: 700;
      cursor: pointer;
      transition: background 0.2s;
    }
    #capture-btn:hover { background: #f0cc00; }
    #capture-btn:disabled { background: #ddd; cursor: not-allowed; }
    #flip-btn {
      background: white;
      border: 2px solid #ddd;
      border-radius: 12px;
      padding: 15px 18px;
      font-size: 20px;
      cursor: pointer;
      transition: background 0.2s;
      line-height: 1;
    }
    #flip-btn:hover { background: #f0f0f0; }

    /* Canvas oculto para capturar el fotograma */
    canvas#snap { display: none; }

    /* ── Foto capturada (previsualización antes de enviar) ── */
    #snap-preview-wrap {
      display: none;
      background: white;
      border-radius: 16px;
      overflow: hidden;
      box-shadow: 0 2px 12px rgba(0,0,0,0.08);
    }
    #snap-preview-wrap img {
      width: 100%;
      max-height: 320px;
      object-fit: contain;
      background: #111;
    }

    /* ── Resultados ── */
    #result {
      display: none;
      flex-direction: column;
      gap: 12px;
    }

    .result-title {
      font-size: 13px;
      font-weight: 700;
      color: #888;
      text-transform: uppercase;
      letter-spacing: 1px;
    }

    .product-card {
      background: white;
      border-radius: 14px;
      padding: 18px 20px;
      box-shadow: 0 2px 10px rgba(0,0,0,0.07);
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 12px;
    }
    .product-card.best { border-left: 5px solid #0058a3; }
    .product-card .info .name {
      font-size: 22px;
      font-weight: 800;
      color: #111;
    }
    .product-card .info .cat {
      font-size: 13px;
      color: #888;
      margin-top: 2px;
    }
    .product-card .badge {
      background: #ffdb00;
      color: #333;
      font-weight: 700;
      font-size: 15px;
      padding: 6px 14px;
      border-radius: 20px;
      white-space: nowrap;
    }
    .product-card.alt .info .name { font-size: 17px; color: #333; }
    .product-card.alt .badge { background: #eee; }

    .unsure-card {
      background: #fff8e1;
      border-left: 5px solid #ffdb00;
      border-radius: 14px;
      padding: 18px 20px;
      font-size: 15px;
      color: #555;
    }
    .unsure-card strong { color: #333; }

    .error-card {
      background: #fff0f0;
      border-left: 5px solid #e53935;
      border-radius: 14px;
      padding: 18px 20px;
      font-size: 15px;
      color: #c62828;
    }

    .loader {
      display: none;
      text-align: center;
      padding: 24px;
      color: #0058a3;
      font-weight: 600;
      font-size: 16px;
    }
    .loader .spinner {
      width: 40px; height: 40px;
      border: 4px solid #d0e4f7;
      border-top-color: #0058a3;
      border-radius: 50%;
      animation: spin 0.8s linear infinite;
      margin: 0 auto 12px;
    }
    /* ── Modo archivo ── */
    #file-section {
      display: flex;
      flex-direction: column;
      gap: 16px;        /* ← Espacio entre upload-area, preview y botón */
    }

    /* ── Modo cámara ── */
    #camera-section {
      display: none;
      flex-direction: column;
      gap: 16px;        /* ← Ya lo tenía en 12px, lo subimos */
    }

    /* ── Resultados ── */
    #result {
      display: none;
      flex-direction: column;
      gap: 16px;        /* ← De 12px a 16px, más aire entre cards */
    }

    /* ── Cards de producto ── */
    .product-card {
      background: white;
      border-radius: 14px;
      padding: 20px 22px;   /* ← Un poco más de padding interno */
      box-shadow: 0 2px 10px rgba(0,0,0,0.07);
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 12px;
    }

    /* ── Loader ── */
    .loader {
      display: none;
      text-align: center;
      padding: 32px 24px;   /* ← Más padding arriba y abajo */
      color: #0058a3;
      font-weight: 600;
      font-size: 16px;
    }
    
    @keyframes spin { to { transform: rotate(360deg); } }
  </style>
</head>
<body>

<header>
  <div class="logo">IKEA</div>
  <h1>Escáner de productos</h1>
</header>

<main>

  <!-- ── Selector de modo ── -->
  <div class="mode-tabs">
    <button class="mode-tab active" id="tab-file" onclick="switchMode('file')">📁 Subir foto</button>
    <button class="mode-tab"        id="tab-cam"  onclick="switchMode('camera')">📷 Usar cámara</button>
  </div>

  <!-- ════════════════ MODO ARCHIVO ════════════════ -->
  <div id="file-section">
    <label class="upload-area" id="drop-area">
      <input type="file" id="file-input" accept="image/*"/>
      <div class="icon">🖼️</div>
      <p><strong>Haz clic o arrastra una foto</strong><br/>de un mueble IKEA</p>
    </label>

    <div id="preview-wrap">
      <img id="preview" src="" alt="preview"/>
    </div>

    <button id="scan-btn" onclick="scanFile()">🔍 Identificar producto</button>
  </div>

  <!-- ════════════════ MODO CÁMARA ════════════════ -->
  <div id="camera-section">

    <div id="camera-wrap">
      <video id="camera-video" autoplay playsinline muted></video>
      <div id="camera-guide"><div class="frame"></div></div>
    </div>

    <p id="cam-hint">Centra el mueble dentro del recuadro y pulsa <strong>Capturar</strong></p>

    <div id="cam-controls">
      <button id="capture-btn" onclick="captureFrame()">📸 Capturar y analizar</button>
      <button id="flip-btn"    onclick="flipCamera()"   title="Cambiar cámara">🔄</button>
    </div>

    <!-- Previsualización del fotograma capturado -->
    <div id="snap-preview-wrap">
      <img id="snap-preview" src="" alt="Captura"/>
    </div>

    <!-- Canvas oculto para captura -->
    <canvas id="snap"></canvas>
  </div>

  <!-- ════════════════ COMPARTIDOS ════════════════ -->
  <div class="loader" id="loader">
    <div class="spinner"></div>
    Analizando imagen…
  </div>

  <div id="result"></div>

</main>

<script>
  /* ────────────────── Estado global ────────────────── */
  let selectedFile  = null;
  let currentMode   = 'file';
  let stream        = null;
  let facingMode    = 'environment';   // 'user' = frontal, 'environment' = trasera

  /* ────────────────── Referencias DOM ────────────────── */
  const fileInput      = document.getElementById('file-input');
  const previewImg     = document.getElementById('preview');
  const previewWrap    = document.getElementById('preview-wrap');
  const scanBtn        = document.getElementById('scan-btn');
  const loader         = document.getElementById('loader');
  const resultDiv      = document.getElementById('result');
  const video          = document.getElementById('camera-video');
  const snapCanvas     = document.getElementById('snap');
  const snapPreview    = document.getElementById('snap-preview');
  const snapPreviewWrap= document.getElementById('snap-preview-wrap');
  const captureBtn     = document.getElementById('capture-btn');

  /* ════════════════ CAMBIO DE MODO ════════════════ */
  async function switchMode(mode) {
    currentMode = mode;

    document.getElementById('tab-file').classList.toggle('active', mode === 'file');
    document.getElementById('tab-cam').classList.toggle('active',  mode === 'camera');
    document.getElementById('file-section').style.display   = mode === 'file'   ? 'flex'  : 'none';
    document.getElementById('camera-section').style.display = mode === 'camera' ? 'flex'  : 'none';

    // Ajustar layout de file-section a columna
    document.getElementById('file-section').style.flexDirection = 'column';
    document.getElementById('file-section').style.gap = '16px';

    clearResult();

    if (mode === 'camera') {
      await startCamera();
    } else {
      stopCamera();
      snapPreviewWrap.style.display = 'none';
    }
  }

  /* ════════════════ MODO ARCHIVO ════════════════ */
  function loadFile(file) {
    if (!file || !file.type.startsWith('image/')) return;
    selectedFile = file;
    previewImg.src = URL.createObjectURL(file);
    previewWrap.style.display = 'block';
    scanBtn.style.display = 'block';
    clearResult();
  }

  fileInput.addEventListener('change', e => loadFile(e.target.files[0]));

  const dropArea = document.getElementById('drop-area');
  dropArea.addEventListener('dragover', e => {
    e.preventDefault();
    dropArea.style.background = '#e8f0fb';
    dropArea.style.borderStyle = 'solid';
  });
  dropArea.addEventListener('dragleave', () => {
    dropArea.style.background = '';
    dropArea.style.borderStyle = 'dashed';
  });
  dropArea.addEventListener('drop', e => {
    e.preventDefault();
    dropArea.style.background = '';
    dropArea.style.borderStyle = 'dashed';
    loadFile(e.dataTransfer.files[0]);
  });

  async function scanFile() {
    if (!selectedFile) return;
    await sendToAPI(selectedFile);
  }

  /* ════════════════ MODO CÁMARA ════════════════ */
  async function startCamera() {
    stopCamera();
    captureBtn.disabled = true;
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode, width: { ideal: 1280 }, height: { ideal: 720 } },
        audio: false
      });
      video.srcObject = stream;
      await video.play();
      captureBtn.disabled = false;
    } catch (err) {
      showError('No se pudo acceder a la cámara. Asegúrate de dar permiso en el navegador.<br/><small>' + err.message + '</small>');
      captureBtn.disabled = false;
    }
  }

  function stopCamera() {
    if (stream) {
      stream.getTracks().forEach(t => t.stop());
      stream = null;
    }
    video.srcObject = null;
  }

  async function flipCamera() {
    facingMode = facingMode === 'environment' ? 'user' : 'environment';
    await startCamera();
  }

  function captureFrame() {
    if (!stream) return;

    // Dibuja el fotograma actual en el canvas
    snapCanvas.width  = video.videoWidth  || 640;
    snapCanvas.height = video.videoHeight || 480;
    const ctx = snapCanvas.getContext('2d');
    ctx.drawImage(video, 0, 0, snapCanvas.width, snapCanvas.height);

    // Muestra previsualización
    const dataUrl = snapCanvas.toDataURL('image/jpeg', 0.92);
    snapPreview.src = dataUrl;
    snapPreviewWrap.style.display = 'block';

    // Convierte a Blob y envía
    snapCanvas.toBlob(async blob => {
      const file = new File([blob], 'captura.jpg', { type: 'image/jpeg' });
      await sendToAPI(file);
    }, 'image/jpeg', 0.92);
  }

  /* ════════════════ API CALL ════════════════ */
  async function sendToAPI(file) {
    setLoading(true);

    const form = new FormData();
    form.append('file', file);

    try {
      const res  = await fetch('/identify', { method: 'POST', body: form });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      renderResult(data);
    } catch (e) {
      showError('Error al conectar con la API. Inténtalo de nuevo.<br/><small>' + e.message + '</small>');
    } finally {
      setLoading(false);
    }
  }

  /* ════════════════ HELPERS ════════════════ */
  function setLoading(on) {
    loader.style.display = on ? 'block' : 'none';
    if (currentMode === 'file') scanBtn.disabled = on;
    if (currentMode === 'camera') captureBtn.disabled = on;
    if (on) clearResult();
  }

  function clearResult() {
    resultDiv.style.display = 'none';
    resultDiv.innerHTML = '';
  }

  function showError(msg) {
    resultDiv.style.display = 'flex';
    resultDiv.innerHTML = `<div class="error-card">⚠️ ${msg}</div>`;
  }

  function renderResult(data) {
    resultDiv.style.display = 'flex';
    const best = data.best_match;
    const conf = best.confidence;

    if (conf < 0.60) {
      resultDiv.innerHTML = `
        <div class="unsure-card">
          <strong>No estoy seguro 🤔</strong><br/>
          No he podido identificar el producto con suficiente confianza.<br/>
          Intenta acercar más la cámara al mueble o mejorar la iluminación.
        </div>`;
      return;
    }

    let html = `<div class="result-title">Mejor coincidencia</div>`;
    html += `
      <div class="product-card best">
        <div class="info">
          <div class="name">${best.nombre}</div>
          <div class="cat">${best.subtitulo}</div>
        </div>
        <div class="badge">${best.confidence_pct}</div>
      </div>`;

    if (data.alternatives && data.alternatives.length > 0) {
      html += `<div class="result-title" style="margin-top:4px">Otras posibilidades</div>`;
      data.alternatives.forEach(alt => {
        html += `
          <div class="product-card alt">
            <div class="info">
              <div class="name">${alt.nombre}</div>
              <div class="cat">${alt.subtitulo}</div>
            </div>
            <div class="badge">${alt.confidence_pct}</div>
          </div>`;
      });
    }

    resultDiv.innerHTML = html;
    // Hacer scroll suave hacia los resultados
    resultDiv.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }

  /* ── Inicialización: mostrar sección archivo por defecto ── */
  document.getElementById('camera-section').style.display = 'none';
</script>

</body>
</html>
"""

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
def ui():
    return HTMLResponse(content=HTML_APP)

@app.get("/model", response_class=HTMLResponse)
def ui_secundaria():
    return HTMLResponse(content=HTML_MODEL)

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
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
