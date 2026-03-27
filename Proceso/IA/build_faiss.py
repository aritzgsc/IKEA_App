"""
build_faiss.py — Construye el índice FAISS a partir del .pkl de OpenCLIP.
Ejecutar UNA vez después de que termine indexer_openclip.py
"""

import pickle
import numpy as np
import faiss  # pip install faiss-cpu
from pathlib import Path

PKL_FILE   = "./ikea_index_openclip.pkl"
FAISS_FILE = "./ikea_faiss.index"
LABELS_FILE = "./ikea_faiss_labels.pkl"

print("📦 Cargando índice pkl...")
with open(PKL_FILE, "rb") as f:
    data = pickle.load(f)

embeddings = np.array(data["embeddings"], dtype=np.float32)
labels     = data["labels"]

print(f"✅ {len(labels)} vectores cargados | dim={embeddings.shape[1]}")

# Verificamos que estén normalizados (deben estarlo, pero por si acaso)
norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
embeddings = embeddings / np.clip(norms, 1e-8, None)

# IndexFlatIP = Inner Product sobre vectores normalizados = Similitud Coseno exacta
# No es aproximado, busca entre TODOS los vectores. Para 350k es perfectamente rápido.
print("🔨 Construyendo índice FAISS...")
dimension = embeddings.shape[1]  # 512 para ViT-B/32
index = faiss.IndexFlatIP(dimension)
index.add(embeddings) # type: ignore[arg-type]

print(f"✅ Índice construido: {index.ntotal} vectores")

# Guardamos el índice FAISS y las etiquetas por separado
faiss.write_index(index, FAISS_FILE)
with open(LABELS_FILE, "wb") as f:
    pickle.dump(labels, f)

print(f"💾 Guardado:")
print(f"   {FAISS_FILE}  ({Path(FAISS_FILE).stat().st_size / 1024 / 1024:.1f} MB)")
print(f"   {LABELS_FILE}")
print("🎉 Listo. Ya puedes lanzar la aplicación.")