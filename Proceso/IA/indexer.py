"""
indexer_openclip.py — Versión OpenCLIP ViT-B/32 del indexador.
Reutiliza TODO el sistema de checkpoints y DataLoader existente.
Tiempo estimado en CPU: 28-32h para 350k imágenes.
"""

import os
import pickle
import time
import numpy as np
from PIL import Image, UnidentifiedImageError
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import open_clip 
from typing import cast
from torchvision.transforms import Compose


# ─────────────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────────────
PRODUCTS_DIR    = "./Dataset_IKEA_Definitivo"
INDEX_FILE      = "./ikea_index_openclip.pkl"
CHECKPOINT_FILE = "./ikea_index_openclip_checkpoint.pkl"
SAVE_EVERY      = 2500
BATCH_SIZE      = 32      # Reducido: ViT-B/32 usa más RAM por imagen que DINOv2-small
NUM_WORKERS     = 4
# ─────────────────────────────────────────────────────

EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
PIN_MEMORY = False

print(f"✅ Dispositivo: {DEVICE.upper()}")
print("⏳ Cargando OpenCLIP ViT-B/32...")

# ── ÚNICO CAMBIO RESPECTO A TU CÓDIGO ANTERIOR ──────

if DEVICE == "cpu":
    torch.set_num_threads(os.cpu_count() or 8)

result = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
clip_model      = cast(open_clip.CLIP, result[0])
clip_preprocess = cast(Compose, result[2])

clip_model = clip_model.to(DEVICE)
clip_model.eval()

# Desactivamos el gradient tracking a nivel de modelo (más eficiente que solo inference_mode)
for param in clip_model.parameters():
    param.requires_grad = False

if DEVICE == "cpu":
    torch.set_num_threads(os.cpu_count() or 8)  # Usar todos los cores disponibles
    print(f"   Usando {os.cpu_count()} threads de CPU")

print("✅ OpenCLIP ViT-B/32 cargado (dim=512)")
# ────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────
# Dataset: usa clip_preprocess en lugar del processor de HuggingFace
# Esta es la otra diferencia clave — OpenCLIP tiene su propio preprocesador
# ─────────────────────────────────────────────────────
class ProductDataset(Dataset):
    def __init__(self, items: list[tuple[str, str]]):
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        path, label = self.items[idx]
        try:
            img = Image.open(path).convert("RGB")
            # clip_preprocess ya hace resize, center_crop y normalización
            tensor = clip_preprocess(img)
            return tensor, label, True, path
        except Exception:
            dummy = torch.zeros(3, 224, 224)
            return dummy, label, False, path


def collate_fn(batch):
    valid         = [(pv, lbl, p) for pv, lbl, ok, p in batch if ok]
    invalid_paths = [p for _, _, ok, p in batch if not ok]
    
    if not valid:
        return None, [], [], invalid_paths
        
    pixel_values = torch.stack([pv for pv, _, _ in valid])
    labels       = [lbl for _, lbl, _ in valid]
    valid_paths  = [p for _, _, p in valid]
    
    return pixel_values, labels, valid_paths, invalid_paths


# ─────────────────────────────────────────────────────
# Las funciones de escaneo, guardado y embeddings son
# IDÉNTICAS a tu código — solo cambia la inferencia
# ─────────────────────────────────────────────────────
def get_image_paths(products_dir: str) -> list[tuple[str, str]]:
    items = []
    base  = Path(products_dir)
    for category_dir in sorted(base.iterdir()):
        if not category_dir.is_dir(): continue
        for product_dir in sorted(category_dir.iterdir()):
            if not product_dir.is_dir(): continue
            label = f"{category_dir.name} | {product_dir.name}"
            for img_path in product_dir.iterdir():
                if img_path.suffix.lower() in EXTENSIONS:
                    items.append((str(img_path), label))
    return items


def save_checkpoint(data_dict, filepath):
    temp_path = filepath + ".tmp"
    with open(temp_path, "wb") as f:
        pickle.dump(data_dict, f)
    os.replace(temp_path, filepath)


def compute_embeddings(items: list[tuple[str, str]]):
    checkpoint_data = {
        "embeddings": [],
        "labels": [],
        "processed_paths": set(),
        "corruptas": 0
    }
    
    if os.path.exists(CHECKPOINT_FILE):
        print("📂 Checkpoint detectado. Reanudando...")
        with open(CHECKPOINT_FILE, "rb") as f:
            checkpoint_data = pickle.load(f)
        print(f"   {len(checkpoint_data['processed_paths'])} imágenes ya procesadas.")

    items_pendientes = [item for item in items if item[0] not in checkpoint_data["processed_paths"]]
    
    if not items_pendientes:
        print("✅ Todas las imágenes ya estaban procesadas.")
        return checkpoint_data

    # Dataset sin processor de HuggingFace
    dataset    = ProductDataset(items_pendientes)
    dataloader = DataLoader(
        dataset,
        batch_size         = BATCH_SIZE,
        num_workers        = NUM_WORKERS,
        pin_memory         = PIN_MEMORY,
        collate_fn         = collate_fn,
        persistent_workers = NUM_WORKERS > 0,
    )

    all_embeddings  = checkpoint_data["embeddings"]
    all_labels      = checkpoint_data["labels"]
    processed_paths = checkpoint_data["processed_paths"]
    corruptas       = checkpoint_data["corruptas"]
    last_save_count = len(processed_paths)

    # Estimación de tiempo realista
    total_pendientes = len(items_pendientes)
    print(f"\n⏱️  Estimación: ~{total_pendientes / 3000:.1f}h en CPU (≈3000 imgs/h para ViT-B/32)")
    print("💡 Consejo: Ejecuta con `nohup python indexer_openclip.py &` para que no pare al cerrar el terminal\n")

    print("🚀 Iniciando procesamiento...")
    t_batch_start = time.time()
    
    with tqdm(total=total_pendientes, desc="Indexando con OpenCLIP", unit="img") as pbar:
        for pixel_values, labels, valid_paths, invalid_paths in dataloader:
            
            corruptas += len(invalid_paths)
            processed_paths.update(invalid_paths)
            pbar.update(len(invalid_paths))

            if pixel_values is not None:
                pixel_values = pixel_values.to(DEVICE)

                with torch.inference_mode():
                    # encode_image es la función nativa de OpenCLIP
                    # Devuelve embeddings de 512 dimensiones (vs 384 de DINOv2-small)
                    emb = clip_model.encode_image(pixel_values)
                    emb = emb / emb.norm(dim=-1, keepdim=True).clamp(min=1e-8)

                all_embeddings.append(emb.cpu().numpy().astype(np.float32))
                all_labels.extend(labels)
                processed_paths.update(valid_paths)
                pbar.update(len(valid_paths))
                
                # Actualización de velocidad real en la barra
                elapsed = time.time() - t_batch_start
                speed = len(processed_paths) / elapsed * 3600
                pbar.set_postfix({"imgs/h": f"{speed:.0f}"})

            if len(processed_paths) - last_save_count >= SAVE_EVERY:
                save_checkpoint({
                    "embeddings": all_embeddings,
                    "labels": all_labels,
                    "processed_paths": processed_paths,
                    "corruptas": corruptas
                }, CHECKPOINT_FILE)
                last_save_count = len(processed_paths)
                tqdm.write(f"   💾 Checkpoint guardado ({len(processed_paths)} imgs)")

    save_checkpoint({
        "embeddings": all_embeddings,
        "labels": all_labels,
        "processed_paths": processed_paths,
        "corruptas": corruptas
    }, CHECKPOINT_FILE)

    return {
        "embeddings": all_embeddings,
        "labels": all_labels,
        "processed_paths": processed_paths,
        "corruptas": corruptas
    }


if __name__ == "__main__":
    print(f"\n📂 Escaneando: {PRODUCTS_DIR}")
    items = get_image_paths(PRODUCTS_DIR)

    if not items:
        print("❌ No se encontraron imágenes.")
        exit(1)

    print(f"✅ {len(items)} imágenes encontradas.\n")

    t_start = time.time()
    result  = compute_embeddings(items)
    t_total = time.time() - t_start

    embeddings_finales = np.vstack(result["embeddings"]) if result["embeddings"] else np.array([])
    labels_finales     = result["labels"]

    print(f"\n💾 Guardando índice final → {INDEX_FILE}")
    with open(INDEX_FILE, "wb") as f:
        pickle.dump({
            "embeddings": embeddings_finales,
            "labels":     labels_finales,
            "modelo":     "openclip-ViT-B-32-laion2b",  # Metadata para saber qué generó este índice
            "dim":        512
        }, f)

    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)

    print(f"\n🎉 Completado en {t_total/3600:.1f} horas")
    print(f"   {len(labels_finales)} imágenes | {len(set(labels_finales))} productos | {result['corruptas']} corruptas")
    print(f"   Tamaño del índice: {os.path.getsize(INDEX_FILE) / 1024 / 1024:.1f} MB")