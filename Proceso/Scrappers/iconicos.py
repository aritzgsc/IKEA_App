"""
filter_iconic.py — Copia solo los productos icónicos de IKEA a una nueva carpeta.
No toca el dataset original.
"""

import shutil
from pathlib import Path

# ─────────────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────────────
ORIGEN  = "./DataSet_IKEA_Definitivo"    # Tu carpeta reorganizada (sin variantes)
DESTINO = "./DataSet_Iconicos"  # Nueva carpeta que se creará
# ─────────────────────────────────────────────────────

# Lista curada de los productos IKEA más icónicos y visualmente distintos.
# Están elegidos por ser reconocibles, populares y visualmente únicos.
# Puedes añadir o quitar nombres según lo que tengas en tu dataset.
PRODUCTOS_ICONICOS = {
    # Estanterías
    "BILLY", "KALLAX", "HEMNES", "EXPEDIT", "BESTA",
    "LACK", "EKET", "IVAR", "TROFAST", "LIXHULT",

    # Sillas
    "POÄNG", "MARKUS", "FLINTAN", "TEODORES", "STEFAN",
    "JANINGE", "EKEDALEN", "INGOLF", "GUNDE", "ADDE",

    # Mesas
    "LACK", "LINNMON", "BEKANT", "MICKE", "ALEX",
    "LISABO", "MOCKELBY", "INGATORP", "BJURSTA", "GLADOM",

    # Camas
    "MALM", "HEMNES", "BRIMNES", "TARVA", "FJELLSE",
    "NEIDEN", "SNIGLAR", "KURA", "VITVAL", "MYDAL",

    # Sofás
    "KLIPPAN", "KIVIK", "EKTORP", "KARLSTAD", "FRIHETEN",
    "VALLENTUNA", "UPPLAND", "LANDSKRONA", "LIDHULT", "VIMLE",

    # Armarios
    "PAX", "BRIMNES", "STUVA", "PLATSA", "DOMBAS",
    "KLEPPSTAD", "RAKKE", "BREIM", "HAUGA", "HEMNES",

    # Iluminación
    "HEKTAR", "RANARP", "FORSA", "ARSTID", "SYMFONISK",
    "TRADFRI", "VARV", "SKURUP", "TERTIAL", "NAVLINGE",

    # Cocina
    "KALLAX", "RASKOG", "BEKVAM", "SUNNERSTA", "BYGEL",
    "GRUNDTAL", "KUNGSFORS", "UTBY", "KARLBY", "VADHOLMA",

    # Baño
    "GODMORGON", "HEMNES", "LILLANGEN", "ENHET", "SILVERAN",
    "MOLGER", "RAGRUND", "FRACK", "KALKGRUND", "SKAREN",

    # Escritorio/Oficina
    "MICKE", "ALEX", "BEKANT", "FREDDE", "SKARSTA",
    "TROTTEN", "IDANSEN", "LAGKAPTEN", "MITTZON", "ELLOVEN",

    # Infantil
    "KURA", "STUVA", "TROFAST", "SNIGLAR", "SUNDVIK",
    "MINNEN", "MYLLRA", "FLISAT", "SMASTAD", "NATTJASMIN",

    # Exterior
    "APPLARO", "TARNO", "HUSARO", "BONDHOLMEN", "FALSTER",
    "SOLLERÖN", "SJÄLLAND", "KUDDARNA", "HÅLLÖ", "ASKHOLMEN",

    # Textil/Alfombras
    "PERSISK", "STOENSE", "LOHALS", "TIPHEDE", "KOLDBY",
    "ALVINE", "VISTOFT", "SPORUP", "LANGSTED", "MORRUM",

    # Almacenaje
    "RASKOG", "VESKEN", "SKUBB", "KUGGIS", "SAMLA",
    "HYVENS", "MOPPE", "REJSA", "DRONA", "TJENA",
}

# ─────────────────────────────────────────────────────

origen  = Path(ORIGEN)
destino = Path(DESTINO)

if destino.exists():
    print(f"⚠️  '{DESTINO}' ya existe. Bórrala primero si quieres regenerarla.")
    exit(1)

total_copiados  = 0
total_ignorados = 0
encontrados     = set()

for category_dir in sorted(origen.iterdir()):
    if not category_dir.is_dir():
        continue

    for product_dir in sorted(category_dir.iterdir()):
        if not product_dir.is_dir():
            continue

        # Comparación case-insensitive
        if product_dir.name.upper() not in {p.upper() for p in PRODUCTOS_ICONICOS}:
            total_ignorados += 1
            continue

        dest_product = destino / category_dir.name / product_dir.name
        dest_product.mkdir(parents=True, exist_ok=True)

        imgs = [f for f in product_dir.iterdir()
                if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}]

        for img in imgs:
            shutil.copy2(img, dest_product / img.name)

        encontrados.add(product_dir.name)
        total_copiados += len(imgs)
        print(f"  ✅ {category_dir.name} / {product_dir.name} — {len(imgs)} imágenes")

print(f"\n✅ Filtrado completado:")
print(f"   {len(encontrados)} productos icónicos copiados")
print(f"   {total_copiados} imágenes en total")
print(f"   {total_ignorados} productos descartados")
print(f"\n⚠️  Productos de la lista NO encontrados en tu dataset:")
no_encontrados = {p for p in PRODUCTOS_ICONICOS if p.upper() not in {e.upper() for e in encontrados}}
for p in sorted(no_encontrados):
    print(f"   - {p}")
print(f"\n➡️  Ahora edita indexer.py y cambia PRODUCTS_DIR a '{DESTINO}'")