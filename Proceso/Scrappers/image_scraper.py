"""
╔══════════════════════════════════════════════════════════════════╗
║          IKEA SCRAPER — Basado en Sitemaps                       ║
║                                                                  ║
║  En lugar de probar IDs a ciegas, descarga los sitemaps          ║
║  oficiales de IKEA que listan TODAS las URLs de productos.       ║
║                                                                  ║
║  Sitemaps: ikea.com/sitemaps/prod-es-ES_1.xml  (y _2, _3...)     ║
║                                                                  ║
║  Instalación:                                                    ║
║    pip install aiohttp beautifulsoup4 lxml                       ║
╚══════════════════════════════════════════════════════════════════╝
"""

import re
import json
import time
import asyncio
import hashlib
import requests
from pathlib import Path
from bs4 import BeautifulSoup
import aiohttp

# ══════════════════════════════════════════════════════════════════
# ⚙️  CONFIGURACIÓN
# ══════════════════════════════════════════════════════════════════

CARPETA_SALIDA   = "Dataset_IKEA"
MAX_IMAGENES     = 15
CONCURRENCIA     = 80
TIMEOUT_GET      = 15
FICHERO_PROGRESO = "progreso_ikea.json"
SITEMAP_BASE     = "https://www.ikea.com/sitemaps/prod-es-ES_{n}.xml"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "es-ES,es;q=0.9",
}


# ══════════════════════════════════════════════════════════════════
# 📋  PASO 1: OBTENER TODAS LAS URLs DEL SITEMAP
# ══════════════════════════════════════════════════════════════════

def obtener_urls_sitemap():
    todas_urls = []
    n = 1
    print("📋 Descargando sitemaps de IKEA España...")

    while True:
        url_sitemap = SITEMAP_BASE.format(n=n)
        try:
            r = requests.get(url_sitemap, headers=HEADERS, timeout=20)
            if r.status_code in (404, 403):
                print(f"   Sitemap {n}: no existe — fin")
                break
            if r.status_code != 200:
                print(f"   Sitemap {n}: HTTP {r.status_code} — skip")
                n += 1
                continue

            soup = BeautifulSoup(r.text, "lxml-xml")
            urls_prod = [
                loc.get_text(strip=True)
                for loc in soup.find_all("loc")
                if "/es/es/p/" in (loc.get_text(strip=True) or "")
            ]
            print(f"   Sitemap {n}: {len(urls_prod)} productos")
            todas_urls.extend(urls_prod)
            n += 1

        except Exception as e:
            print(f"   Sitemap {n}: error ({e}) — fin")
            break

    print(f"\n✅ Total URLs en sitemaps: {len(todas_urls)}\n")
    return todas_urls


# ══════════════════════════════════════════════════════════════════
# 💾  PROGRESO
# ══════════════════════════════════════════════════════════════════

def cargar_progreso():
    if Path(FICHERO_PROGRESO).exists():
        with open(FICHERO_PROGRESO, "r", encoding="utf-8") as f:
            data = json.load(f)
        urls_hechas = set(data.get("urls_procesadas", []))
        total = data.get("total_descargados", 0)
        print(f"📂 Reanudando — {len(urls_hechas)} URLs ya procesadas ({total} productos)")
        return urls_hechas, total
    return set(), 0


def guardar_progreso(urls_procesadas, total_prods, total_imgs):
    with open(FICHERO_PROGRESO, "w", encoding="utf-8") as f:
        json.dump({
            "total_descargados": total_prods,
            "total_imagenes":    total_imgs,
            "urls_procesadas":   sorted(urls_procesadas),
        }, f)


# ══════════════════════════════════════════════════════════════════
# 🌐  PROCESAR CADA URL (async)
# ══════════════════════════════════════════════════════════════════

async def procesar_url(session, semaforo, url):
    async with semaforo:
        try:
            async with session.get(
                url,
                timeout=aiohttp.ClientTimeout(total=TIMEOUT_GET),
            ) as resp:
                if resp.status != 200:
                    return False, url, 0
                html = await resp.text()
        except Exception:
            return False, url, 0

    info = extraer_info_producto(html, url)
    if not info or not info["imagenes"]:
        return False, url, 0

    n_imgs = descargar_imagenes_sync(info, url)
    return n_imgs > 0, url, n_imgs


# ══════════════════════════════════════════════════════════════════
# 🔍  EXTRACCIÓN DE DATOS DEL HTML
# ══════════════════════════════════════════════════════════════════

def extraer_info_producto(html, url_final):
    """
    Estructura real del <h1> de IKEA:

      <h1 ...>
        <span class="...notranslate">KOMPLEMENT</span>
        <span class="...pip...description">
          <span>Caja juego de 4, gris claro, 40x54 cm</span>
        </span>
      </h1>

    → nombre    = texto del span.notranslate
    → categoria = primera parte del description (antes de la primera coma)
    → variante  = resto del description (después de la primera coma)
    """
    soup = BeautifulSoup(html, "lxml")

    h1 = soup.find("h1")
    if not h1:
        return None

    # ── Nombre: span con clase notranslate dentro del h1 ──────────
    nombre_tag = h1.find("span", class_="notranslate")
    nombre = nombre_tag.get_text(strip=True) if nombre_tag else None

    # Fallback: primera palabra del h1 completo
    if not nombre:
        nombre = h1.get_text(" ", strip=True).split()[0]
    if not nombre:
        return None

    # ── Descripción: span con clase *description* dentro del h1 ───
    desc_tag = h1.find("span", class_=re.compile(r"description"))
    desc = desc_tag.get_text(" ", strip=True) if desc_tag else None

    # Fallback: todo el texto del h1 menos el nombre
    if not desc:
        texto_completo = h1.get_text(" ", strip=True)
        desc = texto_completo.replace(nombre, "", 1).strip().lstrip(",- ")

    # ── Parsear "Categoría, Variante" ─────────────────────────────
    categoria = "Sin_categoria"
    variante  = "default"

    if desc:
        partes = [p.strip() for p in desc.split(",")]
        if partes[0]:
            categoria = partes[0]
        if len(partes) > 1:
            variante = ", ".join(p for p in partes[1:] if p)

    # ── Imágenes ──────────────────────────────────────────────────
    slug = re.sub(r"-\d{5,}/?$", "", url_final.rstrip("/"))
    slug = slug.rstrip("/").split("/")[-1]

    imagenes  = []
    pe_vistos = set()

    for img in soup.find_all("img"):
        raw = img.get("src") or img.get("data-src") or ""
        src = (raw if isinstance(raw, str) else " ".join(raw)).strip()

        if not src or "/images/products/" not in src:
            continue

        nombre_archivo = src.split("/")[-1].split("?")[0]
        if slug and not nombre_archivo.startswith(slug):
            continue

        pe    = re.search(r"(pe\d+)", src)
        pe_id = pe.group(1) if pe else nombre_archivo
        if pe_id in pe_vistos:
            continue
        pe_vistos.add(pe_id)
        imagenes.append(src_a_hd(src))

    return {
        "nombre":    sanitizar(nombre),
        "categoria": sanitizar(categoria),
        "variante":  sanitizar(variante),
        "imagenes":  imagenes,
        "url":       url_final,
    }

def descargar_imagenes_sync(info, url):
    ruta = Path(CARPETA_SALIDA) / info["categoria"] / info["nombre"] / info["variante"]
    ruta.mkdir(parents=True, exist_ok=True)

    id_unico    = hashlib.md5(url.encode()).hexdigest()[:8]
    hashes      = set()
    descargadas = 0

    for src in info["imagenes"]:
        if descargadas >= MAX_IMAGENES:
            break
        ruta_tmp   = ruta / f"_tmp_{id_unico}_{descargadas}.jpg"
        ruta_final = ruta / f"img_{descargadas + 1:03d}.jpg"
        try:
            r = requests.get(src, stream=True, timeout=12, headers=HEADERS)
            if r.status_code != 200:
                continue
            with open(ruta_tmp, "wb") as f:
                for chunk in r.iter_content(8192):
                    f.write(chunk)
        except Exception:
            ruta_tmp.unlink(missing_ok=True)
            continue

        h = hashlib.md5(ruta_tmp.read_bytes()).hexdigest()
        if h in hashes:
            ruta_tmp.unlink(missing_ok=True)
            continue
        hashes.add(h)
        ruta_tmp.replace(ruta_final)
        descargadas += 1

    for f in ruta.glob(f"_tmp_{id_unico}_*.jpg"):
        f.unlink(missing_ok=True)

    if descargadas > 0:
        (ruta / "meta.json").write_text(
            json.dumps({
                "url":       url,
                "nombre":    info["nombre"],
                "categoria": info["categoria"],
                "variante":  info["variante"],
                "imagenes":  descargadas,
            }, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
    return descargadas


# ══════════════════════════════════════════════════════════════════
# 🛠️  UTILIDADES
# ══════════════════════════════════════════════════════════════════

def sanitizar(texto):
    texto = re.sub(r'[<>:"/\\|?*]', "_", texto)
    texto = re.sub(r"\s+", " ", texto).strip()
    return texto[:60] or "desconocido"


def src_a_hd(src):
    src = re.sub(r"[?&]f=[^&]*", "", src).rstrip("?&")
    return src + "?f=xxl"


# ══════════════════════════════════════════════════════════════════
# 📊  RESUMEN
# ══════════════════════════════════════════════════════════════════

def imprimir_resumen():
    print("\n" + "=" * 65)
    print("📊  RESUMEN DEL DATASET")
    print("=" * 65)
    total_imgs = total_vars = 0
    dataset = Path(CARPETA_SALIDA)
    if not dataset.exists():
        print("  (carpeta vacía)")
        return
    for cat in sorted(dataset.iterdir()):
        if not cat.is_dir():
            continue
        imgs_cat = vars_cat = 0
        for nombre in cat.iterdir():
            if not nombre.is_dir():
                continue
            for var in nombre.iterdir():
                if var.is_dir():
                    imgs_cat += len(list(var.glob("*.jpg")))
                    vars_cat += 1
        print(f"  📁 {cat.name:<35} {vars_cat:>4} variantes  {imgs_cat:>6} imgs")
        total_vars += vars_cat
        total_imgs += imgs_cat
    print("-" * 65)
    print(f"  TOTAL: {total_vars} variantes  |  {total_imgs} imágenes")
    print("=" * 65)


# ══════════════════════════════════════════════════════════════════
# 🚀  MAIN
# ══════════════════════════════════════════════════════════════════

async def main():
    todas_urls = obtener_urls_sitemap()
    if not todas_urls:
        print("❌ No se encontraron URLs. Revisa la conexión.")
        return

    urls_procesadas, total_prods = cargar_progreso()
    urls_pendientes = [u for u in todas_urls if u not in urls_procesadas]
    total_imgs = 0

    print(f"📦 {len(urls_pendientes)} URLs pendientes")
    print(f"   (de {len(todas_urls)} totales en el sitemap)\n")

    t_inicio = time.time()
    semaforo = asyncio.Semaphore(CONCURRENCIA)
    conector = aiohttp.TCPConnector(limit=CONCURRENCIA, ssl=False)

    async with aiohttp.ClientSession(headers=HEADERS, connector=conector) as session:
        LOTE = 500
        for lote_i in range(0, len(urls_pendientes), LOTE):
            lote   = urls_pendientes[lote_i : lote_i + LOTE]
            tareas = [asyncio.create_task(procesar_url(session, semaforo, u)) for u in lote]

            for tarea in asyncio.as_completed(tareas):
                try:
                    exito, url_ret, n_imgs = await tarea
                except Exception:
                    continue

                urls_procesadas.add(url_ret)
                if exito:
                    total_prods += 1
                    total_imgs  += n_imgs
                    elapsed = time.time() - t_inicio
                    procesadas_total = lote_i + sum(1 for u in lote if u in urls_procesadas)
                    vel = procesadas_total / elapsed if elapsed > 0 else 1
                    eta_min = ((len(urls_pendientes) - procesadas_total) / vel / 60) if vel > 0 else 0
                    nombre_corto = url_ret.rstrip("/").split("/")[-1][:45]
                    print(f"  ✅ {nombre_corto:<45}  {n_imgs} imgs  [{total_prods} prods]  ETA: {eta_min:.0f}min")

            guardar_progreso(urls_procesadas, total_prods, total_imgs)
            pct = min(100, (lote_i + len(lote)) / len(urls_pendientes) * 100)
            print(f"  💾 {pct:.1f}% completado")

    imprimir_resumen()
    elapsed_total = (time.time() - t_inicio) / 60
    print(f"\n🎉 Completado en {elapsed_total:.0f} minutos")
    print(f"   {total_prods} productos  |  {total_imgs} imágenes  |  Dataset: {CARPETA_SALIDA}/")


if __name__ == "__main__":
    print("🚀 IKEA Scraper — Modo Sitemap")
    print(f"   Fuente  : ikea.com/sitemaps/prod-es-ES_N.xml")
    print(f"   Workers : {CONCURRENCIA} conexiones simultáneas\n")
    asyncio.run(main())