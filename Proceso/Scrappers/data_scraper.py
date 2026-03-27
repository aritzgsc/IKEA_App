"""
╔══════════════════════════════════════════════════════════════════╗
║          IKEA SCRAPER — Generador de Catálogo JSON               ║
║                                                                  ║
║  Extrae información de los productos (nombre, categoría, precio, ║
║  imagen, descripción, peso, ubicación) y lo guarda en un JSON.   ║
║                                                                  ║
║  Instalación:                                                    ║
║    pip install aiohttp beautifulsoup4 lxml requests              ║
╚══════════════════════════════════════════════════════════════════╝
"""

import re
import json
import time
import asyncio
import random
import requests
from pathlib import Path
from bs4 import BeautifulSoup
import aiohttp

# ══════════════════════════════════════════════════════════════════
# ⚙️  CONFIGURACIÓN
# ══════════════════════════════════════════════════════════════════

FICHERO_CATALOGO = "catalogo_ikea.json"
FICHERO_PROGRESO = "progreso_scraper_info.json"
CONCURRENCIA     = 80
TIMEOUT_GET      = 15
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
# 💾  ESTADO Y PROGRESO
# ══════════════════════════════════════════════════════════════════

def cargar_estado():
    urls_hechas = set()
    catalogo = {}

    if Path(FICHERO_PROGRESO).exists():
        with open(FICHERO_PROGRESO, "r", encoding="utf-8") as f:
            data = json.load(f)
            urls_hechas = set(data.get("urls_procesadas", []))

    if Path(FICHERO_CATALOGO).exists():
        with open(FICHERO_CATALOGO, "r", encoding="utf-8") as f:
            catalogo = json.load(f)

    if urls_hechas:
        print(f"📂 Reanudando — {len(urls_hechas)} URLs ya procesadas ({len(catalogo)} productos en catálogo)")
    
    return urls_hechas, catalogo


def guardar_estado(urls_procesadas, catalogo):
    with open(FICHERO_PROGRESO, "w", encoding="utf-8") as f:
        json.dump({"urls_procesadas": sorted(list(urls_procesadas))}, f)
    
    with open(FICHERO_CATALOGO, "w", encoding="utf-8") as f:
        json.dump(catalogo, f, ensure_ascii=False, indent=4)


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
                    return False, url, None
                html = await resp.text()
        except Exception:
            return False, url, None

    info = extraer_info_producto(html, url)
    if not info:
        return False, url, None

    return True, url, info


# ══════════════════════════════════════════════════════════════════
# 🔍  EXTRACCIÓN DE DATOS DEL HTML
# ══════════════════════════════════════════════════════════════════

def sanitizar(texto):
    texto = re.sub(r'[<>:"/\\|?*]', "_", texto)
    texto = re.sub(r"\s+", " ", texto).strip()
    return texto[:60] or "desconocido"


def extraer_info_producto(html, url_final):
    soup = BeautifulSoup(html, "lxml")
    h1 = soup.find("h1")
    if not h1:
        return None

    # 1. Extraer Nombre
    nombre_tag = h1.find("span", class_="notranslate")
    nombre = nombre_tag.get_text(strip=True) if nombre_tag else h1.get_text(" ", strip=True).split()[0]
    if not nombre:
        return None

    # 2. Extraer Descripción Corta (Subtítulo) y Categoría
    desc_tag = h1.find("span", class_=re.compile(r"description"))
    desc = desc_tag.get_text(" ", strip=True) if desc_tag else h1.get_text(" ", strip=True).replace(nombre, "", 1).strip().lstrip(",- ")
    
    subtitulo = desc if desc else "Sin especificaciones adicionales"

    categoria = "Sin_categoria"
    if desc:
        partes = [p.strip() for p in desc.split(",")]
        if partes[0]:
            categoria = partes[0]

    nombre_limpio = sanitizar(nombre)
    cat_limpia = sanitizar(categoria)
    id_producto = f"{cat_limpia} | {nombre_limpio}"

    # 3. Extraer URL de la imagen principal
    imagen_url = None
    meta_img = soup.find("meta", property="og:image")
    if meta_img:
        content_val = meta_img.get("content")
        if content_val:
            if isinstance(content_val, list): content_val = content_val[0]
            imagen_url = str(content_val)
            
    if not imagen_url:
        for img in soup.find_all("img"):
            src = img.get("src") or img.get("data-src")
            if src:
                if isinstance(src, list): src = src[0]
                src_str = str(src)
                if "/images/products/" in src_str:
                    imagen_url = src_str.split("?")[0]
                    break

    if imagen_url and imagen_url.startswith("//"):
        imagen_url = "https:" + imagen_url

    # 4. Extraer Precio
    precio = None
    meta_precio = soup.find("meta", itemprop="price")
    if meta_precio:
        content_val = meta_precio.get("content")
        if content_val:
            if isinstance(content_val, list): content_val = content_val[0]
            try: precio = float(str(content_val))
            except ValueError: pass

    if precio is None:
        precio_int_tag = soup.find("span", class_=re.compile(r"pip-temp-price__integer"))
        if precio_int_tag:
            precio_str = re.sub(r"[^\d]", "", precio_int_tag.get_text())
            precio_dec_tag = soup.find("span", class_=re.compile(r"pip-temp-price__decimal"))
            if precio_dec_tag:
                decimales = re.sub(r"[^\d]", "", precio_dec_tag.get_text())
                precio_str += f".{decimales}"
            try:
                if precio_str: precio = float(precio_str)
            except ValueError: pass

    if precio is None:
        scripts = soup.find_all("script", type="application/ld+json")
        for script in scripts:
            if script.string:
                try:
                    data = json.loads(script.string)
                    if isinstance(data, dict): data = [data]
                    for item in data:
                        if item.get("@type") == "Product":
                            ofertas = item.get("offers", {})
                            if isinstance(ofertas, dict) and "price" in ofertas:
                                precio = float(ofertas["price"])
                                break
                            elif isinstance(ofertas, list) and len(ofertas) > 0:
                                precio = float(ofertas[0].get("price", 0))
                                break
                except (json.JSONDecodeError, ValueError, TypeError):
                    continue
            if precio is not None: break

    # 5. Extraer la Descripción Larga
    descripcion_larga = ""
    parrafos_desc = soup.find_all("p", class_=re.compile(r"pipf-product-summary__description"))
    if parrafos_desc:
        textos = [p.get_text(strip=True) for p in parrafos_desc if p.get_text(strip=True)]
        descripcion_larga = " ".join(textos)
        
    if not descripcion_larga:
        meta_desc = soup.find("meta", attrs={"name": "description"})
        if meta_desc:
            desc_val = meta_desc.get("content")
            if desc_val:
                if isinstance(desc_val, list): desc_val = desc_val[0]
                descripcion_larga = str(desc_val).strip()

    if not descripcion_larga:
        descripcion_larga = subtitulo
        
    descripcion_larga = re.sub(r"\s+", " ", descripcion_larga).strip()

    # 🟢 6. Extraer Peso (NUEVO)
    peso_texto = "No especificado"
    # Buscamos en todo el texto de la página la palabra "peso" seguida de un número y "kg"
    peso_match = re.search(r'(?i)peso[^\d]*([\d,.]+\s*kg)', soup.get_text(" "))
    if peso_match:
        peso_texto = peso_match.group(1).lower().replace(",", ".")

    # 🟢 7. Ubicación en tienda (NUEVO - Aleatorio)
    pasillo_aleatorio = random.choice("ABCD")
    estanteria_aleatoria = random.randint(1, 20)

    # Devolvemos el diccionario ampliado
    return {
        "id": id_producto,
        "nombre": nombre,
        "subtitulo": subtitulo,
        "descripcion": descripcion_larga,
        "categoria": cat_limpia,
        "precio": precio,
        "peso": peso_texto,
        "ubicacion": {
            "pasillo": pasillo_aleatorio,
            "estanteria": estanteria_aleatoria
        },
        "url": url_final,
        "imagen": imagen_url
    }

# ══════════════════════════════════════════════════════════════════
# 🚀  MAIN
# ══════════════════════════════════════════════════════════════════

async def main():
    todas_urls = obtener_urls_sitemap()
    if not todas_urls:
        print("❌ No se encontraron URLs. Revisa la conexión.")
        return

    urls_procesadas, catalogo = cargar_estado()
    urls_pendientes = [u for u in todas_urls if u not in urls_procesadas]

    print(f"📦 {len(urls_pendientes)} URLs pendientes")
    print(f"   (de {len(todas_urls)} totales en el sitemap)\n")

    t_inicio = time.time()
    semaforo = asyncio.Semaphore(CONCURRENCIA)
    conector = aiohttp.TCPConnector(limit=CONCURRENCIA, ssl=False)

    async with aiohttp.ClientSession(headers=HEADERS, connector=conector) as session:
        LOTE = 500
        for lote_i in range(0, len(urls_pendientes), LOTE):
            lote = urls_pendientes[lote_i : lote_i + LOTE]
            tareas = [asyncio.create_task(procesar_url(session, semaforo, u)) for u in lote]

            for tarea in asyncio.as_completed(tareas):
                try:
                    exito, url_ret, info = await tarea
                except Exception:
                    continue

                urls_procesadas.add(url_ret)
                
                if exito and info:
                    id_prod = info.pop("id")
                    
                    if id_prod not in catalogo:
                        catalogo[id_prod] = info
                        
                        elapsed = time.time() - t_inicio
                        procesadas_total = lote_i + sum(1 for u in lote if u in urls_procesadas)
                        vel = procesadas_total / elapsed if elapsed > 0 else 1
                        eta_min = ((len(urls_pendientes) - procesadas_total) / vel / 60) if vel > 0 else 0
                        
                        precio_str = f"{info['precio']}€" if info['precio'] else "Sin precio"
                        print(f"  ✅ NUEVO: {id_prod:<30} | {precio_str:>10} | ETA: {eta_min:.0f}min")
                    else:
                        pass

            guardar_estado(urls_procesadas, catalogo)
            pct = min(100, (lote_i + len(lote)) / len(urls_pendientes) * 100)
            print(f"  💾 Guardando lote... ({pct:.1f}% completado)")

    guardar_estado(urls_procesadas, catalogo)
    
    elapsed_total = (time.time() - t_inicio) / 60
    print("\n" + "=" * 65)
    print(f"🎉 Catálogo JSON completado en {elapsed_total:.0f} minutos")
    print(f"   Total de productos únicos indexados: {len(catalogo)}")
    print(f"   Archivo guardado en: {FICHERO_CATALOGO}")
    print("=" * 65)


if __name__ == "__main__":
    print("🚀 IKEA Scraper — Generador de Catálogo JSON")
    print(f"   Workers : {CONCURRENCIA} conexiones simultáneas\n")
    asyncio.run(main())