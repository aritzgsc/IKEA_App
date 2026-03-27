"""
augment.py — Genera variantes aumentadas de cada imagen del dataset.
Las guarda en la misma carpeta con prefijo "aug_".
No toca las imágenes originales.
Después solo tienes que volver a ejecutar indexer.py.

Augmentaciones incluidas (10 tipos + combinaciones):
  0  Rotación leve
  1  Recorte central
  2  Recorte esquina (simula ángulo de cámara)
  3  Brillo
  4  Contraste
  5  Saturación (simula distintos colores de mueble / iluminación)
  6  Nitidez (sharpen — simula fotos bien enfocadas pero distintas)
  7  Temperatura de color (warm/cool — simula luz de habitación)
  8  Perspectiva leve (simula foto tomada en diagonal)
  9  Combinación: brillo + contraste + saturación a la vez
"""

import colorsys
import random
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from pathlib import Path

# ─────────────────────────────────────────────────────
# CONFIGURACIÓN — Cambiar solo esto
# ─────────────────────────────────────────────────────
PRODUCTS_DIR     = "./Dataset_IKEA_Definitivo2"   # Carpeta a augmentar
VARIANTES_X_FOTO = 10                             # Una por cada tipo de augmentación
EXTENSIONS       = {".jpg", ".jpeg", ".png", ".webp"}
# ─────────────────────────────────────────────────────


def shift_hue(img: Image.Image, hue_shift: float) -> Image.Image:
    """
    Rota el tono (hue) de la imagen en grados [-30, 30].
    Útil para simular variantes de color de un mismo mueble.
    """
    arr = np.array(img, dtype=np.float32) / 255.0
    out = np.zeros_like(arr)
    for y in range(arr.shape[0]):
        for x in range(arr.shape[1]):
            r, g, b = arr[y, x]
            h, s, v = colorsys.rgb_to_hsv(r, g, b)
            h = (h + hue_shift / 360.0) % 1.0
            out[y, x] = colorsys.hsv_to_rgb(h, s, v)
    return Image.fromarray((out * 255).astype(np.uint8))


def shift_hue_fast(img: Image.Image, hue_shift: float) -> Image.Image:
    """
    Versión rápida del hue shift usando canales HSV con numpy.
    hue_shift en grados, rango recomendado [-25, 25].
    """
    arr = np.array(img.convert("RGB"), dtype=np.uint8)
    # Convertir a float [0,1]
    rf = arr[:, :, 0] / 255.0
    gf = arr[:, :, 1] / 255.0
    bf = arr[:, :, 2] / 255.0

    maxc = np.maximum(np.maximum(rf, gf), bf)
    minc = np.minimum(np.minimum(rf, gf), bf)
    delta = maxc - minc

    # Value y Saturation
    v = maxc
    s = np.where(maxc != 0, delta / maxc, 0)

    # Hue
    h = np.zeros_like(v)
    mask_r = (maxc == rf) & (delta != 0)
    mask_g = (maxc == gf) & (delta != 0)
    mask_b = (maxc == bf) & (delta != 0)
    h[mask_r] = ((gf[mask_r] - bf[mask_r]) / delta[mask_r]) % 6
    h[mask_g] = (bf[mask_g] - rf[mask_g]) / delta[mask_g] + 2
    h[mask_b] = (rf[mask_b] - gf[mask_b]) / delta[mask_b] + 4
    h = h / 6.0

    # Aplicar shift
    h = (h + hue_shift / 360.0) % 1.0

    # Volver a RGB
    hi = (h * 6).astype(int) % 6
    f  = h * 6 - np.floor(h * 6)
    p  = v * (1 - s)
    q  = v * (1 - f * s)
    t  = v * (1 - (1 - f) * s)

    r2 = np.select([hi==0, hi==1, hi==2, hi==3, hi==4, hi==5], [v, q, p, p, t, v])
    g2 = np.select([hi==0, hi==1, hi==2, hi==3, hi==4, hi==5], [t, v, v, q, p, p])
    b2 = np.select([hi==0, hi==1, hi==2, hi==3, hi==4, hi==5], [p, p, t, v, v, q])

    result = np.stack([r2, g2, b2], axis=-1)
    result = np.clip(result * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(result)


def add_color_temperature(img: Image.Image, warm: bool) -> Image.Image:
    """
    Simula luz cálida (bombilla de habitación) o fría (luz de día / LED).
    Warm: más rojo/amarillo. Cool: más azul.
    """
    arr = np.array(img, dtype=np.float32)
    if warm:
        arr[:, :, 0] = np.clip(arr[:, :, 0] * 1.10, 0, 255)  # R +10%
        arr[:, :, 1] = np.clip(arr[:, :, 1] * 1.05, 0, 255)  # G +5%
        arr[:, :, 2] = np.clip(arr[:, :, 2] * 0.88, 0, 255)  # B -12%
    else:
        arr[:, :, 0] = np.clip(arr[:, :, 0] * 0.90, 0, 255)  # R -10%
        arr[:, :, 1] = np.clip(arr[:, :, 1] * 0.95, 0, 255)  # G -5%
        arr[:, :, 2] = np.clip(arr[:, :, 2] * 1.12, 0, 255)  # B +12%
    return Image.fromarray(arr.astype(np.uint8))


def perspective_transform(img: Image.Image, strength: float = 0.08) -> Image.Image:
    """
    Aplica una transformación de perspectiva leve para simular
    fotos tomadas desde un ángulo ligeramente lateral o superior.
    """
    w, h = img.size
    d = int(min(w, h) * strength)

    # Elegir aleatoriamente qué esquina "comprimir"
    side = random.choice(['left', 'right', 'top', 'bottom'])

    if side == 'left':
        src = [(0,0), (w,0), (w,h), (0,h)]
        dst = [(d, d), (w,0), (w,h), (d, h-d)]
    elif side == 'right':
        src = [(0,0), (w,0), (w,h), (0,h)]
        dst = [(0,0), (w-d, d), (w-d, h-d), (0,h)]
    elif side == 'top':
        src = [(0,0), (w,0), (w,h), (0,h)]
        dst = [(d,0), (w-d,0), (w,h), (0,h)]
    else:
        src = [(0,0), (w,0), (w,h), (0,h)]
        dst = [(0,0), (w,0), (w-d,h), (d,h)]

    # Calcular coeficientes de transformación
    def find_coeffs(source_coords, target_coords):
        matrix = []
        for s, t in zip(source_coords, target_coords):
            matrix.append([t[0], t[1], 1, 0, 0, 0, -s[0]*t[0], -s[0]*t[1]])
            matrix.append([0, 0, 0, t[0], t[1], 1, -s[1]*t[0], -s[1]*t[1]])
        A = np.matrix(matrix, dtype=np.float64)
        B = np.array(source_coords).reshape(8)
        res = np.linalg.solve(A, B)
        return np.array(res).flatten()

    try:
        coeffs = find_coeffs(dst, src)
        return img.transform((w, h), Image.Transform.PERSPECTIVE, coeffs.tolist(),
                             Image.Resampling.BICUBIC)
    except Exception:
        return img  # Si falla la transformación, devolver original


def augment(img: Image.Image, idx: int) -> Image.Image:
    """Aplica una transformación según el índice (0-9)."""
    w, h = img.size

    if idx == 0:
        # Rotación leve
        angle = random.uniform(-12, 12)
        return img.rotate(angle, expand=False, fillcolor=(255, 255, 255))

    elif idx == 1:
        # Recorte central (simula zoom)
        scale  = random.uniform(0.78, 0.93)
        new_w, new_h = int(w * scale), int(h * scale)
        left = (w - new_w) // 2
        top  = (h - new_h) // 2
        return img.crop((left, top, left + new_w, top + new_h)).resize((w, h), Image.Resampling.LANCZOS)

    elif idx == 2:
        # Recorte esquina (simula ángulo de cámara descentrado)
        scale  = random.uniform(0.75, 0.90)
        new_w, new_h = int(w * scale), int(h * scale)
        left = random.randint(0, w - new_w)
        top  = random.randint(0, h - new_h)
        return img.crop((left, top, left + new_w, top + new_h)).resize((w, h), Image.Resampling.LANCZOS)

    elif idx == 3:
        # Brillo variable (simula distintas iluminaciones)
        factor = random.uniform(0.55, 1.55)
        return ImageEnhance.Brightness(img).enhance(factor)

    elif idx == 4:
        # Contraste variable
        factor = random.uniform(0.6, 1.6)
        return ImageEnhance.Contrast(img).enhance(factor)

    elif idx == 5:
        # Saturación (simula colores más vivos o apagados)
        factor = random.uniform(0.3, 1.8)
        return ImageEnhance.Color(img).enhance(factor)

    elif idx == 6:
        # Nitidez (sharpen — fotos bien enfocadas)
        factor = random.uniform(1.5, 3.0)
        return ImageEnhance.Sharpness(img).enhance(factor)

    elif idx == 7:
        # Temperatura de color (luz cálida o fría de habitación)
        warm = random.choice([True, False])
        return add_color_temperature(img, warm)

    elif idx == 8:
        # Perspectiva leve (foto en diagonal)
        strength = random.uniform(0.05, 0.12)
        return perspective_transform(img, strength)

    elif idx == 9:
        # Combinación realista: brillo + contraste + saturación simultáneos
        # Simula una foto real con variaciones naturales de cámara
        result = ImageEnhance.Brightness(img).enhance(random.uniform(0.7, 1.3))
        result = ImageEnhance.Contrast(result).enhance(random.uniform(0.8, 1.3))
        result = ImageEnhance.Color(result).enhance(random.uniform(0.7, 1.4))
        return result

    return img


# ─────────────────────────────────────────────────────
# Ejecución
# ─────────────────────────────────────────────────────
base = Path(PRODUCTS_DIR)
total_originales = 0
total_generadas  = 0
total_skip       = 0

print(f"📂 Augmentando dataset en: {PRODUCTS_DIR}")
print(f"   Variantes por foto: {VARIANTES_X_FOTO}")
print(f"   Tipos: rotación, recortes, brillo, contraste, saturación,")
print(f"          nitidez, temperatura de color, perspectiva, combinación\n")

for img_path in sorted(base.rglob("*")):
    if img_path.suffix.lower() not in EXTENSIONS:
        continue

    # No reaugmentar imágenes ya generadas
    if img_path.stem.startswith("aug_"):
        total_skip += 1
        continue

    total_originales += 1

    try:
        img = Image.open(img_path).convert("RGB")
    except Exception as e:
        print(f"⚠️  Error abriendo {img_path}: {e}")
        continue

    for i in range(VARIANTES_X_FOTO):
        try:
            aug_img  = augment(img, i)
            aug_name = f"aug_{i}__{img_path.stem}{img_path.suffix}"
            aug_path = img_path.parent / aug_name
            aug_img.save(aug_path, quality=92)
            total_generadas += 1
        except Exception as e:
            print(f"⚠️  Error en augmentación {i} de {img_path.name}: {e}")

    print(f"  ✅ {img_path.parent.name} / {img_path.name} → {VARIANTES_X_FOTO} variantes", end="\r")

print(f"\n\n✅ Augmentación completada:")
print(f"   {total_originales} imágenes originales procesadas")
print(f"   {total_generadas} imágenes nuevas generadas")
print(f"   {total_skip} imágenes aug_ saltadas (ya existían)")
print(f"   Total en dataset: {total_originales + total_generadas} imágenes")
print(f"\n➡️  Ahora vuelve a ejecutar indexer.py para regenerar el índice")