import csv
import os
import re
import qrcode
from qrcode.image.styledpil import StyledPilImage
from qrcode.image.styles.moduledrawers.pil import RoundedModuleDrawer
from qrcode.image.styles.colormasks import SolidFillColorMask

archivo_csv = 'Proyecto_IKEA/productos_ikea.csv'
carpeta_salida = 'Proyecto_IKEA/Codigos_QR'
ruta_logo = 'Proyecto_IKEA/IKEA_Logo_BW.jpg' 

def generar_qrs_redondeados():
    if not os.path.exists(carpeta_salida):
        os.makedirs(carpeta_salida)

    usar_logo = os.path.exists(ruta_logo)
    if not usar_logo:
        print("⚠️ No se ha encontrado 'logo.png'. Se harán redondeados pero sin logo central.")

    with open(archivo_csv, 'r', encoding='utf-8') as f:
        lector = csv.DictReader(f)
        contador = 0
        
        for fila in lector:
            url = fila.get('url', '').strip()
            titulo = fila.get('title', 'producto')

            if not url:
                continue

            nombre_archivo = re.sub(r'[^\w\s-]', '_', titulo).strip()
            nombre_archivo = re.sub(r'[-\s]+', '_', nombre_archivo) + '.png'
            ruta_completa = os.path.join(carpeta_salida, nombre_archivo)

            qr = qrcode.QRCode(
                version=5, 
                error_correction=qrcode.ERROR_CORRECT_H, 
                box_size=10,
                border=4,
            )
            qr.add_data(url)
            qr.make(fit=True)

            imagen = qr.make_image(
                image_factory=StyledPilImage,
                module_drawer=RoundedModuleDrawer(), 
                color_mask=SolidFillColorMask(back_color=(255, 255, 255), front_color=(0, 0, 0)), 
                embeded_image_path=ruta_logo if usar_logo else None 
            )

            imagen.save(ruta_completa)
            contador += 1

    print(f"✅ ¡Listo! Se han generado {contador} códigos QR negros y redondeados en '{carpeta_salida}'.")

if __name__ == "__main__":
    generar_qrs_redondeados()