import json
import csv

# Nombres de los archivos
archivo_json = 'catalogo_ikea.json'
archivo_csv = 'productos_ikea.csv'

def json_a_csv():
    try:
        # Abrimos el archivo JSON
        with open(archivo_json, 'r', encoding='utf-8') as f:
            datos = json.load(f)

        # Creamos y abrimos el archivo CSV para escribir
        with open(archivo_csv, 'w', newline='', encoding='utf-8') as f_csv:
            escritor = csv.writer(f_csv)
            
            # Escribimos la cabecera
            escritor.writerow(['url', 'titulo'])
            
            # Recorremos el JSON y escribimos cada fila
            for titulo, info in datos.items():
                url = info.get('url', '') # Obtenemos la url (si no existe, lo deja en blanco)
                escritor.writerow([url, titulo])

        print(f"✅ ¡Éxito! El archivo {archivo_csv} se ha creado correctamente.")

    except FileNotFoundError:
        print(f"❌ Error: No se ha encontrado el archivo {archivo_json}")
    except Exception as e:
        print(f"❌ Ha ocurrido un error: {e}")

if __name__ == "__main__":
    json_a_csv()