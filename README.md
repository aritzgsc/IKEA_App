# 🛒 IKEA App Interactiva - IA & Sensores
**Universidad de Deusto | Asignatura: Interacción y Multimedia**

**Grupo 11:** Iker González García, Ander González García, Oier Unamunzaga Caujapé, Aritz González Santa Cruz.

---

## 📖 Sobre el proyecto
Este proyecto es un prototipo interactivo para transformar la experiencia de compra en IKEA. El objetivo principal es que el dispositivo móvil deje de ser una simple pantalla que requiere atención constante y se convierta en una herramienta ubicua que reacciona a nuestros gestos físicos y a nuestro entorno. 

Para lograrlo, hemos desarrollado una aplicación web (HTML, CSS, JS) conectada a un potente *backend* en Python que integra Inteligencia Artificial avanzada para el reconocimiento de muebles mediante la cámara del móvil.

---

## 🧠 Resumen Técnico del Ecosistema

Nuestro sistema se divide en dos grandes bloques: la interacción en el *frontend* y el motor de visión artificial en el *backend*.

### 1. Interacciones Ubicuas
La interfaz hace uso de los sensores integrados del teléfono para permitir realizar acciones sin mirar la pantalla:
* **Acelerómetro (Agitar para borrar):** Sacudir el móvil hacia los lados elimina el último producto añadido al carrito.
* **Giroscopio (Inclinar para favoritos):** Inclinar el móvil a la derecha guarda un producto en la lista de deseos.
* **Micrófono (Comandos de voz):** Permite ordenar la lista de la compra de forma automática diciendo, por ejemplo, *"Ordenar por ruta"* u *"Ordenar por precio"*.
* **Accesibilidad multisensorial:** Las acciones se confirman mediante *feedback* háptico (vibración) y notificaciones visuales para cubrir diferentes contextos de uso.

### 2. Pipeline de Inteligencia Artificial
Para el reconocimiento de muebles en tiempo real en entornos no controlados (fotografías sacadas con el móvil), hemos diseñado una arquitectura híbrida en Python:
* **Fase de Detección (YOLO-World):** Utilizamos un modelo de vocabulario abierto (*open-vocabulary*) ultraligero que localiza el mueble en la imagen basándose en *prompts* de texto, recortándolo y aislando el fondo para evitar ruido.
* **Fase de Embedding (OpenCLIP + TTA):** El recorte pasa por OpenCLIP (entrenado con LAION-2B). Para solventar la inestabilidad de las fotos reales, aplicamos *Test-Time Augmentation* (TTA), generando múltiples variaciones de la imagen y calculando un vector promedio o centroide muy robusto.
* **Fase de Búsqueda (FAISS):** Comparamos ese centroide contra nuestra base de datos vectorial de 350.000 imágenes (generadas tras aplicar *Data Augmentation* al *scraping* original) utilizando FAISS para obtener el producto exacto en milisegundos.

📄 **[Informe Completo (PDF)](https://drive.google.com/file/d/1PZgX0z7oEtPb9tDEzLJYXsiwq2yIKZSk/view?usp=drive_link)** - *Informe detallado con la investigación, diseño de UX/UI, evolución técnica y conclusiones.*

---

## 📱 Probar la Aplicación en Vivo

La aplicación está conectada a la red mediante un túnel seguro **ngrok**. Esto nos permite procesar la IA en nuestro servidor local mientras la interfaz es accesible desde cualquier dispositivo.

Accede mediante la URL: **[IKEA App](https://parental-mai-snottily.ngrok-free.dev/app)**

O escanea este código QR con tu teléfono para abrir la aplicación web directamente:

![QR App IKEA](App/IKEA_App_QR.png)
> **Nota:** El servidor debe estar iniciado por los administradores para que el enlace esté operativo

---

## 📁 Estructura del Repositorio

A continuación se muestra la estructura del código. Debido al peso de algunos modelos y datasets, **ciertas carpetas y archivos pesados están alojados en Google Drive**. 

```
PROYECTO_IKEA/
├── App/
│   ├── app.py                      # Servidor principal (FastAPI/Flask)
│   ├── catalogo_ikea.json          # Base de datos estructurada
│   ├── IKEA_App_QR.png             # QR de acceso vía ngrok
│   ├── ikea_faiss_labels.pkl       # [Alojado en Drive] Etiquetas de la base de datos vectorial
│   ├── ikea_faiss.index            # [Alojado en Drive] Índice vectorial de búsqueda rápida
│   ├── yolov8s-world.pt            # [Descarga automática en ejecución de app.py] Modelo de detección de vocabulario abierto
│   └── Dataset_IKEA_Definitivo/    # [Alojado en PC de Administrador] Imágenes del catálogo (7GB -> 70 GB) 
├── Proceso/                        
│   ├── IA/
│   │   ├── Modelos/                # [Alojado en Drive] Pruebas con DINOv2 y OpenCLIP
│   │   ├── build_faiss.py          # Creación del índice FAISS
│   │   ├── image_aumentation.py    # Multiplicación del dataset (35k -> 350k)
│   │   └── indexer.py              # Extracción de embeddings (CPU)
│   ├── QRs/
│   │   ├── Codigos QR/             # [Alojado en Drive] QRs generados en masa para etiquetas físicas (y probar la parte de QRs de la app)
│   │   ├── generar_csv_QRs.py      # Generación de QRs en masa
│   │   ├── IKEA_Logo_BW.jpg        # Logo central de los QRs
│   │   ├── productos_ikea.csv      # CSV sacado de 
│   │   └── qr_generator.py         # Script generador de códigos QR
│   └── Scrappers/
│       ├── data_scraper.py         # Extracción de info de sitemaps
│       ├── iconicos.py             # Filtrado de DataSet para crear modelos más rápidos y precisos (no detecta todos los productos)
│       └── image_scraper.py        # Descarga automatizada de 6GB de imágenes
└── .gitignore
```
---

### 🔗 Enlaces a Recursos Externos (Google Drive)
Para facilitar el acceso al proyecto completo, aquí están los enlaces a los archivos que no se han subido directamente a GitHub por cuestiones de tamaño:
* 📱 **[Modelo IA App](https://drive.google.com/drive/folders/14oGHkBdrjiUe4l5x7PC1GAYmsR-Uimll?usp=sharing)**
* 🔗 **[Códigos QR](https://drive.google.com/drive/folders/1RBypIsFg41Bm75hHm5uBhWeXeqtLw0Qi?usp=sharing)**
* 🧠 **[Modelos IA Probados](https://drive.google.com/drive/folders/14oGHkBdrjiUe4l5x7PC1GAYmsR-Uimll?usp=sharing)**

---

## 🚀 Instalación y Despliegue (Levanta tu propio servidor)

Si prefieres ejecutar y alojar el proyecto por tu cuenta en lugar de usar nuestra versión en vivo, hemos preparado un paquete completo (*Ready-to-Run*) para que no tengas que configurar los archivos ni indexar las imágenes desde cero.

📦 **[Descargar Proyecto Completo](https://github.com/aritzgsc/IKEA_App/releases/tag/v1.0)**

**Instrucciones de uso:**
1. Descarga y extrae el archivo `.ZIP`.
2. Asegúrate de tener Python instalado y ejecuta `pip install -r requirements.txt` para instalar las dependencias.
3. Asegúrate de estar en la carpeta raíz del proyecto `cd ruta/a/IKEA_App`
4. Inicia el servidor local ejecutando `python app.py`.
5. Abre tu navegador en el ordenador y accede a la dirección local (por ejemplo, `http://localhost:8000` o la que indique la consola).

> ⚠️ **NOTA IMPORTANTE PARA PRUEBAS EN DISPOSITIVOS MÓVILES:**
> Si quieres acceder a tu servidor local desde un teléfono móvil (algo imprescindible para probar la cámara, el giroscopio y el acelerómetro), **los navegadores web exigen una conexión segura (HTTPS)**. 
> 
> Para solucionar esto, deberás instalarte **[ngrok](https://ngrok.com/)** o una herramienta similar (como Cloudflare Tunnels). Esto te permitirá crear un túnel seguro con certificado SSL hacia el puerto local de tu ordenador y generar un enlace web válido para abrirlo sin restricciones en tu dispositivo móvil.
>
> Una vez instalado simplemente escribe en consola `ngrok http 8000`
