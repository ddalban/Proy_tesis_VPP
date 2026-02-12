# Sistema de Estimación de Punto de Corte para Poda de Vid (VPP)

Repositorio del proyecto de tesis para la estimación automática del punto de corte en sarmientos de vid mediante visión por computador y modelos de aprendizaje profundo.

El sistema integra detección de objetos, estimación de profundidad y procesamiento geométrico para determinar puntos óptimos de poda a partir de imágenes RGB-D.


##  Objetivo

Desarrollar un pipeline de visión artificial capaz de:

- Detectar sarmientos de vid  
- Estimar profundidad de escena  
- Extraer estructura geométrica  
- Calcular puntos de corte de poda  
- Generar resultados reproducibles  


## 🧠 Modelos utilizados

- YOLO (Ultralytics) — detección de sarmientos  
- Depth Anything V2 — estimación de profundidad  
- OpenCV / Scikit-Image — procesamiento de imagen  
- NumPy / SciPy — procesamiento numérico  


## 📁 Estructura del repositorio

models/ — Pesos de modelos utilizados  
Notebooks/ — Notebooks de desarrollo y experimentación  
results/ — Resultados de inferencia y pruebas
dataset_link.txt — Enlace externo al dataset completo  
Dockerfile — Entorno reproducible con contenedor  
requirements.txt — Dependencias Python  

## 📓 Notebooks

El repositorio incluye notebooks que documentan el proceso experimental:

**Notebooks/02_***  
→ integración de detección + profundidad + procesamiento  

**Notebooks/03_estimacion_punto_corte.ipynb**  
→ ⭐ **Notebook principal del sistema**  
→ ejecuta el pipeline completo de estimación del punto de corte  
→ genera los resultados finales
  

## ⚠️ Importante — Rutas de archivos

El Notebook principal fue desarrollado con rutas locales del entorno de trabajo original.

Antes de ejecutar debes actualizar:

- rutas de modelos  
- rutas de imágenes de entrada  
- rutas del dataset  
- rutas de salida de resultados  

Buscar y reemplazar paths locales tipo:

```
C:/Users/...
/home/usuario/...
```

por rutas relativas del repositorio, por ejemplo:

```
models/modelo.pt
data/imagenes/
results/
```

---

## 📦 Dataset

El dataset completo no se incluye en el repositorio debido a su tamaño.

El enlace de descarga se encuentra en:

```
dataset.txt
```

Colocar el dataset descargado dentro de una carpeta:

```
data/
```

o ajustar las rutas en el notebook según la ubicación elegida.


##  Requisitos

- Python 3.10 o superior  
- pip  

Instalar dependencias:

```bash
pip install -r requirements.txt
```

---

## ▶️ Ejecución

### Método 1 — Notebook (recomendado)

```bash
jupyter notebook
```

Abrir:

```
Notebooks/03_estimacion_punto_corte.ipynb
```

Ejecutar las celdas en orden.

---

### Método 2 — Docker

Construir imagen:

```bash
docker build -t vpp_tesis .
```

Ejecutar contenedor:

```bash
docker run -it vpp_tesis
```

---

## 📊 Resultados

Los resultados de pruebas e inferencia se almacenan en:

```
results/
```

Incluyen:

- detecciones  
- estimación de puntos de poda  

##  Reproducibilidad

El repositorio incluye:

- notebooks completos  
- pesos de modelos ligeros  
- requirements.txt  
- Dockerfile  
- ejemplos de resultados  

Esto permite reproducir el pipeline experimental ajustando únicamente las rutas de datos.


## 📌 Notas

Repositorio con fines académicos y de investigación.  
Las rutas de datos y modelos deben ajustarse según el entorno de ejecución.  
El notebook principal documenta el flujo completo del método propuesto.
