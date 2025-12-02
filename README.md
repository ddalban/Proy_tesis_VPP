# Proy_tesis_VPP
# 🌿 Sistema de Poda Inteligente para Robot Podador de Vides

## 📋 Descripción

Sistema de visión artificial híbrida 2D/3D para poda automática de vides basado en el **Método Simonit & Sirch** (Poda Respetuosa). Utiliza detección de estructuras con modelos tipo SAM 2 y cámaras Intel RealSense para calcular puntos de corte precisos en espacio 3D.

### ✨ Características Principales

- ✅ Arquitectura híbrida 2D/3D modular
- ✅ Cálculo de diámetros reales de sarmientos
- ✅ Reglas agronómicas de Poda Respetuosa
- ✅ Transformación de coordenadas cámara → robot
- ✅ Visualización completa para validación de tesis
- ✅ Soporte para datasets RGBD etiquetados
- ✅ Exportación de comandos para controlador robótico

---

## 🗂️ Estructura del Proyecto

```
vine-pruning-system/
├── pruning_system.py          # Módulo principal del sistema
├── dataset_loader.py           # Cargador de datasets RGBD
├── robot_integration.py        # Interfaz con sistema robótico
├── main.py                     # Script principal de ejecución
├── visualization/
│   └── visualizer.py          # Módulo de visualización
├── output/                     # Resultados generados
├── validation_output/          # Reportes de validación
└── README.md                   # Este archivo
```

---

## 🚀 Instalación

### Requisitos Previos

- Python 3.8+
- pip o conda

### Dependencias

```bash
pip install numpy opencv-python scikit-image scipy matplotlib
```

**Lista completa de dependencias:**
- `numpy` >= 1.20.0
- `opencv-python` >= 4.5.0
- `scikit-image` >= 0.18.0
- `scipy` >= 1.7.0
- `matplotlib` >= 3.3.0

### Instalación desde GitHub

```bash
git clone https://github.com/tu-usuario/vine-pruning-system.git
cd vine-pruning-system
pip install -r requirements.txt
```

---

## 📖 Uso Rápido

### 1. Con Datos Sintéticos (Testing)

```python
from pruning_system import run_pruning_system
from visualization.visualizer import visualize_pruning_results

# Ejecutar sistema
results = run_pruning_system()

# Visualizar resultados
visualize_pruning_results(results, save_path="resultado.png")
```

**Línea de comandos:**
```bash
python main.py --synthetic --visualize
```

### 2. Con Dataset Real

```python
from dataset_loader import RealSenseDatasetLoader
from pruning_system import process_cane, AgronomicBrain
from visualization.visualizer import generate_validation_report

# Cargar dataset
loader = RealSenseDatasetLoader("path/to/dataset")
rgb, depth, masks, intrinsics = loader.load_image("0001")

# Procesar (ver main.py para pipeline completo)
# ...

# Generar reporte de validación
generate_validation_report(results, output_dir="validation")
```

**Línea de comandos:**
```bash
# Procesar imagen específica
python main.py --dataset path/to/dataset --image 0001 --visualize

# Procesamiento batch
python main.py --dataset path/to/dataset --batch --output results/
```

### 3. Integración con Robot

```python
from robot_integration import RobotPruningInterface, TransformMatrix
import numpy as np

# Configurar transformación cámara-robot
transform = TransformMatrix(
    translation=np.array([30.0, 0.0, 20.0]),  # cm
    rotation=np.eye(3)
)

# Inicializar interfaz
robot_interface = RobotPruningInterface(
    camera_to_base_transform=transform,
    robot_workspace_limits={
        'x': (-50, 100),
        'y': (-60, 60),
        'z': (0, 150)
    }
)

# Procesar resultados de visión
vision_results = run_pruning_system()
commands = robot_interface.process_vision_results(vision_results)

# Exportar para controlador
robot_interface.export_commands_to_json(commands, "robot_commands.json")
```

---

## 📊 Estructura del Dataset

El sistema espera datasets RGBD con esta estructura:

```
dataset/
├── rgb/
│   ├── img_0001.png
│   ├── img_0002.png
│   └── ...
├── depth/
│   ├── depth_0001.png  (16-bit PNG, valores en mm)
│   ├── depth_0002.png
│   └── ...
├── masks/
│   ├── 0001_cordon.png
│   ├── 0001_sarmiento_1.png
│   ├── 0001_sarmiento_2.png
│   ├── 0001_yema_1.png
│   ├── 0001_yema_2.png
│   └── ...
├── intrinsics.json     (Parámetros de cámara)
└── metadata.json       (Opcional)
```

### Formato de `intrinsics.json`

```json
{
  "fx": 615.123,
  "fy": 615.456,
  "cx": 320.789,
  "cy": 240.012
}
```

### Nomenclatura de Máscaras

Las máscaras deben seguir el patrón: `{image_id}_{objeto}.png`

- **Cordón:** `0001_cordon.png`
- **Sarmientos:** `0001_sarmiento_1.png`, `0001_sarmiento_2.png`, ...
- **Yemas:** `0001_yema_1.png`, `0001_yema_2.png`, ...

---

## 🧠 Arquitectura del Sistema

### Módulos Principales

#### **A. Percepción (RGB-D)**
- **Clase:** `MockRealSenseDetector` / `RealSenseDatasetLoader`
- **Input:** Imagen RGB + Mapa de profundidad
- **Output:** Máscaras de segmentación + Intrínsecos

#### **B. Procesamiento Estructural**
- **Función:** `process_cordon(mask)`
- **Algoritmo:** Esqueletización + Ajuste de curva
- **Output:** Eje central del cordón

#### **C. Geometría Híbrida 2D/3D**
- **Función:** `process_cane(mask, depth, intrinsics)`
- **Pipeline:**
  1. Esqueletización de sarmiento
  2. Distance Transform → Radio en píxeles
  3. Conversión a diámetro real (cm) usando profundidad
  4. Cálculo de vector director 3D
- **Output:** Objeto `Sarmiento` con geometría completa

#### **D. Reglas Agronómicas**
- **Clase:** `AgronomicBrain`
- **Método:** `calculate_cut_point(sarmiento, yema_index)`
- **Reglas de Poda Respetuosa:**
  - **Viabilidad por diámetro:**
    - `< 0.7 cm`: Débil
    - `0.7-1.2 cm`: Viable
    - `> 1.2 cm`: Vigoroso
  - **Punto de corte:** 
    - `P_corte = P_yema + (1.2 × Ø) × V_eje`
  - **Orientación:** Paralela al flujo de savia

#### **E. Pipeline Principal**
- **Función:** `run_pruning_system()`
- **Flujo:**
  1. Cargar datos RGB-D
  2. Detectar estructuras
  3. Procesar cada sarmiento
  4. Asociar yemas
  5. Calcular puntos de corte
  6. Generar visualización

---

## 📈 Visualización para Validación de Tesis

### Función Principal

```python
from visualization.visualizer import generate_validation_report

generate_validation_report(results, output_dir="validation_output/")
```

**Genera:**
- ✅ Imagen completa con 6 subplots (detección, geometría, 3D, etc.)
- ✅ CSV con datos de todos los sarmientos
- ✅ Gráfico de distribución de diámetros
- ✅ Imágenes individuales de alta resolución

### Salidas Generadas

```
validation_output/
├── resultado_completo.png      # Visualización principal
├── datos_poda.csv              # Métricas exportadas
└── distribucion_diametros.png  # Análisis estadístico
```

### Ejemplo de Visualización

La imagen generada incluye:

1. **Imagen RGB Original**
2. **Máscaras de Segmentación** (coloreadas)
3. **Análisis Geométrico 2D** (esqueletos + yemas)
4. **Puntos de Corte Anotados** (coordenadas 3D)
5. **Vista 3D Interactiva** (nube de puntos + vectores)
6. **Tabla de Resultados** (métricas por sarmiento)

---

## 🤖 Integración Robótica

### Transformación de Coordenadas

El sistema calcula puntos de corte en el sistema de coordenadas de la **cámara**. Para usar con un robot:

```python
# 1. Definir transformación cámara → base del robot
camera_to_base = TransformMatrix(
    translation=np.array([x_offset, y_offset, z_offset]),  # cm
    rotation=rotation_matrix_3x3
)

# 2. Transformar coordenadas
cut_point_camera = cut_point.position_3d  # De visión
cut_point_robot = robot_interface.camera_to_robot_coords(cut_point_camera)
```

### Calibración Cámara-Robot

**Métodos recomendados:**
1. **Calibración con patrón de tablero de ajedrez**
2. **Método de 4 puntos** (tocar puntos conocidos con efector)
3. **Optimización con ICP** (Iterative Closest Point)

### Formato de Comandos Exportados

```json
{
  "timestamp": "2025-12-02T10:30:00",
  "num_cuts": 2,
  "commands": [
    {
      "sarmiento_id": 0,
      "agronomic_status": "Viable",
      "diameter_cm": 0.95,
      "camera_coords": [-2.45, 15.32, 48.5],
      "robot_coords": [27.55, 15.32, 68.5],
      "poses": [
        {
          "type": "approach",
          "x": 22.55, "y": 15.32, "z": 68.5,
          "roll": 0.0, "pitch": -0.34, "yaw": 1.23
        },
        {
          "type": "cut",
          "x": 27.55, "y": 15.32, "z": 68.5,
          "roll": 0.0, "pitch": -0.34, "yaw": 1.23
        },
        {
          "type": "retract",
          "x": 22.55, "y": 15.32, "z": 68.5,
          "roll": 0.0, "pitch": -0.34, "yaw": 1.23
        }
      ]
    }
  ]
}
```

---

## 🔬 Validación Científica

### Métricas Calculadas

Para cada sarmiento procesado:

| Métrica | Unidad | Descripción |
|---------|--------|-------------|
| Diámetro real | cm | Grosor del sarmiento en punto de medición |
| Profundidad media | mm | Distancia promedio de la cámara |
| Número de yemas | count | Yemas detectadas y asociadas |
| Coordenadas 3D corte | cm | (X, Y, Z) en espacio de cámara |
| Vector de orientación | unitless | Dirección de aproximación normalizada |
| Distancia distal | cm | Separación del corte respecto a yema |
| Estado agronómico | categórico | Débil / Viable / Vigoroso |

### Exportación de Datos

```python
# CSV con todas las métricas
datos_poda.csv

# Columnas:
# Sarmiento_ID, Diametro_cm, Estado_Agronomico, Num_Yemas,
# Corte_X_cm, Corte_Y_cm, Corte_Z_cm,
# Vector_X, Vector_Y, Vector_Z, Distancia_Distal_cm
```

### Análisis Estadístico

El reporte incluye:
- Histograma de distribución de diámetros
- Umbrales agronómicos visualizados
- Estadísticas descriptivas por categoría

---

## 🧪 Testing y Debugging

### Modo Debug Rápido

```python
from visualization.visualizer import quick_visualize

results = run_pruning_system()
quick_visualize(results)  # Ventana OpenCV simple
```

### Verificar Carga de Dataset

```python
from dataset_loader import RealSenseDatasetLoader

loader = RealSenseDatasetLoader("path/to/dataset")
print(loader.get_statistics())

# Cargar y visualizar
rgb, depth, masks, _ = loader.load_image("0001")
cv2.imshow("RGB", rgb)
cv2.imshow("Depth", depth / depth.max())
cv2.waitKey(0)
```

### Simular Ejecución Robot

```bash
python robot_integration.py
# Sigue las instrucciones interactivas
```

---

## 📚 Referencias

### Poda Respetuosa (Método Simonit & Sirch)

- Simonit, M., & Sirch, P. (2018). *Manual de Poda Respetuosa*
- Principios: Respeto del flujo de savia, prevención de enfermedades de madera

### Intel RealSense

- [Documentación oficial](https://dev.intelrealsense.com/)
- Modelos soportados: D435, D455, L515

### Segment Anything Model (SAM)

- Paper: [Segment Anything (Kirillov et al., 2023)](https://arxiv.org/abs/2304.02643)
- SAM 2: [SAM 2: Segment Anything in Images and Videos](https://arxiv.org/abs/2408.00714)

---

## 🤝 Contribuciones

Para contribuir al proyecto:

1. Fork el repositorio
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

---

## 📄 Licencia

Este proyecto está bajo licencia MIT. Ver `LICENSE` para más detalles.

---

## 👨‍🔬 Autores

**Sistema desarrollado para investigación en robótica agrícola y visión artificial.**

Para citas académicas:
```bibtex
@misc{vine-pruning-system,
  title={Sistema de Poda Inteligente para Robot Podador de Vides},
  author={Tu Nombre},
  year={2025},
  publisher={GitHub},
  url={https://github.com/tu-usuario/vine-pruning-system}
}
```

---

## 📧 Contacto

Para preguntas, sugerencias o colaboraciones:
- **Email:** tu-email@universidad.edu
- **GitHub Issues:** [Reportar problema](https://github.com/tu-usuario/vine-pruning-system/issues)

---

## 🙏 Agradecimientos

- Método Simonit & Sirch por las reglas agronómicas
- Intel RealSense por el hardware de captura
- Comunidad de visión artificial por herramientas open source

---
