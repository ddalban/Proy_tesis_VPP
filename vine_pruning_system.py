"""
Sistema de Poda Inteligente para Robot Podador de Vides
Arquitectura Híbrida 2D/3D con Reglas de Poda Respetuosa (Simonit & Sirch)

Autor: Sistema de Visión Artificial Agrícola
Versión: 1.0
"""

import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy.interpolate import UnivariateSpline
from skimage.morphology import skeletonize


@dataclass
class CameraIntrinsics:
    """Parámetros intrínsecos de la cámara Intel RealSense"""
    fx: float  # Distancia focal en x (píxeles)
    fy: float  # Distancia focal en y (píxeles)
    cx: float  # Centro óptico en x (píxeles)
    cy: float  # Centro óptico en y (píxeles)


@dataclass
class Sarmiento:
    """Estructura de datos para un sarmiento detectado"""
    id: int
    skeleton_points: np.ndarray  # Puntos del esqueleto (Nx2)
    diameter_cm: float  # Diámetro real en cm
    axis_vector_3d: np.ndarray  # Vector director en 3D
    yemas: List['Yema']
    depth_mean: float  # Profundidad media en mm


@dataclass
class Yema:
    """Estructura de datos para una yema"""
    id: int
    position_2d: Tuple[int, int]  # Centro en píxeles (u, v)
    position_3d: np.ndarray  # Posición en espacio 3D (X, Y, Z) en cm
    asociada_a_sarmiento: int


@dataclass
class CutPoint:
    """Punto de corte calculado según Poda Respetuosa"""
    position_3d: np.ndarray  # Coordenadas (X, Y, Z) en cm
    orientation_vector: np.ndarray  # Vector de orientación para el efector
    diameter_check: str  # "Viable", "Débil", "Vigoroso"
    distance_from_bud: float  # Distancia distal en cm
    target_bud_id: int


# ============================================================================
# MÓDULO A: SIMULACIÓN DE PERCEPCIÓN (RGB-D)
# ============================================================================

class MockRealSenseDetector:
    """
    Simula la detección de una cámara Intel RealSense con modelo SAM 2.
    En producción, este módulo se reemplaza por el pipeline real de detección.
    """
    
    def __init__(self, image_path: Optional[str] = None):
        self.image_path = image_path
        # Intrínsecos típicos de Intel RealSense D435
        self.intrinsics = CameraIntrinsics(
            fx=615.0, fy=615.0, cx=320.0, cy=240.0
        )
    
    def generate_synthetic_scene(self) -> Tuple[np.ndarray, np.ndarray, Dict, CameraIntrinsics]:
        """
        Genera una escena sintética de vid con RGB, profundidad y máscaras.
        
        Returns:
            rgb_image: Imagen RGB (H, W, 3)
            depth_map: Mapa de profundidad en mm (H, W)
            masks: Diccionario con máscaras binarias
            intrinsics: Parámetros de la cámara
        """
        # Crear imagen base
        h, w = 480, 640
        rgb_image = np.ones((h, w, 3), dtype=np.uint8) * 200  # Fondo gris
        depth_map = np.ones((h, w), dtype=np.float32) * 2000  # Fondo a 2m
        
        # Simular cordón horizontal (estructura principal)
        cordon_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.line(cordon_mask, (50, 240), (590, 240), 255, 30)
        depth_map[cordon_mask > 0] = 500  # Cordón a 50cm
        rgb_image[cordon_mask > 0] = [139, 90, 43]  # Color madera
        
        # Simular sarmientos (ramas verticales)
        sarmiento_1_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.line(sarmiento_1_mask, (200, 240), (180, 100), 255, 20)
        depth_map[sarmiento_1_mask > 0] = 480
        rgb_image[sarmiento_1_mask > 0] = [120, 80, 40]
        
        sarmiento_2_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.line(sarmiento_2_mask, (400, 240), (420, 120), 255, 18)
        depth_map[sarmiento_2_mask > 0] = 490
        rgb_image[sarmiento_2_mask > 0] = [115, 75, 38]
        
        # Simular yemas (pequeñas protuberancias)
        yema_1_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(yema_1_mask, (190, 180), 8, 255, -1)
        depth_map[yema_1_mask > 0] = 475
        rgb_image[yema_1_mask > 0] = [100, 140, 80]
        
        yema_2_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(yema_2_mask, (185, 140), 8, 255, -1)
        depth_map[yema_2_mask > 0] = 470
        rgb_image[yema_2_mask > 0] = [100, 140, 80]
        
        yema_3_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(yema_3_mask, (410, 180), 8, 255, -1)
        depth_map[yema_3_mask > 0] = 485
        rgb_image[yema_3_mask > 0] = [100, 140, 80]
        
        masks = {
            'cordon': cordon_mask,
            'sarmiento_1': sarmiento_1_mask,
            'sarmiento_2': sarmiento_2_mask,
            'yema_1': yema_1_mask,
            'yema_2': yema_2_mask,
            'yema_3': yema_3_mask,
        }
        
        return rgb_image, depth_map, masks, self.intrinsics


# ============================================================================
# FUNCIONES AUXILIARES DE GEOMETRÍA
# ============================================================================

def deproject_pixel_to_point(u: int, v: int, depth_mm: float, 
                             intrinsics: CameraIntrinsics) -> np.ndarray:
    """
    Convierte coordenadas de píxel (u, v) + profundidad Z a punto 3D (X, Y, Z).
    
    Fórmula de desproyección:
        X = (u - cx) * Z / fx
        Y = (v - cy) * Z / fy
        Z = depth
    
    Args:
        u, v: Coordenadas del píxel
        depth_mm: Profundidad en milímetros
        intrinsics: Parámetros de la cámara
    
    Returns:
        np.ndarray: Punto 3D [X, Y, Z] en centímetros
    """
    depth_m = depth_mm / 1000.0  # Convertir a metros
    x = (u - intrinsics.cx) * depth_m / intrinsics.fx
    y = (v - intrinsics.cy) * depth_m / intrinsics.fy
    z = depth_m
    
    # Retornar en centímetros
    return np.array([x * 100, y * 100, z * 100])


def pixel_size_to_real_size(pixel_size: float, depth_mm: float, 
                            focal_length: float) -> float:
    """
    Convierte tamaño en píxeles a tamaño real en cm usando geometría de cámara.
    
    Args:
        pixel_size: Tamaño en píxeles
        depth_mm: Profundidad en mm
        focal_length: Distancia focal en píxeles
    
    Returns:
        Tamaño real en centímetros
    """
    depth_m = depth_mm / 1000.0
    real_size_m = (pixel_size * depth_m) / focal_length
    return real_size_m * 100  # Convertir a cm


# ============================================================================
# MÓDULO B: PROCESAMIENTO ESTRUCTURAL (CORDÓN)
# ============================================================================

def process_cordon(mask_cordon: np.ndarray) -> np.ndarray:
    """
    Procesa la máscara del cordón para extraer su eje central.
    
    Args:
        mask_cordon: Máscara binaria del cordón
    
    Returns:
        Puntos del eje central ordenados
    """
    # Esqueletizar para obtener el eje central
    skeleton = skeletonize(mask_cordon > 0)
    
    # Extraer coordenadas del esqueleto
    coords = np.column_stack(np.where(skeleton))
    
    if len(coords) == 0:
        return np.array([])
    
    # Ordenar puntos por coordenada x
    coords = coords[coords[:, 1].argsort()]
    
    return coords


# ============================================================================
# MÓDULO C: GEOMETRÍA DEL SARMIENTO (HYBRID 2D/3D)
# ============================================================================

def process_cane(mask_sarmiento: np.ndarray, depth_map: np.ndarray, 
                intrinsics: CameraIntrinsics, cane_id: int) -> Sarmiento:
    """
    Procesa un sarmiento para extraer geometría 2D/3D y diámetro real.
    
    Pipeline:
    1. Esqueletización para obtener eje central
    2. Distance Transform para obtener grosor en píxeles
    3. Conversión a diámetro real usando profundidad
    4. Cálculo de vector director 3D
    
    Args:
        mask_sarmiento: Máscara binaria del sarmiento
        depth_map: Mapa de profundidad en mm
        intrinsics: Parámetros de cámara
        cane_id: Identificador único
    
    Returns:
        Objeto Sarmiento con toda la geometría calculada
    """
    # 1. Esqueletización
    skeleton = skeletonize(mask_sarmiento > 0)
    skeleton_coords = np.column_stack(np.where(skeleton))
    
    if len(skeleton_coords) == 0:
        raise ValueError(f"No se pudo extraer esqueleto del sarmiento {cane_id}")
    
    # 2. Distance Transform para obtener radio en píxeles
    dist_transform = cv2.distanceTransform(mask_sarmiento, cv2.DIST_L2, 5)
    
    # Obtener radio medio en el eje central
    radios_pixels = []
    for coord in skeleton_coords:
        radio = dist_transform[coord[0], coord[1]]
        radios_pixels.append(radio)
    
    radio_medio_pixels = np.mean(radios_pixels)
    
    # 3. Obtener profundidad media en la zona del sarmiento
    depth_valores = depth_map[mask_sarmiento > 0]
    depth_mean = np.mean(depth_valores)
    
    # 4. Convertir radio en píxeles a diámetro real en cm
    radio_real_cm = pixel_size_to_real_size(radio_medio_pixels, depth_mean, intrinsics.fx)
    diameter_cm = radio_real_cm * 2
    
    # 5. Calcular vector director del sarmiento en 2D
    # Ordenar puntos del esqueleto
    if len(skeleton_coords) < 2:
        axis_vector_2d = np.array([0, 1])  # Vector por defecto
    else:
        # Vector desde el punto más bajo al más alto
        punto_inicial = skeleton_coords[np.argmax(skeleton_coords[:, 0])]
        punto_final = skeleton_coords[np.argmin(skeleton_coords[:, 0])]
        axis_vector_2d = punto_final - punto_inicial
        axis_vector_2d = axis_vector_2d / np.linalg.norm(axis_vector_2d)
    
    # 6. Proyectar vector a 3D
    # Tomar dos puntos del esqueleto y proyectarlos
    if len(skeleton_coords) >= 2:
        p1_2d = skeleton_coords[0]
        p2_2d = skeleton_coords[-1]
        d1 = depth_map[p1_2d[0], p1_2d[1]]
        d2 = depth_map[p2_2d[0], p2_2d[1]]
        
        p1_3d = deproject_pixel_to_point(p1_2d[1], p1_2d[0], d1, intrinsics)
        p2_3d = deproject_pixel_to_point(p2_2d[1], p2_2d[0], d2, intrinsics)
        
        axis_vector_3d = p2_3d - p1_3d
        axis_vector_3d = axis_vector_3d / np.linalg.norm(axis_vector_3d)
    else:
        axis_vector_3d = np.array([0, -1, 0])  # Vector por defecto hacia arriba
    
    return Sarmiento(
        id=cane_id,
        skeleton_points=skeleton_coords,
        diameter_cm=diameter_cm,
        axis_vector_3d=axis_vector_3d,
        yemas=[],
        depth_mean=depth_mean
    )


def associate_buds_to_cane(yemas_masks: Dict[str, np.ndarray], 
                          sarmiento: Sarmiento,
                          depth_map: np.ndarray,
                          intrinsics: CameraIntrinsics) -> List[Yema]:
    """
    Asocia yemas a un sarmiento usando distancia punto-línea.
    
    Args:
        yemas_masks: Diccionario con máscaras de yemas
        sarmiento: Sarmiento al que asociar
        depth_map: Mapa de profundidad
        intrinsics: Parámetros de cámara
    
    Returns:
        Lista de yemas asociadas al sarmiento
    """
    yemas_list = []
    
    for yema_id, (yema_name, yema_mask) in enumerate(yemas_masks.items()):
        # Encontrar centro de la yema
        coords = np.column_stack(np.where(yema_mask > 0))
        if len(coords) == 0:
            continue
        
        centro_2d = coords.mean(axis=0).astype(int)
        u, v = centro_2d[1], centro_2d[0]
        
        # Calcular distancia al esqueleto del sarmiento
        distancias = np.linalg.norm(sarmiento.skeleton_points - centro_2d, axis=1)
        min_dist = np.min(distancias)
        
        # Si está cerca del sarmiento (umbral de 30 píxeles)
        if min_dist < 30:
            depth_yema = depth_map[v, u]
            pos_3d = deproject_pixel_to_point(u, v, depth_yema, intrinsics)
            
            yema = Yema(
                id=yema_id,
                position_2d=(u, v),
                position_3d=pos_3d,
                asociada_a_sarmiento=sarmiento.id
            )
            yemas_list.append(yema)
    
    # Ordenar yemas por posición vertical (de abajo hacia arriba)
    yemas_list.sort(key=lambda y: -y.position_2d[1])
    
    return yemas_list


# ============================================================================
# MÓDULO D: REGLAS AGRONÓMICAS (PODA RESPETUOSA)
# ============================================================================

class AgronomicBrain:
    """
    Motor de reglas agronómicas basado en el Método Simonit & Sirch.
    
    Principios de Poda Respetuosa:
    - Respetar el flujo de savia
    - Cortar a distancia adecuada de la yema
    - Considerar vigor del sarmiento
    """
    
    def __init__(self):
        # Umbrales de diámetro según Método Simonit & Sirch
        self.diameter_min_viable = 0.7  # cm
        self.diameter_max_optimal = 1.2  # cm
        
        # Factor de distancia distal (1.2x diámetro)
        self.distal_distance_factor = 1.2
    
    def check_cane_viability(self, diameter_cm: float) -> str:
        """
        Clasifica el sarmiento según su diámetro.
        
        Regla de Poda Respetuosa:
        - < 0.7cm: Débil (dejar más pulgares)
        - 0.7-1.2cm: Viable (poda estándar)
        - > 1.2cm: Vigoroso (reducir carga)
        """
        if diameter_cm < self.diameter_min_viable:
            return "Débil"
        elif diameter_cm <= self.diameter_max_optimal:
            return "Viable"
        else:
            return "Vigoroso"
    
    def calculate_cut_point(self, sarmiento: Sarmiento, 
                           target_bud_index: int = 1) -> CutPoint:
        """
        Calcula el punto de corte según Poda Respetuosa.
        
        Algoritmo:
        1. Seleccionar yema objetivo (por defecto la 2da yema)
        2. Calcular distancia distal: 1.2 × diámetro
        3. Posicionar corte alejándose de la yema siguiendo el eje
        4. Orientar herramienta paralela al eje de crecimiento
        
        Args:
            sarmiento: Sarmiento a podar
            target_bud_index: Índice de la yema objetivo (0-indexed)
        
        Returns:
            CutPoint con coordenadas 3D y orientación
        """
        if not sarmiento.yemas:
            raise ValueError(f"Sarmiento {sarmiento.id} no tiene yemas asociadas")
        
        if target_bud_index >= len(sarmiento.yemas):
            target_bud_index = len(sarmiento.yemas) - 1
        
        yema_objetivo = sarmiento.yemas[target_bud_index]
        
        # 1. Verificar viabilidad
        viability_status = self.check_cane_viability(sarmiento.diameter_cm)
        
        # 2. Calcular distancia distal
        distancia_distal_cm = self.distal_distance_factor * sarmiento.diameter_cm
        
        # 3. Calcular punto de corte
        # P_corte = P_yema + distancia × vector_eje
        # El vector debe apuntar "alejándose" de la base del sarmiento
        cut_position_3d = (yema_objetivo.position_3d + 
                          distancia_distal_cm * sarmiento.axis_vector_3d)
        
        # 4. Vector de orientación para el efector
        # Debe ser PARALELO al eje de crecimiento para respetar flujo de savia
        orientation_vector = sarmiento.axis_vector_3d.copy()
        
        return CutPoint(
            position_3d=cut_position_3d,
            orientation_vector=orientation_vector,
            diameter_check=viability_status,
            distance_from_bud=distancia_distal_cm,
            target_bud_id=yema_objetivo.id
        )


# ============================================================================
# MÓDULO E: PIPELINE PRINCIPAL
# ============================================================================

def run_pruning_system(image_path: Optional[str] = None) -> Dict:
    """
    Pipeline completo del sistema de poda.
    
    Args:
        image_path: Ruta a imagen RGBD (opcional, usa sintética si None)
    
    Returns:
        Diccionario con resultados completos del análisis
    """
    print("=" * 60)
    print("SISTEMA DE PODA INTELIGENTE - Método Simonit & Sirch")
    print("=" * 60)
    
    # 1. PERCEPCIÓN: Cargar/Generar datos RGB-D
    detector = MockRealSenseDetector(image_path)
    rgb_image, depth_map, masks, intrinsics = detector.generate_synthetic_scene()
    print("\n[1] Datos RGB-D cargados correctamente")
    print(f"    Resolución: {rgb_image.shape[1]}x{rgb_image.shape[0]}")
    print(f"    Máscaras detectadas: {len(masks)}")
    
    # 2. PROCESAMIENTO: Analizar cordón
    cordon_axis = process_cordon(masks['cordon'])
    print(f"\n[2] Cordón procesado: {len(cordon_axis)} puntos en eje central")
    
    # 3. PROCESAMIENTO: Analizar cada sarmiento
    sarmientos = []
    yemas_masks = {k: v for k, v in masks.items() if 'yema' in k}
    
    for i, sarmiento_key in enumerate([k for k in masks.keys() if 'sarmiento' in k]):
        print(f"\n[3.{i+1}] Procesando {sarmiento_key}...")
        
        sarmiento = process_cane(
            masks[sarmiento_key], 
            depth_map, 
            intrinsics, 
            cane_id=i
        )
        
        print(f"    → Diámetro estimado: {sarmiento.diameter_cm:.2f} cm")
        print(f"    → Profundidad media: {sarmiento.depth_mean:.1f} mm")
        
        # Asociar yemas
        sarmiento.yemas = associate_buds_to_cane(
            yemas_masks, sarmiento, depth_map, intrinsics
        )
        print(f"    → Yemas asociadas: {len(sarmiento.yemas)}")
        
        sarmientos.append(sarmiento)
    
    # 4. DECISIÓN AGRONÓMICA: Calcular puntos de corte
    agronomic_brain = AgronomicBrain()
    cut_points = []
    
    print("\n" + "=" * 60)
    print("ANÁLISIS AGRONÓMICO Y PUNTOS DE CORTE")
    print("=" * 60)
    
    for sarmiento in sarmientos:
        if not sarmiento.yemas:
            print(f"\nSarmiento {sarmiento.id}: Sin yemas detectadas (no se puede podar)")
            continue
        
        try:
            cut_point = agronomic_brain.calculate_cut_point(sarmiento, target_bud_index=1)
            cut_points.append((sarmiento, cut_point))
            
            print(f"\nSarmiento {sarmiento.id}:")
            print(f"  Estado: {cut_point.diameter_check}")
            print(f"  Diámetro: {sarmiento.diameter_cm:.2f} cm")
            print(f"  Yema objetivo: #{cut_point.target_bud_id + 1}")
            print(f"  Distancia distal: {cut_point.distance_from_bud:.2f} cm")
            print(f"  Punto de corte 3D: X={cut_point.position_3d[0]:.2f}, "
                  f"Y={cut_point.position_3d[1]:.2f}, Z={cut_point.position_3d[2]:.2f} cm")
            print(f"  Vector orientación: [{cut_point.orientation_vector[0]:.3f}, "
                  f"{cut_point.orientation_vector[1]:.3f}, {cut_point.orientation_vector[2]:.3f}]")
        
        except Exception as e:
            print(f"\nSarmiento {sarmiento.id}: Error al calcular corte - {str(e)}")
    
    # 5. Retornar resultados completos
    results = {
        'rgb_image': rgb_image,
        'depth_map': depth_map,
        'masks': masks,
        'intrinsics': intrinsics,
        'sarmientos': sarmientos,
        'cut_points': cut_points,
        'cordon_axis': cordon_axis
    }
    
    print("\n" + "=" * 60)
    print(f"SISTEMA EJECUTADO EXITOSAMENTE")
    print(f"Total sarmientos analizados: {len(sarmientos)}")
    print(f"Total puntos de corte calculados: {len(cut_points)}")
    print("=" * 60 + "\n")
    
    return results


# ============================================================================
# EJEMPLO DE USO
# ============================================================================

if __name__ == "__main__":
    # Ejecutar sistema con datos sintéticos
    results = run_pruning_system()
    
    # Acceder a resultados para integración con robot
    for sarmiento, cut_point in results['cut_points']:
        # Estas son las coordenadas que se enviarían al controlador del robot
        x, y, z = cut_point.position_3d
        ox, oy, oz = cut_point.orientation_vector
        
        print(f"\n→ Comando para robot (Sarmiento {sarmiento.id}):")
        print(f"   MOVE_TO({x:.2f}, {y:.2f}, {z:.2f})")
        print(f"   ORIENT({ox:.3f}, {oy:.3f}, {oz:.3f})")
        print(f"   CUT()")