"""
Módulo para cargar y procesar datasets RGBD etiquetados reales.

Este módulo reemplaza el MockRealSenseDetector cuando se trabaja
con imágenes reales de vides capturadas con Intel RealSense.

Formatos soportados:
- Imágenes RGB: .jpg, .png
- Mapas de profundidad: .png (16-bit), .npy, .raw
- Máscaras: .png (binarias), .json (COCO format), .xml (Pascal VOC)
"""

import numpy as np
import cv2
import json
import os
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
from pathlib import Path


@dataclass
class CameraIntrinsics:
    """Parámetros intrínsecos de la cámara"""
    fx: float
    fy: float
    cx: float
    cy: float


class RealSenseDatasetLoader:
    """
    Carga datasets reales capturados con Intel RealSense D435/D455.
    
    Estructura esperada del dataset:
    
    dataset/
    ├── rgb/
    │   ├── img_0001.png
    │   ├── img_0002.png
    │   └── ...
    ├── depth/
    │   ├── depth_0001.png  (16-bit PNG)
    │   ├── depth_0002.png
    │   └── ...
    ├── masks/
    │   ├── 0001_cordon.png
    │   ├── 0001_sarmiento_1.png
    │   ├── 0001_yema_1.png
    │   └── ...
    ├── intrinsics.json
    └── metadata.json
    """
    
    def __init__(self, dataset_path: str):
        """
        Inicializa el cargador de dataset.
        
        Args:
            dataset_path: Ruta al directorio raíz del dataset
        """
        self.dataset_path = Path(dataset_path)
        self.rgb_dir = self.dataset_path / "rgb"
        self.depth_dir = self.dataset_path / "depth"
        self.masks_dir = self.dataset_path / "masks"
        
        # Validar estructura
        self._validate_structure()
        
        # Cargar intrínsecos de cámara
        self.intrinsics = self._load_intrinsics()
        
        # Listar imágenes disponibles
        self.available_images = self._scan_images()
        
        print(f"✓ Dataset cargado: {len(self.available_images)} imágenes encontradas")
    
    def _validate_structure(self):
        """Valida que el dataset tenga la estructura correcta"""
        required_dirs = [self.rgb_dir, self.depth_dir, self.masks_dir]
        for dir_path in required_dirs:
            if not dir_path.exists():
                raise FileNotFoundError(
                    f"Directorio requerido no encontrado: {dir_path}\n"
                    f"Estructura esperada:\n"
                    f"  dataset/\n"
                    f"    ├── rgb/\n"
                    f"    ├── depth/\n"
                    f"    └── masks/"
                )
    
    def _load_intrinsics(self) -> CameraIntrinsics:
        """
        Carga parámetros intrínsecos desde archivo JSON.
        
        Formato esperado en intrinsics.json:
        {
            "fx": 615.123,
            "fy": 615.456,
            "cx": 320.789,
            "cy": 240.012
        }
        """
        intrinsics_file = self.dataset_path / "intrinsics.json"
        
        if intrinsics_file.exists():
            with open(intrinsics_file, 'r') as f:
                data = json.load(f)
            
            return CameraIntrinsics(
                fx=data['fx'],
                fy=data['fy'],
                cx=data['cx'],
                cy=data['cy']
            )
        else:
            # Valores por defecto para Intel RealSense D435
            print("⚠ intrinsics.json no encontrado, usando valores por defecto D435")
            return CameraIntrinsics(fx=615.0, fy=615.0, cx=320.0, cy=240.0)
    
    def _scan_images(self) -> List[str]:
        """Escanea directorio RGB para listar imágenes disponibles"""
        extensions = ['.png', '.jpg', '.jpeg']
        images = []
        
        for ext in extensions:
            images.extend(self.rgb_dir.glob(f"*{ext}"))
        
        # Extraer IDs de imagen (asumiendo formato img_XXXX.ext)
        image_ids = []
        for img_path in images:
            # Extraer número de la imagen
            stem = img_path.stem  # "img_0001"
            try:
                # Intentar extraer número
                parts = stem.split('_')
                if len(parts) >= 2:
                    image_id = parts[-1]  # "0001"
                else:
                    image_id = stem
                image_ids.append(image_id)
            except:
                image_ids.append(stem)
        
        return sorted(set(image_ids))
    
    def load_image(self, image_id: str) -> Tuple[np.ndarray, np.ndarray, Dict, CameraIntrinsics]:
        """
        Carga una imagen completa del dataset (RGB + Depth + Máscaras).
        
        Args:
            image_id: ID de la imagen (ej: "0001", "img_0001")
        
        Returns:
            rgb_image: Imagen RGB (H, W, 3)
            depth_map: Mapa de profundidad en mm (H, W)
            masks: Diccionario con máscaras binarias
            intrinsics: Parámetros de cámara
        """
        # 1. Cargar RGB
        rgb_image = self._load_rgb(image_id)
        
        # 2. Cargar Depth
        depth_map = self._load_depth(image_id)
        
        # 3. Cargar Máscaras
        masks = self._load_masks(image_id)
        
        # Validar dimensiones
        if rgb_image.shape[:2] != depth_map.shape:
            print(f"⚠ Redimensionando depth map para coincidir con RGB")
            depth_map = cv2.resize(depth_map, (rgb_image.shape[1], rgb_image.shape[0]),
                                  interpolation=cv2.INTER_NEAREST)
        
        return rgb_image, depth_map, masks, self.intrinsics
    
    def _load_rgb(self, image_id: str) -> np.ndarray:
        """Carga imagen RGB"""
        # Intentar diferentes formatos de nombre
        possible_names = [
            f"img_{image_id}.png",
            f"img_{image_id}.jpg",
            f"{image_id}.png",
            f"{image_id}.jpg",
            f"rgb_{image_id}.png",
        ]
        
        for name in possible_names:
            path = self.rgb_dir / name
            if path.exists():
                img = cv2.imread(str(path))
                if img is not None:
                    return img
        
        raise FileNotFoundError(
            f"No se encontró imagen RGB para ID: {image_id}\n"
            f"Buscado en: {self.rgb_dir}\n"
            f"Formatos intentados: {possible_names}"
        )
    
    def _load_depth(self, image_id: str) -> np.ndarray:
        """
        Carga mapa de profundidad.
        
        Soporta:
        - PNG 16-bit (formato nativo RealSense)
        - NPY (numpy array)
        - RAW binario
        """
        possible_paths = [
            self.depth_dir / f"depth_{image_id}.png",
            self.depth_dir / f"{image_id}_depth.png",
            self.depth_dir / f"{image_id}.png",
            self.depth_dir / f"depth_{image_id}.npy",
        ]
        
        for path in possible_paths:
            if path.exists():
                if path.suffix == '.npy':
                    # Cargar numpy array
                    depth = np.load(str(path))
                    return depth.astype(np.float32)
                
                elif path.suffix == '.png':
                    # Cargar PNG 16-bit
                    depth = cv2.imread(str(path), cv2.IMREAD_ANYDEPTH)
                    if depth is not None:
                        return depth.astype(np.float32)
        
        raise FileNotFoundError(
            f"No se encontró mapa de profundidad para ID: {image_id}\n"
            f"Buscado en: {self.depth_dir}"
        )
    
    def _load_masks(self, image_id: str) -> Dict[str, np.ndarray]:
        """
        Carga todas las máscaras asociadas a una imagen.
        
        Busca archivos con formato: {image_id}_{objeto}.png
        Ejemplo: 0001_cordon.png, 0001_sarmiento_1.png
        """
        masks = {}
        
        # Buscar todos los archivos que coincidan con el pattern
        pattern = f"{image_id}_*.png"
        mask_files = list(self.masks_dir.glob(pattern))
        
        # También buscar patrón alternativo
        alt_pattern = f"*_{image_id}_*.png"
        mask_files.extend(self.masks_dir.glob(alt_pattern))
        
        if not mask_files:
            print(f"⚠ No se encontraron máscaras para imagen {image_id}")
            return {}
        
        for mask_file in mask_files:
            # Extraer nombre del objeto
            # Formato: 0001_sarmiento_1.png -> "sarmiento_1"
            parts = mask_file.stem.split('_')
            
            # Intentar identificar el nombre del objeto
            if len(parts) >= 2:
                # Eliminar el ID de imagen del principio
                if parts[0] == image_id or parts[0] == 'img':
                    objeto_name = '_'.join(parts[1:])
                else:
                    objeto_name = '_'.join(parts)
            else:
                objeto_name = mask_file.stem
            
            # Cargar máscara
            mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
            
            if mask is not None:
                # Binarizar (por si acaso)
                _, mask_bin = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
                masks[objeto_name] = mask_bin
        
        print(f"✓ Cargadas {len(masks)} máscaras para imagen {image_id}")
        return masks
    
    def load_batch(self, start_idx: int = 0, 
                   batch_size: int = 10) -> List[Tuple]:
        """
        Carga un lote de imágenes para procesamiento batch.
        
        Args:
            start_idx: Índice inicial
            batch_size: Cantidad de imágenes a cargar
        
        Returns:
            Lista de tuplas (image_id, rgb, depth, masks, intrinsics)
        """
        batch = []
        end_idx = min(start_idx + batch_size, len(self.available_images))
        
        for idx in range(start_idx, end_idx):
            image_id = self.available_images[idx]
            try:
                rgb, depth, masks, intrinsics = self.load_image(image_id)
                batch.append((image_id, rgb, depth, masks, intrinsics))
            except Exception as e:
                print(f"⚠ Error cargando imagen {image_id}: {e}")
        
        return batch
    
    def get_statistics(self) -> Dict:
        """Retorna estadísticas del dataset"""
        return {
            'total_images': len(self.available_images),
            'dataset_path': str(self.dataset_path),
            'intrinsics': {
                'fx': self.intrinsics.fx,
                'fy': self.intrinsics.fy,
                'cx': self.intrinsics.cx,
                'cy': self.intrinsics.cy
            }
        }


class COCOMaskLoader:
    """
    Carga máscaras en formato COCO JSON.
    
    Útil si tu dataset está anotado con herramientas como LabelMe, CVAT, etc.
    
    Formato COCO esperado:
    {
        "images": [{"id": 1, "file_name": "img_0001.jpg"}],
        "annotations": [
            {
                "image_id": 1,
                "category_id": 1,
                "segmentation": [[x1,y1,x2,y2,...]],
                "bbox": [x, y, w, h]
            }
        ],
        "categories": [
            {"id": 1, "name": "cordon"},
            {"id": 2, "name": "sarmiento"}
        ]
    }
    """
    
    def __init__(self, annotations_file: str):
        with open(annotations_file, 'r') as f:
            self.coco_data = json.load(f)
        
        # Crear mapeos
        self.category_map = {cat['id']: cat['name'] 
                           for cat in self.coco_data['categories']}
    
    def load_masks_for_image(self, image_id: int, 
                            image_shape: Tuple[int, int]) -> Dict[str, np.ndarray]:
        """
        Genera máscaras binarias desde anotaciones COCO.
        
        Args:
            image_id: ID de la imagen en COCO
            image_shape: (height, width) de la imagen
        
        Returns:
            Diccionario con máscaras binarias
        """
        masks = {}
        h, w = image_shape
        
        # Filtrar anotaciones para esta imagen
        annotations = [ann for ann in self.coco_data['annotations'] 
                      if ann['image_id'] == image_id]
        
        # Contar instancias de cada categoría
        category_counts = {}
        
        for ann in annotations:
            category_id = ann['category_id']
            category_name = self.category_map[category_id]
            
            # Si hay múltiples instancias, numerarlas
            if category_name in category_counts:
                category_counts[category_name] += 1
                mask_name = f"{category_name}_{category_counts[category_name]}"
            else:
                category_counts[category_name] = 1
                mask_name = category_name
            
            # Convertir segmentación a máscara
            mask = np.zeros((h, w), dtype=np.uint8)
            
            if 'segmentation' in ann:
                # Polígono
                for seg in ann['segmentation']:
                    poly = np.array(seg).reshape(-1, 2).astype(np.int32)
                    cv2.fillPoly(mask, [poly], 255)
            
            masks[mask_name] = mask
        
        return masks


# ============================================================================
# EJEMPLO DE USO
# ============================================================================

if __name__ == "__main__":
    # Ejemplo 1: Cargar dataset real
    print("="*60)
    print("CARGADOR DE DATASET RGBD REAL")
    print("="*60)
    
    # Ruta a tu dataset
    dataset_path = "path/to/your/dataset"  # CAMBIAR ESTO
    
    try:
        loader = RealSenseDatasetLoader(dataset_path)
        
        # Ver estadísticas
        stats = loader.get_statistics()
        print(f"\nEstadísticas del dataset:")
        print(f"  Total imágenes: {stats['total_images']}")
        print(f"  Intrínsecos fx: {stats['intrinsics']['fx']:.2f}")
        
        # Cargar primera imagen
        if loader.available_images:
            image_id = loader.available_images[0]
            print(f"\nCargando imagen: {image_id}")
            
            rgb, depth, masks, intrinsics = loader.load_image(image_id)
            
            print(f"✓ RGB shape: {rgb.shape}")
            print(f"✓ Depth shape: {depth.shape}")
            print(f"✓ Máscaras encontradas: {list(masks.keys())}")
            
            # Integración con sistema de poda
            print("\n" + "="*60)
            print("INTEGRACIÓN CON SISTEMA DE PODA")
            print("="*60)
            print("\nPara usar con el sistema principal:")
            print("    loader = RealSenseDatasetLoader('ruta/dataset')")
            print("    rgb, depth, masks, intrinsics = loader.load_image('0001')")
            print("    # Luego procesar con process_cane(), calculate_cut_point(), etc.")
    
    except FileNotFoundError as e:
        print(f"\n⚠ Error: {e}")
        print("\nEstructura esperada del dataset:")
        print("dataset/")
        print("  ├── rgb/")
        print("  │   ├── img_0001.png")
        print("  │   └── ...")
        print("  ├── depth/")
        print("  │   ├── depth_0001.png")
        print("  │   └── ...")
        print("  ├── masks/")
        print("  │   ├── 0001_cordon.png")
        print("  │   ├── 0001_sarmiento_1.png")
        print("  │   └── ...")
        print("  └── intrinsics.json")