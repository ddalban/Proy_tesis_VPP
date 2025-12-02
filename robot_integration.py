"""
INTEGRACIÓN CON SISTEMA ROBÓTICO
=================================

Este módulo muestra cómo integrar el sistema de poda inteligente
con el controlador de tu robot podador.

Arquitectura:
    Sistema de Poda -> Robot Controller -> Actuadores
    
El sistema calcula:
    - Posición 3D del punto de corte (X, Y, Z) en cm
    - Vector de orientación del efector final
    - Estado agronómico del sarmiento

El controlador del robot debe:
    - Convertir coordenadas de cámara a coordenadas del robot
    - Planificar trayectoria libre de colisiones
    - Ejecutar movimiento y corte
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

# Importar sistema de poda
from pruning_system import (
    run_pruning_system,
    CutPoint,
    Sarmiento,
    CameraIntrinsics
)


# ============================================================================
# DEFINICIONES DE INTERFAZ CON ROBOT
# ============================================================================

class RobotCommand(Enum):
    """Comandos disponibles para el robot"""
    MOVE_TO = "MOVE_TO"           # Mover efector a posición
    ORIENT = "ORIENT"              # Orientar herramienta
    CUT = "CUT"                    # Ejecutar corte
    HOME = "HOME"                  # Volver a posición home
    EMERGENCY_STOP = "E_STOP"      # Parada de emergencia


@dataclass
class RobotPose:
    """Pose del efector final del robot"""
    x: float  # Posición X en cm (base del robot)
    y: float  # Posición Y en cm
    z: float  # Posición Z en cm
    roll: float   # Orientación en radianes
    pitch: float
    yaw: float


@dataclass
class TransformMatrix:
    """Matriz de transformación cámara -> base del robot"""
    translation: np.ndarray  # Vector (3,) en cm
    rotation: np.ndarray     # Matriz (3,3) de rotación


# ============================================================================
# CLASE PRINCIPAL DE INTEGRACIÓN
# ============================================================================

class RobotPruningInterface:
    """
    Interfaz entre el sistema de visión y el controlador del robot.
    
    Responsabilidades:
    1. Transformar coordenadas de cámara a base del robot
    2. Generar secuencia de comandos para el robot
    3. Validar alcanzabilidad de posiciones
    4. Gestionar cola de tareas
    """
    
    def __init__(self, camera_to_base_transform: TransformMatrix,
                 robot_workspace_limits: Dict[str, Tuple[float, float]],
                 safety_distance_cm: float = 5.0):
        """
        Inicializa la interfaz robot-visión.
        
        Args:
            camera_to_base_transform: Transformación de cámara a base del robot
            robot_workspace_limits: Límites del espacio de trabajo
                {'x': (min, max), 'y': (min, max), 'z': (min, max)}
            safety_distance_cm: Distancia de seguridad para aproximación
        """
        self.transform = camera_to_base_transform
        self.workspace_limits = robot_workspace_limits
        self.safety_distance = safety_distance_cm
        
        # Cola de comandos para ejecución secuencial
        self.command_queue: List[Tuple[RobotCommand, Dict]] = []
        
        print("="*60)
        print("Robot Pruning Interface Inicializada")
        print("="*60)
        print(f"Workspace: X={robot_workspace_limits['x']}, "
              f"Y={robot_workspace_limits['y']}, "
              f"Z={robot_workspace_limits['z']}")
    
    def camera_to_robot_coords(self, point_camera: np.ndarray) -> np.ndarray:
        """
        Transforma coordenadas de cámara a coordenadas del robot.
        
        Transformación:
            P_robot = R * P_camera + T
        
        Args:
            point_camera: Punto en coordenadas de cámara [X, Y, Z] (cm)
        
        Returns:
            Punto en coordenadas de base del robot [X, Y, Z] (cm)
        """
        # Aplicar rotación y traslación
        point_robot = (self.transform.rotation @ point_camera + 
                      self.transform.translation)
        
        return point_robot
    
    def vector_camera_to_robot(self, vector_camera: np.ndarray) -> np.ndarray:
        """
        Transforma vector de orientación de cámara a robot.
        
        Solo aplica rotación (vectores no tienen traslación).
        
        Args:
            vector_camera: Vector en coords. de cámara
        
        Returns:
            Vector en coords. de robot
        """
        vector_robot = self.transform.rotation @ vector_camera
        return vector_robot / np.linalg.norm(vector_robot)  # Normalizar
    
    def is_reachable(self, pose: RobotPose) -> bool:
        """
        Verifica si una pose está dentro del espacio de trabajo del robot.
        
        Args:
            pose: Pose a verificar
        
        Returns:
            True si es alcanzable, False caso contrario
        """
        x_ok = self.workspace_limits['x'][0] <= pose.x <= self.workspace_limits['x'][1]
        y_ok = self.workspace_limits['y'][0] <= pose.y <= self.workspace_limits['y'][1]
        z_ok = self.workspace_limits['z'][0] <= pose.z <= self.workspace_limits['z'][1]
        
        return x_ok and y_ok and z_ok
    
    def orientation_to_euler(self, orientation_vector: np.ndarray) -> Tuple[float, float, float]:
        """
        Convierte vector de orientación a ángulos de Euler.
        
        El vector indica la dirección de aproximación de la herramienta.
        Se calculan roll, pitch, yaw para alinear el efector final.
        
        Args:
            orientation_vector: Vector unitario de dirección [x, y, z]
        
        Returns:
            Tupla (roll, pitch, yaw) en radianes
        """
        # Normalizar vector
        v = orientation_vector / np.linalg.norm(orientation_vector)
        
        # Calcular pitch (rotación en Y)
        pitch = np.arcsin(-v[2])
        
        # Calcular yaw (rotación en Z)
        if np.abs(v[0]) > 1e-6 or np.abs(v[1]) > 1e-6:
            yaw = np.arctan2(v[1], v[0])
        else:
            yaw = 0.0
        
        # Roll se mantiene en 0 para esta aplicación (herramienta vertical)
        roll = 0.0
        
        return roll, pitch, yaw
    
    def generate_pruning_sequence(self, cut_point: CutPoint, 
                                  sarmiento: Sarmiento) -> List[RobotPose]:
        """
        Genera secuencia de poses para ejecutar un corte.
        
        Secuencia típica:
        1. Aproximación (safety_distance antes del punto)
        2. Punto de corte
        3. Retracción (volver a safety_distance)
        
        Args:
            cut_point: Punto de corte calculado por sistema de visión
            sarmiento: Sarmiento asociado
        
        Returns:
            Lista de RobotPose para ejecutar
        """
        # Transformar punto de corte a coordenadas del robot
        cut_pos_robot = self.camera_to_robot_coords(cut_point.position_3d)
        
        # Transformar vector de orientación
        orient_robot = self.vector_camera_to_robot(cut_point.orientation_vector)
        
        # Convertir orientación a Euler
        roll, pitch, yaw = self.orientation_to_euler(orient_robot)
        
        # 1. Pose de aproximación (safety_distance antes del corte)
        approach_offset = -orient_robot * self.safety_distance
        approach_pos = cut_pos_robot + approach_offset
        
        approach_pose = RobotPose(
            x=approach_pos[0], y=approach_pos[1], z=approach_pos[2],
            roll=roll, pitch=pitch, yaw=yaw
        )
        
        # 2. Pose de corte
        cut_pose = RobotPose(
            x=cut_pos_robot[0], y=cut_pos_robot[1], z=cut_pos_robot[2],
            roll=roll, pitch=pitch, yaw=yaw
        )
        
        # 3. Pose de retracción (igual a aproximación)
        retract_pose = approach_pose
        
        # Validar alcanzabilidad
        if not self.is_reachable(cut_pose):
            print(f"⚠ WARNING: Punto de corte fuera del workspace!")
            print(f"   Posición: ({cut_pose.x:.2f}, {cut_pose.y:.2f}, {cut_pose.z:.2f})")
            return []
        
        return [approach_pose, cut_pose, retract_pose]
    
    def process_vision_results(self, results: Dict) -> List[Dict]:
        """
        Procesa resultados del sistema de visión y genera comandos para robot.
        
        Args:
            results: Diccionario retornado por run_pruning_system()
        
        Returns:
            Lista de diccionarios con comandos y metadata
        """
        cut_commands = []
        
        for idx, (sarmiento, cut_point) in enumerate(results['cut_points']):
            print(f"\n{'='*60}")
            print(f"Procesando Sarmiento {sarmiento.id}")
            print(f"{'='*60}")
            print(f"Estado Agronómico: {cut_point.diameter_check}")
            print(f"Diámetro: {sarmiento.diameter_cm:.2f} cm")
            
            # Generar secuencia de poses
            pose_sequence = self.generate_pruning_sequence(cut_point, sarmiento)
            
            if not pose_sequence:
                print("⚠ Secuencia no generada (fuera de alcance)")
                continue
            
            # Crear comando estructurado
            command = {
                'sarmiento_id': sarmiento.id,
                'agronomic_status': cut_point.diameter_check,
                'diameter_cm': sarmiento.diameter_cm,
                'camera_coords': cut_point.position_3d.tolist(),
                'robot_coords': self.camera_to_robot_coords(cut_point.position_3d).tolist(),
                'poses': [
                    {
                        'type': 'approach',
                        'x': pose_sequence[0].x,
                        'y': pose_sequence[0].y,
                        'z': pose_sequence[0].z,
                        'roll': pose_sequence[0].roll,
                        'pitch': pose_sequence[0].pitch,
                        'yaw': pose_sequence[0].yaw
                    },
                    {
                        'type': 'cut',
                        'x': pose_sequence[1].x,
                        'y': pose_sequence[1].y,
                        'z': pose_sequence[1].z,
                        'roll': pose_sequence[1].roll,
                        'pitch': pose_sequence[1].pitch,
                        'yaw': pose_sequence[1].yaw
                    },
                    {
                        'type': 'retract',
                        'x': pose_sequence[2].x,
                        'y': pose_sequence[2].y,
                        'z': pose_sequence[2].z,
                        'roll': pose_sequence[2].roll,
                        'pitch': pose_sequence[2].pitch,
                        'yaw': pose_sequence[2].yaw
                    }
                ]
            }
            
            cut_commands.append(command)
            
            # Imprimir resumen
            print(f"\n✓ Secuencia generada:")
            print(f"  1. Approach: ({pose_sequence[0].x:.2f}, {pose_sequence[0].y:.2f}, "
                  f"{pose_sequence[0].z:.2f})")
            print(f"  2. Cut:      ({pose_sequence[1].x:.2f}, {pose_sequence[1].y:.2f}, "
                  f"{pose_sequence[1].z:.2f})")
            print(f"  3. Retract:  ({pose_sequence[2].x:.2f}, {pose_sequence[2].y:.2f}, "
                  f"{pose_sequence[2].z:.2f})")
        
        return cut_commands
    
    def export_commands_to_json(self, commands: List[Dict], 
                               filename: str = "robot_commands.json"):
        """
        Exporta comandos a JSON para el controlador del robot.
        
        Args:
            commands: Lista de comandos generados
            filename: Nombre del archivo de salida
        """
        import json
        
        output = {
            'timestamp': str(np.datetime64('now')),
            'num_cuts': len(commands),
            'commands': commands
        }
        
        with open(filename, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"\n✓ Comandos exportados a: {filename}")


# ============================================================================
# EJEMPLO DE INTEGRACIÓN COMPLETA
# ============================================================================

def example_full_integration():
    """
    Ejemplo completo de integración: Visión -> Procesamiento -> Robot
    """
    print("\n" + "="*70)
    print("EJEMPLO DE INTEGRACIÓN COMPLETA: VISIÓN -> ROBOT")
    print("="*70 + "\n")
    
    # 1. CONFIGURAR TRANSFORMACIÓN CÁMARA-ROBOT
    # Estos valores deben obtenerse mediante calibración
    # Ejemplo: Cámara montada 30cm adelante, 20cm arriba del origen del robot
    camera_to_base = TransformMatrix(
        translation=np.array([30.0, 0.0, 20.0]),  # [X, Y, Z] en cm
        rotation=np.eye(3)  # Sin rotación (cámara alineada con robot)
    )
    
    # Límites del espacio de trabajo del robot (en cm)
    workspace = {
        'x': (-50, 100),   # Adelante/atrás
        'y': (-60, 60),    # Izquierda/derecha
        'z': (0, 150)      # Arriba/abajo
    }
    
    # 2. INICIALIZAR INTERFAZ
    robot_interface = RobotPruningInterface(
        camera_to_base_transform=camera_to_base,
        robot_workspace_limits=workspace,
        safety_distance_cm=5.0
    )
    
    # 3. EJECUTAR SISTEMA DE VISIÓN
    print("\n[1] Ejecutando sistema de visión artificial...")
    vision_results = run_pruning_system()
    
    # 4. PROCESAR RESULTADOS Y GENERAR COMANDOS
    print("\n[2] Generando comandos para robot...")
    robot_commands = robot_interface.process_vision_results(vision_results)
    
    # 5. EXPORTAR COMANDOS
    print("\n[3] Exportando comandos...")
    robot_interface.export_commands_to_json(robot_commands, "robot_commands.json")
    
    # 6. RESUMEN
    print("\n" + "="*70)
    print("RESUMEN DE INTEGRACIÓN")
    print("="*70)
    print(f"Total sarmientos detectados: {len(vision_results['sarmientos'])}")
    print(f"Comandos de corte generados: {len(robot_commands)}")
    print(f"\nPróximos pasos:")
    print("  1. Cargar robot_commands.json en el controlador")
    print("  2. Validar trayectorias con planificador de movimiento")
    print("  3. Ejecutar secuencia de poda")
    print("="*70 + "\n")
    
    return robot_commands


# ============================================================================
# SIMULADOR SIMPLE DE EJECUCIÓN
# ============================================================================

def simulate_robot_execution(commands: List[Dict]):
    """
    Simula la ejecución de comandos en el robot (para testing).
    
    En un sistema real, esto sería reemplazado por llamadas
    al API del controlador del robot (ej: ROS, ABB Rapid, etc.)
    """
    import time
    
    print("\n" + "="*70)
    print("SIMULACIÓN DE EJECUCIÓN EN ROBOT")
    print("="*70 + "\n")
    
    for idx, cmd in enumerate(commands, 1):
        print(f"\n[Corte {idx}/{len(commands)}] Sarmiento {cmd['sarmiento_id']}")
        print(f"Estado: {cmd['agronomic_status']}")
        
        for pose in cmd['poses']:
            pose_type = pose['type'].upper()
            print(f"\n  → {pose_type}:")
            print(f"     Posición: ({pose['x']:.2f}, {pose['y']:.2f}, {pose['z']:.2f})")
            print(f"     Orientación: (R={np.rad2deg(pose['roll']):.1f}°, "
                  f"P={np.rad2deg(pose['pitch']):.1f}°, Y={np.rad2deg(pose['yaw']):.1f}°)")
            
            # Simular tiempo de movimiento
            time.sleep(0.5)
        
        # Simular corte
        print(f"\n  ✂ EJECUTANDO CORTE...")
        time.sleep(0.3)
        print(f"  ✓ Corte completado")
    
    print("\n" + "="*70)
    print("SIMULACIÓN COMPLETADA")
    print("="*70)


# ============================================================================
# PUNTO DE ENTRADA
# ============================================================================

if __name__ == "__main__":
    # Ejecutar ejemplo completo
    commands = example_full_integration()
    
    # Simular ejecución
    print("\n¿Desea simular la ejecución? (s/n): ", end="")
    response = input().strip().lower()
    
    if response == 's':
        simulate_robot_execution(commands)
    
    print("\n✓ Demo de integración completada")