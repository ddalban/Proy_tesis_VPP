"""
Módulo de Visualización para Sistema de Poda Inteligente
Carpeta: visualization/

Este módulo permite visualizar los resultados del sistema para validación
en contexto de tesis doctoral/investigación.

Uso:
    from visualization.visualizer import visualize_pruning_results
    visualize_pruning_results(results, save_path="output/resultado.png")
"""

import numpy as np
import cv2
from typing import Dict, Optional
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrow
import matplotlib.patches as mpatches


def visualize_pruning_results(results: Dict, 
                              save_path: Optional[str] = None,
                              show_plot: bool = True,
                              dpi: int = 150) -> np.ndarray:
    """
    Visualiza los resultados completos del análisis de poda.
    
    Genera una imagen con:
    - RGB original
    - Máscaras detectadas
    - Esqueletos de sarmientos
    - Yemas identificadas
    - Puntos de corte con coordenadas 3D
    - Vectores de orientación
    
    Args:
        results: Diccionario retornado por run_pruning_system()
        save_path: Ruta para guardar imagen (opcional)
        show_plot: Mostrar ventana interactiva
        dpi: Resolución de salida
    
    Returns:
        Imagen visualizada como array numpy
    """
    
    # Extraer datos
    rgb_image = results['rgb_image'].copy()
    masks = results['masks']
    sarmientos = results['sarmientos']
    cut_points = results['cut_points']
    cordon_axis = results['cordon_axis']
    
    # Crear figura con múltiples subplots
    fig = plt.figure(figsize=(16, 10))
    
    # ========== SUBPLOT 1: Imagen RGB Original ==========
    ax1 = plt.subplot(2, 3, 1)
    ax1.imshow(cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB))
    ax1.set_title('Imagen RGB Capturada', fontsize=12, fontweight='bold')
    ax1.axis('off')
    
    # ========== SUBPLOT 2: Máscaras Superpuestas ==========
    ax2 = plt.subplot(2, 3, 2)
    overlay = rgb_image.copy()
    
    # Colorear máscaras
    colors = {
        'cordon': (255, 165, 0),      # Naranja
        'sarmiento_1': (0, 255, 255),  # Cyan
        'sarmiento_2': (255, 0, 255),  # Magenta
        'yema_1': (0, 255, 0),         # Verde
        'yema_2': (0, 255, 0),
        'yema_3': (0, 255, 0),
    }
    
    for name, mask in masks.items():
        color = colors.get(name, (255, 255, 255))
        overlay[mask > 0] = overlay[mask > 0] * 0.5 + np.array(color) * 0.5
    
    ax2.imshow(cv2.cvtColor(overlay.astype(np.uint8), cv2.COLOR_BGR2RGB))
    ax2.set_title('Máscaras de Segmentación', fontsize=12, fontweight='bold')
    ax2.axis('off')
    
    # Leyenda
    legend_elements = [
        mpatches.Patch(color=np.array(colors['cordon'])/255, label='Cordón'),
        mpatches.Patch(color=np.array(colors['sarmiento_1'])/255, label='Sarmientos'),
        mpatches.Patch(color=np.array(colors['yema_1'])/255, label='Yemas')
    ]
    ax2.legend(handles=legend_elements, loc='upper right', fontsize=8)
    
    # ========== SUBPLOT 3: Análisis Geométrico ==========
    ax3 = plt.subplot(2, 3, 3)
    geometric_view = np.ones_like(rgb_image) * 255
    
    # Dibujar cordón
    if len(cordon_axis) > 0:
        for point in cordon_axis:
            cv2.circle(geometric_view, (point[1], point[0]), 2, (255, 165, 0), -1)
    
    # Dibujar sarmientos
    for sarmiento in sarmientos:
        color_sarm = (0, 200, 200)
        
        # Esqueleto
        for point in sarmiento.skeleton_points:
            cv2.circle(geometric_view, (point[1], point[0]), 1, color_sarm, -1)
        
        # Dibujar yemas
        for yema in sarmiento.yemas:
            cv2.circle(geometric_view, yema.position_2d, 8, (0, 255, 0), 2)
            cv2.putText(geometric_view, f"Y{yema.id+1}", 
                       (yema.position_2d[0]+10, yema.position_2d[1]-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 180, 0), 1)
    
    ax3.imshow(cv2.cvtColor(geometric_view, cv2.COLOR_BGR2RGB))
    ax3.set_title('Análisis Geométrico 2D', fontsize=12, fontweight='bold')
    ax3.axis('off')
    
    # ========== SUBPLOT 4: Puntos de Corte Anotados ==========
    ax4 = plt.subplot(2, 3, 4)
    annotated = rgb_image.copy()
    
    for idx, (sarmiento, cut_point) in enumerate(cut_points):
        # Dibujar esqueleto del sarmiento
        for point in sarmiento.skeleton_points:
            cv2.circle(annotated, (point[1], point[0]), 1, (255, 255, 0), -1)
        
        # Dibujar yemas
        for yema in sarmiento.yemas:
            cv2.circle(annotated, yema.position_2d, 8, (0, 255, 0), 2)
        
        # Proyectar punto de corte 3D a 2D (simplificado)
        # En sistema real, usar intrínsecos para proyección precisa
        yema_target = sarmiento.yemas[min(1, len(sarmiento.yemas)-1)]
        
        # Estimar posición 2D del corte (aproximación)
        if len(sarmiento.skeleton_points) > 0:
            # Encontrar punto en esqueleto más cercano a yema objetivo
            distancias = np.linalg.norm(
                sarmiento.skeleton_points - np.array([yema_target.position_2d[1], 
                                                      yema_target.position_2d[0]]), 
                axis=1
            )
            idx_cercano = np.argmin(distancias)
            
            # Moverse hacia adelante en el esqueleto
            offset = min(10, len(sarmiento.skeleton_points) - idx_cercano - 1)
            if idx_cercano + offset < len(sarmiento.skeleton_points):
                cut_2d = sarmiento.skeleton_points[idx_cercano + offset]
            else:
                cut_2d = sarmiento.skeleton_points[-1]
            
            cut_pixel = (cut_2d[1], cut_2d[0])
            
            # Dibujar punto de corte
            cv2.circle(annotated, cut_pixel, 10, (0, 0, 255), -1)
            cv2.circle(annotated, cut_pixel, 12, (255, 255, 255), 2)
            
            # Dibujar flecha indicando dirección
            direction = sarmiento.axis_vector_3d[:2] * 30
            end_point = (int(cut_pixel[0] + direction[0]), 
                        int(cut_pixel[1] + direction[1]))
            cv2.arrowedLine(annotated, cut_pixel, end_point, (255, 0, 0), 2, 
                           tipLength=0.3)
            
            # Anotar coordenadas 3D
            x, y, z = cut_point.position_3d
            text = f"X:{x:.1f} Y:{y:.1f} Z:{z:.1f}cm"
            cv2.putText(annotated, text, 
                       (cut_pixel[0]-80, cut_pixel[1]-20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.putText(annotated, text, 
                       (cut_pixel[0]-81, cut_pixel[1]-21),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
            
            # Estado agronómico
            status_color = {
                'Viable': (0, 255, 0),
                'Débil': (0, 165, 255),
                'Vigoroso': (255, 0, 255)
            }
            color = status_color.get(cut_point.diameter_check, (255, 255, 255))
            cv2.putText(annotated, cut_point.diameter_check,
                       (cut_pixel[0]-30, cut_pixel[1]+30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    ax4.imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
    ax4.set_title('Puntos de Corte Calculados', fontsize=12, fontweight='bold')
    ax4.axis('off')
    
    # ========== SUBPLOT 5: Vista 3D (Proyección) ==========
    ax5 = plt.subplot(2, 3, 5, projection='3d')
    
    # Graficar sarmientos en 3D
    for sarmiento in sarmientos:
        if len(sarmiento.yemas) > 0:
            # Extraer coordenadas 3D de yemas
            yemas_3d = np.array([y.position_3d for y in sarmiento.yemas])
            ax5.scatter(yemas_3d[:, 0], yemas_3d[:, 1], yemas_3d[:, 2], 
                       c='green', marker='o', s=100, label='Yemas' if sarmiento.id == 0 else '')
    
    # Graficar puntos de corte
    for sarmiento, cut_point in cut_points:
        pos = cut_point.position_3d
        ax5.scatter([pos[0]], [pos[1]], [pos[2]], 
                   c='red', marker='X', s=200, edgecolors='white', linewidths=2,
                   label='Corte' if sarmiento.id == 0 else '')
        
        # Vector de orientación
        vec = cut_point.orientation_vector * 5  # Escalar para visualización
        ax5.quiver(pos[0], pos[1], pos[2], vec[0], vec[1], vec[2],
                  color='blue', arrow_length_ratio=0.3, linewidth=2)
    
    ax5.set_xlabel('X (cm)', fontsize=10)
    ax5.set_ylabel('Y (cm)', fontsize=10)
    ax5.set_zlabel('Z (cm)', fontsize=10)
    ax5.set_title('Vista 3D - Espacio de Trabajo', fontsize=12, fontweight='bold')
    ax5.legend(fontsize=8)
    ax5.view_init(elev=20, azim=45)
    
    # ========== SUBPLOT 6: Tabla de Resultados ==========
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('tight')
    ax6.axis('off')
    
    # Crear tabla de datos
    table_data = [['Sarmiento', 'Diám.(cm)', 'Estado', 'Yemas', 'Corte Z(cm)']]
    
    for sarmiento, cut_point in cut_points:
        row = [
            f"S{sarmiento.id}",
            f"{sarmiento.diameter_cm:.2f}",
            cut_point.diameter_check,
            str(len(sarmiento.yemas)),
            f"{cut_point.position_3d[2]:.1f}"
        ]
        table_data.append(row)
    
    table = ax6.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.2, 0.2, 0.2, 0.15, 0.25])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Estilo de encabezado
    for i in range(5):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Colorear filas según estado
    for i, (sarm, cut) in enumerate(cut_points, start=1):
        color = '#e8f5e9' if cut.diameter_check == 'Viable' else \
                '#fff3e0' if cut.diameter_check == 'Débil' else '#fce4ec'
        for j in range(5):
            table[(i, j)].set_facecolor(color)
    
    ax6.set_title('Resumen de Análisis', fontsize=12, fontweight='bold')
    
    # Título general
    fig.suptitle('Sistema de Poda Inteligente - Método Simonit & Sirch\n' + 
                 'Análisis Híbrido 2D/3D con Visión Artificial',
                 fontsize=14, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Guardar si se especifica
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"\n✓ Visualización guardada en: {save_path}")
    
    # Mostrar ventana interactiva
    if show_plot:
        plt.show()
    
    # Convertir figura a imagen numpy
    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    plt.close()
    
    return img


def generate_validation_report(results: Dict, output_dir: str = "validation_output"):
    """
    Genera un reporte completo para validación de tesis.
    
    Crea múltiples visualizaciones y archivos de datos:
    - Imagen principal de resultados
    - Archivo CSV con métricas
    - Gráficos individuales por sarmiento
    
    Args:
        results: Diccionario retornado por run_pruning_system()
        output_dir: Directorio de salida
    """
    import os
    import csv
    
    # Crear directorio si no existe
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Visualización principal
    main_viz_path = os.path.join(output_dir, "resultado_completo.png")
    visualize_pruning_results(results, save_path=main_viz_path, show_plot=False)
    
    # 2. Exportar datos a CSV
    csv_path = os.path.join(output_dir, "datos_poda.csv")
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Sarmiento_ID', 'Diametro_cm', 'Estado_Agronomico', 
                        'Num_Yemas', 'Corte_X_cm', 'Corte_Y_cm', 'Corte_Z_cm',
                        'Vector_X', 'Vector_Y', 'Vector_Z', 'Distancia_Distal_cm'])
        
        for sarmiento, cut_point in results['cut_points']:
            writer.writerow([
                sarmiento.id,
                f"{sarmiento.diameter_cm:.3f}",
                cut_point.diameter_check,
                len(sarmiento.yemas),
                f"{cut_point.position_3d[0]:.2f}",
                f"{cut_point.position_3d[1]:.2f}",
                f"{cut_point.position_3d[2]:.2f}",
                f"{cut_point.orientation_vector[0]:.4f}",
                f"{cut_point.orientation_vector[1]:.4f}",
                f"{cut_point.orientation_vector[2]:.4f}",
                f"{cut_point.distance_from_bud:.2f}"
            ])
    
    print(f"\n✓ Datos exportados a: {csv_path}")
    
    # 3. Gráfico de distribución de diámetros
    diameters = [s.diameter_cm for s, _ in results['cut_points']]
    
    plt.figure(figsize=(10, 6))
    plt.hist(diameters, bins=10, color='#4CAF50', edgecolor='black', alpha=0.7)
    plt.axvline(x=0.7, color='red', linestyle='--', label='Umbral Débil')
    plt.axvline(x=1.2, color='orange', linestyle='--', label='Umbral Vigoroso')
    plt.xlabel('Diámetro (cm)', fontsize=12)
    plt.ylabel('Frecuencia', fontsize=12)
    plt.title('Distribución de Diámetros de Sarmientos', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    dist_path = os.path.join(output_dir, "distribucion_diametros.png")
    plt.savefig(dist_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Gráfico de distribución guardado en: {dist_path}")
    print(f"\n{'='*60}")
    print(f"REPORTE DE VALIDACIÓN GENERADO EN: {output_dir}")
    print(f"{'='*60}\n")


def quick_visualize(results: Dict):
    """
    Visualización rápida para debugging durante desarrollo.
    
    Muestra solo imagen anotada con puntos de corte.
    """
    annotated = results['rgb_image'].copy()
    
    for sarmiento, cut_point in results['cut_points']:
        # Esqueleto
        for point in sarmiento.skeleton_points:
            cv2.circle(annotated, (point[1], point[0]), 1, (255, 255, 0), -1)
        
        # Yemas
        for yema in sarmiento.yemas:
            cv2.circle(annotated, yema.position_2d, 8, (0, 255, 0), 2)
        
        # Punto de corte (estimado en 2D)
        if len(sarmiento.yemas) > 0:
            yema_target = sarmiento.yemas[min(1, len(sarmiento.yemas)-1)]
            cv2.circle(annotated, yema_target.position_2d, 12, (0, 0, 255), -1)
            
            x, y, z = cut_point.position_3d
            text = f"({x:.1f}, {y:.1f}, {z:.1f})"
            cv2.putText(annotated, text, 
                       (yema_target.position_2d[0]-50, yema_target.position_2d[1]-20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    
    cv2.imshow('Resultados Poda', annotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ============================================================================
# EJEMPLO DE USO
# ============================================================================

if __name__ == "__main__":
    # Este archivo debe importarse desde el módulo principal
    print("Módulo de visualización cargado correctamente.")
    print("\nUso desde código principal:")
    print("    from visualization.visualizer import visualize_pruning_results")
    print("    results = run_pruning_system()")
    print("    visualize_pruning_results(results, save_path='output.png')")