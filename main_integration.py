"""
SISTEMA DE PODA INTELIGENTE - INTEGRACIÓN COMPLETA
===================================================

Script principal para ejecutar el sistema con datasets reales o sintéticos.

Uso:
    # Con dataset real:
    python main.py --dataset path/to/dataset --image 0001 --visualize
    
    # Con datos sintéticos:
    python main.py --synthetic --visualize
    
    # Procesamiento batch:
    python main.py --dataset path/to/dataset --batch --output results/
"""

import argparse
import sys
import os
from pathlib import Path

# Importar módulos del sistema
# Asegúrate de que estos archivos estén en el mismo directorio o en PYTHONPATH
from pruning_system import (
    run_pruning_system,
    process_cane,
    associate_buds_to_cane,
    AgronomicBrain,
    CameraIntrinsics
)

from dataset_loader import RealSenseDatasetLoader, COCOMaskLoader
from visualization.visualizer import (
    visualize_pruning_results,
    generate_validation_report,
    quick_visualize
)


def run_with_real_dataset(dataset_path: str, image_id: str, 
                          visualize: bool = True, save_output: bool = True):
    """
    Ejecuta el sistema con una imagen real del dataset.
    
    Args:
        dataset_path: Ruta al dataset
        image_id: ID de la imagen a procesar
        visualize: Si mostrar visualización
        save_output: Si guardar resultados
    """
    print("="*70)
    print("PROCESANDO IMAGEN REAL DEL DATASET")
    print("="*70)
    
    # 1. Cargar dataset
    loader = RealSenseDatasetLoader(dataset_path)
    rgb_image, depth_map, masks, intrinsics = loader.load_image(image_id)
    
    print(f"\n✓ Imagen cargada: {image_id}")
    print(f"  - Resolución RGB: {rgb_image.shape}")
    print(f"  - Máscaras detectadas: {list(masks.keys())}")
    
    # 2. Procesar estructuras
    print("\n" + "-"*70)
    print("PROCESAMIENTO DE ESTRUCTURAS")
    print("-"*70)
    
    # Identificar sarmientos y yemas en las máscaras
    sarmientos = []
    yemas_masks = {}
    
    for mask_name, mask in masks.items():
        if 'sarmiento' in mask_name.lower() or 'cane' in mask_name.lower():
            cane_id = len(sarmientos)
            print(f"\n→ Procesando {mask_name}...")
            
            try:
                sarmiento = process_cane(mask, depth_map, intrinsics, cane_id)
                print(f"  ✓ Diámetro: {sarmiento.diameter_cm:.2f} cm")
                print(f"  ✓ Profundidad: {sarmiento.depth_mean:.1f} mm")
                sarmientos.append(sarmiento)
            except Exception as e:
                print(f"  ✗ Error: {e}")
        
        elif 'yema' in mask_name.lower() or 'bud' in mask_name.lower():
            yemas_masks[mask_name] = mask
    
    # 3. Asociar yemas a sarmientos
    print(f"\n→ Asociando {len(yemas_masks)} yemas a sarmientos...")
    
    for sarmiento in sarmientos:
        sarmiento.yemas = associate_buds_to_cane(
            yemas_masks, sarmiento, depth_map, intrinsics
        )
        print(f"  Sarmiento {sarmiento.id}: {len(sarmiento.yemas)} yemas asociadas")
    
    # 4. Calcular puntos de corte
    print("\n" + "-"*70)
    print("CÁLCULO DE PUNTOS DE CORTE")
    print("-"*70)
    
    brain = AgronomicBrain()
    cut_points = []
    
    for sarmiento in sarmientos:
        if not sarmiento.yemas:
            print(f"\nSarmiento {sarmiento.id}: Sin yemas (omitido)")
            continue
        
        try:
            cut_point = brain.calculate_cut_point(sarmiento, target_bud_index=1)
            cut_points.append((sarmiento, cut_point))
            
            print(f"\nSarmiento {sarmiento.id}:")
            print(f"  Estado: {cut_point.diameter_check}")
            print(f"  Diámetro: {sarmiento.diameter_cm:.2f} cm")
            print(f"  Punto de corte: X={cut_point.position_3d[0]:.2f}, "
                  f"Y={cut_point.position_3d[1]:.2f}, Z={cut_point.position_3d[2]:.2f} cm")
            print(f"  Vector: [{cut_point.orientation_vector[0]:.3f}, "
                  f"{cut_point.orientation_vector[1]:.3f}, "
                  f"{cut_point.orientation_vector[2]:.3f}]")
        
        except Exception as e:
            print(f"\nSarmiento {sarmiento.id}: Error - {e}")
    
    # 5. Construir diccionario de resultados
    results = {
        'rgb_image': rgb_image,
        'depth_map': depth_map,
        'masks': masks,
        'intrinsics': intrinsics,
        'sarmientos': sarmientos,
        'cut_points': cut_points,
        'cordon_axis': []  # No procesado en este ejemplo
    }
    
    # 6. Visualización
    if visualize:
        print("\n" + "="*70)
        print("GENERANDO VISUALIZACIÓN")
        print("="*70)
        
        output_path = None
        if save_output:
            output_dir = Path("output")
            output_dir.mkdir(exist_ok=True)
            output_path = output_dir / f"resultado_{image_id}.png"
        
        visualize_pruning_results(results, save_path=output_path, show_plot=True)
    
    # 7. Generar reporte completo
    if save_output:
        print("\n" + "="*70)
        print("GENERANDO REPORTE DE VALIDACIÓN")
        print("="*70)
        
        output_dir = Path("validation_output") / image_id
        generate_validation_report(results, output_dir=str(output_dir))
    
    return results


def run_batch_processing(dataset_path: str, output_dir: str = "batch_results"):
    """
    Procesa múltiples imágenes del dataset en modo batch.
    
    Args:
        dataset_path: Ruta al dataset
        output_dir: Directorio para guardar resultados
    """
    print("="*70)
    print("PROCESAMIENTO BATCH")
    print("="*70)
    
    loader = RealSenseDatasetLoader(dataset_path)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Crear archivo CSV consolidado
    import csv
    csv_path = output_path / "resultados_batch.csv"
    
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            'Image_ID', 'Sarmiento_ID', 'Diametro_cm', 'Estado',
            'Num_Yemas', 'Corte_X', 'Corte_Y', 'Corte_Z',
            'Vector_X', 'Vector_Y', 'Vector_Z'
        ])
        
        # Procesar cada imagen
        for idx, image_id in enumerate(loader.available_images, 1):
            print(f"\n[{idx}/{len(loader.available_images)}] Procesando {image_id}...")
            
            try:
                results = run_with_real_dataset(
                    dataset_path, image_id,
                    visualize=False, save_output=False
                )
                
                # Guardar visualización individual
                vis_path = output_path / f"{image_id}_resultado.png"
                visualize_pruning_results(results, save_path=str(vis_path), 
                                        show_plot=False, dpi=100)
                
                # Escribir datos al CSV
                for sarmiento, cut_point in results['cut_points']:
                    writer.writerow([
                        image_id,
                        sarmiento.id,
                        f"{sarmiento.diameter_cm:.3f}",
                        cut_point.diameter_check,
                        len(sarmiento.yemas),
                        f"{cut_point.position_3d[0]:.2f}",
                        f"{cut_point.position_3d[1]:.2f}",
                        f"{cut_point.position_3d[2]:.2f}",
                        f"{cut_point.orientation_vector[0]:.4f}",
                        f"{cut_point.orientation_vector[1]:.4f}",
                        f"{cut_point.orientation_vector[2]:.4f}"
                    ])
                
                print(f"  ✓ Completado ({len(results['cut_points'])} sarmientos)")
            
            except Exception as e:
                print(f"  ✗ Error: {e}")
    
    print("\n" + "="*70)
    print(f"BATCH COMPLETADO")
    print(f"Resultados guardados en: {output_path}")
    print(f"CSV consolidado: {csv_path}")
    print("="*70)


def main():
    """Función principal con CLI"""
    parser = argparse.ArgumentParser(
        description='Sistema de Poda Inteligente para Robot Podador de Vides',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  
  # Procesar imagen real del dataset:
  python main.py --dataset path/to/dataset --image 0001 --visualize
  
  # Procesar con datos sintéticos:
  python main.py --synthetic --visualize
  
  # Procesamiento batch de todo el dataset:
  python main.py --dataset path/to/dataset --batch --output results/
  
  # Solo calcular sin visualizar:
  python main.py --dataset path/to/dataset --image 0001 --no-visualize
        """
    )
    
    parser.add_argument('--dataset', type=str, 
                       help='Ruta al dataset RGBD')
    
    parser.add_argument('--image', type=str,
                       help='ID de la imagen a procesar (ej: 0001)')
    
    parser.add_argument('--synthetic', action='store_true',
                       help='Usar datos sintéticos en lugar de dataset real')
    
    parser.add_argument('--batch', action='store_true',
                       help='Procesar todas las imágenes del dataset')
    
    parser.add_argument('--visualize', dest='visualize', action='store_true',
                       help='Mostrar visualización (default)')
    
    parser.add_argument('--no-visualize', dest='visualize', action='store_false',
                       help='No mostrar visualización')
    
    parser.add_argument('--output', type=str, default='output',
                       help='Directorio de salida (default: output)')
    
    parser.add_argument('--report', action='store_true',
                       help='Generar reporte completo de validación')
    
    parser.set_defaults(visualize=True)
    
    args = parser.parse_args()
    
    # Validar argumentos
    if not args.synthetic and not args.dataset:
        parser.error("Debes especificar --dataset o --synthetic")
    
    # Ejecutar según modo
    try:
        if args.synthetic:
            # Modo sintético (del módulo original)
            print("Ejecutando con datos sintéticos...\n")
            results = run_pruning_system()
            
            if args.visualize:
                from visualization.visualizer import visualize_pruning_results
                visualize_pruning_results(results, 
                                        save_path=f"{args.output}/sintetico.png",
                                        show_plot=True)
            
            if args.report:
                from visualization.visualizer import generate_validation_report
                generate_validation_report(results, 
                                         output_dir=f"{args.output}/reporte")
        
        elif args.batch:
            # Modo batch
            run_batch_processing(args.dataset, args.output)
        
        else:
            # Modo imagen individual
            if not args.image:
                # Cargar primera imagen disponible
                loader = RealSenseDatasetLoader(args.dataset)
                args.image = loader.available_images[0]
                print(f"No se especificó --image, usando primera disponible: {args.image}")
            
            results = run_with_real_dataset(
                args.dataset, 
                args.image,
                visualize=args.visualize,
                save_output=args.report
            )
        
        print("\n✓ Ejecución completada exitosamente")
        return 0
    
    except KeyboardInterrupt:
        print("\n\n⚠ Ejecución cancelada por usuario")
        return 1
    
    except Exception as e:
        print(f"\n✗ Error fatal: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())