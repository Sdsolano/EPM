"""
Script para limpiar logs antiguos y mantener solo los más recientes

Uso:
    python scripts/cleanup_old_logs.py
"""

import os
from pathlib import Path
from datetime import datetime
import shutil


def cleanup_old_logs(dry_run: bool = False):
    """
    Limpia logs antiguos del directorio logs/

    Args:
        dry_run: Si True, solo muestra qué archivos se eliminarían sin borrarlos
    """
    logs_dir = Path('logs')

    if not logs_dir.exists():
        print("ERROR: Directorio logs/ no existe")
        return

    print("="*80)
    print("LIMPIEZA DE LOGS ANTIGUOS")
    print("="*80)

    # Patrones de archivos a limpiar (con timestamp)
    patterns_to_clean = [
        '*_20*.log',  # Logs con fecha
        'pipeline_execution_*_20*.json',  # Reportes de ejecución con timestamp
        'training_results_20*.json',  # Resultados de entrenamiento con timestamp
    ]

    # Archivos a MANTENER (nombres fijos sin timestamp)
    files_to_keep = [
        '*_latest.log',
        'pipeline_execution_*_latest.json',
        'training_results.json',
    ]

    total_removed = 0
    total_size = 0

    for pattern in patterns_to_clean:
        files = list(logs_dir.glob(pattern))

        print(f"\nPatron: {pattern}")
        print(f"   Archivos encontrados: {len(files)}")

        for file in files:
            # Verificar que no sea un archivo _latest
            if '_latest' in file.name or not any(c in file.name for c in ['_202', '_201']):
                continue

            size = file.stat().st_size
            total_size += size

            if dry_run:
                print(f"   [DRY RUN] Eliminaria: {file.name} ({size/1024:.1f} KB)")
            else:
                try:
                    file.unlink()
                    print(f"   OK Eliminado: {file.name} ({size/1024:.1f} KB)")
                    total_removed += 1
                except Exception as e:
                    print(f"   ERROR al eliminar {file.name}: {e}")

    print("\n" + "="*80)
    print("RESUMEN")
    print("="*80)
    if dry_run:
        print(f"Archivos que se eliminarían: {total_removed}")
    else:
        print(f"Archivos eliminados: {total_removed}")
    print(f"Espacio liberado: {total_size/1024/1024:.2f} MB")

    # Mostrar archivos que permanecen
    print("\n📌 Archivos que permanecen:")
    remaining = []
    for pattern in files_to_keep:
        remaining.extend(logs_dir.glob(pattern))

    if remaining:
        for file in remaining:
            size = file.stat().st_size
            print(f"   ✓ {file.name} ({size/1024:.1f} KB)")
    else:
        print("   (No hay archivos _latest todavía)")

    print("\n" + "="*80)


def create_backup(backup_dir: str = 'logs_backup'):
    """
    Crea backup de todos los logs antes de limpiar

    Args:
        backup_dir: Directorio donde guardar el backup
    """
    logs_dir = Path('logs')
    backup_path = Path(backup_dir)

    if not logs_dir.exists():
        print("❌ Directorio logs/ no existe")
        return

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_full_path = backup_path / f"logs_backup_{timestamp}"

    print(f"📦 Creando backup en: {backup_full_path}")

    try:
        shutil.copytree(logs_dir, backup_full_path)
        print(f"✓ Backup creado exitosamente")
        print(f"  Ubicación: {backup_full_path.absolute()}")
        return backup_full_path
    except Exception as e:
        print(f"❌ Error al crear backup: {e}")
        return None


if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser(description='Limpia logs antiguos del proyecto')
    parser.add_argument('--dry-run', action='store_true',
                        help='Muestra qué se eliminaría sin borrar archivos')
    parser.add_argument('--backup', action='store_true',
                        help='Crea backup antes de limpiar')
    parser.add_argument('--backup-dir', default='logs_backup',
                        help='Directorio para el backup (default: logs_backup)')

    args = parser.parse_args()

    # Cambiar al directorio raíz del proyecto
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    os.chdir(project_root)

    print(f"Directorio de trabajo: {os.getcwd()}\n")

    # Crear backup si se solicitó
    if args.backup and not args.dry_run:
        backup_path = create_backup(args.backup_dir)
        if not backup_path:
            print("\n❌ Error al crear backup. Abortando limpieza.")
            sys.exit(1)
        print()

    # Ejecutar limpieza
    cleanup_old_logs(dry_run=args.dry_run)

    if args.dry_run:
        print("\n💡 Para ejecutar la limpieza real, ejecuta sin --dry-run:")
        print("   python scripts/cleanup_old_logs.py")
        print("\n💡 Para crear backup antes de limpiar:")
        print("   python scripts/cleanup_old_logs.py --backup")
