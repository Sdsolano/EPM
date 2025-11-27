"""Configuración del sistema EPM"""

from pathlib import Path

# Importar settings (será creado en el siguiente paso)
try:
    from .settings import *
except ImportError:
    pass
