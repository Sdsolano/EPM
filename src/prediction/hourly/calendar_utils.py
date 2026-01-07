"""
Utilidades de Calendario para Clasificación de Días

Usa la API de pronosticos.jmdatalabs.co para obtener festivos por UCP.
Clasifica días según múltiples criterios relevantes para patrones de demanda.
"""

import pandas as pd
from typing import Literal, Optional
from datetime import datetime
import logging

# Importar cliente de festivos API
try:
    from ..festivos_api import FestivosAPIClient
except ImportError:
    try:
        from src.prediction.festivos_api import FestivosAPIClient
    except ImportError:
        FestivosAPIClient = None

logger = logging.getLogger(__name__)


class CalendarClassifier:
    """
    Clasificador de días basado en calendario, festivos y temporadas.

    Usa la API de pronosticos.jmdatalabs.co para obtener festivos por UCP.
    
    IMPORTANTE: El caché de festivos es solo por instancia (en memoria).
    Cada ejecución nueva crea una nueva instancia, por lo que siempre se llama
    a la API al menos una vez por año necesario, garantizando datos frescos.
    """

    def __init__(self, ucp: str = 'Antioquia', country: str = None, years_range: tuple = None):
        """
        Inicializa el clasificador de calendario.

        Args:
            ucp: Nombre del UCP (ej: 'Antioquia', 'Atlantico', 'Oriente'). Default: 'Antioquia'
            country: DEPRECATED - Se mantiene por compatibilidad pero se ignora
            years_range: DEPRECATED - Se mantiene por compatibilidad pero se ignora
        """
        self.ucp = ucp
        
        # Inicializar cliente de festivos API
        if FestivosAPIClient is None:
            logger.warning("⚠ FestivosAPIClient no disponible. Los festivos no funcionarán correctamente.")
            self.festivos_client = None
            self.festivos_cache = set()  # Caché vacío si no hay cliente
            self.loaded_years = set()  # Rastrear años ya cargados
        else:
            self.festivos_client = FestivosAPIClient()
            self.festivos_cache = set()  # Caché para evitar múltiples llamadas a la API
            self.loaded_years = set()  # Rastrear años ya cargados para evitar llamadas repetidas
        
        # Mapeo básico de fechas a nombres de festivos (para get_holiday_name)
        # Se usa cuando la API no proporciona nombres
        self._festivo_names_map = {
            '01-01': 'Año Nuevo',
            '01-06': 'Día de los Reyes Magos',
            '03-19': 'Día de San José',
            '04-17': 'Jueves Santo',
            '04-18': 'Viernes Santo',
            '05-01': 'Día del Trabajo',
            '06-02': 'Ascensión del Señor',
            '06-23': 'Corpus Christi',
            '06-30': 'Sagrado Corazón de Jesús',
            '07-20': 'Día de la Independencia',
            '08-07': 'Batalla de Boyacá',
            '08-17': 'Asunción de la Virgen',
            '10-12': 'Día de la Raza',
            '11-02': 'Día de Todos los Santos',
            '11-16': 'Independencia de Cartagena',
            '12-08': 'Inmaculada Concepción',
            '12-25': 'Navidad',
        }

    def _load_festivos_for_year(self, year: int) -> None:
        """
        Carga festivos para un año específico desde la API y los agrega al caché.
        
        Args:
            year: Año a cargar
        """
        if self.festivos_client is None:
            return
        
        # Si ya cargamos este año, no hacer otra llamada
        if year in self.loaded_years:
            logger.debug(f"Festivos para año {year} ya están en caché, omitiendo llamada a API")
            return
        
        try:
            start_date = f"{year}-01-01"
            end_date = f"{year}-12-31"
            
            logger.debug(f"Cargando festivos desde API para {self.ucp} año {year}...")
            festivos = self.festivos_client.get_festivos(start_date, end_date, self.ucp)
            self.festivos_cache.update(festivos)
            self.loaded_years.add(year)  # Marcar año como cargado
            
            logger.debug(f"Cargados {len(festivos)} festivos para {self.ucp} en {year} (total en caché: {len(self.festivos_cache)})")
        except Exception as e:
            logger.warning(f"Error al cargar festivos para {year}: {e}")
    
    def preload_years(self, years: list) -> None:
        """
        Pre-carga festivos para múltiples años de una vez.
        Útil para evitar múltiples llamadas a la API.
        
        Args:
            years: Lista de años a cargar (ej: [2024, 2025, 2026])
        """
        for year in years:
            self._load_festivos_for_year(year)

    def is_holiday(self, date: pd.Timestamp) -> bool:
        """
        Verifica si una fecha es festivo para el UCP configurado.
        
        Args:
            date: Fecha a verificar
        
        Returns:
            True si es festivo, False en caso contrario
        """
        fecha_str = date.strftime('%Y-%m-%d')
        
        # Verificar caché primero (más rápido)
        if fecha_str in self.festivos_cache:
            return True
        
        # Si no está en caché y no hemos cargado este año aún, cargar festivos
        # pero solo si no lo hemos cargado antes (evita llamadas repetidas)
        if self.festivos_client is not None:
            year = date.year
            if year not in self.loaded_years:
                self._load_festivos_for_year(year)
            
            # Verificar de nuevo después de cargar (o si ya estaba cargado)
            return fecha_str in self.festivos_cache
        
        # Si no hay cliente disponible, retornar False
        return False

    def get_holiday_name(self, date: pd.Timestamp) -> str:
        """
        Obtiene el nombre del festivo si aplica.
        
        Nota: Como la API no proporciona nombres, se usa un mapeo básico.
        
        Args:
            date: Fecha a verificar
        
        Returns:
            Nombre del festivo o "No festivo"
        """
        if not self.is_holiday(date):
            return "No festivo"
        
        # Intentar obtener nombre del mapeo
        month_day = date.strftime('%m-%d')
        nombre = self._festivo_names_map.get(month_day, "Festivo")
        
        return nombre

    def is_weekend(self, date: pd.Timestamp) -> bool:
        """Verifica si es fin de semana (sábado=5, domingo=6)."""
        return date.dayofweek >= 5

    def get_day_type(self, date: pd.Timestamp) -> Literal["laboral", "festivo", "fin_de_semana"]:
        """
        Clasifica el día en 3 categorías principales.

        Returns:
            - "festivo": Festivos oficiales de Colombia
            - "fin_de_semana": Sábados y domingos (no festivos)
            - "laboral": Lunes a viernes (no festivos)
        """
        if self.is_holiday(date):
            return "festivo"
        elif self.is_weekend(date):
            return "fin_de_semana"
        else:
            return "laboral"

    def get_season(self, date: pd.Timestamp) -> Literal["lluviosa", "seca"]:
        """
        Determina la temporada climática en Colombia (Antioquia).

        Temporadas aproximadas:
        - Lluviosa: Abril-Mayo, Septiembre-Noviembre
        - Seca: Diciembre-Marzo, Junio-Agosto

        Returns:
            "lluviosa" o "seca"
        """
        month = date.month

        # Temporada lluviosa: Abril-Mayo (4-5) y Septiembre-Noviembre (9-11)
        if month in [4, 5, 9, 10, 11]:
            return "lluviosa"
        else:
            return "seca"

    def get_day_of_week_name(self, date: pd.Timestamp) -> str:
        """Obtiene el nombre del día de la semana en español."""
        dias = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]
        return dias[date.dayofweek]

    def get_full_classification(self, date: pd.Timestamp) -> dict:
        """
        Retorna clasificación completa del día.

        Returns:
            dict con todas las características del día
        """
        return {
            'fecha': date,
            'dia_semana': self.get_day_of_week_name(date),
            'dia_semana_num': date.dayofweek,  # 0=Lunes, 6=Domingo
            'tipo_dia': self.get_day_type(date),
            'es_festivo': self.is_holiday(date),
            'nombre_festivo': self.get_holiday_name(date),
            'es_fin_semana': self.is_weekend(date),
            'temporada': self.get_season(date),
            'mes': date.month,
            'trimestre': date.quarter,
        }

    def is_special_day(self, date: pd.Timestamp) -> bool:
        """
        Identifica si es un día con patrón especial de demanda.

        Días especiales: festivos importantes que tienen patrones únicos
        (Navidad, Año Nuevo, Día del Trabajo, etc.)
        Incluye el 24 de diciembre (Nochebuena) aunque no sea festivo oficial.
        """
        month_day = date.strftime('%m-%d')
        
        # Días muy especiales (incluye 24 dic aunque no sea festivo oficial)
        very_special_dates = ['01-01', '12-24', '12-25', '12-08']
        if month_day in very_special_dates:
            return True
        
        # Otros festivos con patrones especiales
        if not self.is_holiday(date):
            return False
        
        # Festivos con patrones muy diferentes (verificar por fecha mm-dd)
        special_dates = ['05-01', '07-20']
        
        return month_day in special_dates


# ============== FUNCIONES DE UTILIDAD ==============

def classify_dataframe_dates(df: pd.DataFrame, date_column: str = 'FECHA', ucp: str = 'Antioquia') -> pd.DataFrame:
    """
    Agrega columnas de clasificación de calendario a un DataFrame.

    Args:
        df: DataFrame con columna de fechas
        date_column: Nombre de la columna de fechas
        ucp: Nombre del UCP para obtener festivos (default: 'Antioquia')

    Returns:
        DataFrame con columnas adicionales de clasificación
    """
    classifier = CalendarClassifier(ucp=ucp)
    df = df.copy()

    # Asegurar que la columna es datetime
    df[date_column] = pd.to_datetime(df[date_column])

    # Pre-cargar festivos para todos los años en el DataFrame (optimización)
    if classifier.festivos_client is not None:
        years = df[date_column].dt.year.unique()
        for year in years:
            classifier._load_festivos_for_year(year)

    # Aplicar clasificación
    classifications = df[date_column].apply(classifier.get_full_classification)

    # Convertir a DataFrame
    class_df = pd.DataFrame(classifications.tolist())

    # Concatenar (evitando duplicar la columna fecha)
    return pd.concat([df, class_df.drop('fecha', axis=1)], axis=1)


# ============== EJEMPLO DE USO ==============

if __name__ == "__main__":
    # Crear clasificador con UCP
    ucp = 'Antioquia'  # Puede cambiarse a 'Atlantico', 'Oriente', etc.
    classifier = CalendarClassifier(ucp=ucp)

    # Probar algunas fechas
    fechas_prueba = [
        "2026-01-01",  # Año Nuevo (festivo)
        "2026-05-01",  # Día del Trabajo (festivo)
        "2026-07-20",  # Independencia (festivo)
        "2026-12-25",  # Navidad (festivo)
        "2026-03-15",  # Viernes normal
        "2026-04-20",  # Sábado normal
    ]

    print("=" * 80)
    print(f"CLASIFICACIÓN DE DÍAS - SISTEMA EPM (UCP: {ucp})")
    print("=" * 80)

    for fecha_str in fechas_prueba:
        fecha = pd.to_datetime(fecha_str)
        info = classifier.get_full_classification(fecha)

        print(f"\n📅 {fecha_str}")
        print(f"   Día: {info['dia_semana']}")
        print(f"   Tipo: {info['tipo_dia']}")
        print(f"   Festivo: {info['es_festivo']} - {info['nombre_festivo']}")
        print(f"   Temporada: {info['temporada']}")
        print(f"   Especial: {classifier.is_special_day(fecha)}")

    print("\n" + "=" * 80)
    print("✓ Clasificador de calendario funcionando correctamente")
