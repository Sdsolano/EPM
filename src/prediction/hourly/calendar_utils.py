"""
Utilidades de Calendario para Clasificación de Días

Usa la librería 'holidays' para gestionar festivos de Colombia automáticamente
y clasifica días según múltiples criterios relevantes para patrones de demanda.
"""

import pandas as pd
import holidays
from typing import Literal
from datetime import datetime


class CalendarClassifier:
    """
    Clasificador de días basado en calendario, festivos y temporadas.

    Usa la librería 'holidays' para obtener festivos colombianos automáticamente.
    """

    def __init__(self, country: str = 'CO', years_range: tuple = (2017, 2030)):
        """
        Inicializa el clasificador de calendario.

        Args:
            country: Código ISO del país (por defecto 'CO' = Colombia)
            years_range: Tupla (año_inicio, año_fin) para cargar festivos
        """
        # Cargar festivos de Colombia usando la librería holidays
        self.holidays = holidays.country_holidays(
            country,
            years=range(years_range[0], years_range[1] + 1)
        )

    def is_holiday(self, date: pd.Timestamp) -> bool:
        """Verifica si una fecha es festivo en Colombia."""
        return date.date() in self.holidays

    def get_holiday_name(self, date: pd.Timestamp) -> str:
        """Obtiene el nombre del festivo si aplica."""
        return self.holidays.get(date.date(), "No festivo")

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
        """
        # Festivos con patrones muy diferentes
        special_holidays = [
            'Año Nuevo',
            'Navidad',
            'Día del Trabajo',
            'Día de la Independencia',
            'Inmaculada Concepción',
        ]

        if self.is_holiday(date):
            holiday_name = self.get_holiday_name(date)
            # Verificación flexible (por si nombres no coinciden exactamente)
            return any(special in holiday_name for special in special_holidays)

        return False


# ============== FUNCIONES DE UTILIDAD ==============

def classify_dataframe_dates(df: pd.DataFrame, date_column: str = 'FECHA') -> pd.DataFrame:
    """
    Agrega columnas de clasificación de calendario a un DataFrame.

    Args:
        df: DataFrame con columna de fechas
        date_column: Nombre de la columna de fechas

    Returns:
        DataFrame con columnas adicionales de clasificación
    """
    classifier = CalendarClassifier()
    df = df.copy()

    # Asegurar que la columna es datetime
    df[date_column] = pd.to_datetime(df[date_column])

    # Aplicar clasificación
    classifications = df[date_column].apply(classifier.get_full_classification)

    # Convertir a DataFrame
    class_df = pd.DataFrame(classifications.tolist())

    # Concatenar (evitando duplicar la columna fecha)
    return pd.concat([df, class_df.drop('fecha', axis=1)], axis=1)


# ============== EJEMPLO DE USO ==============

if __name__ == "__main__":
    # Crear clasificador
    classifier = CalendarClassifier()

    # Probar algunas fechas
    fechas_prueba = [
        "2024-01-01",  # Año Nuevo (festivo)
        "2024-05-01",  # Día del Trabajo (festivo)
        "2024-07-20",  # Independencia (festivo)
        "2024-12-25",  # Navidad (festivo)
        "2024-03-15",  # Viernes normal
        "2024-04-20",  # Sábado normal
    ]

    print("=" * 80)
    print("CLASIFICACIÓN DE DÍAS - SISTEMA EPM")
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
