"""
Utilidades para Manejo de Semana Santa y Festivos Móviles

Este módulo proporciona funciones para detectar y mapear días de Semana Santa
a sus equivalentes históricos, resolviendo el problema de festivos móviles
que caen en fechas diferentes cada año.
"""

from datetime import datetime, timedelta, date
from typing import Optional, Literal, Tuple
from dateutil.easter import easter
import logging

logger = logging.getLogger(__name__)


# Tipos de días de Semana Santa que reconocemos
EasterDayType = Literal[
    'Domingo de Ramos',
    'Lunes Santo',
    'Martes Santo',
    'Miércoles Santo',
    'Jueves Santo',
    'Viernes Santo',
    'Sábado Santo',
    'Domingo de Resurrección',
    'Lunes de Pascua'
]


def get_easter_date(year: int) -> date:
    """
    Calcula la fecha de Domingo de Resurrección (Easter) para un año dado.

    Args:
        year: Año para el cual calcular Easter

    Returns:
        Fecha del Domingo de Resurrección

    Example:
        >>> get_easter_date(2026)
        datetime.date(2026, 4, 5)  # Domingo de Resurrección 2026
        >>> get_easter_date(2025)
        datetime.date(2025, 4, 20)  # Domingo de Resurrección 2025
    """
    return easter(year)


def get_easter_week_dates(year: int) -> dict:
    """
    Obtiene todas las fechas de Semana Santa para un año dado.

    Args:
        year: Año para el cual obtener las fechas

    Returns:
        Diccionario con tipo de día → fecha

    Example:
        >>> dates = get_easter_week_dates(2026)
        >>> dates['Jueves Santo']
        datetime.date(2026, 4, 2)
    """
    easter_sunday = get_easter_date(year)

    return {
        'Domingo de Ramos': easter_sunday - timedelta(days=7),
        'Lunes Santo': easter_sunday - timedelta(days=6),
        'Martes Santo': easter_sunday - timedelta(days=5),
        'Miércoles Santo': easter_sunday - timedelta(days=4),
        'Jueves Santo': easter_sunday - timedelta(days=3),
        'Viernes Santo': easter_sunday - timedelta(days=2),
        'Sábado Santo': easter_sunday - timedelta(days=1),
        'Domingo de Resurrección': easter_sunday,
        'Lunes de Pascua': easter_sunday + timedelta(days=1)
    }


def detect_easter_day_type(fecha: datetime) -> Optional[EasterDayType]:
    """
    Detecta si una fecha es parte de Semana Santa y retorna el tipo de día.

    Args:
        fecha: Fecha a verificar (datetime o date)

    Returns:
        Tipo de día de Semana Santa, o None si no es parte de Semana Santa

    Example:
        >>> detect_easter_day_type(datetime(2026, 4, 2))
        'Jueves Santo'
        >>> detect_easter_day_type(datetime(2026, 4, 10))
        None
    """
    # Convertir a date si es datetime
    if isinstance(fecha, datetime):
        fecha = fecha.date()

    # Obtener fechas de Semana Santa para el año de la fecha
    easter_dates = get_easter_week_dates(fecha.year)

    # Buscar si la fecha coincide con algún día de Semana Santa
    for day_type, day_date in easter_dates.items():
        if fecha == day_date:
            return day_type

    return None


def get_historical_easter_date(
    fecha: datetime,
    years_back: int = 1
) -> Optional[datetime]:
    """
    Encuentra la fecha equivalente de Semana Santa en años anteriores.

    Si la fecha es parte de Semana Santa, retorna la fecha del mismo tipo de día
    en años anteriores (ej: Jueves Santo 2026 → Jueves Santo 2025).

    Si la fecha NO es parte de Semana Santa, retorna None.

    Args:
        fecha: Fecha a buscar equivalente histórico
        years_back: Cuántos años atrás buscar (default: 1)

    Returns:
        Fecha equivalente en el año anterior, o None si no es Semana Santa

    Example:
        >>> # Jueves Santo 2026 es abril 2
        >>> get_historical_easter_date(datetime(2026, 4, 2), years_back=1)
        datetime(2025, 4, 17)  # Jueves Santo 2025

        >>> # Viernes Santo 2026 es abril 3
        >>> get_historical_easter_date(datetime(2026, 4, 3), years_back=1)
        datetime(2025, 4, 18)  # Viernes Santo 2025

        >>> # Día normal (no es Semana Santa)
        >>> get_historical_easter_date(datetime(2026, 5, 15), years_back=1)
        None
    """
    # Detectar si es día de Semana Santa
    day_type = detect_easter_day_type(fecha)

    if day_type is None:
        # No es parte de Semana Santa
        return None

    # Calcular el año objetivo
    target_year = fecha.year - years_back

    # Obtener fechas de Semana Santa para ese año
    historical_easter_dates = get_easter_week_dates(target_year)

    # Retornar la fecha del mismo tipo de día
    historical_date = historical_easter_dates[day_type]

    # Convertir a datetime para consistencia
    return datetime.combine(historical_date, datetime.min.time())


def get_smart_historical_lag(
    fecha: datetime,
    df,
    years_back: int = 1,
    column: str = 'demanda_total'
) -> Tuple[float, bool]:
    """
    Obtiene el lag histórico inteligente que maneja Semana Santa.

    Si la fecha es parte de Semana Santa, busca el valor del mismo día litúrgico
    del año anterior (ej: Jueves Santo → Jueves Santo del año pasado).

    Si no es Semana Santa, usa el lag tradicional (misma fecha del año anterior).

    Args:
        fecha: Fecha para la cual calcular el lag
        df: DataFrame con datos históricos (debe tener columnas 'fecha' y column)
        years_back: Cuántos años atrás buscar
        column: Nombre de la columna a extraer

    Returns:
        Tupla (valor_lag, is_easter_adjusted) donde:
        - valor_lag: Valor histórico encontrado (0 si no hay datos)
        - is_easter_adjusted: True si se usó búsqueda de Semana Santa

    Example:
        >>> # Para Jueves Santo 2026
        >>> lag_value, is_easter = get_smart_historical_lag(
        ...     datetime(2026, 4, 2),
        ...     df_historico,
        ...     years_back=1
        ... )
        >>> # lag_value será el valor del Jueves Santo 2025 (abril 17, 2025)
        >>> # is_easter será True
    """
    # Intentar búsqueda de Semana Santa
    historical_easter_date = get_historical_easter_date(fecha, years_back)

    if historical_easter_date is not None:
        # Es Semana Santa - usar fecha equivalente
        day_type = detect_easter_day_type(fecha)
        logger.info(f"   🐣 Detectado {day_type} - Buscando equivalente histórico en {historical_easter_date.date()}")

        # Buscar valor en DataFrame
        mask = df['fecha'].dt.date == historical_easter_date.date()
        if mask.any():
            value = df.loc[mask, column].iloc[-1]
            logger.info(f"   ✓ Encontrado valor histórico de {day_type} {historical_easter_date.year}: {value:.2f} MW")
            return value, True
        else:
            logger.warning(f"   ⚠ No se encontraron datos para {day_type} {historical_easter_date.date()}")
            return 0.0, True
    else:
        # No es Semana Santa - usar lag tradicional
        from pandas import DateOffset
        fecha_lag_anual = fecha - DateOffset(years=years_back)

        mask = df['fecha'].dt.date == fecha_lag_anual.date()
        if mask.any():
            value = df.loc[mask, column].iloc[-1]
            return value, False
        else:
            # Fallback: buscar el día más cercano anterior
            df_antes = df[df['fecha'].dt.date < fecha_lag_anual.date()]
            if len(df_antes) > 0:
                value = df_antes.iloc[-1][column]
                return value, False
            else:
                return 0.0, False


def get_easter_historical_average(
    day_type: EasterDayType,
    df,
    column: str = 'demanda_total',
    max_years_back: int = 5
) -> Optional[float]:
    """
    Calcula el promedio histórico de un tipo de día de Semana Santa.

    Útil como fallback cuando no hay suficientes datos históricos.

    Args:
        day_type: Tipo de día de Semana Santa
        df: DataFrame con datos históricos
        column: Nombre de la columna a promediar
        max_years_back: Cuántos años atrás buscar

    Returns:
        Promedio histórico, o None si no hay datos

    Example:
        >>> # Promedio de todos los Jueves Santos históricos
        >>> avg = get_easter_historical_average('Jueves Santo', df_historico)
        >>> print(f"Promedio histórico de Jueves Santo: {avg:.2f} MW")
    """
    current_year = datetime.now().year
    values = []

    for year in range(current_year - max_years_back, current_year):
        easter_dates = get_easter_week_dates(year)
        target_date = easter_dates.get(day_type)

        if target_date is None:
            continue

        mask = df['fecha'].dt.date == target_date
        if mask.any():
            value = df.loc[mask, column].iloc[-1]
            values.append(value)

    if len(values) > 0:
        import numpy as np
        avg = np.mean(values)
        logger.info(f"   📊 Promedio histórico de {day_type} ({len(values)} años): {avg:.2f} MW")
        return avg
    else:
        logger.warning(f"   ⚠ No hay datos históricos para {day_type}")
        return None


# Función auxiliar para debugging
def print_easter_info(year: int) -> None:
    """
    Imprime información de Semana Santa para un año dado.
    Útil para debugging y verificación.
    """
    dates = get_easter_week_dates(year)
    print(f"\nSemana Santa {year}:")
    print("=" * 50)
    for day_type, day_date in dates.items():
        weekday = day_date.strftime('%A')
        print(f"  {day_type:25} -> {day_date} ({weekday})")
    print("=" * 50)


if __name__ == "__main__":
    # Ejemplos de uso
    print("=" * 70)
    print("UTILIDADES DE SEMANA SANTA - EJEMPLOS")
    print("=" * 70)

    # Mostrar fechas de Semana Santa 2025 y 2026
    print_easter_info(2025)
    print_easter_info(2026)

    # Ejemplo de detección
    print("\nDETECCION DE DIAS DE SEMANA SANTA:")
    print("-" * 70)
    test_dates = [
        datetime(2026, 4, 2),   # Jueves Santo 2026
        datetime(2026, 4, 3),   # Viernes Santo 2026
        datetime(2026, 4, 5),   # Domingo de Resurrección 2026
        datetime(2026, 4, 10),  # Día normal
    ]

    for test_date in test_dates:
        day_type = detect_easter_day_type(test_date)
        if day_type:
            print(f"  {test_date.date()} -> {day_type}")
        else:
            print(f"  {test_date.date()} -> No es Semana Santa")

    # Ejemplo de búsqueda histórica
    print("\nBUSQUEDA DE EQUIVALENTES HISTORICOS:")
    print("-" * 70)
    jueves_santo_2026 = datetime(2026, 4, 2)
    viernes_santo_2026 = datetime(2026, 4, 3)

    for fecha in [jueves_santo_2026, viernes_santo_2026]:
        day_type = detect_easter_day_type(fecha)
        historical = get_historical_easter_date(fecha, years_back=1)
        print(f"  {fecha.date()} ({day_type})")
        print(f"    -> Equivalente 2025: {historical.date()}")
        print()
