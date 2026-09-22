"""
Cálculo de sensación térmica (heat index) a partir de temperatura y humedad.

La API de EPM (clima_new.csv) no entrega sensación térmica — solo
temperatura, humedad, viento y precipitación (ver
docs/MIGRACION_API_EPM_CLIMA.md) — así que se calcula aquí con la
regresión de Rothfusz (NWS Heat Index), el estándar usado por servicios
meteorológicos para climas cálidos/húmedos como el de la costa Caribe.
No incorpora viento: el heat index clásico solo usa temperatura +
humedad relativa (el viento entra como feature climática aparte).
"""
import numpy as np


def calcular_sensacion_termica(temp_c, humedad_pct):
    """
    Calcula la sensación térmica (°C) a partir de temperatura (°C) y
    humedad relativa (%). Vectorizado: acepta escalares, arrays de numpy
    o pandas.Series y devuelve el mismo tipo de forma (ndarray si la
    entrada no era una Series).

    Fórmula: regresión de Rothfusz sobre °F (NWS), con los ajustes
    estándar para humedad baja/alta, luego convertida de vuelta a °C.
    Para temperaturas "aparentes" bajas (<80°F) usa la aproximación
    simple de Steadman en vez de la regresión completa, que solo es
    válida en el rango cálido.
    """
    temp_c_arr = np.asarray(temp_c, dtype=float)
    humedad_arr = np.asarray(humedad_pct, dtype=float)

    temp_f = temp_c_arr * 9.0 / 5.0 + 32.0
    r = humedad_arr

    # Aproximación simple (Steadman) — válida para temperaturas moderadas
    hi_simple = 0.5 * (temp_f + 61.0 + (temp_f - 68.0) * 1.2 + r * 0.094)

    # Regresión completa de Rothfusz — válida para temperaturas "aparentes" altas
    t = temp_f
    hi_full = (
        -42.379
        + 2.04901523 * t
        + 10.14333127 * r
        - 0.22475541 * t * r
        - 0.00683783 * t ** 2
        - 0.05481717 * r ** 2
        + 0.00122874 * t ** 2 * r
        + 0.00085282 * t * r ** 2
        - 0.00000199 * t ** 2 * r ** 2
    )

    # Ajuste NWS: humedad baja (<13%) con temperatura entre 80-112°F
    # (clip a 0 antes del sqrt: fuera de ese rango el radicando puede ser
    # negativo, pero el resultado se descarta igual vía usa_ajuste_baja)
    ajuste_baja = ((13.0 - r) / 4.0) * np.sqrt(np.maximum(17.0 - np.abs(t - 95.0), 0.0) / 17.0)
    usa_ajuste_baja = (r < 13.0) & (t >= 80.0) & (t <= 112.0)
    hi_full = np.where(usa_ajuste_baja, hi_full - ajuste_baja, hi_full)

    # Ajuste NWS: humedad alta (>85%) con temperatura entre 80-87°F
    ajuste_alta = ((r - 85.0) / 10.0) * ((87.0 - t) / 5.0)
    usa_ajuste_alta = (r > 85.0) & (t >= 80.0) & (t <= 87.0)
    hi_full = np.where(usa_ajuste_alta, hi_full + ajuste_alta, hi_full)

    usa_regresion_completa = ((t + hi_simple) / 2.0) >= 80.0
    heat_index_f = np.where(usa_regresion_completa, hi_full, hi_simple)

    heat_index_c = (heat_index_f - 32.0) * 5.0 / 9.0

    if np.isscalar(temp_c) or (np.ndim(temp_c_arr) == 0):
        return float(heat_index_c)
    return heat_index_c
