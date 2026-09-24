"""
Servicio de cálculos FDA/FDP y Clustering.

Este módulo implementa los algoritmos de:
- Curvas típicas: clustering en el histórico para detectar patrones y devolver las N curvas más típicas (forma y nivel).
- FDA (Factor de Demanda Ajustada): Normalización sobre las curvas típicas seleccionadas (suma 1.0).
- FDP (Factor de Demanda Pronóstico): Cos(Atan(Q/P)) sobre las curvas típicas seleccionadas.
- Clustering (agregación): Aplicación de factores multiplicadores a medidas por barra+fecha.
"""

from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np

from app.services import factores_service


# =============================================================================
# CONSTANTES
# =============================================================================

PRECISION_DECIMALES = 5
PERIODOS_COLUMNAS = [f'p{i}' for i in range(1, 25)]
TIPOS_DIA = ["ORDINARIO", "SABADO", "FESTIVO"]


# =============================================================================
# FUNCIONES AUXILIARES - CLUSTERING
# =============================================================================

def aplicar_config_agrupacion(
    valor_crudo: float,
    factor: float,
    dividir_por_1000: bool,
    valor_absoluto: bool,
) -> float:
    """
    Aplica la configuración de una agrupación sobre un valor crudo de
    medida, en el orden dividir_por_1000 -> factor -> valor_absoluto.
    `medidas` guarda siempre el valor tal cual lo manda la API — este
    ajuste se hace al leer/usar el dato, no al insertarlo.
    """
    valor = float(valor_crudo)
    if dividir_por_1000:
        valor = valor / 1000
    valor = valor * float(factor)
    if valor_absoluto:
        valor = abs(valor)
    return valor


def _agrupar_medidas_clusterizadas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrupa medidas por barra+fecha y suma periodos.

    Args:
        df: DataFrame con columnas: barra, fecha, p1-p24

    Returns:
        DataFrame agrupado con periodos sumados y redondeados a 5 decimales
    """
    return df.groupby(['barra', 'fecha'])[PERIODOS_COLUMNAS].sum().round(PRECISION_DECIMALES)


def _curvas_a_matriz(curvas: List[Dict[str, Any]]) -> Tuple[np.ndarray, List[Tuple[str, str]]]:
    """
    Convierte lista de curvas {barra, fecha, periodos} en matriz (n x 24) y lista de (barra, fecha).
    """
    if not curvas:
        return np.array([]).reshape(0, 24), []
    filas = []
    keys = []
    for c in curvas:
        p = c.get("periodos") or c
        if isinstance(p, dict):
            filas.append([float(p.get(f"p{i}", 0)) for i in range(1, 25)])
        else:
            filas.append([float(x) for x in p[:24]])
        keys.append((c["barra"], c["fecha"]))
    return np.array(filas), keys


def _filtrar_outliers_iqr(curvas: List[Dict[str, Any]], factor_iqr: float = 1.5) -> List[Dict[str, Any]]:
    """
    Filtra curvas que tienen valores outliers usando el método IQR.

    Una curva se descarta si tiene AL MENOS UN período fuera de los límites:
    - lower = Q1 - factor_iqr * IQR
    - upper = Q3 + factor_iqr * IQR

    Args:
        curvas: Lista de curvas con periodos p1-p24
        factor_iqr: Multiplicador del IQR (default 1.5)

    Returns:
        Lista de curvas sin outliers
    """
    if not curvas or len(curvas) < 4:
        return curvas

    X, keys = _curvas_a_matriz(curvas)
    n_curvas, n_periodos = X.shape

    # Calcular Q1, Q3, IQR para cada período
    q1 = np.percentile(X, 25, axis=0)
    q3 = np.percentile(X, 75, axis=0)
    iqr = q3 - q1

    lower = q1 - factor_iqr * iqr
    upper = q3 + factor_iqr * iqr

    # Identificar curvas válidas (ningún período fuera de límites)
    curvas_validas = []
    for i in range(n_curvas):
        es_outlier = False
        for j in range(n_periodos):
            if X[i, j] < lower[j] or X[i, j] > upper[j]:
                es_outlier = True
                break
        if not es_outlier:
            curvas_validas.append(curvas[i])

    return curvas_validas


def _seleccionar_curvas_tipicas(
    curvas: List[Dict[str, Any]], n_max: int
) -> List[Dict[str, Any]]:
    """
    De una lista de curvas (barra, fecha, periodos), devuelve hasta n_max más típicas
    por forma Y nivel: calcula distancia euclidiana directa, mide centralidad
    (menor distancia media = más típica).
    Si hay menos de n_max curvas, devuelve todas las encontradas.

    IMPORTANTE:
    - Primero filtra outliers usando IQR antes de calcular tipicidad
    - NO normaliza por L2, por lo que considera tanto forma como magnitud/nivel
    - Curvas similares en patrón Y escala serán seleccionadas como típicas
    """
    if not curvas:
        return []

    # Paso 1: Filtrar outliers por IQR
    curvas_filtradas = _filtrar_outliers_iqr(curvas)

    if not curvas_filtradas:
        return []
    if len(curvas_filtradas) <= n_max:
        return curvas_filtradas

    X, keys = _curvas_a_matriz(curvas_filtradas)

    # NO normalizar - usar valores originales para considerar forma Y nivel
    # Esto permite que curvas con magnitudes similares se agrupen juntas

    # Distancia euclidiana entre todas las filas (menor distancia media = más típica/central)
    n = len(X)
    mean_dists = np.zeros(n)
    for i in range(n):
        d = np.array([np.linalg.norm(X[i] - X[j]) for j in range(n) if j != i])
        mean_dists[i] = float(d.mean()) if len(d) else 0.0

    # Más típicas = menor distancia media (más centrales)
    indices = np.argsort(mean_dists)[:n_max]
    return [curvas_filtradas[i] for i in indices]


# =============================================================================
# FUNCIONES AUXILIARES - FDA
# =============================================================================

def _calcular_fda_normalizado(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica algoritmo FDA: normaliza valores para que cada período sume exactamente 1.0.

    Algoritmo:
    1. Normalizar: dividir cada valor por la suma de su período
    2. Redondear a PRECISION_DECIMALES
    3. Calcular diferencia real con 1.0 (causada por redondeo)
    4. Aplicar ajuste al valor máximo de cada período para que sume exactamente 1.0
    5. Resultado: cada período suma exactamente 1.0

    Args:
        df: DataFrame con periodos p1-p24 (valores absolutos de potencia)

    Returns:
        DataFrame con factores FDA normalizados (cada período suma EXACTAMENTE 1.0)
    """
    # Paso 1: Normalizar dividiendo por la suma de cada período
    sumas_por_periodo = df[PERIODOS_COLUMNAS].sum()

    # Evitar división por cero
    sumas_por_periodo = sumas_por_periodo.replace(0, 1)

    # Normalizar: cada valor / suma del período
    df_normalizado = df[PERIODOS_COLUMNAS].div(sumas_por_periodo)

    # Paso 2: REDONDEAR PRIMERO (esto puede causar que la suma no sea exactamente 1.0)
    df_redondeado = df_normalizado.round(PRECISION_DECIMALES)

    # Paso 3: Calcular la diferencia REAL después del redondeo
    sumas_redondeadas = df_redondeado.sum()
    ajustes_necesarios = 1.0 - sumas_redondeadas

    # Paso 4: Identificar el índice del máximo en cada columna
    idx_maximos = df_redondeado.idxmax()

    # Paso 5: Aplicar el ajuste al máximo de cada período
    df_ajustado = df_redondeado.copy()
    for col in PERIODOS_COLUMNAS:
        if abs(ajustes_necesarios[col]) > 1e-10:  # Solo ajustar si hay diferencia significativa
            idx_max = idx_maximos[col]
            df_ajustado.loc[idx_max, col] = round(
                df_ajustado.loc[idx_max, col] + ajustes_necesarios[col],
                PRECISION_DECIMALES
            )

    return df_ajustado


# =============================================================================
# FUNCIONES AUXILIARES - FDP
# =============================================================================

def _calcular_fdp_vectorizado(df_a: pd.DataFrame, df_r: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula FDP para todos los periodos usando numpy vectorizado.

    FDP = Cos(Atan(Potencia_Reactiva / Potencia_Activa))

    Casos especiales:
    - Si P=0 y Q=0: FDP = 1.0
    - Si P=0 y Q≠0: FDP = 0.0

    Args:
        df_a: DataFrame con medidas activas (tipo A); debe tener fecha y opcionalmente barra
        df_r: DataFrame con medidas reactivas (tipo R)

    Returns:
        DataFrame con columnas fdp_p1 a fdp_p24 (y fecha, barra si aplica)
    """
    on_cols = ['barra', 'fecha'] if 'barra' in df_a.columns and 'barra' in df_r.columns else ['fecha']
    cols_a = on_cols + PERIODOS_COLUMNAS
    cols_r = on_cols + PERIODOS_COLUMNAS
    # LEFT JOIN: barras sin datos reactivos quedan con NaN en columnas _r → FDP = 1.0
    df_merged = pd.merge(
        df_a[cols_a],
        df_r[cols_r],
        on=on_cols,
        how='left',
        suffixes=('_a', '_r')
    )

    # Calcular FDP para cada periodo usando numpy vectorizado
    for i in range(1, 25):
        col_a = f'p{i}_a'
        col_r = f'p{i}_r'
        col_fdp = f'fdp_p{i}'

        P = df_merged[col_a].values
        Q = df_merged[col_r].fillna(0).values  # NaN (sin reactiva) → Q=0 → FDP=1.0

        # Vectorizado con numpy.where para manejar división por cero
        df_merged[col_fdp] = np.where(
            P == 0,
            np.where(Q == 0, 1.0, 0.0),  # Casos especiales
            np.cos(np.arctan(Q / P))      # Cálculo normal
        )

    # Seleccionar solo columnas FDP y redondear
    cols_fdp = [f'fdp_p{i}' for i in range(1, 25)]
    out_cols = on_cols + cols_fdp
    return df_merged[out_cols].round(PRECISION_DECIMALES)


# =============================================================================
# FUNCIONES AUXILIARES - CONVERSIÓN
# =============================================================================

def _df_to_response(df: pd.DataFrame, tipo_dia: str, ajuste: float = None) -> Dict[str, Any]:
    """
    Convierte DataFrame a estructura de respuesta JSON.

    Args:
        df: DataFrame con resultados
        tipo_dia: ORDINARIO, SABADO o FESTIVO
        ajuste: Ajuste aplicado (opcional, solo para FDA)

    Returns:
        Diccionario con estructura de respuesta
    """
    response = {
        "tipo_dia": tipo_dia,
        "n_registros": len(df),
        "factores": df.to_dict('index'),
        "suma_total": round(df[PERIODOS_COLUMNAS].sum().sum(), PRECISION_DECIMALES)
    }

    if ajuste is not None:
        response["ajuste_aplicado"] = round(ajuste, PRECISION_DECIMALES)

    return response


# =============================================================================
# FUNCIONES PRINCIPALES - CLUSTERING
# =============================================================================

def _procesar_medidas_con_factores(
    medidas: List[Dict[str, Any]],
    factores: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Aplica la config de cada agrupación (dividir_por_1000 -> factor ->
    valor_absoluto) sobre medidas crudas y agrupa por barra+fecha sumando
    los periodos. Lógica compartida entre aplicar_clustering (todos los
    codigo_rpm configurados en una barra) y aplicar_clustering_generador
    (un solo codigo_rpm, para aislar el aporte de un generador puntual).
    """
    df = pd.DataFrame(medidas)

    # Crear diccionario de config (factor, dividir_por_1000, valor_absoluto)
    # de cada agrupación para lookup
    config_map = {
        (f['codigo_rpm'], f['flujo']): {
            'factor': float(f['factor']),
            'dividir_por_1000': bool(f['dividir_por_1000']),
            'valor_absoluto': bool(f['valor_absoluto']),
        }
        for f in factores
    }
    config_default = {'factor': 1.0, 'dividir_por_1000': False, 'valor_absoluto': False}

    # Aplicar dividir_por_1000 -> factor -> valor_absoluto por periodo
    # Las columnas de la consulta son: mep1, mep2, ..., mep24
    for i in range(1, 25):
        col_medida = f'mep{i}'
        col_resultado = f'p{i}'
        if col_medida in df.columns:
            def _aplicar(row, col_medida=col_medida):
                cfg = config_map.get((row['mecodigo_rpm'], row['meflujo']), config_default)
                return round(
                    aplicar_config_agrupacion(
                        row[col_medida], cfg['factor'], cfg['dividir_por_1000'], cfg['valor_absoluto']
                    ),
                    PRECISION_DECIMALES
                )
            df[col_resultado] = df.apply(_aplicar, axis=1)

    # Renombrar columnas para uniformidad
    df = df.rename(columns={'babarra': 'barra', 'mefecha': 'fecha'})

    # Agrupar por barra+fecha
    df_agrupado = _agrupar_medidas_clusterizadas(df)
    df_agrupado = df_agrupado.reset_index()

    # Convertir a lista de dicts
    resultado = []
    for _, row in df_agrupado.iterrows():
        periodos_dict = {f'p{i}': row[f'p{i}'] for i in range(1, 25)}
        resultado.append({
            'barra': row['barra'],
            'fecha': str(row['fecha']),
            'periodos': periodos_dict
        })

    return resultado


def aplicar_clustering(
    fecha_inicial: str,
    fecha_final: str,
    mc: str,
    barra: str,
    flujo_tipo: str,
    tipo_dia: str = "",
    dsn: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Aplica factores multiplicadores a medidas y agrupa por barra+fecha.

    Este es el "clustering" que multiplica cada medida por su factor correspondiente
    y luego agrupa sumando los periodos.

    Args:
        fecha_inicial: Formato YYYY-MM-DD
        fecha_final: Formato YYYY-MM-DD
        mc: Código de mercado/centro
        barra: Nombre de la barra
        flujo_tipo: "A" (Activa) o "R" (Reactiva)
        tipo_dia: ORDINARIO, SABADO, FESTIVO o vacío para todos

    Returns:
        Lista de medidas clusterizadas por fecha
    """
    # Obtener códigos RPM de la barra
    codigos = factores_service.consultar_barra_nombre(barra, dsn=dsn)
    if not codigos:
        return []

    codigo_rpm = [row['codigo_rpm'] for row in codigos]

    # Obtener factores
    factores = factores_service.consultar_barra_factor_nombre(barra, flujo_tipo, codigo_rpm, dsn=dsn)
    if not factores:
        return []

    # Obtener medidas
    flujos = [f['flujo'] for f in factores]
    medidas = factores_service.consultar_medidas_calcular_completo(
        fecha_inicial, fecha_final, mc, flujos, tipo_dia, codigo_rpm, barra, False, dsn=dsn
    )

    if not medidas:
        return []

    return _procesar_medidas_con_factores(medidas, factores)


def aplicar_clustering_generador(
    fecha_inicial: str,
    fecha_final: str,
    mc: str,
    barra: str,
    flujo_tipo: str,
    codigo_rpm_generador: str,
    tipo_dia: str = "",
    dsn: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Igual que aplicar_clustering, pero para UN SOLO codigo_rpm (un
    generador/circuito puntual de la barra, ya configurado en
    agrupaciones) en vez de sumar todos los codigo_rpm de la barra —
    aísla su aporte individual, usado por calcular_ajuste_fp_generador
    para saber cuánto debe cambiar ese generador específicamente.
    """
    factores = factores_service.consultar_barra_factor_nombre(
        barra, flujo_tipo, [codigo_rpm_generador], dsn=dsn
    )
    if not factores:
        return []

    flujos = [f['flujo'] for f in factores]
    medidas = factores_service.consultar_medidas_calcular_completo(
        fecha_inicial, fecha_final, mc, flujos, tipo_dia, [codigo_rpm_generador], barra, False, dsn=dsn
    )
    if not medidas:
        return []

    return _procesar_medidas_con_factores(medidas, factores)


def obtener_curvas_tipicas(
    fecha_inicial: str,
    fecha_final: str,
    mc: str,
    tipo_dia: str,
    flujo_tipo: str,
    n_max: int,
    barra: Optional[str] = None,
    dsn: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Obtiene las N curvas más típicas del histórico (por forma y nivel).

    1. Obtiene todas las curvas clusterizadas en el rango/MC/tipo_dia (una barra o todas las del MC).
    2. Normaliza por L2 y mide centralidad (distancia media a las demás).
    3. Devuelve hasta n_max más típicas; si el cluster solo encuentra menos, devuelve esas.

    Returns:
        Lista de {barra, fecha, periodos} con las curvas más típicas.
    """
    if flujo_tipo not in ("A", "R"):
        return []

    if barra:
        barras_a_usar = [{"barra": barra}]
    else:
        barras_a_usar = factores_service.consultar_barras_index_xmc(mc, dsn=dsn)
        if not barras_a_usar:
            return []

    curvas_todas = []
    for b in barras_a_usar:
        nombre_barra = b.get("barra")
        if not nombre_barra:
            continue
        medidas = aplicar_clustering(
            fecha_inicial, fecha_final, mc, nombre_barra, flujo_tipo, tipo_dia, dsn=dsn
        )
        for m in medidas:
            curvas_todas.append({
                "barra": m["barra"],
                "fecha": m["fecha"],
                "periodos": m["periodos"],
            })

    return _seleccionar_curvas_tipicas(curvas_todas, n_max)


def obtener_curvas_tipicas_ucp(
    fecha_inicial: str,
    fecha_final: str,
    mc: str,
    tipo_dia: str,
    n_max: int,
    dsn: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Igual que obtener_curvas_tipicas (misma selección: filtro IQR +
    centralidad por distancia euclidiana), pero sobre la demanda TOTAL diaria
    del mercado (tabla actualizaciondatos) en vez de clusterizar por barra —
    para comparar contra la curva "Demanda Real (DB)" que se muestra en
    Actualización de datos, que es justo esa suma total.

    Returns:
        Lista de {barra, fecha, periodos} con las curvas más típicas —
        "barra" acá es el propio nombre del mercado, solo para etiquetar.
    """
    filas = factores_service.consultar_actualizaciondatos_completo(
        fecha_inicial, fecha_final, mc, tipo_dia, dsn=dsn
    )
    curvas = [
        {
            "barra": mc,
            "fecha": f["fecha"],
            "periodos": {f"p{i}": float(f.get(f"p{i}") or 0) for i in range(1, 25)},
        }
        for f in filas
    ]
    return _seleccionar_curvas_tipicas(curvas, n_max)


# =============================================================================
# FUNCIONES PRINCIPALES - FDA
# =============================================================================

def _filtrar_medidas_por_curvas_tipicas(
    medidas: List[Dict[str, Any]],
    curvas_tipicas: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Filtra medidas para conservar solo las (barra, fecha) que están en curvas_tipicas."""
    if not curvas_tipicas:
        return []
    set_ref = {(c["barra"], str(c["fecha"])) for c in curvas_tipicas}
    return [
        m for m in medidas
        if (m["barra"], str(m["fecha"])) in set_ref
    ]


def _obtener_medidas_clusterizadas_para_curvas_tipicas(
    fecha_inicial: str,
    fecha_final: str,
    mc: str,
    tipo_dia: str,
    curvas_tipicas: List[Dict[str, Any]],
    flujo_tipo: str,
    dsn: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Obtiene medidas clusterizadas solo para las (barra, fecha) indicadas en curvas_tipicas.
    """
    if not curvas_tipicas:
        return []
    barras_unicas = list({c["barra"] for c in curvas_tipicas})
    todas = []
    for barra in barras_unicas:
        medidas = aplicar_clustering(
            fecha_inicial, fecha_final, mc, barra, flujo_tipo, tipo_dia, dsn=dsn
        )
        todas.extend(medidas)
    return _filtrar_medidas_por_curvas_tipicas(todas, curvas_tipicas)


def _obtener_medidas_generador_para_curvas_tipicas(
    fecha_inicial: str,
    fecha_final: str,
    mc: str,
    tipo_dia: str,
    curvas_tipicas: List[Dict[str, Any]],
    flujo_tipo: str,
    barra: str,
    codigo_rpm_generador: str,
    dsn: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Igual que _obtener_medidas_clusterizadas_para_curvas_tipicas, pero
    aislando un solo codigo_rpm (generador) en vez de toda la barra —
    usado por calcular_ajuste_fp_generador.
    """
    if not curvas_tipicas:
        return []
    medidas = aplicar_clustering_generador(
        fecha_inicial, fecha_final, mc, barra, flujo_tipo, codigo_rpm_generador, tipo_dia, dsn=dsn
    )
    return _filtrar_medidas_por_curvas_tipicas(medidas, curvas_tipicas)


def calcular_fda_para_tipo_dia(
    fecha_inicial: str,
    fecha_final: str,
    mc: str,
    tipo_dia: str,
    curvas_tipicas: List[Dict[str, Any]],
    dsn: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Calcula FDA (Factor de Demanda Ajustada) solo sobre las curvas típicas indicadas.

    El algoritmo FDA normaliza los factores para que la suma sea exactamente 1.0,
    aplicando el ajuste únicamente al valor máximo de cada periodo.

    Args:
        fecha_inicial: Formato YYYY-MM-DD
        fecha_final: Formato YYYY-MM-DD
        mc: Código de mercado/centro
        tipo_dia: ORDINARIO, SABADO o FESTIVO
        curvas_tipicas: Lista de {barra, fecha} (salida de curvas-tipicas). FDA se calcula solo sobre estas.
        dsn: URL de conexión a BD alternativa (opcional)

    Returns:
        Diccionario con factores FDA normalizados
    """
    medidas_clusterizadas = _obtener_medidas_clusterizadas_para_curvas_tipicas(
        fecha_inicial, fecha_final, mc, tipo_dia, curvas_tipicas, "A", dsn=dsn
    )

    if not medidas_clusterizadas:
        return {
            "tipo_dia": tipo_dia,
            "n_registros": 0,
            "factores": {},
            "suma_total": 0.0,
            "ajuste_aplicado": 0.0
        }

    # Convertir a DataFrame
    df_list = []
    for medida in medidas_clusterizadas:
        row_data = {'barra': medida['barra'], 'fecha': medida['fecha']}
        row_data.update(medida['periodos'])
        df_list.append(row_data)

    df = pd.DataFrame(df_list)

    # Aplicar normalización FDA
    df_normalizado = _calcular_fda_normalizado(df)

    # Calcular ajuste real aplicado (diferencia entre suma normalizada y 1.0)
    # Este valor debería ser muy cercano a 0 después de la normalización
    sumas_finales = df_normalizado[PERIODOS_COLUMNAS].sum()
    ajustes_reales = (1.0 - sumas_finales).abs()
    ajuste_promedio = ajustes_reales.mean()  # Promedio de ajustes por período

    # Agregar barra y fecha de vuelta
    df_normalizado['barra'] = df['barra'].values
    df_normalizado['fecha'] = df['fecha'].values

    # Convertir a respuesta
    return _df_to_response(df_normalizado, tipo_dia, ajuste_promedio)


# =============================================================================
# FUNCIONES PRINCIPALES - FDP
# =============================================================================

def calcular_fdp_para_tipo_dia(
    fecha_inicial: str,
    fecha_final: str,
    mc: str,
    tipo_dia: str,
    curvas_tipicas: List[Dict[str, Any]],
    dsn: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Calcula FDP (Factor de Demanda Pronóstico) solo sobre las curvas típicas indicadas.

    FDP = Cos(Atan(Potencia_Reactiva / Potencia_Activa))

    Requiere medidas tanto de tipo A (activa) como R (reactiva) para esas curvas.

    Args:
        fecha_inicial: Formato YYYY-MM-DD
        fecha_final: Formato YYYY-MM-DD
        mc: Código de mercado/centro
        tipo_dia: ORDINARIO, SABADO o FESTIVO
        curvas_tipicas: Lista de {barra, fecha}. FDP se calcula solo sobre estas.
        dsn: URL de conexión a BD alternativa (opcional)

    Returns:
        Diccionario con factores FDP calculados
    """
    medidas_a = _obtener_medidas_clusterizadas_para_curvas_tipicas(
        fecha_inicial, fecha_final, mc, tipo_dia, curvas_tipicas, "A", dsn=dsn
    )
    medidas_r = _obtener_medidas_clusterizadas_para_curvas_tipicas(
        fecha_inicial, fecha_final, mc, tipo_dia, curvas_tipicas, "R", dsn=dsn
    )

    if not medidas_a or not medidas_r:
        return {
            "tipo_dia": tipo_dia,
            "n_registros": 0,
            "factores": {}
        }

    # Convertir a DataFrames (incluir barra para merge correcto con varias barras)
    df_a_list = []
    for m in medidas_a:
        row = {'barra': m['barra'], 'fecha': m['fecha']}
        row.update(m['periodos'])
        df_a_list.append(row)
    df_a = pd.DataFrame(df_a_list)

    df_r_list = []
    for m in medidas_r:
        row = {'barra': m['barra'], 'fecha': m['fecha']}
        row.update(m['periodos'])
        df_r_list.append(row)
    df_r = pd.DataFrame(df_r_list)

    # Calcular FDP vectorizado
    df_fdp = _calcular_fdp_vectorizado(df_a, df_r)

    # Convertir columnas fdp_p* a p*
    for i in range(1, 25):
        df_fdp[f'p{i}'] = df_fdp[f'fdp_p{i}']
        df_fdp = df_fdp.drop(columns=[f'fdp_p{i}'])

    # Convertir a respuesta
    return _df_to_response(df_fdp, tipo_dia)


# =============================================================================
# FUNCIONES PRINCIPALES - AJUSTE FP POR GENERADOR
# =============================================================================

def calcular_ajuste_fp_generador(
    fecha_inicial: str,
    fecha_final: str,
    mc: str,
    tipo_dia: str,
    curvas_tipicas: List[Dict[str, Any]],
    barra: str,
    codigo_rpm_generador: str,
    fp_objetivo: float,
    dsn: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Calcula, por período (p1..p24) y por cada fecha típica seleccionada, el
    factor multiplicador y la variación absoluta de potencia ACTIVA que
    habría que aplicarle a UN generador puntual (codigo_rpm_generador) para
    que el FP de la barra alcance fp_objetivo — Modo 2 del algoritmo de
    ajuste de factor de potencia (se ajusta la activa del generador,
    manteniendo constante la reactiva total de la barra):

        FP_actual = P_barra / sqrt(P_barra^2 + Q_barra^2)
        P_objetivo_barra = |Q_barra| * fp_objetivo / sqrt(1 - fp_objetivo^2)
        delta_P = P_objetivo_barra - P_barra
        P_generador_nuevo = P_generador_actual + delta_P
        factor_ajuste = P_generador_nuevo / P_generador_actual

    Periodos donde FP_actual ya es >= fp_objetivo: factor=1.0, variación=0.0
    (no requieren ajuste).

    Es un cálculo puramente informativo — no persiste nada, ya que
    agrupaciones.factor es un único valor fijo y no puede representar un
    ajuste distinto por período. El promedio final entre las fechas
    típicas (para mostrar una sola curva de 24 periodos) queda a cargo de
    quien consuma esta respuesta, igual que ya se hace hoy con FDP.

    Un mismo generador puede estar repartido entre varias barras con
    factores que suman 1 (p.ej. 0.85 en una y 0.15 en la otra) — se
    devuelve también el factor_actual configurado y, si existe, la
    barra_complementaria con SU factor actual, para poder calcular del
    otro lado (1 - factor_nuevo_recomendado) sin tener que consultarlo
    aparte. El "factor" de cada período es relativo al factor_actual
    (factor_nuevo_absoluto = factor_actual * factor); quien consuma esta
    respuesta decide cómo resumir las 24 horas en un solo valor a aplicar
    (p.ej. el peor caso, el máximo de los 24 factores).

    Returns:
        {
          "tipo_dia", "barra", "codigo_rpm_generador", "fp_objetivo",
          "factor_actual",           # factor hoy configurado en agrupaciones
          "barra_complementaria",    # {barra, factor_actual} | None
          "n_registros",
          "factores": {idx: {barra, fecha, p1..p24}},     # multiplicador relativo al factor_actual
          "variaciones": {idx: {barra, fecha, p1..p24}},  # delta absoluto (kW/MW, misma unidad que las medidas)
        }
    """
    medidas_p_total = _obtener_medidas_clusterizadas_para_curvas_tipicas(
        fecha_inicial, fecha_final, mc, tipo_dia, curvas_tipicas, "A", dsn=dsn
    )
    medidas_q_total = _obtener_medidas_clusterizadas_para_curvas_tipicas(
        fecha_inicial, fecha_final, mc, tipo_dia, curvas_tipicas, "R", dsn=dsn
    )
    medidas_generador = _obtener_medidas_generador_para_curvas_tipicas(
        fecha_inicial, fecha_final, mc, tipo_dia, curvas_tipicas, "A",
        barra, codigo_rpm_generador, dsn=dsn
    )

    # Factor actualmente configurado en agrupaciones para (barra,
    # generador) — el "factor_ajuste" que arma cada periodo es relativo a
    # lo que ya está configurado; hace falta este valor para poder
    # convertirlo en el nuevo factor absoluto (factor_actual * ajuste).
    factor_actual_rows = factores_service.consultar_barra_factor_nombre(
        barra, "A", [codigo_rpm_generador], dsn=dsn
    )
    factor_actual = (
        float(factor_actual_rows[0]["factor"]) if factor_actual_rows else None
    )

    # Un mismo generador puede repartirse entre varias barras con
    # factores que suman 1 (p.ej. 0.85 en una y 0.15 en la otra) — se
    # busca esa contraparte para poder mostrar también su factor
    # resultante (1 - factor_nuevo_recomendado).
    barra_complementaria = None
    if factor_actual is not None:
        otras_barras = factores_service.consultar_barras_por_codigo_rpm(
            codigo_rpm_generador, "A", dsn=dsn
        )
        for fila in otras_barras:
            if fila["barra"] != barra:
                barra_complementaria = {
                    "barra": fila["barra"],
                    "factor_actual": float(fila["factor"]),
                }
                break

    vacio = {
        "tipo_dia": tipo_dia,
        "barra": barra,
        "codigo_rpm_generador": codigo_rpm_generador,
        "fp_objetivo": fp_objetivo,
        "factor_actual": factor_actual,
        "barra_complementaria": barra_complementaria,
        "n_registros": 0,
        "factores": {},
        "variaciones": {},
    }
    if not medidas_p_total or not medidas_q_total or not medidas_generador:
        return vacio

    def _a_df(medidas: List[Dict[str, Any]]) -> pd.DataFrame:
        filas = []
        for m in medidas:
            fila = {'barra': m['barra'], 'fecha': m['fecha']}
            fila.update(m['periodos'])
            filas.append(fila)
        return pd.DataFrame(filas)

    df_p = _a_df(medidas_p_total)
    df_q = _a_df(medidas_q_total)
    df_gen = _a_df(medidas_generador).rename(
        columns={f'p{i}': f'p{i}_gen' for i in range(1, 25)}
    )

    df = df_p.merge(df_q, on=['barra', 'fecha'], suffixes=('_p', '_q'))
    df = df.merge(
        df_gen[['fecha'] + [f'p{i}_gen' for i in range(1, 25)]],
        on='fecha', how='left',
    )

    if df.empty:
        return vacio

    denom_objetivo = float(np.sqrt(max(1e-12, 1.0 - float(fp_objetivo) ** 2)))

    factor_cols: Dict[str, np.ndarray] = {}
    variacion_cols: Dict[str, np.ndarray] = {}
    for i in range(1, 25):
        P = df[f'p{i}_p'].to_numpy(dtype=float)
        Q = df[f'p{i}_q'].to_numpy(dtype=float)
        col_gen = f'p{i}_gen'
        P_gen = (
            df[col_gen].to_numpy(dtype=float)
            if col_gen in df.columns else np.full(len(df), np.nan)
        )

        Q_mag = np.abs(Q)
        S = np.sqrt(P ** 2 + Q_mag ** 2)
        fp_actual = np.where(S == 0, 1.0, np.divide(P, S, out=np.ones_like(P), where=S != 0))

        p_objetivo_barra = Q_mag * float(fp_objetivo) / denom_objetivo
        delta_p = p_objetivo_barra - P

        ya_cumple = fp_actual >= (float(fp_objetivo) - 1e-9)

        p_gen_nuevo = P_gen + delta_p
        with np.errstate(divide='ignore', invalid='ignore'):
            factor = np.where(
                ya_cumple,
                1.0,
                np.where(P_gen == 0, np.nan, p_gen_nuevo / P_gen),
            )
        variacion = np.where(ya_cumple, 0.0, delta_p)

        factor_cols[f'p{i}'] = np.round(factor, PRECISION_DECIMALES)
        variacion_cols[f'p{i}'] = np.round(variacion, PRECISION_DECIMALES)

    df_factor = pd.DataFrame(factor_cols)
    df_factor['barra'] = df['barra'].values
    df_factor['fecha'] = df['fecha'].values

    df_variacion = pd.DataFrame(variacion_cols)
    df_variacion['barra'] = df['barra'].values
    df_variacion['fecha'] = df['fecha'].values

    return {
        "tipo_dia": tipo_dia,
        "barra": barra,
        "codigo_rpm_generador": codigo_rpm_generador,
        "fp_objetivo": fp_objetivo,
        "factor_actual": factor_actual,
        "barra_complementaria": barra_complementaria,
        "n_registros": len(df),
        "factores": df_factor.where(pd.notna(df_factor), None).to_dict('index'),
        "variaciones": df_variacion.where(pd.notna(df_variacion), None).to_dict('index'),
    }
