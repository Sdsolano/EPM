"""
Modelo de NIVEL para pronóstico de demanda DIARIA (/predict-daily)
==================================================================

Los árboles entrenados sobre el TOTAL crudo no extrapolan tendencia/nivel
(la demanda crece ~12% interanual en mercados jóvenes), por eso aquí se
predice el RATIO  TOTAL_t / ancla(origen)  donde el ancla es el promedio de
los últimos ANCHOR_DAYS (21) días de demanda "normal" (sin festivos ni
temporada navideña) conocidos al origen. La demanda final = ratio * ancla,
de modo que el nivel siempre se re-ancla a la demanda reciente y el modelo
solo aprende la forma (día de semana, calendario, clima, horizonte).

Módulo auto-contenido: NO depende de FeatureEngineer ni de los conectores
del pipeline horario. Un ÚNICO código de calendario/clima/anclas sirve para
entrenar y para predecir (evita skew train/predict).
"""

import logging
import os
import threading
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import lightgbm as lgb
from dateutil.easter import easter
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import RidgeCV

from src.models.metrics import calculate_all_metrics
from src.utils.weather import calcular_sensacion_termica

logger = logging.getLogger(__name__)

MODEL_KIND = 'daily_level_ratio_v1'
HORIZONS_TRAIN = [1, 4, 8, 15, 22, 30]
MAX_HORIZON = 30
MIN_DAYS = 120            # mínimo de días de demanda para entrenar
MIN_TRAIN_ROWS = 100      # mínimo de filas (t,h) para entrenar
WARMUP_DAYS = 35          # primer origen válido = primer día + 35
ANCHOR_DAYS = 21          # ventana (días) del ancla de nivel al origen
WEATHER_COLS = ['tmean', 'tmin', 'tmax', 'himean', 'himax', 'wind', 'rain', 'cloud']
MIN_WEATHER_COVERAGE = 0.5  # cobertura mínima de clima real en train; si no, solo calendario
CORRUPT_RATE_MAX = 0.5    # un periodo es "confiable" si su tasa de corrupción < 0.5
# En producción solo hay clima observado + pronóstico de ~12 días: para h mayor se usa climatología
WEATHER_KNOWN_HORIZON = 12


# ---------------------------------------------------------------------------
# Clima diario
# ---------------------------------------------------------------------------

def load_daily_weather(path) -> pd.DataFrame:
    """
    Clima diario ya parseado, con caché por ruta+mtime+tamaño (evita re-parsear el
    mismo clima_new.csv en cada request/predicción; el constructor del forecaster ya
    lee el mismo archivo). Ver `_load_daily_weather_uncached`.
    """
    try:
        ruta = os.path.abspath(os.fspath(path))
        st = os.stat(ruta)
    except OSError:
        return _load_daily_weather_uncached(path)
    with _WEATHER_CACHE_GUARD:
        entrada = _WEATHER_CACHE.get(ruta)
        if entrada is not None and entrada[0] == st.st_mtime_ns and entrada[1] == st.st_size:
            return entrada[2]
    df = _load_daily_weather_uncached(path)
    with _WEATHER_CACHE_GUARD:
        _WEATHER_CACHE[ruta] = (st.st_mtime_ns, st.st_size, df)
    return df


# Caché de clima diario ya parseado (una entrada por ruta; se invalida si cambia mtime/tamaño).
_WEATHER_CACHE: Dict[str, Tuple[int, int, pd.DataFrame]] = {}
_WEATHER_CACHE_GUARD = threading.Lock()


def _load_daily_weather_uncached(path) -> pd.DataFrame:
    """
    Lee clima_new.csv (fecha,periodo,p_t,p_h,p_v,p_i) y lo agrega por día.

    El feed trae filas corruptas (p_h == p_t: humedad copiada de la
    temperatura, y p_t/p_v rellenados). Si existen, se detectan los periodos
    confiables (tasa de corrupción < 0.5 medida desde la primera fecha
    corrupta) y se usan SOLO esas filas no corruptas para TODOS los días
    (así min/max/media son homogéneos entre días). Sin corrupción se usan
    todas las filas.

    p_i NO son mm de lluvia sino códigos de condición tipo OpenWeather:
    200-599 lluvia/tormenta, 800 despejado, 801-804 nubes.

    Returns:
        DataFrame indexado por fecha (Timestamp normalizado) con columnas
        tmean,tmin,tmax,hmean,himean,himax,wind,rain,cloud. Vacío si el
        archivo no existe o no tiene datos.
    """
    try:
        c = pd.read_csv(path)
    except (FileNotFoundError, OSError, pd.errors.EmptyDataError):
        return pd.DataFrame()
    if c.empty or not {'fecha', 'periodo', 'p_t', 'p_h'} <= set(c.columns):
        return pd.DataFrame()
    for col in ['p_v', 'p_i']:          # columnas opcionales: si faltan quedan NaN (sin KeyError)
        if col not in c.columns:
            c[col] = np.nan

    c['fecha'] = pd.to_datetime(c['fecha'], errors='coerce').dt.normalize()
    for col in ['periodo', 'p_t', 'p_h', 'p_v', 'p_i']:
        c[col] = pd.to_numeric(c[col], errors='coerce')
    c = c.dropna(subset=['fecha', 'periodo'])
    c = c.dropna(subset=['p_t', 'p_h'], how='all')
    if c.empty:
        return pd.DataFrame()

    corrupta = (c['p_h'] == c['p_t']) & c['p_t'].notna()
    if corrupta.any():
        primera = c.loc[corrupta, 'fecha'].min()
        desde = c[c['fecha'] >= primera]
        tasa_periodo = corrupta[desde.index].groupby(desde['periodo']).mean()
        confiables = tasa_periodo[tasa_periodo < CORRUPT_RATE_MAX].index
        logger.warning(
            f"⚠ Clima corrupto (p_h == p_t): {corrupta.mean()*100:.1f}% de las filas "
            f"(desde {primera.date()}). Usando solo periodos confiables {sorted(int(p) for p in confiables)} "
            f"y filas no corruptas."
        )
        c = c[c['periodo'].isin(confiables) & ~corrupta]
        if c.empty:
            return pd.DataFrame()

    c = c.copy()
    c['hi'] = calcular_sensacion_termica(c['p_t'], c['p_h'])
    c['rain'] = ((c['p_i'] >= 200) & (c['p_i'] < 600)).astype(float).where(c['p_i'].notna())
    c['cloud'] = (c['p_i'] - 800).clip(lower=0).where(c['p_i'] >= 800, 0.0).where(c['p_i'].notna())

    g = c.groupby('fecha').agg(
        tmean=('p_t', 'mean'), tmin=('p_t', 'min'), tmax=('p_t', 'max'),
        hmean=('p_h', 'mean'), himean=('hi', 'mean'), himax=('hi', 'max'),
        wind=('p_v', 'mean'), rain=('rain', 'mean'), cloud=('cloud', 'mean'),
    )
    return g.sort_index()


# ---------------------------------------------------------------------------
# Calendario y anclas de nivel
# ---------------------------------------------------------------------------

def _to_festivo_index(festivos: Optional[Iterable]) -> pd.DatetimeIndex:
    if festivos is None:
        return pd.DatetimeIndex([])
    return pd.DatetimeIndex(pd.to_datetime(list(festivos))).normalize()


def _navidad_mask(idx: pd.DatetimeIndex) -> np.ndarray:
    return np.asarray(((idx.month == 12) & (idx.day >= 23)) | ((idx.month == 1) & (idx.day <= 6)))


def build_calendar_features(dates, festivos) -> pd.DataFrame:
    """Features de calendario (mismo código para train y predict)."""
    idx = pd.DatetimeIndex(pd.to_datetime(dates)).normalize()
    fest = _to_festivo_index(festivos)
    one = pd.Timedelta(days=1)
    d = pd.DataFrame(index=idx)
    d['dow'] = idx.dayofweek
    d['month'] = idx.month
    d['doy'] = idx.dayofyear
    d['fest'] = idx.isin(fest).astype(int)
    d['wkend'] = (d['dow'] >= 5).astype(int)
    d['post_fest'] = (idx - one).isin(fest).astype(int)
    d['pre_fest'] = (idx + one).isin(fest).astype(int)
    d['post2'] = (idx - 2 * one).isin(fest).astype(int)
    d['puente'] = ((d['fest'] == 1) & d['dow'].isin([0, 4])).astype(int)
    d['navidad'] = _navidad_mask(idx).astype(int)
    d['dow_s'] = np.sin(2 * np.pi * d['dow'] / 7)
    d['dow_c'] = np.cos(2 * np.pi * d['dow'] / 7)
    d['m_s'] = np.sin(2 * np.pi * d['doy'] / 365.25)
    d['m_c'] = np.cos(2 * np.pi * d['doy'] / 365.25)
    pascua = pd.DatetimeIndex([pd.Timestamp(easter(int(y))) for y in idx.year])
    off = np.asarray((idx - pascua).days)
    d['sem_santa'] = ((off >= -7) & (off <= 1)).astype(int)    # Domingo de Ramos .. Lunes de Pascua
    d['santo'] = np.isin(off, [-3, -2]).astype(int)            # Jueves / Viernes Santo
    return d


class _Anchors:
    """Anclas de nivel al origen: ancla (media de ANCHOR_DAYS días), m7=a7/ancla,
    m14=a14/ancla (demanda <= origen, excluyendo temporada navideña y festivos)."""

    def __init__(self, demand: pd.Series, festivos):
        dem = demand.dropna().sort_index()
        dem.index = pd.DatetimeIndex(dem.index).normalize()
        fest = _to_festivo_index(festivos)
        ok = ~_navidad_mask(dem.index) & ~dem.index.isin(fest)
        self._idx = dem.index[ok]
        self._val = dem.values[ok]
        self._cache: Dict[pd.Timestamp, Tuple[float, float, float]] = {}

    def first_valid(self):
        return self._idx[0] if len(self._idx) else None

    def at(self, origin) -> Tuple[float, float, float]:
        """(ancla, m7, m14) al origen. ValueError si no hay días válidos <= origen."""
        origin = pd.Timestamp(origin).normalize()
        if origin not in self._cache:
            n = self._idx.searchsorted(origin, side='right')
            h = self._val[:n]
            if len(h) == 0:
                raise ValueError(
                    f"No hay días de demanda válidos (sin festivos ni temporada navideña) "
                    f"hasta {origin.date()} para calcular el nivel de referencia.")
            ancla = h[-ANCHOR_DAYS:].mean()
            self._cache[origin] = (ancla, h[-7:].mean() / ancla, h[-14:].mean() / ancla)
        return self._cache[origin]

    def level_strict(self, origin) -> float:
        """Ancla al origen solo si hay >=ANCHOR_DAYS días válidos y la ventana es reciente
        (<=60 días); si no, NaN."""
        origin = pd.Timestamp(origin).normalize()
        n = self._idx.searchsorted(origin, side='right')
        if n < ANCHOR_DAYS or (origin - self._idx[n - ANCHOR_DAYS]).days > 60:
            return np.nan
        return float(self._val[n - ANCHOR_DAYS:n].mean())


def yoy_growth_factor(demand: pd.Series, festivos, origin, lo: float = 0.8, hi: float = 1.3) -> float:
    """Crecimiento interanual del nivel: ancla(origen) / ancla(origen - 1 año), clip [lo, hi].
    Las ventanas excluyen festivos y navidad (igual que las anclas). 1.0 si alguna de las dos no existe."""
    anchors = _Anchors(clean_demand(demand), festivos)
    origin = pd.Timestamp(origin).normalize()
    a_now = anchors.level_strict(origin)
    a_prev = anchors.level_strict(origin - pd.DateOffset(years=1))
    if not (np.isfinite(a_now) and np.isfinite(a_prev) and a_prev > 0):
        return 1.0
    return float(np.clip(a_now / a_prev, lo, hi))


# ---------------------------------------------------------------------------
# Modelo (ensamble de ratio)
# ---------------------------------------------------------------------------

class _RidgeMember:
    """Ridge con estandarización y dow one-hot (sin dow/month/doy crudos)."""

    def fit(self, X: pd.DataFrame, y):
        self.cols_ = [c for c in X.columns if c not in ('dow', 'month', 'doy')]
        self.med_ = X[self.cols_].median()
        A = self._design(X)
        self.mu_, self.sd_ = A.mean(), A.std().replace(0, 1)
        self.model_ = RidgeCV(alphas=np.logspace(-2, 3, 20)).fit((A - self.mu_) / self.sd_, y)
        return self

    def _design(self, X: pd.DataFrame) -> pd.DataFrame:
        dummies = pd.DataFrame(
            {f'd_{k}': (X['dow'].astype(int).values == k).astype(float) for k in range(7)},
            index=X.index)
        return pd.concat([X[self.cols_].fillna(self.med_), dummies], axis=1)

    def predict(self, X: pd.DataFrame):
        return self.model_.predict((self._design(X) - self.mu_) / self.sd_)


class DailyLevelModel:
    """Ensamble (promedio simple) LightGBM + ExtraTrees + Ridge sobre el ratio."""

    def __init__(self):
        self.feature_names: List[str] = []
        self.weather_cols: List[str] = []
        self.climatology: Optional[pd.DataFrame] = None   # media mensual del clima de entrenamiento
        self.members: Dict[str, object] = {}
        self.median_: Optional[pd.Series] = None

    def fit(self, X: pd.DataFrame, y) -> 'DailyLevelModel':
        self.feature_names = list(X.columns)
        self.median_ = X.median()
        self.members = {
            'lightgbm': lgb.LGBMRegressor(
                n_estimators=300, learning_rate=0.03, num_leaves=8, min_child_samples=20,
                subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=5, verbose=-1),
            'extratrees': ExtraTreesRegressor(
                n_estimators=100, min_samples_leaf=3, max_features=0.7, n_jobs=min(4, os.cpu_count() or 1), random_state=0),
            'ridge': _RidgeMember(),
        }
        Xf = X.fillna(self.median_)
        self.members['lightgbm'].fit(X, y)
        self.members['extratrees'].fit(Xf, y)
        # En predicción (<=30 filas) el despacho multihilo de joblib cuesta más que el cálculo
        # (~3x más lento que n_jobs=1); el fit sí se benefició de n_jobs>1. El atributo queda en el
        # payload, así que se fija a 1 tras entrenar (no altera las predicciones del árbol).
        self.members['extratrees'].set_params(n_jobs=1)
        self.members['ridge'].fit(X, y)
        return self

    def predict_members(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        X = X[self.feature_names]
        Xf = X.fillna(self.median_)
        return {
            'lightgbm': self.members['lightgbm'].predict(X),
            'extratrees': self.members['extratrees'].predict(Xf),
            'ridge': self.members['ridge'].predict(X),
        }

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.mean(list(self.predict_members(X).values()), axis=0)


# ---------------------------------------------------------------------------
# Construcción de filas
# ---------------------------------------------------------------------------

def _climatology(weather: pd.DataFrame, dates: pd.DatetimeIndex, cols: List[str]) -> pd.DataFrame:
    """Media mensual (12 filas) del clima sobre las fechas de entrenamiento."""
    w = weather.reindex(dates)[cols]
    clim = w.groupby(w.index.month).mean().reindex(range(1, 13))
    return clim.fillna(w.mean()).fillna(0.0)


def _weather_block(targets: pd.DatetimeIndex, weather: pd.DataFrame, cols: List[str],
                   clim: pd.DataFrame, horizons) -> np.ndarray:
    """Clima del MISMO día; NaN o h > WEATHER_KNOWN_HORIZON -> climatología mensual del set
    de entrenamiento (igual que en producción, donde no hay clima más allá de ~12 días)."""
    vals = weather.reindex(index=targets, columns=cols).to_numpy(dtype=float)
    clim_v = clim.loc[targets.month, cols].to_numpy(dtype=float)
    usar_clim = np.isnan(vals) | (np.asarray(horizons)[:, None] > WEATHER_KNOWN_HORIZON)
    return np.where(usar_clim, clim_v, vals)


def _assemble(targets, origins, horizons, anchors: _Anchors, festivos, weather,
              weather_cols, clim) -> Tuple[pd.DataFrame, np.ndarray]:
    """Matriz de features para pares (t, origen, h) y vector de ancla."""
    targets = pd.DatetimeIndex(targets)
    cal = build_calendar_features(targets, festivos)
    X = cal.reset_index(drop=True)
    if weather_cols:
        w = _weather_block(targets, weather, weather_cols, clim, horizons)
        for j, col in enumerate(weather_cols):
            X[col] = w[:, j]
    a = np.array([anchors.at(o) for o in origins], dtype=float).reshape(-1, 3)
    X['h'] = np.asarray(horizons, dtype=int)
    X['m7'] = a[:, 1]
    X['m14'] = a[:, 2]
    return X, a[:, 0]


def _train_rows(dem: pd.Series, anchors: _Anchors, festivos, weather, weather_cols, clim):
    first = dem.index[0] + pd.Timedelta(days=WARMUP_DAYS)
    first_ok = anchors.first_valid()
    if first_ok is None:
        raise ValueError("No hay días de demanda válidos (sin festivos ni temporada navideña) para entrenar.")
    first = max(first, first_ok)
    t_list, o_list, h_list = [], [], []
    for h in HORIZONS_TRAIN:
        for t in dem.index:
            o = t - pd.Timedelta(days=h)
            if o >= first:
                t_list.append(t); o_list.append(o); h_list.append(h)
    if not t_list:
        return pd.DataFrame(), np.array([])
    X, ancla = _assemble(t_list, o_list, h_list, anchors, festivos, weather, weather_cols, clim)
    y = dem.loc[t_list].values / ancla
    ok = np.isfinite(y) & np.isfinite(X['m7'].values) & np.isfinite(X['m14'].values)
    return X[ok].reset_index(drop=True), y[ok]


def clean_demand(demand: pd.Series) -> pd.Series:
    """Serie de demanda válida: sin NaN, índice normalizado y único (keep='last'), solo > 0."""
    dem = pd.Series(demand).dropna().astype(float)
    dem.index = pd.DatetimeIndex(dem.index).normalize()
    # sort estable: preserva el orden de entrada en los empates, de modo que keep='last'
    # conserva de forma determinista la última ocurrencia del input para fechas duplicadas.
    dem = dem.sort_index(kind='stable')
    dem = dem[~dem.index.duplicated(keep='last')]
    return dem[dem > 0]


# ---------------------------------------------------------------------------
# Entrenamiento / validación
# ---------------------------------------------------------------------------

def train_daily_level(demand: pd.Series, festivos: Iterable, weather: Optional[pd.DataFrame] = None):
    """
    Entrena el modelo de nivel con validación cronológica 80/20 (orígenes
    rodantes cada 30 días sobre el 20% final) y lo reajusta con el 100%.

    Returns:
        (payload, metrics). payload: dict listo para joblib.dump con
        {'model','feature_names','model_kind','trained_until','metrics'}.
        metrics: {'ensemble': {mape,rmape,r2,mae,...}, 'members': {nombre: {...}}}.

    Raises:
        ValueError: si hay menos de MIN_DAYS días de demanda o muy pocas filas.
    """
    dem = clean_demand(demand)
    if len(dem) < MIN_DAYS:
        raise ValueError(
            f"Se necesitan al menos {MIN_DAYS} días de demanda para entrenar el modelo "
            f"diario (hay {len(dem)})."
        )
    weather = weather if weather is not None else pd.DataFrame()
    fest_idx = _to_festivo_index(festivos)
    weather_cols = [c for c in WEATHER_COLS if c in weather.columns and weather[c].notna().any()]
    # Cobertura de clima real POR COLUMNA en los días de entrenamiento: se descartan individualmente
    # las columnas con cobertura < MIN_WEATHER_COVERAGE; si cae tmean se descarta todo el clima.
    cob_col = {c: float(weather.reindex(dem.index)[c].notna().mean()) for c in weather_cols}
    cobertura = float(np.mean(list(cob_col.values()))) if cob_col else 0.0
    clima_descartado = False
    descartadas = [c for c, v in cob_col.items() if v < MIN_WEATHER_COVERAGE]
    if descartadas:
        weather_cols = [c for c in weather_cols if c not in descartadas]
        if 'tmean' not in weather_cols:
            weather_cols = []
            clima_descartado = True
        logger.warning(f"⚠ Cobertura de clima real en los días de entrenamiento < {MIN_WEATHER_COVERAGE*100:.0f}% "
                       f"en {', '.join(f'{c} ({cob_col[c]*100:.0f}%)' for c in descartadas)}. "
                       + ("Se descarta todo el clima (modelo solo con calendario)." if clima_descartado
                          else "Se descartan solo esas columnas."))
    anchors = _Anchors(dem, fest_idx)

    # --- Validación: fit en el 80% inicial, evaluar en el 20% final ---
    n_train = int(len(dem) * 0.8)
    split = dem.index[n_train - 1]
    dem_tr = dem.iloc[:n_train]
    clim_tr = _climatology(weather, dem_tr.index, weather_cols) if weather_cols else None
    Xtr, ytr = _train_rows(dem_tr, anchors, fest_idx, weather, weather_cols, clim_tr)
    if len(Xtr) < MIN_TRAIN_ROWS:
        raise ValueError("Muy pocas filas de entrenamiento para el modelo diario.")
    val_model = DailyLevelModel().fit(Xtr, ytr)

    val = dem.iloc[n_train:]
    k = ((val.index - split).days - 1) // MAX_HORIZON
    origins = split + pd.to_timedelta(k * MAX_HORIZON, unit='D')
    hs = (val.index - origins).days
    Xv, ancla_v = _assemble(val.index, origins, hs, anchors, fest_idx, weather, weather_cols, clim_tr)
    members = val_model.predict_members(Xv)
    members['ensemble'] = np.mean(list(members.values()), axis=0)
    y_true = val.values
    all_m = {name: calculate_all_metrics(y_true, p * ancla_v) for name, p in members.items()}
    metrics = {'ensemble': all_m.pop('ensemble'), 'members': all_m,
               'clima': {'columnas': list(weather_cols), 'cobertura': cobertura,
                         'cobertura_por_columna': cob_col, 'descartadas': descartadas,
                         'descartado': clima_descartado}}
    logger.info(f"  Validación modelo diario ({len(val)} días): MAPE ensamble "
                f"{metrics['ensemble']['mape']:.2f}% | " +
                ", ".join(f"{n} {m['mape']:.2f}%" for n, m in metrics['members'].items()))

    # --- Reajuste con el 100% ---
    clim = _climatology(weather, dem.index, weather_cols) if weather_cols else None
    X, y = _train_rows(dem, anchors, fest_idx, weather, weather_cols, clim)
    if len(X) < MIN_TRAIN_ROWS:
        raise ValueError("Muy pocas filas de entrenamiento para el modelo diario.")
    model = DailyLevelModel().fit(X, y)
    model.weather_cols = weather_cols
    model.climatology = clim

    payload = {
        'model': model,
        'feature_names': list(model.feature_names),
        'model_kind': MODEL_KIND,
        'trained_until': str(dem.index.max().date()),
        'metrics': metrics,
    }
    return payload, metrics


# ---------------------------------------------------------------------------
# Predicción
# ---------------------------------------------------------------------------

def predict_daily_level(model: DailyLevelModel, demand_history: pd.Series, target_dates,
                        festivos: Iterable, weather: Optional[pd.DataFrame] = None) -> pd.Series:
    """
    Demanda diaria = ratio * ancla. El origen es el último día de historia, el
    horizonte h = clip(días desde el origen, 1, 30) y las anclas quedan
    congeladas en el origen.
    """
    dem = clean_demand(demand_history)
    if dem.empty:
        raise ValueError("Historia de demanda vacía: no se puede predecir.")
    targets = pd.DatetimeIndex(pd.to_datetime(target_dates)).normalize()
    origin = dem.index.max()
    hs = np.clip((targets - origin).days, 1, MAX_HORIZON)
    fest_idx = _to_festivo_index(festivos)
    anchors = _Anchors(dem, fest_idx)
    weather = weather if weather is not None else pd.DataFrame()
    if model.weather_cols:
        dentro = hs <= WEATHER_KNOWN_HORIZON
        if len(weather):
            con_dato = weather.reindex(targets)[model.weather_cols[0]].notna().to_numpy()
        else:
            con_dato = np.zeros(len(targets), dtype=bool)
        n_real = int((dentro & con_dato).sum())
        logger.info(f"  Clima real usado en {n_real}/{len(targets)} días a predecir "
                    f"({int(dentro.sum())} dentro del horizonte h<={WEATHER_KNOWN_HORIZON}); "
                    f"el resto usa climatología mensual")
    X, ancla = _assemble(targets, [origin] * len(targets), hs, anchors, fest_idx, weather,
                         model.weather_cols, model.climatology)
    ratio = model.predict(X)
    return pd.Series(ratio * ancla, index=targets)
