"""
Tests del modelo de nivel diario (/predict-daily): src/models/daily_level_model.py
y su despacho en ForecastPipeline. Sin red.
"""
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

# Añadir raíz del proyecto al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.daily_level_model import (
    MODEL_KIND, _Anchors, build_calendar_features, load_daily_weather,
    predict_daily_level, train_daily_level,
)

FESTIVOS = ['2025-03-24', '2025-04-17', '2025-04-18', '2025-05-01']


def _demanda_sintetica(n=260, inicio='2025-01-01', seed=0, escala=1.0):
    idx = pd.date_range(inicio, periods=n)
    rng = np.random.RandomState(seed)
    dow = idx.dayofweek.values
    base = 1000 + 0.8 * np.arange(n) + np.where(dow >= 5, -120, 0) + 30 * np.sin(2 * np.pi * idx.dayofyear / 365.25)
    y = np.asarray(base + rng.normal(0, 8, n))
    y[idx.isin(pd.to_datetime(FESTIVOS))] -= 150
    return pd.Series(y * escala, index=idx)


def _clima_csv(path, corrupto=True, ndias=40):
    """Clima sintético: 24 periodos x 40 días; desde el día 20 los periodos no múltiplos de 3
    tienen p_h == p_t (humedad copiada de la temperatura)."""
    filas = []
    for d, fecha in enumerate(pd.date_range('2025-06-01', periods=ndias)):
        for p in range(1, 25):
            t, h = 28.0 + (p % 5), 70.0 + (p % 4)
            if corrupto and d >= 20 and p % 3 != 0:
                h = t
            filas.append((fecha.date(), p, t, h, 2.0, 500 if p == 6 else 800))
    pd.DataFrame(filas, columns=['fecha', 'periodo', 'p_t', 'p_h', 'p_v', 'p_i']).to_csv(path, index=False)


def test_load_daily_weather_limpia_corruptos(tmp_path):
    p = tmp_path / 'clima.csv'
    _clima_csv(p, corrupto=True)
    w = load_daily_weather(p)
    assert len(w) == 40
    assert not w[['tmean', 'hmean', 'himean', 'wind']].isna().any().any()
    # humedad plausible (70-73) también en los días posteriores a la corrupción
    assert w['hmean'].between(65, 75).all()


def test_load_daily_weather_sin_corrupcion_usa_todas(tmp_path):
    p = tmp_path / 'clima.csv'
    _clima_csv(p, corrupto=False)
    w = load_daily_weather(p)
    # con todas las filas, tmin/tmax reflejan el rango completo de periodos (28..32)
    assert w['tmin'].iloc[0] == 28.0 and w['tmax'].iloc[0] == 32.0


def test_load_daily_weather_archivo_inexistente(tmp_path):
    assert load_daily_weather(tmp_path / 'no_existe.csv').empty


def test_rain_cloud_desde_codigos(tmp_path):
    p = tmp_path / 'clima.csv'
    filas = [('2025-06-01', per, 30.0, 70.0, 1.0, cod) for per, cod in
             zip(range(1, 5), [500, 800, 802, 200])]
    pd.DataFrame(filas, columns=['fecha', 'periodo', 'p_t', 'p_h', 'p_v', 'p_i']).to_csv(p, index=False)
    w = load_daily_weather(p).iloc[0]
    assert w['rain'] == pytest.approx(0.5)                    # 500 y 200 son lluvia
    assert w['cloud'] == pytest.approx((0 + 0 + 2 + 0) / 4)   # solo p_i>=800 aporta (p_i-800)


def test_calendario_post_pre_navidad():
    f = ['2025-12-25', '2025-05-01']
    fechas = pd.to_datetime(['2025-12-24', '2025-12-25', '2025-12-26', '2025-12-27',
                             '2025-05-01', '2025-05-02', '2025-06-10'])
    c = build_calendar_features(fechas, f)
    assert c.loc['2025-12-24', 'pre_fest'] == 1 and c.loc['2025-12-24', 'navidad'] == 1
    assert c.loc['2025-12-25', 'fest'] == 1
    assert c.loc['2025-12-26', 'post_fest'] == 1
    assert c.loc['2025-12-27', 'post2'] == 1
    assert c.loc['2025-05-01', 'puente'] == 0            # jueves
    assert c.loc['2025-05-02', 'post_fest'] == 1
    assert c.loc['2025-06-10', 'navidad'] == 0 and c.loc['2025-06-10', 'fest'] == 0


def test_anclas_excluyen_festivos_y_navidad():
    idx = pd.date_range('2025-11-01', '2026-01-20')
    dem = pd.Series(100.0, index=idx)
    fest = ['2025-12-08']
    dem.loc['2025-12-08'] = 9999.0                       # festivo: debe ignorarse
    dem.loc['2025-12-23':'2026-01-06'] = 5000.0          # navidad: debe ignorarse
    a28, m7, m14 = _Anchors(dem, fest).at('2026-01-20')
    assert a28 == pytest.approx(100.0) and m7 == pytest.approx(1.0) and m14 == pytest.approx(1.0)


@pytest.fixture(scope='module')
def entrenado():
    dem = _demanda_sintetica()
    payload, metrics = train_daily_level(dem, FESTIVOS, pd.DataFrame())
    return dem, payload, metrics


def test_entrena_y_devuelve_metricas(entrenado):
    _, payload, metrics = entrenado
    assert payload['model_kind'] == MODEL_KIND
    assert payload['feature_names'] == payload['model'].feature_names
    assert {'mape', 'rmape', 'r2', 'mae'} <= set(metrics['ensemble'])
    assert set(metrics['members']) == {'lightgbm', 'extratrees', 'ridge'}
    assert metrics['ensemble']['mape'] < 10


def test_invariancia_de_escala(entrenado):
    dem, payload, _ = entrenado
    p2, _ = train_daily_level(dem * 1.2, FESTIVOS, pd.DataFrame())
    fechas = pd.date_range(dem.index.max() + pd.Timedelta(days=1), periods=30)
    a = predict_daily_level(payload['model'], dem, fechas, FESTIVOS)
    b = predict_daily_level(p2['model'], dem * 1.2, fechas, FESTIVOS)
    assert np.allclose(b.values / a.values, 1.2, rtol=0.03)


def test_ida_y_vuelta_joblib(entrenado, tmp_path):
    dem, payload, _ = entrenado
    p = tmp_path / 'daily_level.joblib'
    joblib.dump(payload, p)
    cargado = joblib.load(p)
    fechas = pd.date_range(dem.index.max() + pd.Timedelta(days=1), periods=10)
    a = predict_daily_level(payload['model'], dem, fechas, FESTIVOS)
    b = predict_daily_level(cargado['model'], dem, fechas, FESTIVOS)
    assert np.allclose(a.values, b.values)
    assert cargado['trained_until'] == str(dem.index.max().date())


def test_clima_ausente_usa_climatologia(tmp_path):
    p = tmp_path / 'clima.csv'
    _clima_csv(p, corrupto=False, ndias=200)
    w = load_daily_weather(p)
    dem = _demanda_sintetica(n=200, inicio='2025-06-01')
    payload, _ = train_daily_level(dem, [], w)
    assert payload['model'].weather_cols
    fechas = pd.date_range(dem.index.max() + pd.Timedelta(days=1), periods=5)   # sin clima
    pred = predict_daily_level(payload['model'], dem, fechas, [], w)
    assert pred.notna().all() and (pred > 0).all()


def test_error_con_pocos_dias():
    with pytest.raises(ValueError):
        train_daily_level(_demanda_sintetica(n=100), FESTIVOS, pd.DataFrame())


class _FestivosFalso:
    festivos = FESTIVOS

    def get_festivos(self, *args, **kwargs):
        return type(self).festivos


def _crear_pipeline(tmp_path, monkeypatch, payload, hist=None, festivos_api=None):
    import src.prediction.forecaster as F
    _FestivosFalso.festivos = festivos_api if festivos_api is not None else FESTIVOS
    monkeypatch.setattr(F, 'FestivosAPIClient', _FestivosFalso)
    if hist is None:
        dem = _demanda_sintetica()
        hist = pd.DataFrame({'FECHA': dem.index, 'TOTAL': dem.values,
                             'is_festivo': dem.index.isin(pd.to_datetime(FESTIVOS)).astype(int)})
    hist_path = tmp_path / 'hist.csv'
    hist.to_csv(hist_path, index=False)
    model_path = tmp_path / 'model.joblib'
    joblib.dump(payload, model_path)
    return F.ForecastPipeline(model_path=str(model_path), historical_data_path=str(hist_path),
                              enable_hourly_disaggregation=False,
                              raw_climate_path=str(tmp_path / 'sin_clima.csv'), ucp='Test')


def _lanza_legacy(*args, **kwargs):
    raise RuntimeError('legacy')


def test_despacho_forecast_pipeline_camino_nuevo(entrenado, tmp_path, monkeypatch):
    _, payload, _ = entrenado
    pipe = _crear_pipeline(tmp_path, monkeypatch, payload)
    assert pipe.model_kind == MODEL_KIND
    # el camino legacy necesita pronóstico de clima: si se usara, revienta
    monkeypatch.setattr(pipe, 'generate_climate_forecast', _lanza_legacy)
    out = pipe.predict_next_n_days(14)
    assert len(out) == 14
    for col in ['fecha', 'demanda_predicha', 'is_festivo', 'is_weekend', 'dayofweek',
                'temp_mean', 'metodo_desagregacion', 'cluster_id', 'P1', 'P24']:
        assert col in out.columns
    assert (out['demanda_predicha'] > 0).all()
    assert out['fecha'].iloc[0] == _demanda_sintetica().index.max() + pd.Timedelta(days=1)


def test_despacho_otro_modelo_usa_camino_legacy(entrenado, tmp_path, monkeypatch):
    _, payload, _ = entrenado
    otro = dict(payload, model_kind='otro')
    pipe = _crear_pipeline(tmp_path, monkeypatch, otro)
    assert pipe.model_kind == 'otro'
    monkeypatch.setattr(pipe, '_predict_next_n_days_daily_level',
                        lambda *a, **k: pytest.fail('no debe usar el camino nuevo'))
    monkeypatch.setattr(pipe, 'generate_climate_forecast', _lanza_legacy)
    with pytest.raises(RuntimeError, match='legacy'):
        pipe.predict_next_n_days(5)


def _parchear_nivel_constante(monkeypatch, valor=1000.0):
    """Reemplaza el modelo por una predicción constante para aislar la lógica de mezcla."""
    import src.prediction.forecaster as F
    monkeypatch.setattr(F, 'predict_daily_level',
                        lambda model, hist, fechas, festivos, weather=None: pd.Series(valor, index=pd.DatetimeIndex(fechas)))


def test_semana_santa_alineada_a_pascua_y_escalada_por_crecimiento(tmp_path, monkeypatch):
    _parchear_nivel_constante(monkeypatch)
    idx = pd.date_range('2024-01-01', '2025-04-10')
    dem = pd.Series(np.where(idx.year == 2024, 1000.0, 1100.0), index=idx)
    dem.loc['2024-03-28'] = 700.0      # Jueves Santo 2024 (festivo)
    dem.loc['2024-04-17'] = 950.0      # misma fecha calendario que Jueves Santo 2025: NO debe usarse
    hist = pd.DataFrame({'FECHA': idx, 'TOTAL': dem.values,
                         'is_festivo': idx.isin(pd.to_datetime(['2024-03-28', '2024-03-29'])).astype(int)})
    pipe = _crear_pipeline(tmp_path, monkeypatch, {'model': 1, 'model_kind': MODEL_KIND}, hist,
                           festivos_api=['2025-04-17', '2025-04-18'])
    out = pipe.predict_next_n_days(14).set_index('fecha')
    # 0.60 * (valor Jueves Santo 2024 * crecimiento 1.1) + 0.40 * nivel del modelo
    assert out.loc['2025-04-17', 'demanda_predicha'] == pytest.approx(0.6 * 700 * 1.1 + 0.4 * 1000, rel=0.01)
    # día normal: sin mezcla
    assert out.loc['2025-04-15', 'demanda_predicha'] == pytest.approx(1000.0)


def test_mezcla_no_aplica_a_festivo_ordinario(tmp_path, monkeypatch):
    _parchear_nivel_constante(monkeypatch)
    pipe = _crear_pipeline(tmp_path, monkeypatch, {'model': 1, 'model_kind': MODEL_KIND})
    out = pipe.predict_next_n_days(3)
    assert (out['demanda_predicha'] == 1000.0).all()


def test_despacho_ultimo_dia_invalido(tmp_path, monkeypatch):
    _parchear_nivel_constante(monkeypatch)
    dem = _demanda_sintetica()
    tot = dem.values.copy()
    tot[-2:] = [0.0, np.nan]                      # últimas filas sin demanda válida
    hist = pd.DataFrame({'FECHA': dem.index, 'TOTAL': tot, 'is_festivo': 0})
    pipe = _crear_pipeline(tmp_path, monkeypatch, {'model': 1, 'model_kind': MODEL_KIND}, hist)
    out = pipe.predict_next_n_days(5)
    # arranca después del último día VÁLIDO (origen del modelo), no después de la última fila
    assert out['fecha'].iloc[0] == dem.index[-3] + pd.Timedelta(days=1)


def test_anclas_sin_dias_validos_lanza_valueerror():
    idx = pd.date_range('2025-12-23', '2026-01-05')        # todo navidad
    with pytest.raises(ValueError):
        _Anchors(pd.Series(100.0, index=idx), []).at('2026-01-05')
    with pytest.raises(ValueError):
        train_daily_level(pd.Series(100.0, index=pd.date_range('2025-01-01', periods=130)),
                          pd.date_range('2025-01-01', periods=130), pd.DataFrame())


def _df_features(n, demanda=None):
    dem = _demanda_sintetica(n=n) if demanda is None else demanda
    return pd.DataFrame({'FECHA': dem.index, 'TOTAL': dem.values,
                         'is_festivo': dem.index.isin(pd.to_datetime(FESTIVOS)).astype(int)})


def test_migracion_champion_legacy(tmp_path, monkeypatch):
    import src.api.main as M
    monkeypatch.chdir(tmp_path)
    reg = tmp_path / 'models' / 'U' / 'registry'
    reg.mkdir(parents=True)
    legacy = reg / 'champion_model_diario.joblib'
    joblib.dump({'model': 1, 'feature_names': []}, legacy)        # sin model_kind = legacy
    df = _df_features(200)
    path, m = M.train_model_if_needed(df, 'U', variant='daily', raw_climate_path=str(tmp_path / 'x.csv'))
    assert m['modelo_seleccionado'] == 'daily_level_ensemble'
    assert joblib.load(path)['model_kind'] == MODEL_KIND
    # segunda llamada: ya es el modelo nuevo, no reentrena
    path2, m2 = M.train_model_if_needed(df, 'U', variant='daily', raw_climate_path=str(tmp_path / 'x.csv'))
    assert m2 == {} and path2 == path


def test_champion_legacy_con_pocas_filas_no_migra(tmp_path, monkeypatch):
    import src.api.main as M
    monkeypatch.chdir(tmp_path)
    reg = tmp_path / 'models' / 'U' / 'registry'
    reg.mkdir(parents=True)
    joblib.dump({'model': 1}, reg / 'champion_model_diario.joblib')
    path, m = M.train_model_if_needed(_df_features(100), 'U', variant='daily')
    assert m == {} and 'model_kind' not in joblib.load(path)


def test_datos_insuficientes_cae_a_legacy(tmp_path, monkeypatch):
    import src.api.main as M
    monkeypatch.chdir(tmp_path)

    class _LegacyLlamado(Exception):
        pass

    class _TrainerFalso:
        def __init__(self, *a, **k):
            raise _LegacyLlamado()

    monkeypatch.setattr(M, 'ModelTrainer', _TrainerFalso)
    # 130 filas pero casi toda la demanda es 0 -> < MIN_DAYS días válidos -> ValueError -> legacy
    dem = _demanda_sintetica(n=130)
    dem.iloc[:60] = 0.0
    with pytest.raises(_LegacyLlamado):
        M.train_model_if_needed(_df_features(130, dem), 'U', variant='daily', force_retrain=True)


def test_tipo_produccion_tendencia_creciente():
    """Serie con ~12% de crecimiento anual + estacionalidad semanal: sin ningún ajuste hardcodeado el
    MAPE a 30 días es < 5% y el nivel sigue la tendencia (sin subestimar > 3% en promedio)."""
    idx = pd.date_range('2025-01-01', periods=450)
    rng = np.random.RandomState(1)
    dow = idx.dayofweek.values
    y = 1000 * (1.12 ** (np.arange(450) / 365)) * np.where(dow >= 5, 0.85, 1.0) * (1 + rng.normal(0, 0.01, 450))
    dem = pd.Series(y, index=idx)
    corte = 420
    payload, _ = train_daily_level(dem.iloc[:corte], [], pd.DataFrame())
    fechas = idx[corte:corte + 30]
    pred = predict_daily_level(payload['model'], dem.iloc[:corte], fechas, [])
    real = dem.iloc[corte:corte + 30]
    assert (abs(pred - real) / real * 100).mean() < 5
    assert ((pred - real) / real * 100).mean() > -3


# ---------------------------------------------------------------------------
# Robustez (ronda 2): migración, sidecar, locks, sanitización, clima
# ---------------------------------------------------------------------------

def _champion_legacy(tmp_path, monkeypatch, contenido=None):
    monkeypatch.chdir(tmp_path)
    reg = tmp_path / 'models' / 'U' / 'registry'
    reg.mkdir(parents=True)
    legacy = reg / 'champion_model_diario.joblib'
    if contenido is None:
        joblib.dump({'model': 1, 'feature_names': []}, legacy)
    else:
        legacy.write_bytes(contenido)
    return reg, legacy


def _falla(*a, **k):
    raise RuntimeError('boom')


def test_migracion_con_excepcion_no_valueerror_conserva_champion(tmp_path, monkeypatch):
    import src.api.main as M
    _, legacy = _champion_legacy(tmp_path, monkeypatch)
    monkeypatch.setattr(M, '_train_daily_level_model', _falla)
    path, m = M.train_model_if_needed(_df_features(200), 'U', variant='daily')
    assert path.resolve() == legacy.resolve() and m == {}


def test_primer_entrenamiento_con_excepcion_cae_a_legacy(tmp_path, monkeypatch):
    import src.api.main as M
    monkeypatch.chdir(tmp_path)

    class _LegacyLlamado(Exception):
        pass

    class _TrainerFalso:
        def __init__(self, *a, **k):
            raise _LegacyLlamado()

    monkeypatch.setattr(M, '_train_daily_level_model', _falla)
    monkeypatch.setattr(M, 'ModelTrainer', _TrainerFalso)
    with pytest.raises(_LegacyLlamado):
        M.train_model_if_needed(_df_features(200), 'U', variant='daily')


def test_champion_ilegible_se_reentrena_y_se_respalda(tmp_path, monkeypatch):
    import src.api.main as M
    reg, champion = _champion_legacy(tmp_path, monkeypatch, contenido=b'no es un joblib')
    path, m = M.train_model_if_needed(_df_features(200), 'U', variant='daily',
                                      raw_climate_path=str(tmp_path / 'x.csv'))
    assert m['modelo_seleccionado'] == 'daily_level_ensemble'
    assert joblib.load(path)['model_kind'] == MODEL_KIND
    assert (reg / 'champion_model_diario.corrupt.joblib').read_bytes() == b'no es un joblib'
    assert not (reg / 'champion_model_diario.legacy.joblib').exists()


def test_champion_ilegible_con_fallo_queda_en_backoff(tmp_path, monkeypatch):
    import src.api.main as M
    _, champion = _champion_legacy(tmp_path, monkeypatch, contenido=b'no es un joblib')
    llamadas = []
    monkeypatch.setattr(M, '_train_daily_level_model', lambda *a, **k: (llamadas.append(1), _falla())[1])
    for _ in range(3):
        path, m = M.train_model_if_needed(_df_features(200), 'U', variant='daily')
        assert m == {} and path.resolve() == champion.resolve()
    assert len(llamadas) == 1 and 'U' in M._DAILY_FAILURES           # 2 y 3 en backoff


def test_migracion_respalda_legacy_y_escribe_sidecar(tmp_path, monkeypatch):
    import src.api.main as M
    reg, legacy = _champion_legacy(tmp_path, monkeypatch)
    path, m = M.train_model_if_needed(_df_features(200), 'U', variant='daily',
                                      raw_climate_path=str(tmp_path / 'x.csv'))
    backup = reg / 'champion_model_diario.legacy.joblib'
    assert backup.exists() and joblib.load(backup) == {'model': 1, 'feature_names': []}
    meta = json.loads((reg / 'champion_model_diario.meta.json').read_text(encoding='utf-8'))
    assert meta['model_kind'] == MODEL_KIND and meta['trained_until'] and meta['size'] == path.stat().st_size
    # un reentrenamiento posterior NO pisa el respaldo del legacy
    M.train_model_if_needed(_df_features(200), 'U', variant='daily', force_retrain=True,
                            raw_climate_path=str(tmp_path / 'x.csv'))
    assert joblib.load(backup) == {'model': 1, 'feature_names': []}


def test_sidecar_evita_joblib_load_y_se_regenera(tmp_path, monkeypatch):
    import src.api.main as M
    reg, _ = _champion_legacy(tmp_path, monkeypatch)
    path, _ = M.train_model_if_needed(_df_features(200), 'U', variant='daily',
                                      raw_climate_path=str(tmp_path / 'x.csv'))
    with monkeypatch.context() as mp:
        mp.setattr(joblib, 'load', _falla)
        assert M._champion_model_kind(path) == MODEL_KIND          # solo con el sidecar
    meta = reg / 'champion_model_diario.meta.json'
    meta.unlink()                                                   # champion de la versión anterior: sin sidecar
    assert M._champion_model_kind(path) == MODEL_KIND
    assert meta.exists()                                            # se regenera al leerlo


def test_lock_por_ucp_un_solo_entrenamiento(tmp_path, monkeypatch):
    import threading
    import src.api.main as M
    monkeypatch.chdir(tmp_path)
    original = M._train_daily_level_model
    llamadas = []

    def _contado(*a, **k):
        llamadas.append(1)
        return original(*a, **k)

    monkeypatch.setattr(M, '_train_daily_level_model', _contado)
    df = _df_features(200)
    barrera = threading.Barrier(2)
    salidas = []

    def _worker():
        barrera.wait()
        salidas.append(M.train_model_if_needed(df, 'U', variant='daily',
                                               raw_climate_path=str(tmp_path / 'x.csv')))

    hilos = [threading.Thread(target=_worker) for _ in range(2)]
    for h in hilos:
        h.start()
    for h in hilos:
        h.join()
    assert len(llamadas) == 1 and len(salidas) == 2
    assert sorted(len(m) > 0 for _, m in salidas) == [False, True]   # uno entrenó, el otro reutilizó


def test_sanitiza_metricas_no_finitas():
    import src.api.main as M
    m = M._finite_or_none({'mape': 3.0, 'rmape': float('inf'), 'x': {'r2': float('nan'), 'mae': np.float64(2.0)}})
    assert m == {'mape': 3.0, 'rmape': None, 'x': {'r2': None, 'mae': 2.0}}
    json.dumps(m, allow_nan=False)


def test_clima_baja_cobertura_entrena_solo_calendario():
    dem = _demanda_sintetica(n=200)
    fechas = dem.index[:10]                                     # 5% de cobertura
    w = pd.DataFrame({c: 1.0 for c in ['tmean', 'tmin', 'tmax', 'himean', 'himax', 'wind', 'rain', 'cloud']},
                     index=fechas)
    payload, _ = train_daily_level(dem, [], w)
    assert payload['model'].weather_cols == []
    assert not any(c in payload['feature_names'] for c in ['tmean', 'rain'])


def test_clima_sin_columnas_opcionales(tmp_path):
    p = tmp_path / 'clima.csv'
    filas = [('2025-06-01', per, 30.0, 70.0) for per in range(1, 7)]
    pd.DataFrame(filas, columns=['fecha', 'periodo', 'p_t', 'p_h']).to_csv(p, index=False)   # sin p_v ni p_i
    w = load_daily_weather(p)
    assert len(w) == 1 and w['tmean'].iloc[0] == 30.0
    assert np.isnan(w['wind'].iloc[0]) and np.isnan(w['rain'].iloc[0]) and np.isnan(w['cloud'].iloc[0])


def test_yoy_growth_factor_publico():
    from src.models.daily_level_model import yoy_growth_factor
    idx = pd.date_range('2024-01-01', '2025-04-10')
    dem = pd.Series(np.where(idx.year == 2024, 1000.0, 1100.0), index=idx)
    assert yoy_growth_factor(dem, [], '2025-04-10') == pytest.approx(1.1)
    assert yoy_growth_factor(dem.loc['2025-01-01':], [], '2025-04-10') == 1.0     # sin ventana de hace 1 año
    assert yoy_growth_factor(dem, [], '2025-04-10', hi=1.05) == pytest.approx(1.05)


def test_indice_duplicado_no_rompe_el_despacho(tmp_path, monkeypatch):
    _parchear_nivel_constante(monkeypatch)
    dem = _demanda_sintetica()
    hist = pd.DataFrame({'FECHA': dem.index, 'TOTAL': dem.values, 'is_festivo': 0})
    hist = pd.concat([hist, hist.tail(3)], ignore_index=True)       # últimas 3 fechas duplicadas
    pipe = _crear_pipeline(tmp_path, monkeypatch, {'model': 1, 'model_kind': MODEL_KIND}, hist)
    out = pipe.predict_next_n_days(4)
    assert len(out) == 4 and out['fecha'].iloc[0] == dem.index[-1] + pd.Timedelta(days=1)


# ---------------------------------------------------------------------------
# Ronda 3: imports tolerantes, fast-path sin lock, backoff, force, champion ilegible, metadatos de clima
# ---------------------------------------------------------------------------

REPO = Path(__file__).resolve().parent.parent

_SCRIPT_IMPORT = (
    "import sys\n"
    "src, root, modo = sys.argv[1], sys.argv[2], sys.argv[3]\n"
    "base = [p for p in sys.path if p not in ('', root, src)]\n"
    "sys.path[:] = ([src] if modo == 'solo_src' else [root, src]) + base\n"
    "import prediction.forecaster as f\n"
    "print(f.predict_daily_level is not None, f.DAILY_LEVEL_MODEL_KIND)\n"
)


@pytest.mark.parametrize('modo', ['solo_src', 'raiz_y_src'])
def test_import_forecaster_con_solo_src_en_path(tmp_path, modo):
    """Regresión: en HEAD `import prediction.forecaster` con solo src/ en sys.path funcionaba."""
    import os
    import subprocess
    env = {k: v for k, v in os.environ.items() if k != 'PYTHONPATH'}
    out = subprocess.run([sys.executable, '-c', _SCRIPT_IMPORT, str(REPO / 'src'), str(REPO), modo],
                         cwd=tmp_path, env=env, capture_output=True, text=True, timeout=180)
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.strip().splitlines()[-1] == f'True {MODEL_KIND}'


@pytest.fixture(autouse=True)
def _limpiar_fallos_diarios():
    import src.api.main as M
    M._DAILY_FAILURES.clear()
    M._DAILY_LOCKS.clear()
    yield
    M._DAILY_FAILURES.clear()
    M._DAILY_LOCKS.clear()


def _entrenar_champion_nivel(M, tmp_path, monkeypatch, n=200):
    monkeypatch.chdir(tmp_path)
    df = _df_features(n)
    path, _ = M.train_model_if_needed(df, 'U', variant='daily', raw_climate_path=str(tmp_path / 'x.csv'))
    return df, path


def test_fast_path_sin_lock_mientras_otro_hilo_entrena(tmp_path, monkeypatch):
    import threading
    import src.api.main as M
    df, champion = _entrenar_champion_nivel(M, tmp_path, monkeypatch)
    original = M._train_daily_level_model
    entro, liberar = threading.Event(), threading.Event()

    def _bloqueado(*a, **k):
        entro.set()
        assert liberar.wait(timeout=60)
        return original(*a, **k)

    monkeypatch.setattr(M, '_train_daily_level_model', _bloqueado)
    hilo = threading.Thread(target=lambda: M.train_model_if_needed(
        df, 'U', variant='daily', force_retrain=True, raw_climate_path=str(tmp_path / 'x.csv')))
    hilo.start()
    try:
        assert entro.wait(timeout=30)                                   # el hilo tiene el lock
        res = []
        t = threading.Thread(target=lambda: res.append(
            M.train_model_if_needed(df, 'U', variant='daily', raw_climate_path=str(tmp_path / 'x.csv'))))
        t.start()
        t.join(timeout=15)
        assert not t.is_alive(), 'la llamada normal esperó al lock'
        assert res[0][1] == {} and res[0][0].resolve() == champion.resolve()
    finally:
        liberar.set()
        hilo.join(timeout=120)


def test_backoff_de_migracion_fallida(tmp_path, monkeypatch):
    """Backoff con reloj simulado + caída a legacy con force_retrain."""
    import src.api.main as M
    _, legacy = _champion_legacy(tmp_path, monkeypatch)
    reloj = [1000.0]
    monkeypatch.setattr(M.time, 'monotonic', lambda: reloj[0])
    llamadas = []
    monkeypatch.setattr(M, '_train_daily_level_model', lambda *a, **k: (llamadas.append(1), _falla())[1])
    df = _df_features(200)
    for _ in range(3):
        path, m = M.train_model_if_needed(df, 'U', variant='daily')
        assert m == {} and path.resolve() == legacy.resolve()
    assert len(llamadas) == 1                                           # 2 y 3 en backoff
    reloj[0] += M._DAILY_RETRY_INTERVAL_S - 1
    M.train_model_if_needed(df, 'U', variant='daily')
    assert len(llamadas) == 1                                           # aún dentro del intervalo
    monkeypatch.setattr(M, 'ModelTrainer', lambda *a, **k: (_ for _ in ()).throw(KeyError('legacy')))
    with pytest.raises(KeyError, match='legacy'):                       # force con legacy que falla: cae a legacy
        M.train_model_if_needed(df, 'U', variant='daily', force_retrain=True)
    assert len(llamadas) == 2                                           # force_retrain salta el backoff
    reloj[0] += M._DAILY_RETRY_INTERVAL_S + 1
    M.train_model_if_needed(df, 'U', variant='daily')
    assert len(llamadas) == 3                                           # vencido el intervalo, reintenta


def test_force_con_champion_nivel_y_fallo_conserva_champion(tmp_path, monkeypatch):
    import src.api.main as M
    df, champion = _entrenar_champion_nivel(M, tmp_path, monkeypatch)
    monkeypatch.setattr(M, '_train_daily_level_model', _falla)
    monkeypatch.setattr(M, 'ModelTrainer', lambda *a, **k: pytest.fail('no debe degradar a legacy'))
    path, m = M.train_model_if_needed(df, 'U', variant='daily', force_retrain=True)
    assert m == {} and path.resolve() == champion.resolve()
    assert joblib.load(path)['model_kind'] == MODEL_KIND


def test_metricas_incluyen_info_de_clima(tmp_path, monkeypatch):
    import src.api.main as M
    monkeypatch.chdir(tmp_path)
    clima = tmp_path / 'clima.csv'
    _clima_csv(clima, corrupto=False, ndias=200)
    df = _df_features(200, _demanda_sintetica(n=200, inicio='2025-06-01'))
    _, m = M.train_model_if_needed(df, 'U', variant='daily',
                                   raw_climate_path=str(clima))
    assert m['columnas_clima'] and m['cobertura_clima'] > 0.9 and 'clima_descartado' not in m
    # cobertura baja: clima descartado
    _clima_csv(clima, corrupto=False, ndias=40)
    _, m2 = M.train_model_if_needed(df, 'U2', variant='daily', raw_climate_path=str(clima))
    assert m2['columnas_clima'] == [] and m2['cobertura_clima'] < 0.5 and m2['clima_descartado'] is True
    json.dumps(m2, allow_nan=False)


def test_log_clima_distingue_horizonte(tmp_path, caplog):
    import logging
    clima = tmp_path / 'clima.csv'
    _clima_csv(clima, corrupto=False, ndias=200)
    w = load_daily_weather(clima)
    dem = _demanda_sintetica(n=200, inicio='2025-06-01')
    payload, _ = train_daily_level(dem, [], w)
    fechas = pd.date_range(dem.index.max() + pd.Timedelta(days=1), periods=20)
    with caplog.at_level(logging.INFO, logger='src.models.daily_level_model'):
        predict_daily_level(payload['model'], dem, fechas, [], w)
    texto = ' '.join(r.getMessage() for r in caplog.records)
    assert '0/20' in texto and '12 dentro del horizonte' in texto


def test_champion_ilegible_en_flujo_diario_da_500(tmp_path, monkeypatch):
    import asyncio
    from fastapi import HTTPException
    import src.api.main as M
    import src.prediction.forecaster as F
    _, champion = _champion_legacy(tmp_path, monkeypatch, contenido=b'no es un joblib')
    monkeypatch.setattr(M, '_train_daily_level_model', _falla)      # reentrenamiento falla: sigue ilegible
    (tmp_path / 'data' / 'features' / 'U').mkdir(parents=True)
    monkeypatch.setattr(F, 'FestivosAPIClient', _FestivosFalso)
    monkeypatch.setattr(M, 'full_update_csv_diario', lambda ucp: None)
    monkeypatch.setattr(M, 'run_automated_pipeline', lambda **k: (_df_features(200), None))
    with pytest.raises(HTTPException) as e:
        asyncio.run(M.run_predict_daily_flow(M.PredictRequest(ucp='U', n_days=5)))
    assert e.value.status_code == 500 and 'force_retrain=true' in e.value.detail
    assert list((tmp_path / 'data' / 'features' / 'U').glob('temp_api_features_daily*')) == []   # sin huérfanos


def test_cobertura_por_columna_descarta_solo_las_bajas():
    dem = _demanda_sintetica(n=200)
    cols = ['tmean', 'tmin', 'tmax', 'himean', 'himax', 'wind', 'rain', 'cloud']
    w = pd.DataFrame({c: 25.0 for c in cols}, index=dem.index)
    w.loc[w.index[:150], 'wind'] = np.nan                        # viento con 25% de cobertura
    payload, m = train_daily_level(dem, [], w)
    assert 'wind' not in payload['model'].weather_cols and 'tmean' in payload['model'].weather_cols
    assert m['clima']['descartadas'] == ['wind'] and m['clima']['descartado'] is False
    assert m['clima']['cobertura_por_columna']['wind'] == pytest.approx(0.25)
    # si cae tmean se descarta todo el clima
    w2 = w.copy()
    w2.loc[w2.index[:150], 'tmean'] = np.nan
    payload2, m2 = train_daily_level(dem, [], w2)
    assert payload2['model'].weather_cols == [] and m2['clima']['descartado'] is True


def test_metricas_y_sidecar_con_cobertura_por_columna(tmp_path, monkeypatch):
    import src.api.main as M
    monkeypatch.chdir(tmp_path)
    clima = tmp_path / 'clima.csv'
    _clima_csv(clima, corrupto=False, ndias=200)
    df = _df_features(200, _demanda_sintetica(n=200, inicio='2025-06-01'))
    path, m = M.train_model_if_needed(df, 'U', variant='daily', raw_climate_path=str(clima))
    assert set(m['cobertura_clima_por_columna']) == set(m['columnas_clima'])
    meta = json.loads(path.with_name('champion_model_diario.meta.json').read_text(encoding='utf-8'))
    assert meta['columnas_clima'] == m['columnas_clima']


def test_fallback_legacy_toma_el_lock_por_ucp(tmp_path, monkeypatch):
    """El entrenamiento legacy del diario (<120 filas) no corre en paralelo con otro del mismo UCP."""
    import threading
    import src.api.main as M
    monkeypatch.chdir(tmp_path)
    activos, maximo, lk = [0], [0], threading.Lock()

    def _legacy_falso(df, ucp, force_retrain=False, variant='hourly'):
        with lk:
            activos[0] += 1
            maximo[0] = max(maximo[0], activos[0])
        time.sleep(0.3)
        with lk:
            activos[0] -= 1
        return Path('x'), {}

    monkeypatch.setattr(M, '_train_model_legacy', _legacy_falso)
    df = _df_features(100)                                       # < 120 filas -> camino legacy
    barrera = threading.Barrier(2)

    def _worker():
        barrera.wait()
        M.train_model_if_needed(df, 'U', variant='daily')

    hilos = [threading.Thread(target=_worker) for _ in range(2)]
    for h in hilos:
        h.start()
    for h in hilos:
        h.join()
    assert maximo[0] == 1


def test_archivo_temporal_unico_y_borrado_si_falla(tmp_path, monkeypatch):
    import asyncio
    from fastapi import HTTPException
    import src.api.main as M
    monkeypatch.chdir(tmp_path)
    carpeta = tmp_path / 'data' / 'features' / 'U'
    carpeta.mkdir(parents=True)
    rutas = []

    class _PipelineFalso:
        def __init__(self, *a, historical_data_path=None, **k):
            rutas.append(historical_data_path)
            assert Path(historical_data_path).exists()
            raise RuntimeError('falla el constructor')

    monkeypatch.setattr(M, 'ForecastPipeline', _PipelineFalso)
    monkeypatch.setattr(M, 'full_update_csv_diario', lambda ucp: None)
    monkeypatch.setattr(M, 'run_automated_pipeline', lambda **k: (_df_features(200), None))
    for _ in range(2):
        with pytest.raises(HTTPException):
            asyncio.run(M.run_predict_daily_flow(M.PredictRequest(ucp='U', n_days=5)))
    assert len(set(rutas)) == 2 and all(Path(r).parent.resolve() == carpeta.resolve() for r in rutas)
    assert list(carpeta.glob('temp_api_features_daily*')) == []
