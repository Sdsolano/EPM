"""Tests de src/adjustments/retro.py (P1: diagnóstico informativo)."""

from datetime import date, timedelta

import numpy as np
import pytest

from src.adjustments import retro

LUNES0 = date(2026, 1, 5)  # lunes


def _real(d: date, base: float = 1000.0) -> list:
    return [base / 24 * (1 + 0.1 * (d.weekday() == 0))] * 24


def _serie(n_dias: int, factor, festivos=(), inicio: date = LUNES0, base: float = 1000.0):
    """Emisiones (una por semana) y reales; reportado = real * factor(d)."""
    reales, curvas = {}, {}
    for i in range(n_dias):
        d = inicio + timedelta(days=i)
        r = np.array(_real(d, base))
        reales[d.isoformat()] = r.tolist()
        curvas[d.isoformat()] = (r * factor(d)).tolist()
    emis = []
    for k in range(0, n_dias, 7):
        lunes = inicio + timedelta(days=k)
        sem = {f: v for f, v in curvas.items() if lunes <= date.fromisoformat(f) < lunes + timedelta(days=7)}
        emis.append({"semana_inicio": lunes.isoformat(), "version": 1, "curvas": sem})
    return emis, reales


def test_sintetico_signo_y_magnitud():
    emis, reales = _serie(84, lambda d: 1.02)
    r = retro.resumen_aprendizaje("X", emis, reales, [], semanas=12, hoy="2026-04-01")
    assert r["estado"] == "ok" and r["modo"] == "error"
    assert r["sesgo_global"] == {"pct": 2.0, "wape_pct": 2.0, "n": 84}
    assert r["base"]["n_dias"] == 84 and r["base"]["semanas"] == 12
    assert [v["semanas"] for v in r["ventanas"]] == [4, 8, 12]
    assert all(v["sesgo_pct"] == 2.0 and v["n"] == 7 * v["semanas"] for v in r["ventanas"])
    assert r["ema"] == {"h56": 2.0, "h112": 2.0}
    assert len(r["ultima_semana"]) == 7
    assert r["ultima_semana"][0]["error_pct"] == 2.0
    assert [b["bloque"] for b in r["por_bloque"]] == ["madrugada", "manana", "tarde", "noche"]
    assert all(b["sesgo_pct"] == 2.0 for b in r["por_bloque"])
    assert set(r) >= {"base", "sesgo_global", "ventanas", "ema", "por_tipo", "por_bloque",
                      "ultima_semana", "autocorr_lag", "min_casos", "avisos"}


def test_signo_negativo_cuando_reportado_debajo():
    emis, reales = _serie(28, lambda d: 0.97)
    r = retro.resumen_aprendizaje("X", emis, reales, [], semanas=4, hoy="2026-02-20")
    assert r["sesgo_global"]["pct"] == -3.0


def test_ema_serie_conocida():
    # dos puntos: edad 0 (peso 1) con 4 y edad h (peso .5) con 1 -> (4+0.5)/1.5 = 3
    errs = {"2026-01-10": 1.0, "2026-01-20": 4.0}
    assert retro.sesgo_ema(errs, 10, "2026-01-20") == pytest.approx(3.0)
    assert retro.sesgo_ema({}, 10) is None
    with pytest.raises(ValueError):
        retro.sesgo_ema(errs, 0)


def test_winsorize():
    assert retro.winsorizar(12.0) == 5.0 and retro.winsorizar(-9.0) == -5.0 and retro.winsorizar(2.0) == 2.0
    # un día atípico +50 % cuenta como +5 en el promedio
    errs = {"2026-01-01": 0.0, "2026-01-02": 50.0}
    assert retro.sesgo_ema(errs, 1e9, "2026-01-02") == pytest.approx(2.5, abs=1e-6)
    emis, reales = _serie(28, lambda d: 1.5 if d == LUNES0 else 1.0)
    r = retro.resumen_aprendizaje("X", emis, reales, [], semanas=4, hoy="2026-02-20")
    assert r["sesgo_global"]["pct"] == pytest.approx(5.0 / 28, abs=0.01)


def test_sin_fuga_de_futuro():
    # los últimos 7 días son un disparate; con hasta anterior no deben influir
    f = lambda d: 3.0 if d >= LUNES0 + timedelta(days=21) else 1.01
    emis, reales = _serie(28, f)
    r = retro.resumen_aprendizaje("X", emis, reales, [], semanas=4, hasta="2026-01-25", hoy="2026-02-20")
    assert r["base"]["hasta"] == "2026-01-25" and r["base"]["n_dias"] == 21
    assert r["sesgo_global"]["pct"] == 1.0 and r["ema"]["h56"] == 1.0
    assert r["base"]["ultimo_real"] == "2026-01-25"
    # sesgo_ema ignora fechas posteriores a hasta
    assert retro.sesgo_ema({"2026-01-01": 1.0, "2026-02-01": 5.0}, 10, "2026-01-15") == pytest.approx(1.0)


def test_tipos_de_dia():
    fest = ["2026-01-12"]  # lunes festivo -> martes post_festivo; domingo 11 es domingo
    emis, reales = _serie(28, lambda d: 1.0, festivos=fest)
    r = retro.resumen_aprendizaje("X", emis, reales, fest, semanas=4, min_casos=1, hoy="2026-02-20")
    tipos = {x["fecha"]: x["tipo_dia"] for x in r["ultima_semana"]}
    hist = {h["fecha"]: h["tipo_dia"] for h in retro.historial_error(
        {k: v for k, v in _emis_map(emis).items()}, reales, fest)}
    assert hist["2026-01-12"] == "festivo"
    assert hist["2026-01-13"] == "post_festivo"
    assert hist["2026-01-11"] == "domingo"
    assert hist["2026-01-10"] == "sabado"
    assert hist["2026-01-14"] == "laboral"
    assert r["por_tipo"]["festivo"]["n"] == 1 and r["por_tipo"]["post_festivo"]["n"] == 1
    assert tipos  # última semana presente


def _emis_map(emis):
    out = {}
    for e in emis:
        out.update(e["curvas"])
    return out


def test_tipo_insuficiente_nulo():
    fest = ["2026-01-12"]
    emis, reales = _serie(28, lambda d: 1.02)
    r = retro.resumen_aprendizaje("X", emis, reales, fest, semanas=4, min_casos=8, hoy="2026-02-20")
    f = r["por_tipo"]["festivo"]
    assert f == {"sesgo_pct": None, "wape_pct": None, "n": 1, "suficiente": False}
    assert r["por_tipo"]["laboral"]["suficiente"] is True and r["por_tipo"]["laboral"]["sesgo_pct"] == 2.0
    codigos = {a["codigo"] for a in r["avisos"]}
    assert "RETRO_TIPO_INSUFICIENTE" in codigos


def test_avisos_muestra_corta_y_rezago():
    emis, reales = _serie(14, lambda d: 1.01)
    r = retro.resumen_aprendizaje("X", emis, reales, [], semanas=12, hoy="2026-03-01")
    codigos = {a["codigo"] for a in r["avisos"]}
    assert {"RETRO_MUESTRA_CORTA", "RETRO_REZAGO_REAL"} <= codigos
    r2 = retro.resumen_aprendizaje("X", emis, reales, [], semanas=12, hoy="2026-01-19")  # ultimo real 18/01
    assert "RETRO_REZAGO_REAL" not in {a["codigo"] for a in r2["avisos"]}


def test_sin_datos():
    r = retro.resumen_aprendizaje("X", [], {})
    assert set(r) == {"estado", "ucp", "motivo", "avisos"} and r["motivo"]
    assert [x["codigo"] for x in r["avisos"]] == ["RETRO_SIN_HISTORIAL"]
    emis, reales = _serie(7, lambda d: 1.0)
    r = retro.resumen_aprendizaje("X", emis, {}, [])
    assert r["estado"] == "sin_datos"
    # emisiones y reales en fechas disjuntas
    r = retro.resumen_aprendizaje("X", emis, {"2030-01-01": [1.0] * 24}, [])
    assert r["estado"] == "sin_datos"
    # hasta antes de todo
    r = retro.resumen_aprendizaje("X", emis, reales, [], hasta="2020-01-01")
    assert r["estado"] == "sin_datos"


def test_version_menor_es_ex_ante():
    emis, reales = _serie(7, lambda d: 1.00)
    final = {"semana_inicio": LUNES0.isoformat(), "version": 3,
             "curvas": {k: [x * 2 for x in v] for k, v in emis[0]["curvas"].items()}}
    r = retro.resumen_aprendizaje("X", [final] + emis, reales, [], semanas=4, hoy="2026-01-12")
    assert r["sesgo_global"]["pct"] == 0.0


def test_autocorr_y_ultima_semana_rangos():
    rng = np.random.default_rng(1)
    ruido = {(LUNES0 + timedelta(days=i)): 1 + rng.normal(0, 0.02) for i in range(84)}
    emis, reales = _serie(84, lambda d: ruido[d])
    r = retro.resumen_aprendizaje("X", emis, reales, [], semanas=12, hoy="2026-04-01")
    for v in r["autocorr_lag"].values():
        assert v is None or -1 <= v <= 1
    assert r["ultima_semana"][-1]["fecha"] == "2026-03-29"


@pytest.mark.parametrize("kw,msg", [
    (dict(ucp=" "), "ucp"),
    (dict(semanas=3), "semanas"),
    (dict(semanas=27), "semanas"),
    (dict(min_casos=0), "min_casos"),
    (dict(hasta="2026-13-40"), "hasta"),
    (dict(hasta="2026-01-05xyz"), "hasta"),
    (dict(reales={"2026-01-05basura": [1.0] * 24}), "fecha inválida"),
    (dict(emisiones=[{"semana_inicio": "2026-01-06", "curvas": {}}]), "no es lunes"),
    (dict(emisiones=[{"semana_inicio": "xx", "curvas": {}}]), "semana_inicio"),
    (dict(emisiones=[{"semana_inicio": "2026-01-05", "curvas": {"2026-01-05": [1.0] * 23}}]), "24 valores"),
    (dict(emisiones=[{"semana_inicio": "2026-01-05", "curvas": {"2026-01-05": [-1.0] * 24}}]), "negativos"),
    (dict(emisiones=[{"semana_inicio": "2026-01-05", "curvas": {"2026-01-05": [float("nan")] * 24}}]), "NaN"),
    (dict(emisiones=[{"semana_inicio": "2026-01-05", "curvas": {"2026-01-99": [1.0] * 24}}]), "fecha inválida"),
    (dict(reales={"2026-01-05": [1.0] * 5}), "24 valores"),
    (dict(reales={"nope": [1.0] * 24}), "fecha inválida"),
    (dict(festivos="2026-01-01"), "festivos"),
])
def test_entrada_invalida(kw, msg):
    args = dict(ucp="X", emisiones=[], reales={})
    args.update(kw)
    with pytest.raises(ValueError, match=msg):
        retro.validar_entrada(**args)
    with pytest.raises(ValueError, match=msg):
        retro.resumen_aprendizaje(**{k: v for k, v in args.items()})


def test_no_muta_entrada():
    emis, reales = _serie(14, lambda d: 1.02)
    import copy
    e0, r0 = copy.deepcopy(emis), copy.deepcopy(reales)
    retro.resumen_aprendizaje("X", emis, reales, [], hoy="2026-02-01")
    assert emis == e0 and reales == r0


def test_hasta_muy_posterior_a_los_datos_es_sin_datos():
    emis, reales = _serie(28, lambda d: 1.01)
    r = retro.resumen_aprendizaje("X", emis, reales, [], semanas=12, hasta="2027-01-04", hoy="2027-01-04")
    assert r["estado"] == "sin_datos" and "semanas" in r["motivo"]


def test_semanas_sin_off_by_one_y_muestra_corta_por_faltantes():
    # 12 semanas completas terminando en jueves: base.semanas == 12 (no 13) y sin aviso de muestra corta
    emis, reales = _serie(88, lambda d: 1.01)  # 88 dias: termina jueves 2026-04-02
    r = retro.resumen_aprendizaje("X", emis, reales, [], semanas=12, hoy="2026-04-03")
    assert r["base"]["semanas"] == 12 and r["base"]["n_dias"] == 84
    assert "RETRO_MUESTRA_CORTA" not in {a["codigo"] for a in r["avisos"]}
    # falta una semana completa en medio de la ventana: avisa
    emis2, reales2 = _serie(88, lambda d: 1.01)
    for k in range(7):
        reales2.pop((LUNES0 + timedelta(days=46 + k)).isoformat())
    r2 = retro.resumen_aprendizaje("X", emis2, reales2, [], semanas=12, hoy="2026-04-03")
    assert r2["base"]["semanas"] == 11
    assert "RETRO_MUESTRA_CORTA" in {a["codigo"] for a in r2["avisos"]}


def test_ultimo_real_es_el_ultimo_dia_con_real_no_con_emision():
    emis, reales = _serie(28, lambda d: 1.0)
    reales["2026-02-05"] = [10.0] * 24  # real posterior a la ultima emision (2026-02-01)
    r = retro.resumen_aprendizaje("X", emis, reales, [], semanas=4, hoy="2026-02-05")
    assert r["base"]["hasta"] == "2026-02-01" and r["base"]["ultimo_real"] == "2026-02-05"
    assert "RETRO_REZAGO_REAL" not in {a["codigo"] for a in r["avisos"]}


def test_ventanas_mayores_a_semanas_no_disponibles():
    emis, reales = _serie(28, lambda d: 1.02)
    r = retro.resumen_aprendizaje("X", emis, reales, [], semanas=4, hoy="2026-02-01")
    v = {x["semanas"]: x for x in r["ventanas"]}
    assert v[4]["sesgo_pct"] == 2.0 and v[4]["n"] == 28
    for w in (8, 12):
        assert v[w] == {"semanas": w, "sesgo_pct": None, "wape_pct": None, "n": 0}


def test_hoy_por_defecto_usa_bogota(monkeypatch):
    emis, reales = _serie(14, lambda d: 1.0)
    r = retro.resumen_aprendizaje("X", emis, reales, [], semanas=4)  # hoy real: rezago enorme
    assert "RETRO_REZAGO_REAL" in {a["codigo"] for a in r["avisos"]}
    assert str(retro.TZ_BOGOTA) == "America/Bogota"
