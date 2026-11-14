"""Tests de la lógica pura de insights (src/adjustments/insights.py) con datos sintéticos."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pytest

from src.adjustments import data, insights as I
from src.adjustments.types import Alert

LUNES0 = date(2026, 1, 5)  # lunes
N_SEMANAS = 20
FORMA = np.array([0.8, 0.75, 0.72, 0.7, 0.72, 0.8, 0.9, 1.0, 1.05, 1.08, 1.1, 1.12,
                  1.1, 1.08, 1.07, 1.06, 1.05, 1.1, 1.25, 1.3, 1.22, 1.1, 0.98, 0.88])
DOW = [1.0, 1.01, 1.01, 1.02, 1.0, 0.9, 0.8]


def _real(d: date) -> np.ndarray:
    return 400.0 * FORMA * DOW[d.weekday()]


def _datos(sesgo_sabado: float = -0.03, n_semanas: int = N_SEMANAS, festivos=()) -> I.Datos:
    """Pronóstico = real × (1+error): el sábado queda `sesgo_sabado` corto; el resto, ±0.2 % alterno."""
    emitidos, semana_de, reales = {}, {}, {}
    for w in range(n_semanas):
        lunes = LUNES0 + timedelta(weeks=w)
        for i in range(7):
            d = lunes + timedelta(days=i)
            f = d.isoformat()
            r = _real(d)
            err = sesgo_sabado if d.weekday() == 5 else (0.002 if (w + i) % 2 else -0.002)
            reales[f] = r
            emitidos[f] = r * (1 + err)
            semana_de[f] = lunes.isoformat()
    return I.Datos(emitidos=emitidos, semana_de=semana_de, reales=reales, festivos=set(festivos))


def _corrida(semana: date, estado: str = "borrador", estados_paso=None, editadas=(), alertas=(),
             escala: float = 1.0, festivos=()) -> I.Corrida:
    fechas = [(semana + timedelta(days=i)).isoformat() for i in range(7)]
    base = {f: [float(v) for v in _real(date.fromisoformat(f)) * escala] for f in fechas}
    return I.Corrida(
        ucp="Antioquia", semana_inicio=semana, estado=estado, version=2, base=base, final={f: list(v) for f, v in base.items()},
        alertas=list(alertas), editadas=set(editadas),
        estados_paso=estados_paso or {"base": "aplicado", "referencia": "aplicado", "nivel": "aplicado",
                                      "forma": "aplicado", "especiales": "aplicado", "control": "aplicado"},
        ref_inicio=None, festivos=set(festivos))


# ---------------------------------------------------------------------------
# precisión
# ---------------------------------------------------------------------------

def test_precision_sin_datos_degrada():
    p = I.precision(I.Datos())
    assert p["semanas"] == [] and p["por_tipo_dia"] == []
    assert p["resumen"]["semanas"] == 0 and p["resumen"]["wape_pct"] is None
    assert p["resumen"]["tendencia"] == "sin_datos"
    assert p["por_hora"]["sesgo_pct"] == [None] * 24


def test_precision_forma_y_signo():
    d = _datos()
    p = I.precision(d, semanas=5)
    assert [s["semana_inicio"] for s in p["semanas"]] == sorted(s["semana_inicio"] for s in p["semanas"])
    assert len(p["semanas"]) == 5 and p["hasta"] == p["semanas"][-1]["semana_inicio"]
    sem = p["semanas"][-1]
    assert sem["fuente"] == "modelo_historico" and sem["dias_con_datos"] == 7 and len(sem["dias"]) == 7
    # sábados 3 % cortos → error de energía negativo en el sábado, y por tipo de día
    sab = [x for x in sem["dias"] if x["tipo_dia"] == "sabado"][0]
    assert sab["error_energia_pct"] == pytest.approx(-3.0, abs=0.05)
    por_tipo = {t["tipo_dia"]: t for t in p["por_tipo_dia"]}
    assert por_tipo["sabado"]["sesgo_energia_pct"] < -2.5 and por_tipo["sabado"]["n_dias"] == 5
    assert len(p["por_hora"]["sesgo_pct"]) == 24 and len(p["por_hora"]["wape_pct"]) == 24
    assert p["resumen"]["wape_pct"] is not None and p["resumen"]["semanas"] == 5


def test_precision_hasta_y_semanas_incompletas():
    d = _datos()
    ultimo = LUNES0 + timedelta(weeks=N_SEMANAS - 1)
    d.reales.pop((ultimo + timedelta(days=3)).isoformat())  # la última semana queda sin un día real
    p = I.precision(d, semanas=52)
    assert p["hasta"] == (ultimo - timedelta(weeks=1)).isoformat() and len(p["semanas"]) == N_SEMANAS - 1
    p2 = I.precision(d, semanas=52, hasta=LUNES0 + timedelta(weeks=3))
    assert len(p2["semanas"]) == 4 and p2["hasta"] == (LUNES0 + timedelta(weeks=3)).isoformat()


def test_precision_aprobada_prevalece_y_marca_fuente():
    d = _datos()
    lunes = LUNES0 + timedelta(weeks=4)
    d.aprobadas = {lunes.isoformat(): {(lunes + timedelta(days=i)).isoformat(): list(_real(lunes + timedelta(days=i)))
                                       for i in range(7)}}
    p = I.precision(d, semanas=52)
    s = {x["semana_inicio"]: x for x in p["semanas"]}
    assert s[lunes.isoformat()]["fuente"] == "corrida_aprobada"
    assert s[lunes.isoformat()]["wape_pct"] == 0.0  # la aprobada era exacta
    assert s[(lunes + timedelta(weeks=1)).isoformat()]["fuente"] == "modelo_historico"


def test_tendencia():
    assert I._tendencia([5, 5, 4.9]) == "sin_datos"
    assert I._tendencia([5.0, 5.0, 3.0, 3.0]) == "mejorando"
    assert I._tendencia([3.0, 3.0, 5.0, 5.0]) == "empeorando"
    assert I._tendencia([3.0, 3.1, 3.0, 3.1]) == "estable"


def test_nivel_confianza_umbrales():
    assert [I.nivel_confianza(x) for x in (None, 2.9, 3.0, 4.0, 6.0, 6.1)] == [
        "sin_datos", "alta", "alta", "media", "media", "baja"]


# ---------------------------------------------------------------------------
# banda
# ---------------------------------------------------------------------------

def test_banda_cuantiles_y_festivo_usa_domingo():
    d = _datos()
    t = I.tablas_banda(d)
    sab = t["sabado"]
    assert sab["n_dias"] == N_SEMANAS
    # real/pronóstico − 1 con pronóstico 3 % corto = +3.09 %
    assert sab["p50"][10] == pytest.approx(1 / 0.97 - 1, abs=1e-9)
    assert np.all(sab["p10"] <= sab["p50"]) and np.all(sab["p50"] <= sab["p90"])
    assert t["festivo"]["n_dias"] == 0 and "nota" in t["festivo"]  # sin festivos: banda de domingo con nota
    j = I.banda_json(t, d)
    assert set(j["por_tipo_dia"]) == set(I.TIPOS_DIA) and j["n_semanas_historial"] == N_SEMANAS
    assert len(j["por_tipo_dia"]["laboral"]["p10_pct"]) == 24 and j["nivel_confianza"] == 0.8
    assert "nota" in j["por_tipo_dia"]["festivo"]


def test_banda_sin_datos_arrays_null():
    j = I.banda_json(I.tablas_banda(I.Datos()), I.Datos())
    for t in I.TIPOS_DIA:
        assert j["por_tipo_dia"][t]["n_dias"] == 0 and j["por_tipo_dia"][t]["p10_pct"] is None
    assert j["n_semanas_historial"] == 0
    assert I.aplicar_banda([1.0] * 24, "laboral", I.tablas_banda(I.Datos())) is None


def test_banda_filtros_fecha_y_tipo():
    d = _datos()
    t = I.tablas_banda(d, fecha_inicio=LUNES0.isoformat(), fecha_fin=(LUNES0 + timedelta(days=13)).isoformat())
    assert t["sabado"]["n_dias"] == 2
    j = I.banda_json(I.tablas_banda(d), d, tipo="sabado")
    assert list(j["por_tipo_dia"]) == ["sabado"]


def test_aplicar_banda_envuelve_la_curva():
    d = _datos()
    t = I.tablas_banda(d)
    curva = [float(v) for v in _real(date(2026, 6, 1))]
    inf, sup = I.aplicar_banda(curva, "laboral", t)
    assert len(inf) == 24 and all(a < c < b for a, c, b in zip(inf, curva, sup))
    assert I.aplicar_banda([1.0] * 5, "laboral", t) is None  # curva inválida


# ---------------------------------------------------------------------------
# sesgo y recomendaciones
# ---------------------------------------------------------------------------

def test_sesgo_por_segmento_mide_el_sabado():
    sg = I.sesgo_por_segmento(_datos())
    assert sg["sábado"]["n"] == I.OCURRENCIAS_DIA_SEMANA
    assert sg["sábado"]["sesgo_pct"] == pytest.approx(-3.0, abs=0.05)
    assert sg["sábado"]["consistencia"] == 1.0 and sg["sábado"]["factor_nivel"] == pytest.approx(1 / 0.97)
    assert sg["sábado"]["error_nivel_pct"] < sg["sábado"]["error_antes_pct"]
    assert len(sg["sábado"]["factores_hora"]) == 24


def test_recomendacion_sabado_con_parche_valido():
    d = _datos()
    semana = LUNES0 + timedelta(weeks=N_SEMANAS)  # la semana siguiente al historial
    c = _corrida(semana)
    out = I.recomendaciones(c, d)
    recs = out["recomendaciones"]
    assert [r["fecha"] for r in recs] == [(semana + timedelta(days=5)).isoformat()]  # solo el sábado
    r = recs[0]
    assert r["id"] == f"sesgo-{r['fecha']}" and r["categoria"] == "nivel" and r["tipo_dia"] == "sabado"
    assert r["titulo"] == "El sábado suele quedar corto" and "por debajo" in r["explicacion"]
    assert "α" not in r["explicacion"] and "ratio" not in r["explicacion"]
    assert r["evidencia"] == {"n": 8, "sesgo_pct": pytest.approx(-3.0, abs=0.05), "fuente": "historial"}
    assert r["aplicable"] is True and r["prioridad"] in ("alta", "media")
    aj = r["ajuste"]
    assert aj["delta_energia_pct"] == pytest.approx(3.0, abs=0.2) and aj["delta_energia_mwh"] > 0
    assert aj["patch"]["paso"] == "forma" and aj["patch"]["body"]["estado"] == "editado"
    assert aj["patch"]["body"]["curvas"] == {r["fecha"]: aj["curva_sugerida"]}
    cs = np.asarray(aj["curva_sugerida"])
    assert cs.shape == (24,) and np.all(np.isfinite(cs)) and np.all(cs > 0)
    assert cs.sum() == pytest.approx(sum(c.final[r["fecha"]]) * 1.03, rel=2e-3)
    assert out["ya_resueltas"] == []


def test_sin_evidencia_no_recomienda():
    d = _datos(sesgo_sabado=0.0)  # sin sesgo sistemático
    assert I.recomendaciones(_corrida(LUNES0 + timedelta(weeks=N_SEMANAS)), d)["recomendaciones"] == []
    d2 = _datos(n_semanas=5)  # menos de 6 ocurrencias
    assert I.recomendaciones(_corrida(LUNES0 + timedelta(weeks=5)), d2)["recomendaciones"] == []
    # efecto menor a 0.5 %
    d3 = _datos(sesgo_sabado=-0.004)
    assert I.recomendaciones(_corrida(LUNES0 + timedelta(weeks=N_SEMANAS)), d3)["recomendaciones"] == []
    # evidencia vieja (más de 20 semanas antes)
    d4 = _datos()
    assert I.recomendaciones(_corrida(LUNES0 + timedelta(weeks=N_SEMANAS + 30)), d4)["recomendaciones"] == []
    # sin historial en absoluto
    assert I.recomendaciones(_corrida(LUNES0), I.Datos()) == {"recomendaciones": [], "ya_resueltas": []}


def test_ajuste_ya_incorporado_no_se_recomienda():
    d = _datos()
    semana = LUNES0 + timedelta(weeks=N_SEMANAS)
    c = _corrida(semana)
    for f in c.final:  # el analista/el pipeline ya subió todo 3 %: el sesgo quedó cubierto
        c.final[f] = [v * 1.03 for v in c.final[f]]
    assert I.recomendaciones(c, d)["recomendaciones"] == []


def test_aprobada_informativa_y_editadas_resueltas():
    d = _datos()
    semana = LUNES0 + timedelta(weeks=N_SEMANAS)
    sab = (semana + timedelta(days=5)).isoformat()
    r = I.recomendaciones(_corrida(semana, estado="aprobada"), d)["recomendaciones"][0]
    assert r["aplicable"] is False and r["ajuste"] is not None
    out = I.recomendaciones(_corrida(semana, editadas=[sab]), d)
    assert out["recomendaciones"] == [] and out["ya_resueltas"] == [f"sesgo-{sab}"]


def test_paso_parche_segun_estado_de_pasos():
    d = _datos()
    semana = LUNES0 + timedelta(weeks=N_SEMANAS)
    base = {"base": "aplicado", "forma": "omitido", "especiales": "aplicado"}
    r = I.recomendaciones(_corrida(semana, estados_paso=base), d)["recomendaciones"][0]
    assert r["ajuste"]["patch"]["paso"] == "especiales"
    base["especiales"] = "omitido"
    r = I.recomendaciones(_corrida(semana, estados_paso=base), d)["recomendaciones"][0]
    assert r["ajuste"] is None and r["aplicable"] is False  # editar un paso omitido cambiaría otros días


def test_proveedor_de_sesgo_reemplazable():
    d = _datos()
    out = I.recomendaciones(_corrida(LUNES0 + timedelta(weeks=N_SEMANAS)), d, proveedor_sesgo=lambda *_a, **_k: {})
    assert out["recomendaciones"] == []


def test_alertas_a_recomendaciones_y_textos():
    semana = date(2026, 6, 29)
    fest = "2026-06-29"
    alertas = [
        ("control", Alert("info", "FESTIVO_EN_SEMANA", "x", [fest])),
        ("nivel", Alert("warning", "SIN_EVIDENCIA_NIVEL", "x (α = 0)", ["2026-07-01"])),
        ("control", Alert("warning", "TENDENCIA_FUERA_BANDA", "x", ["2026-07-02"], {"desvio_pp": 2.3})),
        ("nivel", Alert("info", "NIVEL_ANCLA_ALTERNA", "técnico", ["2026-07-03"])),
        ("control", Alert("bloqueo", "RECONCILIACION_ROTA", "x", [])),
    ]
    c = _corrida(semana, alertas=alertas, festivos=[fest])
    ins = I.insights_de_alertas(c)
    assert [i["nivel"] for i in ins] == sorted([i["nivel"] for i in ins], key=lambda n: I._NIVEL_ORD[n])
    ids = {i["id"] for i in ins}
    assert f"festivo-{fest}" in ids and "nivel_ancla_alterna-2026-07-03" not in ids  # técnico oculto
    fes = next(i for i in ins if i["id"] == f"festivo-{fest}")
    assert fes["titulo"].startswith("Lunes festivo") and "menos demanda" in fes["texto"]
    todos = " ".join(i["texto"] + i["titulo"] for i in ins)
    for jerga in ("α", "ratio", "ancla", "alpha"):
        assert jerga not in todos
    assert "+2.3" in next(i for i in ins if i["codigo"] == "TENDENCIA_FUERA_BANDA")["texto"]
    recs = I.recomendaciones(c, I.Datos())["recomendaciones"]
    assert {r["categoria"] for r in recs} == {"festivo", "alerta"}
    assert all(r["ajuste"] is None and r["aplicable"] is False for r in recs)
    assert recs[0]["prioridad"] == "alta"  # el bloqueo primero


# ---------------------------------------------------------------------------
# resumen y series
# ---------------------------------------------------------------------------

def test_titulo_semana():
    c = _corrida(date(2026, 6, 29))
    assert I._titulo_semana(c, 28412.3, 1.84) == (
        "Semana del 29 jun al 5 jul: demanda prevista 28.4 GWh, un 1.8 % más que la semana pasada")
    assert "menos que la semana pasada" in I._titulo_semana(c, 28000.0, -2.0)
    assert I._titulo_semana(c, 28000.0, None).endswith("28.0 GWh")
    assert I._titulo_semana(_corrida(date(2026, 11, 9)), 1.0, None).startswith("Semana del 9 al 15 nov")
    assert "sin pronóstico" in I._titulo_semana(c, None, None)


def test_resumen_forma_y_valores():
    d = _datos()
    semana = LUNES0 + timedelta(weeks=N_SEMANAS)
    c = _corrida(semana)
    r = I.resumen(c, d)
    assert r["semana_inicio"] == semana.isoformat() and r["semana_fin"] == (semana + timedelta(days=6)).isoformat()
    assert r["confianza"]["nivel"] == "alta" and r["confianza"]["semanas_evaluadas"] == 8
    assert "últimas 8 semanas" in r["confianza"]["texto"]
    k = r["kpis"]
    esperado = sum(float(_real(semana + timedelta(days=i)).sum()) for i in range(7))
    assert k["energia_semana_mwh"] == pytest.approx(esperado, abs=0.1)
    assert k["energia_semana_anterior_mwh"] is not None and k["variacion_vs_anterior_pct"] == pytest.approx(0.0, abs=1.0)
    assert 0 <= k["pico_hora"] <= 23 and k["pico_fecha"] in c.fechas
    assert k["error_reciente_tendencia"] in ("mejorando", "estable", "empeorando", "sin_datos")
    assert len(r["dias"]) == 7 and r["dias"][5]["estado"] == "revisar" and r["dias"][5]["n_recomendaciones"] == 1
    assert r["dias"][0]["estado"] == "ok" and r["dias"][0]["ajuste_pct"] == 0.0
    assert r["siguiente_paso"]["accion"] == "revisar_recomendaciones"
    assert k["recomendaciones"]["total"] == 1
    assert I.resumen(_corrida(semana, estado="aprobada"), d)["siguiente_paso"]["accion"] == "ya_aprobada"
    sin = I.resumen(_corrida(semana), _datos(sesgo_sabado=0.0))
    assert sin["siguiente_paso"]["accion"] == "aprobar"


def test_resumen_sin_historial():
    r = I.resumen(_corrida(date(2026, 6, 29)), I.Datos())
    assert r["confianza"]["nivel"] == "sin_datos" and r["kpis"]["error_reciente_pct"] is None
    assert r["kpis"]["energia_semana_anterior_mwh"] is None and r["kpis"]["variacion_vs_anterior_pct"] is None
    assert r["kpis"]["energia_semana_mwh"] is not None


def test_series_168_puntos():
    d = _datos()
    semana = LUNES0 + timedelta(weeks=N_SEMANAS)  # futura: sin real
    s = I.series(_corrida(semana), d)
    for k in ("base", "ajustado", "real", "banda_inferior", "banda_superior"):
        assert len(s["series"][k]) == 168
    assert all(v is None for v in s["series"]["real"]) and s["series"]["referencia"] is None
    assert all(v is not None for v in s["series"]["ajustado"])
    assert all(i <= u for i, u in zip(s["series"]["banda_inferior"], s["series"]["banda_superior"]))
    assert s["etiquetas_eje"][0] == {"i": 0, "texto": f"lun {semana.day}"} and len(s["etiquetas_eje"]) == 7
    assert len(s["energia_diaria"]["ajustado"]) == 7 and s["energia_diaria"]["real"] == [None] * 7
    assert [x["tipo_dia"] for x in s["dias"]][5] == "sabado"


def test_series_con_real_referencia_y_sin_banda():
    d = _datos()
    semana = LUNES0 + timedelta(weeks=3)  # pasada: hay real
    c = _corrida(semana)
    c.ref_inicio = LUNES0
    s = I.series(c, d)
    assert all(v is not None for v in s["series"]["real"])
    assert s["series"]["referencia_semana_inicio"] == LUNES0.isoformat() and len(s["series"]["referencia"]) == 168
    s2 = I.series(c, I.Datos())
    assert s2["series"]["banda_inferior"] is None and s2["series"]["real"] == [None] * 168
    assert s2["series"]["referencia"] is None and s2["series"]["referencia_semana_inicio"] is None


def test_cargar_emitidos_csv_real_y_cache(tmp_path):
    (tmp_path / "X_k1.csv").write_text(
        "fecha,semana_inicio," + ",".join(f"P{i}" for i in range(1, 25)) + "\n"
        + "2026-01-05,2026-01-05," + ",".join(["10.0"] * 24) + "\n"
        + "2026-01-06,2026-01-05," + ",".join(["10.0"] * 23 + ["nan"]) + "\n")
    em, sd = I.cargar_emitidos("X", tmp_path)
    assert list(em) == ["2026-01-05"] and sd["2026-01-05"] == "2026-01-05"  # la fila con NaN se descarta
    assert I.cargar_emitidos("X", tmp_path)[0] is em  # caché
    assert I.cargar_emitidos("NoExiste", tmp_path) == ({}, {})


def test_csv_reales_del_repo_son_validos():
    for ucp in ("Antioquia", "Atlantico"):
        em, sd = I.cargar_emitidos(ucp)
        assert len(em) == 364 and len(set(sd.values())) == 52
        for f, s in sd.items():
            assert 0 <= (date.fromisoformat(f) - date.fromisoformat(s)).days <= 6
            assert date.fromisoformat(s).weekday() == 0
