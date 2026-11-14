"""Tests de la lógica pura de insights (src/adjustments/insights.py) con datos sintéticos."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pytest

from src.adjustments import data, insights
from src.adjustments.types import Alert

LUNES0 = date(2026, 1, 5)  # lunes
N_SEMANAS = 20
FORMA = np.array([0.8, 0.75, 0.72, 0.7, 0.72, 0.8, 0.9, 1.0, 1.05, 1.08, 1.1, 1.12,
                  1.1, 1.08, 1.07, 1.06, 1.05, 1.1, 1.25, 1.3, 1.22, 1.1, 0.98, 0.88])
DOW = [1.0, 1.01, 1.01, 1.02, 1.0, 0.9, 0.8]


def _real(d: date) -> np.ndarray:
    return 400.0 * FORMA * DOW[d.weekday()]


def _datos(sesgo_sabado=-0.03, n_semanas: int = N_SEMANAS, festivos=()) -> insights.Datos:
    """Pronóstico = real × (1+error): el sábado queda `sesgo_sabado` corto; el resto, ±0.2 % alterno."""
    emitidos, semana_de, reales = {}, {}, {}
    for w in range(n_semanas):
        lunes = LUNES0 + timedelta(weeks=w)
        for i in range(7):
            d = lunes + timedelta(days=i)
            f = d.isoformat()
            r = _real(d)
            sab = sesgo_sabado(w) if callable(sesgo_sabado) else sesgo_sabado
            err = sab if d.weekday() == 5 else (0.002 if (w + i) % 2 else -0.002)
            reales[f] = r
            emitidos[f] = r * (1 + err)
            semana_de[f] = lunes.isoformat()
    return insights.Datos(emitidos=emitidos, semana_de=semana_de, reales=reales, festivos=set(festivos))


def _corrida(semana: date, estado: str = "borrador", estados_paso=None, editadas=(), alertas=(),
             escala: float = 1.0, festivos=()) -> insights.Corrida:
    fechas = [(semana + timedelta(days=i)).isoformat() for i in range(7)]
    base = {f: [float(v) for v in _real(date.fromisoformat(f)) * escala] for f in fechas}
    return insights.Corrida(
        ucp="Antioquia", semana_inicio=semana, estado=estado, version=2, base=base, final={f: list(v) for f, v in base.items()},
        alertas=list(alertas), editadas=set(editadas),
        estados_paso=estados_paso or {"base": "aplicado", "referencia": "aplicado", "nivel": "aplicado",
                                      "forma": "aplicado", "especiales": "aplicado", "control": "aplicado"},
        ref_inicio=None, festivos=set(festivos))


# ---------------------------------------------------------------------------
# precisión
# ---------------------------------------------------------------------------

def test_precision_sin_datos_degrada():
    p = insights.precision(insights.Datos())
    assert p["semanas"] == [] and p["por_tipo_dia"] == []
    assert p["resumen"]["semanas"] == 0 and p["resumen"]["wape_pct"] is None
    assert p["resumen"]["tendencia"] == "sin_datos"
    assert p["por_hora"]["sesgo_pct"] == [None] * 24


def test_precision_forma_y_signo():
    d = _datos()
    p = insights.precision(d, semanas=5)
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
    p = insights.precision(d, semanas=52)
    assert p["hasta"] == (ultimo - timedelta(weeks=1)).isoformat() and len(p["semanas"]) == N_SEMANAS - 1
    p2 = insights.precision(d, semanas=52, hasta=LUNES0 + timedelta(weeks=3))
    assert len(p2["semanas"]) == 4 and p2["hasta"] == (LUNES0 + timedelta(weeks=3)).isoformat()


def test_precision_aprobada_prevalece_y_marca_fuente():
    d = _datos()
    lunes = LUNES0 + timedelta(weeks=4)
    d.aprobadas = {lunes.isoformat(): {(lunes + timedelta(days=i)).isoformat(): list(_real(lunes + timedelta(days=i)))
                                       for i in range(7)}}
    p = insights.precision(d, semanas=52)
    s = {x["semana_inicio"]: x for x in p["semanas"]}
    assert s[lunes.isoformat()]["fuente"] == "corrida_aprobada"
    assert s[lunes.isoformat()]["wape_pct"] == 0.0  # la aprobada era exacta
    assert s[(lunes + timedelta(weeks=1)).isoformat()]["fuente"] == "modelo_historico"


def test_tendencia():
    assert insights._tendencia([5, 5, 4.9]) == "sin_datos"
    assert insights._tendencia([5.0, 5.0, 3.0, 3.0]) == "mejorando"
    assert insights._tendencia([3.0, 3.0, 5.0, 5.0]) == "empeorando"
    assert insights._tendencia([3.0, 3.1, 3.0, 3.1]) == "estable"


def test_nivel_confianza_umbrales():
    assert [insights.nivel_confianza(x) for x in (None, 2.9, 3.0, 4.0, 6.0, 6.1)] == [
        "sin_datos", "alta", "alta", "media", "media", "baja"]


# ---------------------------------------------------------------------------
# banda
# ---------------------------------------------------------------------------

def test_banda_cuantiles_y_festivo_usa_domingo():
    d = _datos()
    t = insights.tablas_banda(d)
    sab = t["sabado"]
    assert sab["n_dias"] == N_SEMANAS
    # real/pronóstico − 1 con pronóstico 3 % corto = +3.09 %
    assert sab["p50"][10] == pytest.approx(1 / 0.97 - 1, abs=1e-9)
    assert np.all(sab["p10"] <= sab["p50"]) and np.all(sab["p50"] <= sab["p90"])
    assert t["festivo"]["n_dias"] == 0 and "nota" in t["festivo"]  # sin festivos: banda de domingo con nota
    j = insights.banda_json(t, d)
    assert set(j["por_tipo_dia"]) == set(data.TIPOS_DIA) and j["n_semanas_historial"] == N_SEMANAS
    assert len(j["por_tipo_dia"]["laboral"]["p10_pct"]) == 24 and j["nivel_confianza"] == 0.8
    assert "nota" in j["por_tipo_dia"]["festivo"]


def test_banda_sin_datos_arrays_null():
    j = insights.banda_json(insights.tablas_banda(insights.Datos()), insights.Datos())
    for t in data.TIPOS_DIA:
        assert j["por_tipo_dia"][t]["n_dias"] == 0 and j["por_tipo_dia"][t]["p10_pct"] is None
    assert j["n_semanas_historial"] == 0
    assert insights.aplicar_banda([1.0] * 24, "laboral", insights.tablas_banda(insights.Datos())) is None


def test_banda_filtros_fecha_y_tipo():
    d = _datos()
    t = insights.tablas_banda(d, fecha_inicio=LUNES0.isoformat(), fecha_fin=(LUNES0 + timedelta(days=13)).isoformat())
    assert t["sabado"]["n_dias"] == 2
    j = insights.banda_json(insights.tablas_banda(d), d, tipo="sabado")
    assert list(j["por_tipo_dia"]) == ["sabado"]


def test_aplicar_banda_envuelve_la_curva():
    d = _datos()
    t = insights.tablas_banda(d)
    curva = [float(v) for v in _real(date(2026, 6, 1))]
    inf, sup = insights.aplicar_banda(curva, "laboral", t)
    assert len(inf) == 24 and all(a < c < b for a, c, b in zip(inf, curva, sup))
    assert insights.aplicar_banda([1.0] * 5, "laboral", t) is None  # curva inválida


def test_banda_calibrada_escala_los_cuantiles(monkeypatch):
    d = _datos()
    monkeypatch.setattr(insights, "ESCALA_BANDA", 1.0)
    t1 = insights.tablas_banda(d)
    monkeypatch.setattr(insights, "ESCALA_BANDA", 1.2)
    t2 = insights.tablas_banda(d)
    for tipo in ("laboral", "sabado"):
        a, b = t1[tipo], t2[tipo]
        assert np.all(b["p10"] <= a["p10"] + 1e-12) and np.all(b["p90"] >= a["p90"] - 1e-12)
        assert np.all(b["p10"] <= b["p50"]) and np.all(b["p50"] <= b["p90"])
        assert np.allclose(b["p50"], a["p50"])
        ancho1, ancho2 = a["p90"] - a["p10"], b["p90"] - b["p10"]
        assert np.all(ancho2 >= ancho1 - 1e-12) and ancho2.sum() > ancho1.sum()
    # los p10/p90 del laboral cambian exactamente por el factor cuando no cruzan la mediana
    lab1, lab2 = t1["laboral"], t2["laboral"]
    assert np.allclose(lab2["p90"], np.maximum(lab1["p90"] * 1.2, lab1["p50"]))
    assert np.allclose(lab2["p10"], np.minimum(lab1["p10"] * 1.2, lab1["p50"]))


def test_banda_json_incluye_escala_y_cobertura_medida():
    d = _datos()
    j = insights.banda_json(insights.tablas_banda(d), d)
    assert j["escala_calibracion"] == 1.2 == insights.ESCALA_BANDA
    cob = j["cobertura_medida"]
    assert cob["nominal"] == 0.8 and cob["antioquia_prueba"] == 0.876 and cob["atlantico_prueba"] == 0.802
    assert cob["antioquia_sin_escala"] == 0.813 and cob["atlantico_sin_escala"] == 0.732
    assert "esta implementación" in cob["nota"] and "21 semanas" in cob["nota"]
    jv = insights.banda_json(insights.tablas_banda(insights.Datos()), insights.Datos())  # sin datos: los campos siguen presentes
    assert jv["escala_calibracion"] == 1.2 and jv["cobertura_medida"]["nominal"] == 0.8


# ---------------------------------------------------------------------------
# sesgo y recomendaciones
# ---------------------------------------------------------------------------

def test_sesgo_por_segmento_mide_el_sabado():
    sg = insights.sesgo_por_segmento(_datos())
    assert sg["sábado"]["n"] == insights.OCURRENCIAS_DIA_SEMANA
    assert sg["sábado"]["sesgo_pct"] == pytest.approx(-3.0, abs=0.05)
    assert sg["sábado"]["consistencia"] == 1.0 and sg["sábado"]["factor_nivel"] == pytest.approx(1 / 0.97)
    assert sg["sábado"]["error_nivel_pct"] < sg["sábado"]["error_antes_pct"]
    assert len(sg["sábado"]["factores_hora"]) == 24


def test_recomendacion_sabado_con_parche_valido():
    d = _datos()
    semana = LUNES0 + timedelta(weeks=N_SEMANAS)  # la semana siguiente al historial
    c = _corrida(semana)
    out = insights.recomendaciones(c, d)
    recs = out["recomendaciones"]
    assert [r["fecha"] for r in recs] == [(semana + timedelta(days=5)).isoformat()]  # solo el sábado
    r = recs[0]
    assert r["id"] == f"sesgo-{r['fecha']}" and r["categoria"] == "nivel" and r["tipo_dia"] == "sabado"
    assert r["titulo"] == "Tendencia reciente: el sábado quedó corto" and "por debajo" in r["explicacion"]
    assert insights.AVISO_TENDENCIA in r["explicacion"] and "sobre lo que muestra hoy" in r["explicacion"]
    assert "prueba con semanas anteriores" not in r["explicacion"] and "el período con el que se eligió la regla" in r["explicacion"]
    assert r["respaldo"] == "tendencia_reciente" and r["validado_fuera_de_muestra"] is False
    assert "validacion" not in r["evidencia"] and r["evidencia"]["ventana_seleccion"]["dias_evaluados"] >= 8
    assert "α" not in r["explicacion"] and "ratio" not in r["explicacion"]
    assert {k: r["evidencia"][k] for k in ("n", "sesgo_pct", "fuente")} == {
        "n": 8, "sesgo_pct": pytest.approx(-3.0, abs=0.05), "fuente": "historial"}
    assert r["aplicable"] is True and r["prioridad"] == "baja"  # tendencia observada: siempre "baja"
    assert "Reduce" not in r["impacto_esperado"] and "%" not in r["impacto_esperado"]
    assert "no garantiza mejora" in r["impacto_esperado"]
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
    assert insights.recomendaciones(_corrida(LUNES0 + timedelta(weeks=N_SEMANAS)), d)["recomendaciones"] == []
    d2 = _datos(n_semanas=5)  # menos de 6 ocurrencias
    assert insights.recomendaciones(_corrida(LUNES0 + timedelta(weeks=5)), d2)["recomendaciones"] == []
    # efecto menor a 0.5 %
    d3 = _datos(sesgo_sabado=-0.004)
    assert insights.recomendaciones(_corrida(LUNES0 + timedelta(weeks=N_SEMANAS)), d3)["recomendaciones"] == []
    # evidencia vieja (más de 20 semanas antes)
    d4 = _datos()
    assert insights.recomendaciones(_corrida(LUNES0 + timedelta(weeks=N_SEMANAS + 30)), d4)["recomendaciones"] == []
    # sin historial en absoluto
    assert insights.recomendaciones(_corrida(LUNES0), insights.Datos()) == {"recomendaciones": [], "ya_resueltas": []}


def test_ajuste_ya_incorporado_no_se_recomienda():
    d = _datos()
    semana = LUNES0 + timedelta(weeks=N_SEMANAS)
    c = _corrida(semana)
    for f in c.final:  # el analista/el pipeline ya subió todo 3 %: el sesgo quedó cubierto
        c.final[f] = [v * 1.03 for v in c.final[f]]
    assert insights.recomendaciones(c, d)["recomendaciones"] == []


def test_aprobada_informativa_y_editadas_resueltas():
    d = _datos()
    semana = LUNES0 + timedelta(weeks=N_SEMANAS)
    sab = (semana + timedelta(days=5)).isoformat()
    r = insights.recomendaciones(_corrida(semana, estado="aprobada"), d)["recomendaciones"][0]
    assert r["aplicable"] is False and r["ajuste"] is not None
    out = insights.recomendaciones(_corrida(semana, editadas=[sab]), d)
    assert out["recomendaciones"] == [] and out["ya_resueltas"] == [f"sesgo-{sab}"]


def test_paso_parche_segun_estado_de_pasos():
    d = _datos()
    semana = LUNES0 + timedelta(weeks=N_SEMANAS)
    base = {"base": "aplicado", "forma": "omitido", "especiales": "aplicado"}
    r = insights.recomendaciones(_corrida(semana, estados_paso=base), d)["recomendaciones"][0]
    assert r["ajuste"]["patch"]["paso"] == "especiales"
    base["especiales"] = "omitido"
    r = insights.recomendaciones(_corrida(semana, estados_paso=base), d)["recomendaciones"][0]
    assert r["ajuste"] is None and r["aplicable"] is False  # editar un paso omitido cambiaría otros días


def test_sin_validacion_rolling_no_se_recomienda():
    """Cambio de régimen: el sesgo de las últimas 8 semanas es consistente, pero la regla no habría funcionado antes."""
    d = _datos(sesgo_sabado=lambda w: 0.03 if w < 12 else -0.03)
    semana = LUNES0 + timedelta(weeks=N_SEMANAS)
    hasta = (semana - timedelta(days=1)).isoformat()
    sg = insights.sesgo_por_segmento(d, hasta=hasta)["sábado"]
    assert insights._hay_evidencia(sg)  # pasa el gating simple
    assert insights.validar_rolling(d, "sábado", hasta)["pasa"] is False
    assert insights.recomendaciones(_corrida(semana), d)["recomendaciones"] == []


def test_validar_rolling_sesgo_estable_pasa():
    d = _datos()
    val = insights.validar_rolling(d, "sábado", (LUNES0 + timedelta(weeks=N_SEMANAS, days=-1)).isoformat())
    assert val["pasa"] is True and val["pct_dias_mejora"] == 1.0
    assert val["wape_despues_pct"] < val["wape_antes_pct"] and val["dias_evaluados"] == insights.ROLLING_SEMANAS
    vacio = insights.validar_rolling(insights.Datos(), "sábado", "2026-06-01")
    assert vacio["pasa"] is False and vacio["dias_evaluados"] == 0


def test_rolling_no_mira_el_dia_evaluado():
    """Un único sábado atípico al final: la regla no puede 'adivinarlo', no mejora ese día."""
    d = _datos(sesgo_sabado=lambda w: -0.03 if w < N_SEMANAS - 1 else 0.03)
    val = insights.validar_rolling(d, "sábado", (LUNES0 + timedelta(weeks=N_SEMANAS, days=-1)).isoformat())
    assert val["dias_mejora"] < val["dias_evaluados"]


def test_corte_de_la_corrida_acota_la_evidencia():
    d = _datos()
    semana = LUNES0 + timedelta(weeks=N_SEMANAS)
    c = _corrida(semana)
    assert c.fecha_fin_sin_fuga == (semana - timedelta(days=1)).isoformat()
    c.fecha_corte = LUNES0 + timedelta(weeks=3)  # la corrida se emitió con solo 3 semanas de real
    assert c.fecha_fin_sin_fuga == c.fecha_corte.isoformat()
    assert insights.recomendaciones(c, d)["recomendaciones"] == []


def test_nivel_se_aplica_sobre_el_final_actual_y_el_delta_es_el_real():
    d = _datos()
    semana = LUNES0 + timedelta(weeks=N_SEMANAS)
    c = _corrida(semana)
    sab = (semana + timedelta(days=5)).isoformat()
    c.final = {f: ([v * 1.01 for v in cv] if f == sab else cv) for f, cv in c.final.items()}  # el pipeline ya subió 1 %
    r = insights.recomendaciones(c, d)["recomendaciones"][0]
    e_final = sum(c.final[sab])
    cs = np.asarray(r["ajuste"]["curva_sugerida"])
    assert cs.sum() == pytest.approx(e_final / 0.97, rel=2e-3)  # factor sobre lo que muestra final
    assert r["ajuste"]["delta_energia_pct"] == pytest.approx((cs.sum() / e_final - 1) * 100, abs=0.06)
    assert f"{r['ajuste']['delta_energia_mwh']:+.0f} MWh" in r["explicacion"]
    # si el pipeline ya recogió el sesgo, no se sugiere
    c.final = {f: ([v * 1.03 for v in cv] if f == sab else cv) for f, cv in c.final.items()}
    assert insights.recomendaciones(c, d)["recomendaciones"] == []


def test_sin_dias_suficientes_en_la_ventana_no_se_emite_ajuste():
    """Con 13 semanas hay evidencia simple (>= 6 sábados) pero la regla solo actúa en 7 de los 8 días de la ventana."""
    d = _datos(n_semanas=13)
    semana = LUNES0 + timedelta(weeks=13)
    hasta = (semana - timedelta(days=1)).isoformat()
    assert insights._hay_evidencia(insights.sesgo_por_segmento(d, hasta=hasta)["sábado"])
    val = insights.validar_rolling(d, "sábado", hasta)
    assert val["dias_evaluados"] < insights.ROLLING_MIN_DIAS and val["pasa"] is False
    assert insights.recomendaciones(_corrida(semana), d)["recomendaciones"] == []


def test_filtro_exige_mejora_minima_de_wape_y_pct_de_dias(monkeypatch):
    d = _datos()
    hasta = (LUNES0 + timedelta(weeks=N_SEMANAS, days=-1)).isoformat()
    assert insights.validar_rolling(d, "sábado", hasta)["pasa"] is True
    monkeypatch.setattr(insights, "ROLLING_MIN_MEJORA_WAPE_PP", 50.0)
    assert insights.validar_rolling(d, "sábado", hasta)["pasa"] is False
    monkeypatch.undo()
    assert insights.ROLLING_MIN_DIAS >= 8 and insights.ROLLING_MIN_MEJORA >= 0.75
    assert insights.ROLLING_MIN_MEJORA_WAPE_PP == pytest.approx(0.3)


def test_tendencia_no_empuja_el_siguiente_paso_ni_cuenta_como_alta():
    d = _datos()
    semana = LUNES0 + timedelta(weeks=N_SEMANAS)
    r = insights.resumen(_corrida(semana), d)
    k = r["kpis"]["recomendaciones"]
    assert k == {"total": 1, "alta": 0, "del_sistema": 0, "tendencia_reciente": 1}
    assert r["siguiente_paso"]["accion"] == "aprobar"
    assert r["dias"][5]["n_recomendaciones"] == 1 and r["dias"][5]["estado"] == "ok"


def test_clima_info_no_es_recomendacion_accionable():
    semana = date(2026, 6, 29)
    a = Alert("info", "CLIMA_OBSERVADO", "x", [], {"temp_media": 24.0})
    c = _corrida(semana, alertas=[("control", a)])
    assert [i["id"] for i in insights.insights_de_alertas(c)] == ["clima-semana"]
    assert insights.recomendaciones(c, insights.Datos())["recomendaciones"] == []
    r = insights.resumen(c, insights.Datos())
    assert r["siguiente_paso"]["accion"] == "aprobar" and all(x["estado"] == "ok" for x in r["dias"])
    # el clima extremo (warning) sí es accionable
    w = _corrida(semana, alertas=[("control", Alert("warning", "CLIMA_OBSERVADO", "x", [], {"temp_media": 35.0}))])
    assert insights.resumen(w, insights.Datos())["siguiente_paso"]["accion"] == "revisar_recomendaciones"


def test_corrida_reemplazada_responde_ya_aprobada():
    c = _corrida(date(2026, 6, 29), estado="reemplazada",
                 alertas=[("control", Alert("warning", "SIN_EVIDENCIA_NIVEL", "x", ["2026-07-01"]))])
    r = insights.resumen(c, insights.Datos())
    assert r["siguiente_paso"]["accion"] == "ya_aprobada" and "aprobada" in r["siguiente_paso"]["accion"]


def test_textos_festivos_fuente_alterna_y_bloqueos_por_curva():
    a = Alert("info", "FESTIVOS_FUENTE_ALTERNA", "técnico", [], {"fuente_festivos": "x"})
    ins = insights._expandir_alerta(a)
    assert len(ins) == 1 and "calendario" in ins[0]["texto"].lower() and ins[0]["titulo"] != "Aviso del sistema"
    b = Alert("bloqueo", "VALORES_NEGATIVOS", "x", ["2026-07-01"])
    t = insights._expandir_alerta(b)[0]
    assert t["nivel"] == "critico" and "miércoles 1" in t["texto"] and "{" not in t["texto"]
    sf = Alert("warning", "SIN_FORMA", "técnico", ["2026-07-01"])
    assert len(insights._expandir_alerta(sf)) == 1  # ya no se oculta
    assert "{dias}" not in insights._expandir_alerta(Alert("bloqueo", "RECONCILIACION_ROTA", "x", []))[0]["texto"]


def test_loo_forma_conserva_la_energia_como_corregir_curva():
    rng = np.random.default_rng(1)
    R = 400 * FORMA * (1 + 0.02 * rng.standard_normal((6, 24)))
    F = R * 0.9 * (1 + 0.01 * rng.standard_normal((6, 24)))
    esperado = []
    for i in range(6):
        m = np.arange(6) != i
        sg = {"factores_hora": np.mean(R[m] / F[m], axis=0)}
        esperado.append(insights.wape(insights._corregir_curva(sg, F[i], "forma"), R[i]))
    assert insights._loo_error(F, R, True) == pytest.approx(float(np.mean(esperado)))


def test_corte_semana_y_helpers():
    assert data.corte_semana(date(2026, 7, 3)) == date(2026, 6, 27)  # sábado antes del lunes 29 jun
    assert data.corte_semana(date(2026, 6, 29)) == date(2026, 6, 27)
    assert insights.festivos_por_defecto(["2026-06-29"]) and True


def test_cobertura_medida_con_fecha():
    assert "2026-11-14" in insights.COBERTURA_MEDIDA["detalle_tecnico"]


def test_banda_json_n_semanas_respeta_filtros():
    d = _datos()
    t = insights.tablas_banda(d)
    assert insights.banda_json(t, d)["n_semanas_historial"] == N_SEMANAS
    fi, ff = LUNES0.isoformat(), (LUNES0 + timedelta(days=13)).isoformat()
    j = insights.banda_json(insights.tablas_banda(d, fi, ff), d, None, fi, ff)
    assert j["n_semanas_historial"] == 2
    js = insights.banda_json(t, d, "sabado", fi, ff)
    assert js["n_semanas_historial"] == 2 and list(js["por_tipo_dia"]) == ["sabado"]


def test_corrida_desde_version_y_fecha_corte():
    c = insights.corrida_desde({"ucp": "X", "semana_inicio": "2026-06-29"}, [], {})
    assert c.version == 1 and c.fecha_corte is None and c.estado == "borrador"
    c2 = insights.corrida_desde({"ucp": "X", "semana_inicio": "2026-06-29", "version": 4, "fecha_corte": "2026-06-27"}, [], {})
    assert c2.version == 4 and c2.fecha_corte == date(2026, 6, 27) and c2.fecha_fin_sin_fuga == "2026-06-27"


def test_elecciones_ia_tiene_texto_de_negocio():
    a = Alert("info", "ELECCIONES_IA", "técnico", ["2026-06-28"], {"eventos": {"2026-06-28": "consulta"}})
    ins = insights._expandir_alerta(a)
    assert len(ins) == 1 and ins[0]["titulo"] == "Posibles elecciones o votación"
    assert "no están verificados" in ins[0]["texto"] and "domingo 28" in ins[0]["texto"]


def test_segmento_unificado():
    assert insights._segmento("festivo", "2026-06-29") == "festivo"
    assert insights._segmento("laboral", "2026-06-29") == "lunes"
    c = _corrida(date(2026, 6, 29), festivos=["2026-06-29"])
    assert c.tipo("2026-06-29") == "festivo" and c.tipo("2026-06-30") == "laboral"


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
    ins = insights.insights_de_alertas(c)
    assert [i["nivel"] for i in ins] == sorted([i["nivel"] for i in ins], key=lambda n: insights._NIVEL_ORD[n])
    ids = {i["id"] for i in ins}
    assert f"festivo-{fest}" in ids and "nivel_ancla_alterna-2026-07-03" not in ids  # técnico oculto
    fes = next(i for i in ins if i["id"] == f"festivo-{fest}")
    assert fes["titulo"].startswith("Lunes festivo") and "menos demanda" in fes["texto"]
    todos = " ".join(i["texto"] + i["titulo"] for i in ins)
    for jerga in ("α", "ratio", "ancla", "alpha"):
        assert jerga not in todos
    assert "+2.3" in next(i for i in ins if i["codigo"] == "TENDENCIA_FUERA_BANDA")["texto"]
    recs = insights.recomendaciones(c, insights.Datos())["recomendaciones"]
    assert {r["categoria"] for r in recs} == {"festivo", "alerta"}
    assert all(r["ajuste"] is None and r["aplicable"] is False for r in recs)
    assert recs[0]["prioridad"] == "alta"  # el bloqueo primero


# ---------------------------------------------------------------------------
# resumen y series
# ---------------------------------------------------------------------------

def test_titulo_semana():
    c = _corrida(date(2026, 6, 29))
    assert insights._titulo_semana(c, 28412.3, 1.84) == (
        "Semana del 29 jun al 5 jul: demanda prevista 28.4 GWh, un 1.8 % más que la semana pasada")
    assert "menos que la semana pasada" in insights._titulo_semana(c, 28000.0, -2.0)
    assert insights._titulo_semana(c, 28000.0, None).endswith("28.0 GWh")
    assert insights._titulo_semana(_corrida(date(2026, 11, 9)), 1.0, None).startswith("Semana del 9 al 15 nov")
    assert "sin pronóstico" in insights._titulo_semana(c, None, None)


def test_resumen_forma_y_valores():
    d = _datos()
    semana = LUNES0 + timedelta(weeks=N_SEMANAS)
    c = _corrida(semana)
    r = insights.resumen(c, d)
    assert r["semana_inicio"] == semana.isoformat() and r["semana_fin"] == (semana + timedelta(days=6)).isoformat()
    assert r["confianza"]["nivel"] == "alta" and r["confianza"]["semanas_evaluadas"] == 8
    assert "últimas 8 semanas" in r["confianza"]["texto"]
    k = r["kpis"]
    esperado = sum(float(_real(semana + timedelta(days=i)).sum()) for i in range(7))
    assert k["energia_semana_mwh"] == pytest.approx(esperado, abs=0.1)
    assert k["energia_semana_anterior_mwh"] is not None and k["variacion_vs_anterior_pct"] == pytest.approx(0.0, abs=1.0)
    assert 0 <= k["pico_hora"] <= 23 and k["pico_fecha"] in c.fechas
    assert k["error_reciente_tendencia"] in ("mejorando", "estable", "empeorando", "sin_datos")
    assert len(r["dias"]) == 7 and r["dias"][5]["estado"] == "ok" and r["dias"][5]["n_recomendaciones"] == 1
    assert r["dias"][0]["estado"] == "ok" and r["dias"][0]["ajuste_pct"] == 0.0
    assert r["siguiente_paso"]["accion"] == "aprobar"  # solo una tendencia reciente: no empuja la revisión
    assert k["recomendaciones"]["total"] == 1
    assert insights.resumen(_corrida(semana, estado="aprobada"), d)["siguiente_paso"]["accion"] == "ya_aprobada"
    sin = insights.resumen(_corrida(semana), _datos(sesgo_sabado=0.0))
    assert sin["siguiente_paso"]["accion"] == "aprobar"


def test_resumen_sin_historial():
    r = insights.resumen(_corrida(date(2026, 6, 29)), insights.Datos())
    assert r["confianza"]["nivel"] == "sin_datos" and r["kpis"]["error_reciente_pct"] is None
    assert r["kpis"]["energia_semana_anterior_mwh"] is None and r["kpis"]["variacion_vs_anterior_pct"] is None
    assert r["kpis"]["energia_semana_mwh"] is not None


def test_series_168_puntos():
    d = _datos()
    semana = LUNES0 + timedelta(weeks=N_SEMANAS)  # futura: sin real
    s = insights.series(_corrida(semana), d)
    for k in ("base", "ajustado", "real", "banda_inferior", "banda_superior"):
        assert len(s["series"][k]) == 168
    assert all(v is None for v in s["series"]["real"]) and s["series"]["referencia"] is None
    assert all(v is not None for v in s["series"]["ajustado"])
    assert all(i <= u for i, u in zip(s["series"]["banda_inferior"], s["series"]["banda_superior"]))
    # coherente con la banda calibrada de banda_corrida (misma tabla, mismas curvas finales)
    bc = insights.banda_corrida(_corrida(semana), insights.tablas_banda(d))
    assert s["series"]["banda_inferior"] == [v for f in s["fechas"] for v in bc[f]["inferior"]]
    assert s["series"]["banda_superior"] == [v for f in s["fechas"] for v in bc[f]["superior"]]
    assert s["etiquetas_eje"][0] == {"i": 0, "texto": f"lun {semana.day}"} and len(s["etiquetas_eje"]) == 7
    assert len(s["energia_diaria"]["ajustado"]) == 7 and s["energia_diaria"]["real"] == [None] * 7
    assert [x["tipo_dia"] for x in s["dias"]][5] == "sabado"


def test_series_con_real_referencia_y_sin_banda():
    d = _datos()
    semana = LUNES0 + timedelta(weeks=3)  # pasada: hay real
    c = _corrida(semana)
    c.ref_inicio = LUNES0
    s = insights.series(c, d)
    assert all(v is not None for v in s["series"]["real"])
    assert s["series"]["referencia_semana_inicio"] == LUNES0.isoformat() and len(s["series"]["referencia"]) == 168
    s2 = insights.series(c, insights.Datos())
    assert s2["series"]["banda_inferior"] is None and s2["series"]["real"] == [None] * 168
    assert s2["series"]["referencia"] is None and s2["series"]["referencia_semana_inicio"] is None


def test_cargar_emitidos_csv_real_y_cache(tmp_path):
    (tmp_path / "X_k1.csv").write_text(
        "fecha,semana_inicio," + ",".join(f"P{i}" for i in range(1, 25)) + "\n"
        + "2026-01-05,2026-01-05," + ",".join(["10.0"] * 24) + "\n"
        + "2026-01-06,2026-01-05," + ",".join(["10.0"] * 23 + ["nan"]) + "\n")
    em, sd = insights.cargar_emitidos("X", tmp_path)
    assert list(em) == ["2026-01-05"] and sd["2026-01-05"] == "2026-01-05"  # la fila con NaN se descarta
    assert insights.cargar_emitidos("X", tmp_path)[0] is em  # caché
    assert insights.cargar_emitidos("NoExiste", tmp_path) == ({}, {})


def test_csv_reales_del_repo_son_validos():
    for ucp in ("Antioquia", "Atlantico"):
        em, sd = insights.cargar_emitidos(ucp)
        assert len(em) == 364 and len(set(sd.values())) == 52
        for f, s in sd.items():
            assert 0 <= (date.fromisoformat(f) - date.fromisoformat(s)).days <= 6
            assert date.fromisoformat(s).weekday() == 0


# ---------------------------------------------------------------------------
# B6: confianza con tope de antigüedad, clima sin cifras crudas, guardas y textos
# ---------------------------------------------------------------------------

def test_confianza_historial_viejo_es_sin_datos():
    d = _datos()
    semana = LUNES0 + timedelta(weeks=N_SEMANAS + 30)  # el historial termina hace 30 semanas
    r = insights.resumen(_corrida(semana), d)
    assert r["confianza"]["nivel"] == "sin_datos" and r["kpis"]["error_reciente_pct"] is None
    assert r["confianza"]["semanas_evaluadas"] == 0


def test_confianza_texto_hasta_fecha_si_la_ultima_semana_es_vieja():
    d = _datos()
    cerca = insights.resumen(_corrida(LUNES0 + timedelta(weeks=N_SEMANAS)), d)
    assert "de las últimas 8 semanas" in cerca["confianza"]["texto"]
    lejos = insights.resumen(_corrida(LUNES0 + timedelta(weeks=N_SEMANAS + 11)), d)  # última evaluada: hace 12 semanas
    assert lejos["confianza"]["nivel"] != "sin_datos"
    assert "hasta el" in lejos["confianza"]["texto"] and "últimas" not in lejos["confianza"]["texto"]


def test_clima_no_imprime_cifra_cruda_de_lluvia():
    det = {"temp_media": 24.0, "lluvia_mm": 129507.0, "lluvia_max_diaria_mm": 30000.0, "p90_lluvia_mm": 20000.0}
    i = insights._expandir_alerta(Alert("warning", "CLIMA_OBSERVADO", "x", [], det))[0]
    assert "129507" not in i["texto"] and "mm" not in i["texto"] and "lluvia por encima de lo habitual" in i["texto"]
    c = _corrida(date(2026, 6, 29), alertas=[("control", Alert("warning", "CLIMA_OBSERVADO", "x", [], det))])
    assert "129507" not in str(insights.recomendaciones(c, insights.Datos()))


def test_dias_con_real_cero_no_rompen():
    d = _datos()
    f = [x for x in d.reales if date.fromisoformat(x).weekday() == 5][-1]
    d.reales[f] = np.zeros(24)
    assert f not in insights._pares(d)
    insights.sesgo_por_segmento(d)
    insights.tablas_banda(d)
    insights.resumen(_corrida(LUNES0 + timedelta(weeks=N_SEMANAS)), d)
    assert insights._stats_segmento(d, [f, f, f]) is None


def test_textos_plural_desvio_y_jerga():
    def txt(n):
        alertas = [("control", Alert("bloqueo", f"CURVA_BASE_NULA", "x", [f"2026-07-0{i + 1}"])) for i in range(n)]
        return insights.resumen(_corrida(date(2026, 6, 29), alertas=alertas), insights.Datos())["siguiente_paso"]["texto"]
    assert txt(1) == "Hay 1 problema que impide aprobar: revísalo primero."
    assert txt(2).startswith("Hay 2 problemas que impiden aprobar")
    sin = insights._expandir_alerta(Alert("warning", "TENDENCIA_FUERA_BANDA", "x", ["2026-07-02"]))[0]["texto"]
    assert "  " not in sin and "quedó distinta" in sin
    assert "quedó +2.3 % distinta" in insights._expandir_alerta(
        Alert("warning", "TENDENCIA_FUERA_BANDA", "x", ["2026-07-02"], {"desvio_pp": 2.3}))[0]["texto"]
    assert insights.banda_json(insights.tablas_banda(_datos()), _datos())["cobertura_medida"]["detalle_tecnico"]
