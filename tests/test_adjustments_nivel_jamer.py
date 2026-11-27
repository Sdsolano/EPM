"""Tests de nivel.modo="reciente" (E2): g dinámico, peso_modelo, diagnósticos y alertas."""

import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.adjustments import referencia_reciente as rr
from src.adjustments.data import clase_dia, curvas_reales, dias_semana, festivos_colombia, iso
from src.adjustments.pipeline import build_context
from src.adjustments.steps import nivel
from src.adjustments.types import Alert
from tests.ajustes_sinteticos import historico_sintetico

HIST = historico_sintetico()
FESTIVOS = set(festivos_colombia(range(2014, 2027)))
SEMANA = date(2025, 9, 8)  # sin festivos; corte 2025-09-03


def _ctx(semana=SEMANA):
    return build_context("Sintetica", semana, fecha_emision=semana - timedelta(days=4), historico=HIST)


def _prev(ctx, mw=1000.0):
    return {iso(d): [mw] * 24 for d in dias_semana(ctx.semana_inicio)}


def _step(ctx, params=None, prev=None):
    p = {"modo": "reciente", **(params or {})}
    return nivel.step(ctx, prev if prev is not None else _prev(ctx), p)


# ---------------------------------------------------------------- params

def test_params_default_no_cambian_ni_se_persisten():
    assert nivel.validar_params({}) == {"alpha": 0.25}
    assert nivel.validar_params({"modo": "alpha", "peso_modelo": 0.0, "g_k": 1.0}) == {"alpha": 0.25}
    p = nivel.validar_params({"modo": "reciente", "peso_modelo": 0.3, "g_fuente": "interanual"})
    # con modo="reciente" se persisten los valores efectivos (también los iguales al default): ver EA1
    assert p == {"alpha": 0.25, "modo": "reciente", "peso_modelo": 0.3, "saltar_puente": False,
                 **_crec_mod.DEFAULTS_G}
    # idempotente: re-validar lo ya validado devuelve lo mismo
    assert nivel.validar_params(p) == p


@pytest.mark.parametrize(
    "mal",
    [
        {"modo": "otro"},
        {"peso_modelo": 1.5},
        {"peso_modelo": -0.1},
        {"peso_modelo": None},
        {"g_fuente": "x"},
        {"g_k": -1},
        {"g_min": 0.05, "g_max": 0.01},
        {"g_ventana_dias": 10},
        {"g_min_pares": 0},
        {"saltar_puente": "si"},
        {"g_fijo": float("nan")},
        {"clave_rara": 1},
    ],
)
def test_params_invalidos(mal):
    with pytest.raises(ValueError):
        nivel.validar_params(mal)


# ---------------------------------------------------------------- regresión modo alpha

def test_modo_alpha_default_identico_y_sin_claves_nuevas():
    ctx = _ctx()
    ref = nivel.step(ctx, _prev(ctx), {})
    exp = nivel.step(ctx, _prev(ctx), {"modo": "alpha", "peso_modelo": 0.4, "g_k": 2.0})
    assert ref.curves == exp.curves and ref.diagnostics == exp.diagnostics
    assert "referencias" not in ref.diagnostics and "crecimiento" not in ref.diagnostics


# ---------------------------------------------------------------- fórmula

def test_reciente_w0_energia_es_sumaR1_por_1_mas_g_y_forma_intacta():
    ctx = _ctx()
    prev = {iso(d): list(np.linspace(900, 1100, 24)) for d in dias_semana(ctx.semana_inicio)}
    res = _step(ctx, prev=prev)
    cv = curvas_reales(ctx.historico)
    for f, dd in res.diagnostics["dias"].items():
        e_r1 = float(cv[dd["r1"]["fecha"]].sum())
        assert dd["r1"]["energia"] == pytest.approx(e_r1)
        assert dd["E_obj"] == pytest.approx(e_r1 * (1 + dd["g_usado"]))
        assert sum(res.curves[f]) == pytest.approx(dd["E_obj"], rel=1e-9)
        assert dd["E"] == dd["E_final"] == pytest.approx(dd["E_obj"])
        # forma proporcional: curva final / curva previa constante
        ratio = np.asarray(res.curves[f]) / np.asarray(prev[f])
        assert np.allclose(ratio, ratio[0])
        assert dd["modo"] == "reciente" and dd["regla"] == "reciente" and dd["acotado"] is False
        # lag y g dinámico: (1+Y)^(lag/364) − 1 con Y del diagnóstico
        Y = res.diagnostics["crecimiento"]["Y"]
        assert dd["g_raw"] == pytest.approx((1 + Y) ** (dd["lag_dias"] / 364) - 1)
        assert dd["lag_dias"] == (date.fromisoformat(f) - date.fromisoformat(dd["r1"]["fecha"])).days


def test_peso_modelo_mezcla_y_limites():
    ctx = _ctx()
    prev = _prev(ctx, 1000.0)  # E_base = 24000
    r0 = _step(ctx, {"peso_modelo": 0.0})
    r1 = _step(ctx, {"peso_modelo": 1.0})
    rm = _step(ctx, {"peso_modelo": 0.25})
    for f in prev:
        assert sum(r1.curves[f]) == pytest.approx(24000.0)  # w=1 → solo modelo
        e_obj = r0.diagnostics["dias"][f]["E_obj"]
        assert sum(rm.curves[f]) == pytest.approx(0.75 * e_obj + 0.25 * 24000.0)
        assert rm.diagnostics["dias"][f]["E_base"] == 24000.0
        assert rm.diagnostics["dias"][f]["alpha"] == pytest.approx(0.75)
    assert r1.diagnostics["peso_modelo"] == 1.0


def test_g_fijo_y_fuentes():
    ctx = _ctx()
    fijo = _step(ctx, {"g_fuente": "fijo", "g_fijo": 0.004})
    assert all(d["g_usado"] == pytest.approx(0.004) for d in fijo.diagnostics["dias"].values())
    assert not [a for a in fijo.alerts if a.codigo.startswith("G_")]
    base = _step(ctx)
    corto = _step(ctx, {"g_fuente": "interanual_corto"})
    est = _step(ctx, {"g_fuente": "interanual_estacional"})
    assert corto.diagnostics["crecimiento"]["ventana"] != base.diagnostics["crecimiento"]["ventana"]
    assert corto.diagnostics["crecimiento"]["fuente_g"] == "interanual_corto"
    assert est.diagnostics["crecimiento"]["fuente_g"] == "interanual_estacional"
    assert all(np.isfinite(d["g_usado"]) for d in est.diagnostics["dias"].values())
    # g_k escala el g crudo
    k2 = _step(ctx, {"g_k": 2.0})
    for f, d in k2.diagnostics["dias"].items():
        assert d["g_usado"] == pytest.approx(2 * base.diagnostics["dias"][f]["g_usado"])


def test_g_acotado_emite_alerta_unica_con_fechas():
    res = _step(_ctx(), {"g_k": 1000.0, "g_max": 0.01})
    assert all(d["acotado"] and d["g_usado"] == pytest.approx(0.01) for d in res.diagnostics["dias"].values())
    al = [a for a in res.alerts if a.codigo == "G_ACOTADO"]
    assert len(al) == 1 and len(al[0].fechas) == 7 and al[0].nivel == "info"
    assert set(al[0].detalle["por_fecha"]) == set(al[0].fechas)


def test_g_sin_evidencia_usa_g_fijo():
    res = _step(_ctx(), {"g_ventana_dias": 7, "g_min_pares": 30, "g_fijo": 0.003})
    assert res.diagnostics["crecimiento"]["fallback"] is True
    assert res.diagnostics["crecimiento"]["Y"] is None
    assert all(d["g_usado"] == pytest.approx(0.003) for d in res.diagnostics["dias"].values())
    assert any(a.codigo == "G_SIN_EVIDENCIA" for a in res.alerts)


def test_sin_r1_conserva_base_con_alerta():
    ctx = _ctx()
    ctx.historico = ctx.historico.iloc[0:0]
    prev = _prev(ctx)
    res = _step(ctx, prev=prev)
    assert res.curves == prev
    al = [a for a in res.alerts if a.codigo == "SIN_EVIDENCIA_NIVEL"]
    assert al and len(al[0].fechas) == 7
    assert all(d["regla"] == "sin_evidencia" and d["r1"] is None for d in res.diagnostics["dias"].values())


def test_base_nula_no_divide_por_cero():
    ctx = _ctx()
    prev = {iso(d): [0.0] * 24 for d in dias_semana(ctx.semana_inicio)}
    res = _step(ctx, prev=prev)
    assert res.curves == prev and any(a.codigo == "CURVA_BASE_NULA" for a in res.alerts)


# ---------------------------------------------------------------- diagnósticos y referencias

def test_referencias_y_crecimiento():
    ctx = _ctx()
    res = _step(ctx)
    cv = curvas_reales(ctx.historico)
    c = res.diagnostics["crecimiento"]
    assert set(c) == {"Y", "ventana", "n_pares", "fuente_g", "fallback"} and c["n_pares"] >= 14
    refs = res.diagnostics["referencias"]
    assert set(refs) == set(res.curves)
    for f, r in refs.items():
        d = date.fromisoformat(f)
        r1 = date.fromisoformat(r["r1"]["fecha"])
        assert len(r["r1"]["curva"]) == 24 and r1 <= ctx.fecha_corte
        assert r["r1"]["curva"] == [float(x) for x in cv[iso(r1)]]
        assert r["r3"]["fecha"] == iso(r1 - timedelta(days=364))
        assert r["r2"]["fecha"] == iso(d - timedelta(days=364)) and len(r["r2"]["curva"]) == 24
        assert [e["fecha"] for e in r["r1_extra"]] == [iso(r1 - timedelta(days=7))]
        assert res.diagnostics["dias"][f]["ratio"] == pytest.approx(sum(r["r2"]["curva"]) / sum(r["r3"]["curva"]))


def test_r1_misma_clase_con_festivos():
    semana = date(2025, 11, 17)  # lun 17/11 festivo; mar 18 post-festivo
    ctx = _ctx(semana)
    res = _step(ctx)
    for f, dd in res.diagnostics["dias"].items():
        assert dd["clase"] == clase_dia(f, FESTIVOS)
        assert clase_dia(dd["r1"]["fecha"], FESTIVOS) == dd["clase"]
    lun = res.diagnostics["dias"]["2025-11-17"]
    assert lun["clase"] == "festivo" and lun["r1"]["fecha"] != iso(ctx.fecha_corte)


def test_alertas_de_e1_se_propagan(monkeypatch):
    ctx = _ctx()
    orig = rr.seleccionar_recientes

    def falso(d, *a, **k):
        r = orig(d, *a, **k)
        r.alertas.append(Alert(nivel="info", codigo="R1_SIN_MISMA_CLASE", mensaje="x", fechas=[iso(d)]))
        return r

    monkeypatch.setattr(rr, "seleccionar_recientes", falso)
    res = _step(ctx)
    al = [a for a in res.alerts if a.codigo == "R1_SIN_MISMA_CLASE"]
    assert len(al) == 1 and len(al[0].fechas) == 7


def test_no_modifica_prev():
    ctx = _ctx()
    prev = _prev(ctx)
    copia = {k: list(v) for k, v in prev.items()}
    _step(ctx, prev=prev)
    assert prev == copia


# ---------------------------------------------------------------- caso réplica (semana 23–29/11, datos reales opcionales)

JAMER = Path(__file__).resolve().parents[1] / "reports" / "ajustes" / "jamer"
_hay = all((JAMER / n).exists() for n in ("historico_replica.csv", "base_sintetica_23_29_nov.csv", "verdad_jamer.csv", "fixture_replica.json"))


@pytest.mark.skipif(not _hay, reason="datos de réplica (reports/ajustes/jamer: histórico, base, verdad y fixture; ignorados por git) no disponibles")
def test_replica_semana_23_29_nov_vs_jamer():
    from scripts.backtest_adjustments import cargar_archivo

    hist = pd.read_csv(JAMER / "historico_replica.csv")
    ctx = build_context("Antioquia", date(2026, 11, 23), fecha_emision=date(2026, 11, 15), historico=hist)
    assert ctx.fecha_corte == date(2026, 11, 14)
    base = {f: v for f, v in cargar_archivo(JAMER / "base_sintetica_23_29_nov.csv").items() if f >= "2026-11-23"}
    res = nivel.step(ctx, base, {"modo": "reciente"})
    ses = pd.read_csv(JAMER / "verdad_jamer.csv")
    ses = ses[ses["tipo"] == "P"]
    jamer = {r["fecha"]: float(sum(r[f"p{i}"] for i in range(1, 25))) for _, r in ses.iterrows()}
    r1 = {f: d["r1"]["fecha"] for f, d in res.diagnostics["dias"].items()}
    assert r1 == json.loads((JAMER / "fixture_replica.json").read_text(encoding="utf-8"))["r1_esperado"]
    assert res.diagnostics["dias"]["2026-11-24"]["r1_extra"][0]["fecha"] == "2026-10-27"
    assert {"fecha": "2026-11-03", "motivo": "post_festivo"} in res.diagnostics["dias"]["2026-11-24"]["saltados"]
    err = {f: sum(res.curves[f]) / jamer[f] - 1 for f in res.curves}
    sem = sum(sum(v) for v in res.curves.values()) / sum(jamer[f] for f in res.curves) - 1
    assert abs(sem) <= 0.005
    assert all(abs(e) <= 0.01 for e in err.values())
    assert sum(abs(e) <= 0.005 for e in err.values()) >= 5
    gs = [d["g_usado"] for d in res.diagnostics["dias"].values()]
    assert all(0.002 <= g <= 0.009 for g in gs)
    assert not res.alerts


# ---------------------------------------------------------------- correcciones de la ronda 1 (E3, E4, E6, E9)

from src.adjustments import referencia_reciente as _rr_mod  # noqa: E402
from src.adjustments.data import PERIODOS, LAG_ANIO  # noqa: E402
from src.adjustments.types import RunContext  # noqa: E402

FESTIVOS_26 = sorted(festivos_colombia(range(2024, 2028)))
CORTE_26 = date(2026, 11, 14)
LUNES_26 = date(2026, 11, 23)


def _curva_26(d: date, factor: float = 1.0):
    h = np.arange(24)
    return (1000 + 40 * d.weekday() + 300 * np.sin((h - 5) / 24 * 2 * np.pi)) * factor


def _ctx_26(crec=0.04, semana_ref=LUNES_26 - timedelta(days=364)) -> RunContext:
    """Histórico sintético 2025-06-01..corte con 2026 = (1+crec)·2025 y festivos reales de 2025 y 2026."""
    filas, d = [], date(2025, 6, 1)
    while d <= CORTE_26:
        v = _curva_26(d, (1 + crec) if d.year == 2026 else 1.0)
        filas.append({"fecha": pd.Timestamp(d), **{c: float(x) for c, x in zip(PERIODOS, v)}, "TOTAL": float(v.sum())})
        d += timedelta(days=1)
    return RunContext(ucp="Antioquia", semana_inicio=LUNES_26, fecha_emision=LUNES_26, fecha_corte=CORTE_26,
                      historico=pd.DataFrame(filas), festivos=FESTIVOS_26, semana_ref_inicio=semana_ref)


def _prev_26(ctx):
    return {iso(d): [1000.0] * 24 for d in dias_semana(ctx.semana_inicio)}


def test_ventana_g_corto_escala_el_minimo_de_pares():
    assert nivel._ventana_g("interanual", 28, 14) == (28, 14)
    assert nivel._ventana_g("fijo", 28, 14) == (28, 14)
    assert nivel._ventana_g("interanual_corto", 28, 14) == (14, 7)   # 50 % de la ventana
    assert nivel._ventana_g("interanual_corto", 14, 14) == (14, 14)
    assert nivel._ventana_g("interanual_corto", 28, 1) == (14, 1)


def test_interanual_corto_corte_14_nov_2026_da_Y_valido_y_no_cae_a_g_fijo():
    """E3: ventana 14 d (01–14/nov) con el festivo 02/11 y post-festivo 03/11 de 2026 y el de 2025: ≥ 7 pares → Y válido."""
    ctx = _ctx_26(crec=0.04)
    res = nivel.step(ctx, _prev_26(ctx), {"modo": "reciente", "g_fuente": "interanual_corto"})
    c = res.diagnostics["crecimiento"]
    assert c["fallback"] is False and c["Y"] == pytest.approx(0.04, abs=1e-9)
    assert c["ventana"] == ["2026-11-01", "2026-11-14"] and 7 <= c["n_pares"] < 14
    assert not any(a.codigo == "G_SIN_EVIDENCIA" for a in res.alerts)
    # antes de la corrección: min_pares=14 sobre 14 días con festivos → siempre fallback
    viejo = _crec_mod.crecimiento_interanual(
        curvas_reales(ctx.historico), CORTE_26, set(ctx.festivos), ventana_dias=14, min_pares=14)
    assert viejo.fallback is True


def test_interanual_estacional_calcula_recientes_una_vez_por_dia(monkeypatch):
    ctx = _ctx_26()
    llamadas = []
    original = _rr_mod.seleccionar_recientes

    def espia(d, *a, **k):
        llamadas.append(iso(d) if hasattr(d, "isoformat") else d)
        return original(d, *a, **k)

    monkeypatch.setattr(_rr_mod, "seleccionar_recientes", espia)
    res = nivel.step(ctx, _prev_26(ctx), {"modo": "reciente", "g_fuente": "interanual_estacional"})
    assert len(llamadas) == 7 and len(set(llamadas)) == 7
    assert res.diagnostics["crecimiento"]["fuente_g"] == "interanual_estacional"


from src.adjustments import crecimiento as _crec_mod  # noqa: E402


def test_desfase_unico_y_r2_consistente_con_semana_ref_igual_a_semana_objetivo():
    """E6: con semana_ref = semana objetivo el desfase efectivo es 364 en diagnostics y en R2 (antes: 0 vs 364)."""
    ctx = _ctx_26(semana_ref=LUNES_26)
    res = nivel.step(ctx, _prev_26(ctx), {"modo": "reciente"})
    assert res.diagnostics["desfase_dias"] == LAG_ANIO
    r2 = res.diagnostics["referencias"]["2026-11-24"]["r2"]["fecha"]
    assert r2 == (date(2026, 11, 24) - timedelta(days=LAG_ANIO)).isoformat()
    sin_ref = _ctx_26(semana_ref=None)  # sin semana de referencia R2 usa d − 364 y el diagnóstico lo dice
    assert nivel.step(sin_ref, _prev_26(sin_ref), {"modo": "reciente"}).diagnostics["desfase_dias"] == LAG_ANIO
    con_ref = _ctx_26(semana_ref=LUNES_26 - timedelta(days=28))
    r = nivel.step(con_ref, _prev_26(con_ref), {"modo": "reciente"})
    assert r.diagnostics["desfase_dias"] == 28
    assert r.diagnostics["referencias"]["2026-11-24"]["r2"]["fecha"] == "2026-10-27"


def test_saltar_puente_booleano_estricto_en_nivel_y_forma():
    from src.adjustments.steps import forma

    for malo in (1, 0, "true", None):
        with pytest.raises(ValueError, match="saltar_puente"):
            nivel.validar_params({"modo": "reciente", "saltar_puente": malo})
        with pytest.raises(ValueError, match="saltar_puente"):
            forma.validar_params({"saltar_puente": malo})
    assert nivel.validar_params({"modo": "reciente", "saltar_puente": np.bool_(True)})["saltar_puente"] is True
    assert forma.validar_params({"saltar_puente": True})["saltar_puente"] is True


def test_unir_alertas_mensaje_por_dia_no_el_del_primero():
    from src.adjustments.types import Alert

    a1 = Alert("info", "X", "dia uno", ["2026-11-23"], {"v": 1})
    a2 = Alert("info", "X", "dia dos", ["2026-11-24"], {"v": 2})
    (u,) = _rr_mod.unir_alertas([a1, a2])
    assert u.fechas == ["2026-11-23", "2026-11-24"]
    assert u.mensaje not in ("dia uno", "dia dos")  # genérico: no el del primero
    assert u.detalle["por_fecha"]["2026-11-24"] == {"v": 2, "mensaje": "dia dos"}
    (igual,) = _rr_mod.unir_alertas([a1, Alert("info", "X", "dia uno", ["2026-11-24"])])
    assert igual.mensaje == "dia uno"


def test_ancla_analogos_en_modo_reciente_se_ignora_con_alerta_informativa():
    ctx = _ctx_26()
    con = nivel.step(ctx, _prev_26(ctx), {"modo": "reciente", "ancla": "analogos"})
    (a,) = [x for x in con.alerts if x.codigo == "ANCLA_IGNORADA_MODO_RECIENTE"]
    assert a.nivel == "info" and a.detalle == {"ancla": "analogos"}
    sin = nivel.step(ctx, _prev_26(ctx), {"modo": "reciente"})
    assert not [x for x in sin.alerts if x.codigo == "ANCLA_IGNORADA_MODO_RECIENTE"]
    assert con.curves == sin.curves  # solo avisa: el cálculo no cambia
