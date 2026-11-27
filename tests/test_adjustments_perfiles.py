"""Tests de `src/adjustments/perfiles.py` (expansión de perfiles a params por paso)."""

from __future__ import annotations

import copy

import pytest

from src.adjustments import crecimiento, perfiles
from src.adjustments.steps import control, forma, nivel


def test_perfiles_conocidos():
    assert perfiles.PERFILES == ("jamer", "jamer_modelo")


def test_jamer_expande_replica_pura():
    p = perfiles.expandir("jamer", None)
    assert p["nivel"] == {"modo": "reciente", "g_fuente": "interanual", "peso_modelo": 0.0, "festivo_por_tipo": True}
    assert p["forma"] == {"fuente": "reciente_clase", "n": 1, "delta": "plano", "tramos": "off",
                          "festivo_por_tipo": True}
    assert p["control"] == {"banda_salto_mw": 40.0, "periodos_valle": [3, 4, 5], "crecimiento_ventana_dias": 28,
                            "crecimiento_min_pares": 14, "banda_error_pct": 5.0}
    assert set(p) == {"nivel", "forma", "control"}


def test_jamer_modelo_solo_cambia_peso_modelo():
    # jamer_modelo difiere de jamer solo en peso_modelo y en no activar festivo_por_tipo (festivos_r1).
    a, b = perfiles.expandir("jamer", None), perfiles.expandir("jamer_modelo", None)
    assert b["nivel"]["peso_modelo"] == perfiles.PESO_MODELO_JAMER_MODELO == 0.3
    b["nivel"]["peso_modelo"] = 0.0
    assert a["nivel"].pop("festivo_por_tipo") is True and a["forma"].pop("festivo_por_tipo") is True
    assert a == b


def test_sin_perfil_devuelve_params_tal_cual():
    assert perfiles.expandir(None, None) is None
    p = {"nivel": {"alpha": 0.5}}
    assert perfiles.expandir(None, p) is p


def test_usuario_gana_clave_a_clave_y_conserva_el_resto():
    usuario = {"nivel": {"peso_modelo": 0.5, "g_k": 1.2}, "forma": {"delta": "proporcional"},
               "especiales": {"x": 1}}
    antes = copy.deepcopy(usuario)
    p = perfiles.expandir("jamer", usuario)
    assert p["nivel"] == {"modo": "reciente", "g_fuente": "interanual", "peso_modelo": 0.5, "g_k": 1.2,
                          "festivo_por_tipo": True}
    assert p["forma"]["delta"] == "proporcional" and p["forma"]["fuente"] == "reciente_clase"
    assert p["control"]["banda_salto_mw"] == 40.0 and p["control"]["periodos_valle"] == [3, 4, 5]
    assert p["especiales"] == {"x": 1}
    assert usuario == antes  # no muta la entrada


def test_expandir_no_comparte_estado_entre_llamadas():
    a = perfiles.expandir("jamer", None)
    a["nivel"]["peso_modelo"] = 0.9
    assert perfiles.expandir("jamer", None)["nivel"]["peso_modelo"] == 0.0


def test_perfil_desconocido_value_error():
    with pytest.raises(ValueError, match="perfil desconocido"):
        perfiles.expandir("otro", None)
    with pytest.raises(ValueError):
        perfiles.params_perfil("JAMER")


@pytest.mark.parametrize("perfil", perfiles.PERFILES)
def test_las_claves_del_perfil_las_aceptan_los_pasos_reales(perfil):
    """Las claves del perfil existen y validan en nivel/forma/control (no se inventan nombres)."""
    p = perfiles.expandir(perfil, None)
    n = nivel.validar_params(p["nivel"])
    assert n["modo"] == "reciente"
    f = forma.validar_params(p["forma"])
    assert f["fuente"] == "reciente_clase" and f["delta"] == "plano"
    c = control.validar_params(p["control"])
    assert c["banda_salto_mw"] == 40.0


# ------------------------------------------------ params efectivos congelados en steps[].params (EA1)

def _correr(params):
    from datetime import date, timedelta

    from src.adjustments.pipeline import build_context, run_pipeline
    from tests.ajustes_sinteticos import ProveedorPlano, historico_sintetico

    semana = date(2025, 9, 8)
    ctx = build_context("Sintetica", semana, fecha_emision=semana - timedelta(days=4), historico=historico_sintetico())
    return ctx, run_pipeline(ctx, ProveedorPlano(), params)


@pytest.mark.parametrize("perfil", perfiles.PERFILES)
def test_con_perfil_los_params_efectivos_quedan_en_los_pasos(perfil):
    _, recs = _correr(perfiles.expandir(perfil, None))
    por = {r.paso: r.params for r in recs}
    for k, v in crecimiento.DEFAULTS_G.items():
        assert por["nivel"][k] == v
    assert por["nivel"]["modo"] == "reciente" and "peso_modelo" in por["nivel"]
    fijados = {"delta": "plano"}  # lo fija el perfil; el resto, default
    if perfil == perfiles.PERFIL_JAMER:
        fijados["festivo_por_tipo"] = True  # festivos_r1: solo jamer la activa
    for k, v in forma.DEFAULTS_JAMER.items():
        assert por["forma"][k] == fijados.get(k, v)
    for k in ("banda_salto_mw", "periodos_valle", "crecimiento_ventana_dias", "crecimiento_min_pares", "banda_error_pct"):
        assert k in por["control"]
    assert por["control"]["periodos_valle"] == [3, 4, 5]


def test_recalibrar_defaults_no_cambia_una_corrida_ya_creada(monkeypatch):
    """PATCH recalcula con `steps[].params`: tras cambiar los defaults del código el resultado no se mueve."""
    from src.adjustments.pipeline import rerun_from
    from src.adjustments.types import StepRecord
    from tests.ajustes_sinteticos import ProveedorPlano

    ctx, recs = _correr(perfiles.expandir("jamer", None))
    antes = {r.paso: {f: list(c) for f, c in r.result.curves.items()} for r in recs}
    antes_diag = {r.paso: r.result.diagnostics for r in recs}

    monkeypatch.setitem(crecimiento.DEFAULTS_G, "g_k", 3.0)
    monkeypatch.setitem(crecimiento.DEFAULTS_G, "g_min", -0.2)
    monkeypatch.setitem(crecimiento.DEFAULTS_G, "g_max", 0.2)
    monkeypatch.setitem(crecimiento.DEFAULTS_G, "g_fijo", 0.15)
    monkeypatch.setitem(crecimiento.DEFAULTS_G, "g_ventana_dias", 14)
    monkeypatch.setitem(nivel.DEFAULTS_RECIENTE, "g_k", 3.0)
    monkeypatch.setitem(nivel.DEFAULTS_RECIENTE, "g_max", 0.2)
    monkeypatch.setitem(nivel.DEFAULTS_RECIENTE, "g_min", -0.2)
    monkeypatch.setitem(nivel.DEFAULTS_RECIENTE, "g_fijo", 0.15)
    monkeypatch.setitem(nivel.DEFAULTS_RECIENTE, "g_ventana_dias", 14)
    monkeypatch.setitem(forma.DEFAULTS_JAMER, "umbral_tramo_mw", 1.0)
    monkeypatch.setitem(forma.DEFAULTS_JAMER, "suavizado_borde", 4)
    monkeypatch.setitem(control.DEFAULTS_JAMER, "crecimiento_ventana_dias", 56)
    monkeypatch.setitem(control.DEFAULTS_JAMER, "periodos_valle", [1, 2])

    nuevo = next(r for r in recs if r.paso == "nivel")  # con sus params persistidos
    rec2 = rerun_from(ctx, recs, "nivel", ProveedorPlano(),
                      StepRecord(nuevo.orden, nuevo.paso, nuevo.estado, dict(nuevo.params), nuevo.result))
    despues = {r.paso: {f: list(c) for f, c in r.result.curves.items()} for r in rec2}
    assert despues == antes
    assert {r.paso: r.result.diagnostics for r in rec2} == antes_diag


def test_sin_perfil_los_params_siguen_sin_persistir_defaults():
    """Sin perfil nada cambia: los params de los pasos son los de siempre (solo lo que difiere del default)."""
    _, recs = _correr(None)
    por = {r.paso: r.params for r in recs}
    assert por["nivel"] == {"alpha": 0.25}
    assert "g_k" not in por["nivel"] and "umbral_tramo_mw" not in por["forma"]
    assert "periodos_valle" not in por["control"] and "banda_salto_mw" not in por["control"]


def test_jamer_activa_festivo_por_tipo_en_nivel_y_forma():
    # festivos_r1: la regla pasó el criterio de aceptación solo para `jamer`.
    p = perfiles.params_perfil("jamer")
    assert p["nivel"]["festivo_por_tipo"] is True
    assert p["forma"]["festivo_por_tipo"] is True
    assert "r1_limpia" not in p["nivel"] and "r1_limpia" not in p["forma"]
    pm = perfiles.params_perfil("jamer_modelo")
    assert "festivo_por_tipo" not in pm["nivel"] and "festivo_por_tipo" not in pm["forma"]
