"""Tests de funciones puras de scripts/backtest_mesa_vs_jamer.py (datos sintéticos; sin red ni datos reales)."""

from __future__ import annotations

import importlib.util
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.adjustments import data
from tests.ajustes_sinteticos import historico_sintetico

ROOT = Path(__file__).resolve().parent.parent


def _cargar_script():
    spec = importlib.util.spec_from_file_location(
        "backtest_mesa_vs_jamer", ROOT / "scripts" / "backtest_mesa_vs_jamer.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["backtest_mesa_vs_jamer"] = mod
    spec.loader.exec_module(mod)
    return mod


bmj = _cargar_script()

PER = [f"P{i}" for i in range(1, 25)]


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def hist():
    return data.normalizar_historico(historico_sintetico(date(2024, 1, 1), date(2025, 11, 30)), ucp="Antioquia")


@pytest.fixture(scope="module")
def festivos():
    return data.festivos_colombia(range(2023, 2027))


@pytest.fixture(scope="module")
def reales(hist):
    return data.curvas_reales(hist)


def _envio(sesion, envio, lunes, valor=1000.0, version=1):
    return [{"sesion": sesion, "envio": pd.Timestamp(envio), "version": version,
             "fecha": (lunes + timedelta(days=i)).isoformat(), **{p: valor for p in PER}} for i in range(7)]


# ---------------------------------------------------------------------------
# selección de semanas / VF / final / corte
# ---------------------------------------------------------------------------

def test_seleccionar_envios_vf_final_y_descartes():
    L = date(2025, 11, 10)
    filas = (_envio(1, "2025-11-05 20:00", L) + _envio(2, "2025-11-07 10:00", L, version=2)
             + _envio(3, "2025-11-12 09:00", L, version=3)        # dentro de la semana: cuenta como final
             + _envio(4, "2025-11-20 09:00", L, version=4)        # posterior al domingo: retroactivo, no cuenta
             + _envio(5, "2025-11-01 09:00", date(2025, 11, 4))   # la sesión no empieza en lunes
             + _envio(6, "2025-10-29 09:00", date(2025, 11, 3)))
    sem = bmj.seleccionar_envios(pd.DataFrame(filas))
    assert list(sem["lunes"]) == [date(2025, 11, 3), L]
    s = sem.set_index("lunes").loc[L]
    assert s["vf_sesion"] == 1 and s["final_sesion"] == 3 and s["n_envios"] == 3
    assert list(bmj.seleccionar_envios(pd.DataFrame(filas), desde=L)["lunes"]) == [L]


def test_cortes_y_dias_a_evaluar():
    assert bmj.corte_de_envio(pd.Timestamp("2025-11-05 20:00")) == date(2025, 11, 4)
    assert bmj.corte_martes(date(2025, 11, 10)) == date(2025, 11, 4)
    L = date(2025, 11, 10)
    reales = {(L + timedelta(days=i)).isoformat(): np.ones(24) for i in range(5)}  # faltan sáb y dom
    assert bmj.dias_a_evaluar(L, reales) == [(L + timedelta(days=i)).isoformat() for i in range(5)]
    assert bmj.dias_a_evaluar(L, reales, pd.Timestamp("2025-11-12 09:00")) == ["2025-11-13", "2025-11-14"]


def test_con_tramo_60_40_plan():
    # 33 semanas del PLAN → 20 de calibración y 13 de prueba (partición de backtest_jamer.partir_calibracion)
    lunes = [(date(2026, 4, 6) + timedelta(days=7 * i)).isoformat() for i in range(33)]
    df = bmj.con_tramo(pd.DataFrame({"lunes": lunes[::-1]}))
    assert (df["tramo"] == "calibracion").sum() == 20 and (df["tramo"] == "prueba").sum() == 13
    assert set(df[df["tramo"] == "prueba"]["lunes"]) == set(lunes[20:])
    # al menos una semana por lado
    assert sorted(bmj.con_tramo(pd.DataFrame({"lunes": ["a", "b"]}))["tramo"]) == ["calibracion", "prueba"]
    assert bmj.con_tramo(pd.DataFrame()).empty


def test_curvas_envio():
    L = date(2025, 11, 10)
    env = pd.DataFrame(_envio(9, "2025-11-05", L, valor=5.0))
    c = bmj.curvas_envio(env, 9)
    assert sorted(c) == [(L + timedelta(days=i)).isoformat() for i in range(7)]
    assert c[L.isoformat()].sum() == pytest.approx(120.0)


# ---------------------------------------------------------------------------
# corrida sin fuga y configs no soportadas
# ---------------------------------------------------------------------------

def test_corredor_sin_fuga_y_salta_config_desconocida(hist, festivos):
    configs = (bmj.Config("jamer", {}, None),
               bmj.Config("jamer+inexistente", bmj._ambos({"clave_que_no_existe": True}), "festivo"),
               bmj.Config("jamer+saltar_puente", bmj._ambos({"saltar_puente": True}), "r1_cambia"))
    corr = bmj.Corredor(hist, festivos, configs)
    L = date(2025, 11, 17)
    curvas, r1s, corte = corr.correr(L, date(2025, 11, 11))
    assert corte <= date(2025, 11, 11)
    assert set(curvas) == {"jamer", "jamer+saltar_puente"}
    assert "jamer+inexistente" in corr.saltadas and "desconocid" in corr.saltadas["jamer+inexistente"]
    # R1 de nivel siempre <= corte (sin fuga) y las curvas cubren la semana objetivo
    for f in [(L + timedelta(days=i)).isoformat() for i in range(7)]:
        assert f in curvas["jamer"]
        r1 = r1s["jamer"][f]["r1"]
        assert r1 is None or date.fromisoformat(r1) <= corte
    # la config saltada no se vuelve a intentar
    curvas2, _, _ = corr.correr(date(2025, 11, 24), date(2025, 11, 18))
    assert set(curvas2) == {"jamer", "jamer+saltar_puente"}


def test_recortar_sin_fuga(hist):
    hc = bmj.recortar_sin_fuga(hist, date(2025, 6, 30))
    assert hc["fecha"].max().date() == date(2025, 6, 30)


# ---------------------------------------------------------------------------
# métricas
# ---------------------------------------------------------------------------

def test_errores_dia():
    real = np.full(24, 100.0)
    pred = real.copy()
    pred[:12] = 110.0
    m = bmj.errores_dia(pred, real)
    assert m["mape_h"] == pytest.approx(5.0)
    assert m["err_dia"] == pytest.approx(5.0)
    assert m["energia"] == pytest.approx(2520.0) and m["real"] == pytest.approx(2400.0)


def _df_metricas():
    filas = []
    for k, lunes in enumerate(["2025-01-06", "2025-01-13", "2025-01-20", "2025-01-27"]):
        for j, tipo in enumerate(["laboral", "sabado", "festivo"]):
            f = f"{lunes[:8]}{int(lunes[8:]) + j:02d}"
            for serie, mape in (("jamer", 3.0 + k * 0.1), ("cfg", 2.5 + k * 0.1), ("jamer_envio", 2.8)):
                filas.append({"parte": 1, "lunes": lunes, "momento": "VF", "fecha": f, "tipo_dia": tipo,
                              "tipo_festivo": "fijo" if tipo == "festivo" else None,
                              "serie": serie, "mape_h": mape + (0.5 if (serie == "cfg" and tipo == "sabado") else 0),
                              "err_dia": 1.0, "energia": 101.0, "real": 100.0, "r1_saltada": tipo == "laboral",
                              "r1": "2024-12-30" if serie == "cfg" and tipo == "laboral" else "2024-12-23",
                              "r1_jamer": "2024-12-23"})
    return pd.DataFrame(filas)


def test_resumen_y_segmentos():
    df = _df_metricas()
    r = bmj.resumen(df).set_index("serie")
    assert r.loc["jamer", "dias"] == 12
    assert r.loc["jamer", "sesgo_energia"] == pytest.approx(1.0)
    assert r.loc["jamer", "p90_abs_err_dia"] == pytest.approx(1.0)
    fest = bmj.comparar_segmento(df, "festivo", "cfg")
    assert fest["dias"] == 4 and fest["dif_pp"] == pytest.approx(-0.5)
    # R1 saltada = días cuya R1 de nivel en la corrida de la config difiere de la de jamer (solo los laborales)
    r1 = bmj.comparar_segmento(df, "r1_saltada", "cfg")
    assert r1["dias"] == 4 and r1["dif_pp"] == pytest.approx(-0.5)
    assert bmj.comparar_segmento(df, "festivo_tipo", "cfg") == fest  # todos los festivos del fixture son "fijo"
    lunes = df.assign(tipo_festivo=np.where(df["tipo_dia"] == "festivo", "lunes", None))
    assert bmj.comparar_segmento(lunes, "festivo_tipo", "cfg")["dias"] == 0  # los festivos lunes no cuentan
    amb = bmj.comparar_segmento(df, "festivo_tipo|r1_saltada", "cfg")
    assert amb["dias"] == 8 and amb["dif_pp"] == pytest.approx(-0.5)
    # la mitad R1 de `ambas` excluye los festivos fijo/semana_santa aunque su R1 cambie
    df_amb = df.assign(r1=np.where(df["serie"] == "cfg", "2024-12-30", "2024-12-23"))
    assert bmj.comparar_segmento(df_amb, "r1_saltada", "cfg")["dias"] == 12
    assert bmj.comparar_segmento(df_amb, "r1_saltada_no_festivo_tipo", "cfg")["dias"] == 8
    # sin ningún día con R1 distinta el segmento queda vacío (sin error)
    igual = df.assign(r1="2024-12-23")
    assert bmj.comparar_segmento(igual, "r1_saltada", "cfg")["dias"] == 0
    cambia = bmj.comparar_segmento(df, "r1_cambia", "cfg")  # solo los laborales cambian de R1
    assert cambia["dias"] == 4 and cambia["dif_pp"] == pytest.approx(-0.5)
    tipos = bmj.dif_por_tipo_dia(df, "cfg")
    assert tipos["sabado"] == pytest.approx(0.0) and tipos["laboral"] == pytest.approx(-0.5)


def test_bootstrap_determinista_y_pareado():
    dif = np.array([-0.5, -0.2, 0.1, -0.3, -0.4])
    a = bmj.bootstrap_media(dif, semilla=7, n_boot=500)
    assert a == bmj.bootstrap_media(dif, semilla=7, n_boot=500)
    assert a != bmj.bootstrap_media(dif, semilla=8, n_boot=500)
    assert a[0] <= dif.mean() <= a[1]
    p = bmj.pareado(_df_metricas(), "cfg", "jamer", semilla=7, n_boot=500)
    assert p["semanas"] == 4 and p["gana_a"] == 4
    assert p["dif_media_pp"] == pytest.approx((-0.5 * 3 + 0.5) / 3)
    assert p == bmj.pareado(_df_metricas(), "cfg", "jamer", semilla=7, n_boot=500)


# ---------------------------------------------------------------------------
# criterio de aceptación
# ---------------------------------------------------------------------------

def _ok_inputs():
    return ([("ampliada", {"dias": 30, "dif_pp": -0.8}), ("prueba", {"dias": 3, "dif_pp": -0.4})],
            {"semanas": 33, "dif_media_pp": -0.02, "ic95": (-0.1, 0.05)},
            {"laboral": 0.0, "sabado": 0.05, "domingo": -0.1, "festivo": -1.2})


def test_veredicto_pasa_todo():
    v = bmj.veredicto_reglas(*_ok_inputs())
    assert v["pasa"] and all(r["pasa"] for r in v["reglas"])
    # los días de cada parte de la regla 1 van junto al veredicto
    assert v["reglas"][0]["valor"] == "ampliada -0.80 pp (30 d); prueba -0.40 pp (3 d)"


def _con_parte(i, seg):
    return lambda p, g, t: ([x if k != i else (x[0], seg) for k, x in enumerate(p)], g, t)


@pytest.mark.parametrize("cambio, regla", [
    (_con_parte(0, {"dias": 30, "dif_pp": 0.1}), 0),            # no mejora en la primera parte
    (_con_parte(1, {"dias": 3, "dif_pp": 0.0}), 0),             # empate en la segunda
    (_con_parte(1, {"dias": 0, "dif_pp": float("nan")}), 0),    # sin días en la segunda
    (lambda p, g, t: ([], g, t), 0),                            # sin partes
    (lambda p, g, t: (p, {"semanas": 33, "dif_media_pp": 0.01, "ic95": (-0.05, 0.08)}, t), 1),
    (lambda p, g, t: (p, {"semanas": 33, "dif_media_pp": -0.0, "ic95": (0.01, 0.08)}, t), 1),
    (lambda p, g, t: (p, g, {**t, "sabado": 0.11}), 2),
    (lambda p, g, t: (p, g, {}), 2),
])
def test_veredicto_falla_por_regla(cambio, regla):
    v = bmj.veredicto_reglas(*cambio(*_ok_inputs()))
    assert not v["pasa"]
    assert [r["pasa"] for r in v["reglas"]] == [i != regla for i in range(3)]


def test_veredicto_regla2_estricta():
    p, _, t = _ok_inputs()
    # IC que toca 0 por abajo y media 0 → pasa; media > 0 aunque el IC incluya 0 → no pasa
    assert bmj.veredicto_reglas(p, {"semanas": 33, "dif_media_pp": 0.0, "ic95": (0.0, 0.2)}, t)["reglas"][1]["pasa"]
    assert not bmj.veredicto_reglas(p, {"semanas": 33, "dif_media_pp": 0.001, "ic95": (-0.2, 0.2)},
                                    t)["reglas"][1]["pasa"]


def _df_festivos_anios():
    """Parte 2 sintética: festivos fijos en 2023 (cfg mejora 1 pp) y 2025 (cfg mejora 0.5 pp) + un festivo lunes."""
    filas = []
    for fecha, tipo_f, mejora in (("2023-05-01", "fijo", 1.0), ("2023-08-07", "semana_santa", 1.0),
                                  ("2025-05-01", "fijo", 0.5), ("2025-11-17", "lunes", -3.0),
                                  ("2021-12-08", "fijo", -9.0)):
        for serie, mape in (("jamer", 4.0), ("cfg", 4.0 - mejora)):
            filas.append({"parte": 2, "lunes": fecha, "momento": "martes", "fecha": fecha, "tipo_dia": "festivo",
                          "tipo_festivo": tipo_f, "serie": serie, "mape_h": mape, "r1": "x", "r1_jamer": "x"})
    return pd.DataFrame(filas)


def test_partes_regla1_festivos_por_corte_temporal():
    cfg = bmj.Config("cfg", {}, "festivo_tipo")
    partes = bmj.partes_regla1(cfg, _df_festivos_anios(), pd.DataFrame())
    assert [e for e, _ in partes] == ["festivos 2022-2024 (ampliada, calibración)", "festivos 2025-2026 (ampliada, prueba)"]
    (_, cal), (_, pru) = partes
    # 2021 queda fuera; el lunes no cuenta
    assert cal["dias"] == 2 and cal["dif_pp"] == pytest.approx(-1.0)
    assert pru["dias"] == 1 and pru["dif_pp"] == pytest.approx(-0.5)
    v = bmj.veredicto_reglas(partes, {"semanas": 33, "dif_media_pp": -0.01, "ic95": (-0.1, 0.1)}, {"laboral": 0.0})
    assert v["reglas"][0]["pasa"]


def test_partes_regla1_ambas_exige_las_dos_mitades():
    cfg = bmj.Config("cfg", {}, "festivo_tipo|r1_saltada")
    df = _df_festivos_anios()
    partes = bmj.partes_regla1(cfg, df, pd.DataFrame())
    assert len(partes) == 4 and partes[2][0] == "R1 distinta ampliada"
    # la mitad R1 no tiene días (ninguna R1 distinta) → la regla 1 no pasa aunque los festivos mejoren
    assert partes[2][1]["dias"] == 0 and partes[3][1]["dias"] == 0
    assert not bmj.veredicto_reglas(partes, {"semanas": 1, "dif_media_pp": 0.0, "ic95": (-1, 1)},
                                    {"laboral": 0.0})["reglas"][0]["pasa"]
    with pytest.raises(ValueError):
        bmj.partes_regla1(bmj.Config("cfg", {}, "otro"), df, pd.DataFrame())


def test_nota_festivos_prueba_calculada():
    filas = []
    for lunes, fecha, tipo_dia, tipo_f in (("2026-11-09", "2026-11-09", "laboral", None),
                                           ("2026-11-16", "2026-11-16", "festivo", "lunes")):
        filas.append({"lunes": lunes, "fecha": fecha, "tipo_dia": tipo_dia, "tipo_festivo": tipo_f, "serie": "jamer"})
    df = pd.DataFrame(filas)
    nota = bmj.nota_festivos_prueba(df)
    assert nota is not None and "Semanas de prueba de la parte 1 (2)" in nota and "1 día(s)" in nota
    con_fijo = pd.concat([df, pd.DataFrame([{"lunes": "2026-12-07", "fecha": "2026-12-08", "tipo_dia": "festivo",
                                             "tipo_festivo": "fijo", "serie": "jamer"}])])
    assert bmj.nota_festivos_prueba(con_fijo) is None
    assert bmj.nota_festivos_prueba(pd.DataFrame()) is None


# ---------------------------------------------------------------------------
# clasificación de días
# ---------------------------------------------------------------------------

def test_r1_cambia_con_limpia_usa_la_regla_de_produccion(reales, festivos):
    fset = set(festivos)
    # mié 23/abr/2025 con corte 16/abr: la R1 de siempre es el Miércoles Santo 16/abr, que r1_limpia salta
    assert bmj.r1_cambia_con_limpia(date(2025, 4, 23), reales, date(2025, 4, 16), fset)
    assert not bmj.r1_cambia_con_limpia(date(2025, 10, 15), reales, date(2025, 10, 9), fset)
    alterado = dict(reales)
    alterado["2025-10-08"] = reales["2025-10-08"] * 1.07  # atípico (> 6 %)
    assert bmj.r1_cambia_con_limpia(date(2025, 10, 15), alterado, date(2025, 10, 9), fset)


def test_filas_dia_r1_saltada_desde_la_corrida_real():
    reales = {"2025-10-15": np.full(24, 100.0)}
    series = {"jamer": {"2025-10-15": np.full(24, 101.0)}, "jamer+r1_limpia": {"2025-10-15": np.full(24, 99.0)}}
    r1s = {"jamer": {"2025-10-15": {"r1": "2025-10-08", "saltados": []}},
           "jamer+r1_limpia": {"2025-10-15": {"r1": "2025-10-01",
                                              "saltados": [{"fecha": "2025-10-08", "motivo": "atipico"}]}}}
    filas = bmj._filas_dia({"parte": 1}, "2025-10-15", series, r1s, reales, set())
    assert {f["serie"] for f in filas} == {"jamer", "jamer+r1_limpia"}
    assert all(f["r1_saltada"] is True and f["motivo_r1"] == "atipico" for f in filas)
    assert {f["serie"]: f["r1"] for f in filas} == {"jamer": "2025-10-08", "jamer+r1_limpia": "2025-10-01"}
    # sin la corrida de r1_limpia la columna informativa queda en None
    solo = bmj._filas_dia({"parte": 1}, "2025-10-15", {"jamer": series["jamer"]}, {"jamer": r1s["jamer"]},
                          reales, set())
    assert solo[0]["r1_saltada"] is None and solo[0]["motivo_r1"] is None


def test_semanas_ampliadas_sin_semanas_no_falla(reales, festivos):
    vacio = bmj.semanas_ampliadas(reales, festivos, date(2030, 1, 6), n_control=5, semilla=3)
    assert vacio.empty and list(vacio.columns) == bmj.COLUMNAS_SEMANAS_P2
    assert "sin datos" in "".join(bmj.md_parte2(pd.DataFrame(), vacio, bmj.CONFIGS, date(2030, 1, 6)))


def test_parte1_vf_solo_dias_posteriores_al_corte(hist, reales):
    # VF enviado el miércoles de la semana → corte martes: lunes y martes se descartan (y se cuentan)
    L = date(2025, 11, 17)
    env = pd.DataFrame(_envio(1, "2025-11-19 10:00", L))
    corr = bmj.Corredor(hist, data.festivos_colombia(range(2023, 2027)), (bmj.Config("jamer", {}, None),))
    descartes = {}
    df = bmj.correr_parte1(corr, env, reales, descartes_vf=descartes)
    vf = df[df["momento"] == "VF"]
    assert sorted(vf["fecha"].unique()) == [(L + timedelta(days=i)).isoformat() for i in range(2, 7)]
    assert (pd.to_datetime(vf["fecha"]) > pd.to_datetime(vf["corte"])).all()
    assert descartes == {L.isoformat(): 2}
    assert "2 día(s) VF con fecha <= corte descartados" in "".join(
        bmj.md_parte1(df, bmj.CONFIGS, 7, descartes_vf=descartes))


def test_semanas_ampliadas_banderas_y_control_con_semilla(reales, festivos):
    a = bmj.semanas_ampliadas(reales, festivos, date(2025, 1, 1), n_control=5, semilla=3)
    b = bmj.semanas_ampliadas(reales, festivos, date(2025, 1, 1), n_control=5, semilla=3)
    pd.testing.assert_frame_equal(a, b)
    idx = a.set_index("lunes")
    assert idx.loc[date(2025, 4, 14), "semana_santa"] and idx.loc[date(2025, 4, 14), "festivo"]
    assert idx.loc[date(2025, 8, 18), "festivo"]
    assert int(a["control"].sum()) == 5
    assert not (a["control"] & (a["festivo"] | a["semana_santa"] | a["r1_saltada"])).any()
    # la semana de Pascua (corte martes santo) toma R1 en Semana Santa → r1_saltada
    assert idx.loc[date(2025, 4, 21), "r1_saltada"]
    c = bmj.semanas_ampliadas(reales, festivos, date(2025, 1, 1), n_control=5, semilla=4)
    assert set(c[c["control"]]["lunes"]) != set(a[a["control"]]["lunes"])


# ---------------------------------------------------------------------------
# extremo a extremo (sintético, corto)
# ---------------------------------------------------------------------------

def test_help_no_falla_en_cp1252(capsys):
    with pytest.raises(SystemExit) as e:
        bmj.main(["--help"])
    assert e.value.code == 0
    out = capsys.readouterr().out
    assert "--muestra-informativa" in out and "2026-04-06" in out
    out.encode("cp1252")  # todo el texto de ayuda es representable en la consola de Windows
    assert bmj.__doc__.encode("cp1252")


def test_main_sintetico(tmp_path, hist):
    h = hist.copy()
    h["FECHA"] = h["fecha"].dt.strftime("%Y-%m-%d")
    h[["FECHA", *PER]].to_csv(tmp_path / "hist.csv", index=False)
    fest = data.festivos_colombia(range(2023, 2027))
    pd.DataFrame({"fecha": fest, "nombre": ""}).to_csv(tmp_path / "fest.csv", index=False)
    filas = []
    for k, L in enumerate([date(2025, 11, 3), date(2025, 11, 10), date(2025, 11, 17), date(2025, 11, 24)]):
        reales = data.curvas_reales(hist)
        for v, dias in ((1, 5), (2, 3)):
            env = datetime.combine(L - timedelta(days=dias), datetime.min.time()) + timedelta(hours=18)
            for i in range(7):
                f = L + timedelta(days=i)
                curva = reales[f.isoformat()] * (1.01 + 0.01 * v)
                filas.append({"sesion": 10 * k + v, "envio": env, "version": v, "fecha": f.isoformat(),
                              **{p: float(x) for p, x in zip(PER, curva)}})
    pd.DataFrame(filas).to_csv(tmp_path / "env.csv", index=False)
    rc = bmj.main(["--historico", str(tmp_path / "hist.csv"), "--festivos", str(tmp_path / "fest.csv"),
                   "--envios", str(tmp_path / "env.csv"), "--conjunto", "festivos_r1",
                   "--desde", "2025-11-17", "--muestra-informativa",
                   "--desde-informativa", "2025-11-03", "--desde-ampliada", "2025-10-01", "--n-control", "1",
                   "--fecha", "20990101", "--salida", str(tmp_path), "--salida-detalle", str(tmp_path / "det")])
    assert rc == 0
    md = (tmp_path / "backtest_mesa_vs_jamer_20990101.md").read_text(encoding="utf-8")
    assert "Criterio de aceptación" in md and "Parte 1" in md and "Parte 2" in md
    assert "Enmienda 2026-11-27" in md and "solo contienen festivos lunes" in md
    assert "festivos 2022-2024 (ampliada, calibración)" in md
    # la muestra del PLAN (desde 17/nov: 2 semanas) decide; la informativa (4 semanas) se rotula aparte
    plan, info = md.split("Parte 1 INFORMATIVO — no decide")
    assert "Semanas: 2 (2025-11-17 .. 2025-11-24)" in plan
    assert "Semanas: 4 (2025-11-03 .. 2025-11-24)" in info
    det = pd.read_csv(tmp_path / "det" / "backtest_mesa_vs_jamer_20990101_detalle.csv")
    assert set(det["parte"]) == {1, 2}
    assert set(det[det["parte"] == 1]["muestra_plan"]) == {True, False}
    p1 = det[det["parte"] == 1]
    assert (pd.to_datetime(p1["corte"]) < pd.to_datetime(p1["fecha"])).all()
    assert {"jamer_envio", "jamer"} <= set(p1["serie"])
    # jamer_envio en VF = real * 1.02 → error de energía del día ≈ +2 %
    assert p1[(p1["serie"] == "jamer_envio") & (p1["momento"] == "VF")]["err_dia"].mean() == pytest.approx(2.0, abs=1e-6)


# ---------------------------------------------------------------------------
# factor post-festivo: conjuntos, segmento, elección de variante
# ---------------------------------------------------------------------------

def test_conjuntos_de_configs():
    assert bmj.CONFIGS is bmj.CONJUNTOS["post_festivo"]
    pf = {c.nombre: c for c in bmj.CONJUNTOS["post_festivo"]}
    assert set(pf) == {"jamer", "jamer+pf_homologo", "jamer+pf_recientes", "jamer+pf_combinado"}
    assert pf["jamer"].extra == {}  # base = perfil jamer actual (trae festivo_por_tipo)
    for fuente in ("homologo", "recientes", "combinado"):
        c = pf[f"jamer+pf_{fuente}"]
        assert c.segmento == "post_festivo" and c.grupo == "post_festivo"
        esperado = {"post_festivo_factor": True, "post_festivo_fuente": fuente}
        assert c.extra == {"nivel": esperado, "forma": esperado}
    # festivos_r1 reproduce esa fase: la base y todas sus configs sin festivo_por_tipo salvo las que lo prueban
    fr = {c.nombre: c for c in bmj.CONJUNTOS["festivos_r1"]}
    assert fr["jamer"].extra["nivel"] == {"festivo_por_tipo": False} == fr["jamer"].extra["forma"]
    assert fr["jamer+r1_limpia"].extra["nivel"] == {"festivo_por_tipo": False, "r1_limpia": True}
    assert fr["jamer+festivo_por_tipo"].extra["forma"]["festivo_por_tipo"] is True


def _df_post_festivo():
    """Parte 2 sintética con post-festivos en 2023 y 2025 para jamer y las 3 variantes (+ un día normal)."""
    mape = {  # serie -> (2023-11-07, 2024-11-05, 2025-11-04, normal 2023-11-14)
        "jamer": (5.0, 5.0, 5.0, 2.0),
        "jamer+pf_homologo": (4.0, 4.4, 4.5, 9.0),
        "jamer+pf_recientes": (4.5, 4.5, 3.0, 9.0),
        "jamer+pf_combinado": (4.2, 4.2, 5.5, 9.0),
    }
    filas = []
    for serie, vals in mape.items():
        for (fecha, pf), v in zip((("2023-11-07", True), ("2024-11-05", True), ("2025-11-04", True),
                                   ("2023-11-14", False)), vals):
            filas.append({"parte": 2, "lunes": fecha, "momento": "martes", "fecha": fecha, "tipo_dia": "laboral",
                          "tipo_festivo": None, "post_festivo": pf, "serie": serie, "mape_h": v,
                          "r1": "x", "r1_jamer": "x"})
    return pd.DataFrame(filas)


def test_segmento_post_festivo_y_partes_regla1():
    df = _df_post_festivo()
    seg = bmj.comparar_segmento(df, "post_festivo", "jamer+pf_homologo")
    assert seg["dias"] == 3 and seg["dif_pp"] == pytest.approx((4.0 + 4.4 + 4.5) / 3 - 5.0)
    cfg = next(c for c in bmj.CONFIGS if c.nombre == "jamer+pf_homologo")
    partes = bmj.partes_regla1(cfg, df, pd.DataFrame())
    assert [e for e, _ in partes] == ["post-festivos 2022-2024 (ampliada, calibración)",
                                      "post-festivos 2025-2026 (ampliada, prueba)"]
    (_, cal), (_, pru) = partes
    assert cal["dias"] == 2 and cal["dif_pp"] == pytest.approx(-0.8)  # el día normal no cuenta
    assert pru["dias"] == 1 and pru["dif_pp"] == pytest.approx(-0.5)
    info = bmj.info_regla1(cfg, pd.DataFrame())
    assert info[0][0] == "post-festivos prueba parte 1 (VF, informativo)" and info[0][1]["dias"] == 0


def test_elegir_variante_solo_con_calibracion():
    df = _df_post_festivo()
    variantes = [c for c in bmj.CONFIGS if c.grupo == "post_festivo"]
    elegida, medias = bmj.elegir_variante(df, variantes)
    # calibración 2023-2024: homologo 4.2, recientes 4.5, combinado 4.2 → empate, gana la primera (homologo);
    # recientes es la mejor en 2025 (prueba) pero la prueba no elige
    assert elegida == "jamer+pf_homologo"
    assert medias["jamer+pf_recientes"] == {"dias": 2, "mape_h": pytest.approx(4.5)}
    assert bmj.elegir_variante(df[pd.to_datetime(df["fecha"]).dt.year == 2025], variantes)[0] is None
    assert bmj.elegir_variante(pd.DataFrame(), variantes)[0] is None


def test_evaluar_criterios_solo_la_elegida_tiene_veredicto():
    p2 = _df_post_festivo()
    crit = bmj.evaluar_criterios(pd.DataFrame(columns=["serie", "momento", "tramo"]), p2, bmj.CONFIGS, 7)
    assert set(crit) == {"jamer+pf_homologo", "jamer+pf_recientes", "jamer+pf_combinado"}
    assert crit["jamer+pf_homologo"]["informativa"] is False and crit["jamer+pf_homologo"]["pasa"] is False
    for n in ("jamer+pf_recientes", "jamer+pf_combinado"):
        assert crit[n]["informativa"] is True and crit[n]["pasa"] is None
        assert "informativa" in bmj.veredicto_txt(crit[n])
    assert {c["eleccion"]["elegida"] for c in crit.values()} == {"jamer+pf_homologo"}
    md = "".join(bmj.md_criterios(crit, {}))
    assert "Elección de variante" in md and "**sí**" in md and "informativo (no decide)" in md
    assert "Factor post-festivo" in md
    # sin datos de calibración: ninguna variante recibe veredicto
    sin_cal = bmj.evaluar_criterios(pd.DataFrame(columns=["serie", "momento", "tramo"]),
                                    p2[pd.to_datetime(p2["fecha"]).dt.year == 2025], bmj.CONFIGS, 7)
    assert all(c["informativa"] and c["pasa"] is None for c in sin_cal.values())
    assert "Sin datos de calibración" in "".join(bmj.md_criterios(sin_cal, {}))


def test_semanas_ampliadas_incluye_semanas_con_post_festivo(reales, festivos):
    a = bmj.semanas_ampliadas(reales, festivos, date(2025, 1, 1), n_control=5, semilla=3)
    idx = a.set_index("lunes")
    # dom 20/jul/2025 festivo → lun 21/jul post-festivo: semana sin festivos que entra solo por la bandera
    assert idx.loc[date(2025, 7, 21), "post_festivo"] and not idx.loc[date(2025, 7, 21), "festivo"]
    assert idx.loc[date(2025, 11, 3), "post_festivo"]
    assert not (a["control"] & a["post_festivo"]).any()


def test_main_sintetico_post_festivo(tmp_path, hist):
    h = hist.copy()
    h["FECHA"] = h["fecha"].dt.strftime("%Y-%m-%d")
    h[["FECHA", *PER]].to_csv(tmp_path / "hist.csv", index=False)
    pd.DataFrame({"fecha": data.festivos_colombia(range(2023, 2027)), "nombre": ""}).to_csv(
        tmp_path / "fest.csv", index=False)
    rc = bmj.main(["--historico", str(tmp_path / "hist.csv"), "--festivos", str(tmp_path / "fest.csv"),
                   "--partes", "2", "--desde-ampliada", "2025-10-27", "--n-control", "0", "--fecha", "20990102",
                   "--salida", str(tmp_path), "--salida-detalle", str(tmp_path / "det")])
    assert rc == 0
    md = (tmp_path / "backtest_mesa_vs_jamer_20990102.md").read_text(encoding="utf-8")
    assert "--conjunto post_festivo" in md and "Factor post-festivo" in md
    assert "`jamer+pf_combinado`" in md and "post-festivos" in md
    assert "solo contienen festivos lunes" not in md
    det = pd.read_csv(tmp_path / "det" / "backtest_mesa_vs_jamer_20990102_detalle.csv")
    assert {"jamer", "jamer+pf_homologo", "jamer+pf_recientes", "jamer+pf_combinado"} == set(det["serie"])
    assert det.loc[det["fecha"] == "2025-11-04", "post_festivo"].all()
    assert det["sem_post_festivo"].any()
