"""Tests del backtest de días análogos (scripts/backtest_analogos.py) con datos sintéticos."""

from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import backtest_analogos as ba  # noqa: E402
from backtest_adjustments import proveedor_ingenuo  # noqa: E402
from src.adjustments.data import festivos_colombia  # noqa: E402
from tests.ajustes_sinteticos import historico_sintetico  # noqa: E402

FESTIVOS = festivos_colombia(range(2024, 2027))


def test_particionar_mitad_antigua_calibracion():
    lunes = [date(2025, 1, 6) + timedelta(days=7 * i) for i in range(6)]
    p = ba.particionar(list(reversed(lunes)))
    assert [p[l] for l in lunes] == ["calibracion"] * 3 + ["holdout"] * 3
    p5 = ba.particionar(lunes[:5])
    assert sum(v == "calibracion" for v in p5.values()) == 2
    assert p5[lunes[4]] == "holdout"


@pytest.mark.parametrize("dia, esperado", [
    (date(2025, 6, 2), {"tipo_dia": "festivo", "relacion_festivo": "festivo", "normal": False}),
    (date(2025, 6, 3), {"tipo_dia": "laboral", "relacion_festivo": "despues", "distancia_festivo": 1}),
    (date(2025, 12, 25), {"relacion_festivo": "festivo", "diciembre": True, "normal": False}),
    (date(2025, 4, 17), {"relacion_festivo": "festivo", "semana_santa": True}),
    (date(2025, 3, 5), {"tipo_dia": "laboral", "relacion_festivo": "ninguna", "semana_santa": False,
                        "diciembre": False, "normal": True}),
])
def test_etiquetas_dia(dia, esperado):
    et = ba.etiquetas_dia(dia, FESTIVOS)
    for k, v in esperado.items():
        assert et[k] == v, (k, et)


def test_mascaras_segmentos():
    dias = [date(2025, 6, 2), date(2025, 6, 3), date(2025, 12, 25), date(2025, 4, 17), date(2025, 3, 5)]
    df = pd.DataFrame([ba.etiquetas_dia(d, FESTIVOS) for d in dias])
    m = ba.mascaras_segmentos(df)
    assert m["todos"].sum() == 5
    assert m["despues_dist1"].tolist() == [False, True, False, False, False]
    assert m["rel:festivo"].sum() == 3
    assert m["semana_santa"].sum() == 1 and m["diciembre"].sum() == 1 and m["normal"].sum() == 1


def test_configs_barrido():
    cfgs = ba.configs_barrido([3, 5], [0, 45], [0.25, 0.5])
    nombres = [c.nombre for c in cfgs]
    assert nombres[0] == "default" and len(set(nombres)) == len(nombres)
    assert "forma_analogos_n5_t45" in nombres and "nivel_analogos_a025" in nombres
    assert "nivel_actual_a025" not in nombres  # es el default
    t45 = next(c for c in cfgs if c.nombre == "forma_analogos_n3_t45")
    assert t45.params["forma"]["filtros_analogos"]["temporada_dias"] == 45
    assert next(c for c in cfgs if c.nombre == "nivel_analogos_a050").pasos_afectados() == ("nivel", "forma")
    assert t45.pasos_afectados() == ("forma",)


def _comp_falso(mejora_hold_atl: float) -> pd.DataFrame:
    filas = []
    for ucp in ("Antioquia", "Atlantico"):
        for part in ba.PARTICIONES:
            m = mejora_hold_atl if (ucp, part) == ("Atlantico", "holdout") else 5.0
            filas.append({"segmento": "todos", "base": "ingenuo", "ucp": ucp, "particion": part,
                          "config": "forma_analogos_n3", "paso": "forma", "n_dias": 364,
                          "mejora_rel_pct": m, "delta_pico_pp": -0.1})
    return pd.DataFrame(filas)


def test_evaluar_regla():
    cfgs = ba.configs_barrido([3], [0], [0.25])
    ok = ba.evaluar_regla(_comp_falso(2.5), ["Antioquia", "Atlantico"], 2.0, cfgs)
    assert bool(ok["cumple_regla"].iloc[0]) and ok["celdas"].iloc[0] == "4/4"
    no = ba.evaluar_regla(_comp_falso(1.0), ["Antioquia", "Atlantico"], 2.0, cfgs)
    assert not bool(no["cumple_regla"].iloc[0]) and no["celdas_que_ganan"].iloc[0] == 3


def test_corrida_dos_semanas_sintetica(tmp_path):
    hist = historico_sintetico(date(2023, 1, 1), date(2025, 6, 29))
    semanas = [date(2025, 6, 16), date(2025, 6, 23)]
    parts = ba.particionar(semanas)
    cfgs = [c for c in ba.configs_barrido([3], [0], [0.25]) if c.nombre in
            ("default", "forma_analogos_n3", "nivel_analogos_a025")]
    df = ba.correr("Antioquia", hist, semanas, parts, lambda ucp, h: proveedor_ingenuo(h), "ingenuo", cfgs)
    # default: base+nivel+forma; variantes: nivel+forma; 7 días x 2 semanas
    assert len(df) == 2 * 7 * (3 + 2 + 2)
    assert set(df["particion"]) == {"calibracion", "holdout"}
    assert (df["fecha_corte"] < df["semana_inicio"]).all()
    assert {"relacion_festivo", "semana_santa", "diciembre", "normal", "abs_err_horario"} <= set(df.columns)
    forma_an = df[(df["config"] == "forma_analogos_n3") & (df["paso"] == "forma")]
    assert (forma_an["forma_fuente_usada"] == "analogos").sum() >= 10  # festivos: perfil homólogo

    comp = ba.tabla_comparacion(df)
    d = comp[comp["config"] == "default"]
    assert (d["mejora_rel_pct"].abs() < 1e-9).all()
    regla = ba.evaluar_regla(comp, ["Antioquia"], 2.0, cfgs)
    assert set(regla["config"]) == {"forma_analogos_n3", "nivel_analogos_a025"}

    md = tmp_path / "r.md"
    segs = list(ba.mascaras_segmentos(df.head(1)).keys())
    ba.escribir_reporte(df, comp, regla, cfgs, md, {
        "fecha": "test", "csv_nombre": "x.csv", "bases": ["ingenuo"], "semanas": 2, "umbral": 2.0,
        "tiempo_s": 1.0, "segmentos": segs,
        "eleccion": ba.texto_eleccion(comp, {"forma": None, "nivel": None, "combo": None}, ["Antioquia"]),
        "dificiles": ba.texto_dificiles(comp, cfgs, ["Antioquia"]),
        "conclusion": ba.texto_conclusion(regla, 2.0),
    })
    texto = md.read_text(encoding="utf-8")
    assert "## 6. Conclusión" in texto and "forma_analogos_n3" in texto

    # regenerar el .md desde el CSV (sin correr el pipeline)
    csv = tmp_path / "backtest_analogos_t.csv"
    df.to_csv(csv, index=False)
    assert ba.main(["--desde-csv", str(csv), "--out-dir", str(tmp_path), "--fecha", "t", "--ucp", "Antioquia",
                    "--semanas", "2"]) == 0
    assert "## 5. Regla" in (tmp_path / "backtest_analogos_t.md").read_text(encoding="utf-8")


def test_lectura_cumplidas_alpha_vs_ancla():
    ok = pd.DataFrame([
        {"segmento": "tipo:festivo", "config": "nivel_analogos_a050", "mejora_media_pct": 18.0},
        {"segmento": "tipo:festivo", "config": "nivel_actual_a050", "mejora_media_pct": 17.0},
    ])
    texto = "\n".join(ba._lectura_cumplidas(ok))
    assert "+1.0 pp" in texto and "no son interpretables" in texto
