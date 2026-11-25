"""Clasificación de lluvia desde p_i (códigos de condición, no mm) y su uso en análogos / alerta F06."""
import math
from datetime import date, timedelta

import pandas as pd
import pytest

from src.adjustments import analogos, clima


@pytest.mark.parametrize("codigo,esperado", [
    (500, True), (501, True), (200, True), (301, True), (599, True),
    (800, False), (804, False), (803, False), (701, False), (600, False),
    (0, False), (float("nan"), False), (None, False), ("x", False),
    (1, False), (12, True), (33, False), (39, True), (18, True), (45, False),
])
def test_periodo_con_lluvia(codigo, esperado):
    assert clima.periodo_con_lluvia(codigo) is esperado


def test_periodo_conocido_cero_y_nan_son_faltantes():
    assert not clima.periodo_conocido(0) and not clima.periodo_conocido(float("nan"))
    assert clima.periodo_conocido(804) and clima.periodo_conocido(1)


def test_resumen_dia_con_lluvia():
    r = clima.resumen_lluvia_dia([500] * 3 + [804] * 21)
    assert r == {"periodos_lluvia": 3, "periodos_conocidos": 24, "frac_lluvia": 3 / 24,
                 "con_lluvia": True, "calidad": "suficiente"}


def test_resumen_dia_bajo_umbral_periodos():
    r = clima.resumen_lluvia_dia([500] * 2 + [803] * 22)
    assert r["con_lluvia"] is False and r["periodos_lluvia"] == 2


def test_resumen_dia_pocos_conocidos_es_desconocido():
    r = clima.resumen_lluvia_dia([500] * 5 + [0] * 13 + [float("nan")] * 6)
    assert r["periodos_conocidos"] == 5 and r["calidad"] == "insuficiente"
    assert r["con_lluvia"] is None and r["frac_lluvia"] == 1.0
    assert clima.resumen_lluvia_dia([0] * 24) == {"periodos_lluvia": 0, "periodos_conocidos": 0,
                                                  "frac_lluvia": None, "con_lluvia": None, "calidad": "sin_dato"}


def test_resumen_dia_12_conocidos_basta():
    assert clima.resumen_lluvia_dia([804] * 12 + [0] * 12)["calidad"] == "suficiente"


def test_resumen_legado():
    assert clima.resumen_lluvia_dia([12] * 4 + [1] * 20)["con_lluvia"] is True
    assert clima.resumen_lluvia_dia([33] * 24)["con_lluvia"] is False


def _filas(fecha, codigos):
    return [{"fecha": fecha, "periodo": p + 1, "p_t": 22.0, "p_h": 80.0, "p_v": 1.0, "p_i": c}
            for p, c in enumerate(codigos)]


def test_clima_diario_semana_noviembre_like_sin_mm_absurdos():
    filas = []
    for k, cods in enumerate([[804] * 24, [803] * 12 + [500] * 12, [500] * 24, [804] * 20 + [0] * 4]):
        filas += _filas(f"2026-11-{23 + k}", cods)
    c = analogos.clima_diario(pd.DataFrame(filas))
    assert all("lluvia_mm" not in v for v in c.values())
    assert c["2026-11-23"]["con_lluvia"] is False
    assert c["2026-11-24"]["lluvia_periodos"] == 12 and c["2026-11-24"]["frac_lluvia"] == 0.5
    assert c["2026-11-25"]["frac_lluvia"] == 1.0
    assert c["2026-11-26"]["periodos_conocidos"] == 20


def test_clima_diario_dia_sin_datos_de_lluvia_conserva_temperatura():
    c = analogos.clima_diario(pd.DataFrame(_filas("2026-09-05", [0] * 24)))
    d = c["2026-09-05"]
    assert d["temp_media"] == pytest.approx(22.0) and d["con_lluvia"] is None and d["calidad_lluvia"] == "sin_dato"


def test_analogos_con_lluvia_no_es_verdadero_para_todo_dia():
    def dia(c):
        return {"temp_media": 22.0, "con_lluvia": c, "frac_lluvia": None if c is None else float(c)}
    base = analogos.descriptor_dia("2025-06-10", [], {"2025-06-10": dia(True)})
    sec = analogos.descriptor_dia("2025-06-09", [], {"2025-06-09": dia(False)})
    desc = analogos.descriptor_dia("2025-06-08", [], {"2025-06-08": dia(True)})
    ign = analogos.descriptor_dia("2025-06-07", [], {"2025-06-07": dia(None)})
    assert "con lluvia" in analogos.puntuar(base, desc)[2]
    assert "con lluvia" not in analogos.puntuar(base, sec)[2]
    assert "con lluvia" not in analogos.puntuar(base, ign)[2]


def test_no_quedan_rastros_de_la_regla_vieja():
    assert not hasattr(analogos, "LLUVIA_UMBRAL_MM")
