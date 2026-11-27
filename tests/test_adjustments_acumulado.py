import pytest

from src.adjustments.acumulado import (
    calcular_acumulado_anual,
    error_pct_dia,
)


def _serie(n_fuera, n_total=10, anio=2026):
    real = {f"{anio}-01-{d:02d}": 100.0 for d in range(1, n_total + 1)}
    apr = {f: (110.0 if i < n_fuera else 101.0) for i, f in enumerate(real)}
    return apr, real


def test_apagado_por_defecto():
    apr, real = _serie(5)
    r = calcular_acumulado_anual(apr, real, 2026)
    assert r["estado"] == "apagado" and r["alerta"] is None and r["dias_fuera"] == 0


def test_banda_sin_cupo_cuenta_pero_no_alerta():
    apr, real = _serie(4)
    r = calcular_acumulado_anual(apr, real, 2026, banda_pct=5.0)
    assert r["estado"] == "sin_cupo" and r["dias_fuera"] == 4 and r["dias_evaluados"] == 10
    assert r["alerta"] is None and r["restantes"] is None


@pytest.mark.parametrize("fuera,estado,alerta", [(7, "dentro", False), (8, "cerca", True),
                                                 (10, "cerca", True), (11, "excedido", True)])
def test_estados_con_cupo(fuera, estado, alerta):
    apr, real = _serie(fuera, n_total=12)
    r = calcular_acumulado_anual(apr, real, 2026, banda_pct=5.0, cupo_dias=10)
    assert r["estado"] == estado
    assert (r["alerta"] is not None) is alerta
    if alerta:
        assert r["alerta"]["nivel"] == "info" and r["alerta"]["codigo"] == "ACUMULADO_ANUAL_FUERA_RANGO"
        assert len(r["alerta"]["fechas"]) == fuera


def test_frontera_de_banda_es_estricta():
    r = calcular_acumulado_anual({"2026-01-01": 105.0}, {"2026-01-01": 100.0}, 2026, banda_pct=5.0)
    assert r["dias_fuera"] == 0
    r = calcular_acumulado_anual({"2026-01-01": 105.01}, {"2026-01-01": 100.0}, 2026, banda_pct=5.0)
    assert r["dias_fuera"] == 1


def test_no_evaluables_y_otro_anio_y_hasta():
    apr = {"2026-01-01": 50.0, "2026-01-02": 50.0, "2025-12-31": 50.0, "2026-03-01": 50.0, "2026-01-03": 50.0}
    real = {"2026-01-01": 100.0, "2026-01-02": 0.0, "2025-12-31": 100.0, "2026-03-01": 100.0}
    r = calcular_acumulado_anual(apr, real, 2026, banda_pct=5.0, hasta="2026-02-01")
    assert (r["dias_fuera"], r["dias_evaluados"], r["dias_no_evaluables"]) == (1, 1, 1)


def test_cupo_cero_y_validaciones():
    apr, real = _serie(1)
    assert calcular_acumulado_anual(apr, real, 2026, banda_pct=5.0, cupo_dias=0)["estado"] == "excedido"
    assert calcular_acumulado_anual(*_serie(0), 2026, banda_pct=5.0, cupo_dias=0)["estado"] == "dentro"
    for kw in ({"banda_pct": -1.0}, {"banda_pct": 5.0, "cupo_dias": -1}, {"banda_pct": 5.0, "cupo_dias": 2.5},
               {"banda_pct": 5.0, "fraccion_aviso": 0}, {"banda_pct": 5.0, "hasta": "no-fecha"}):
        with pytest.raises(ValueError):
            calcular_acumulado_anual(apr, real, 2026, **kw)


def test_helpers():
    assert error_pct_dia(95.0, 100.0) == pytest.approx(5.0)
    assert error_pct_dia(1.0, 0.0) is None
