"""Tests de la configuración R2 (`automatico.py`): valores del backtest, API e import aislado."""

from __future__ import annotations

import copy
import importlib
import subprocess
import sys
import types
from pathlib import Path
from typing import Dict, List, Tuple

import pytest

from src.adjustments import automatico
from src.adjustments.steps import base, control, especiales, forma, nivel, referencia

_MODULOS_PASOS = (base, referencia, nivel, forma, especiales, control)
_RAIZ = Path(__file__).resolve().parents[1]


def _esperado(cambios: List[Tuple[str, str]]) -> Dict[str, Dict[str, str]]:
    """Los 4 tipos × 2 pasos en `sugerido` salvo los pares marcados automáticos."""
    config = {tipo: {paso: "sugerido" for paso in automatico.PASOS} for tipo in automatico.TIPOS}
    for tipo, paso in cambios:
        config[tipo][paso] = "automatico"
    return config


def _esperado_segmentos(cambios: List[Tuple[str, str, str]]) -> Dict[str, Dict[str, str]]:
    """Los 4 tipos × 2 pasos en `sugerido` salvo los pares (tipo, paso, recomendación)."""
    config = {tipo: {paso: "sugerido" for paso in automatico.PASOS} for tipo in automatico.TIPOS}
    for tipo, paso, rec in cambios:
        config[tipo][paso] = rec
    return config


ANTIOQUIA = [("laboral", "forma")]
ATLANTICO = [("laboral", "forma"), ("domingo", "forma"), ("sabado", "nivel")]
ANTIOQUIA_SEG = [("laboral", "forma", "automatico"), ("domingo", "nivel", "no_aplicar")]
ATLANTICO_SEG = [
    ("laboral", "nivel", "no_aplicar"),
    ("laboral", "forma", "automatico"),
    ("sabado", "nivel", "automatico"),
    ("domingo", "forma", "automatico"),
]


def _recomendaciones(cambios: List[Tuple[str, str, str]]) -> List[Dict[str, str]]:
    """Lista de recomendaciones explícitas en el orden TIPOS × PASOS."""
    return [
        {"tipo_dia": tipo, "paso": paso, "recomendacion": rec}
        for tipo, paso, rec in cambios
    ]


# ---------------------------------------------------------------- constantes

def test_tipos_pasos_y_estados():
    assert automatico.TIPOS == ("laboral", "sabado", "domingo", "festivo")
    assert automatico.PASOS == ("nivel", "forma")
    assert automatico.ESTADOS == ("automatico", "sugerido")
    assert automatico.RECOMENDACIONES == ("automatico", "sugerido", "no_aplicar")


def test_config_r2_coincide_con_el_backtest():
    assert automatico.CONFIG_R2 == {
        "Antioquia": {"laboral": {"forma": "automatico"}},
        "Atlantico": {
            "laboral": {"forma": "automatico"},
            "domingo": {"forma": "automatico"},
            "sabado": {"nivel": "automatico"},
        },
    }
    assert automatico.CONFIG_NO_APLICAR == {
        "Antioquia": {"domingo": ("nivel",)},
        "Atlantico": {"laboral": ("nivel",)},
    }


def test_metadatos_de_fuente():
    assert "backtest_modelo_20261113.md" in automatico.FUENTE
    assert "8." in automatico.SECCION
    assert "calibración" in automatico.CRITERIO and "prueba" in automatico.CRITERIO


# ---------------------------------------------------------------- configuracion

@pytest.mark.parametrize(
    "ucp, cambios, seg_cambios",
    [
        ("Antioquia", ANTIOQUIA, ANTIOQUIA_SEG),
        ("Atlantico", ATLANTICO, ATLANTICO_SEG),
        ("Desconocida", [], []),
        ("atlantico", [], []),  # la UCP se compara exacta
        (None, [], []),
    ],
)
def test_configuracion_completa_por_tipo(ucp, cambios, seg_cambios):
    cfg = automatico.configuracion(ucp)
    assert cfg["ucp"] == ucp
    assert cfg["fuente"] == automatico.FUENTE
    assert cfg["seccion"] == automatico.SECCION
    assert cfg["criterio"] == automatico.CRITERIO
    assert set(cfg["tipos"]) == set(automatico.TIPOS)
    for tipo in automatico.TIPOS:
        assert list(cfg["tipos"][tipo]) == list(automatico.PASOS)
        assert set(cfg["tipos"][tipo].values()) <= set(automatico.ESTADOS)
    assert cfg["tipos"] == _esperado(cambios)
    assert cfg["segmentos"] == _esperado_segmentos(seg_cambios)
    assert cfg["recomendaciones"] == _recomendaciones(seg_cambios)


def test_configuracion_devuelve_estructuras_nuevas():
    primera = automatico.configuracion("Antioquia")
    primera["tipos"]["laboral"]["forma"] = "tocado"
    primera["segmentos"]["laboral"]["forma"] = "tocado"
    primera["recomendaciones"][0]["recomendacion"] = "tocado"
    segunda = automatico.configuracion("Antioquia")
    assert segunda["tipos"]["laboral"]["forma"] == "automatico"
    assert segunda["segmentos"]["laboral"]["forma"] == "automatico"
    assert segunda["recomendaciones"][0]["recomendacion"] == "automatico"
    assert automatico.CONFIG_R2["Antioquia"]["laboral"]["forma"] == "automatico"


# ---------------------------------------------------------------- estado

@pytest.mark.parametrize(
    "ucp, tipo_dia, paso, esperado",
    [
        ("Antioquia", "laboral", "forma", "automatico"),
        ("Antioquia", "laboral", "nivel", "sugerido"),
        ("Antioquia", "sabado", "nivel", "sugerido"),
        ("Antioquia", "sabado", "forma", "sugerido"),
        ("Antioquia", "domingo", "nivel", "sugerido"),
        ("Antioquia", "domingo", "forma", "sugerido"),
        ("Antioquia", "festivo", "nivel", "sugerido"),
        ("Antioquia", "festivo", "forma", "sugerido"),
        ("Atlantico", "laboral", "forma", "automatico"),
        ("Atlantico", "laboral", "nivel", "sugerido"),
        ("Atlantico", "sabado", "nivel", "automatico"),
        ("Atlantico", "sabado", "forma", "sugerido"),
        ("Atlantico", "domingo", "forma", "automatico"),
        ("Atlantico", "domingo", "nivel", "sugerido"),
        ("Atlantico", "festivo", "nivel", "sugerido"),
        ("Atlantico", "festivo", "forma", "sugerido"),
        ("Desconocida", "laboral", "forma", "sugerido"),
        ("", "sabado", "nivel", "sugerido"),
        (None, "domingo", "forma", "sugerido"),
        ("Antioquia ", "laboral", "forma", "sugerido"),  # sin normalización de espacios
    ],
)
def test_estado_por_ucp_tipo_paso(ucp, tipo_dia, paso, esperado):
    assert automatico.estado(ucp, tipo_dia, paso) == esperado


@pytest.mark.parametrize("tipo_dia", ["lunes", "", "Laboral", None, 5])
def test_estado_tipo_invalido(tipo_dia):
    with pytest.raises(ValueError, match="tipo_dia inválido"):
        automatico.estado("Antioquia", tipo_dia, "forma")


@pytest.mark.parametrize("paso", ["especiales", "base", "", "Forma", None, 1])
def test_estado_paso_invalido(paso):
    with pytest.raises(ValueError, match="paso inválido"):
        automatico.estado("Antioquia", "laboral", paso)


def test_estado_valida_claves_antes_de_la_ucp():
    with pytest.raises(ValueError, match="paso inválido"):
        automatico.estado("Desconocida", "laboral", "control")
    with pytest.raises(ValueError, match="tipo_dia inválido"):
        automatico.estado("Desconocida", "martes", "forma")


# ---------------------------------------------------------------- recomendacion

@pytest.mark.parametrize(
    "ucp, tipo_dia, paso, esperado",
    [
        # 4 segmentos respaldados por el backtest
        ("Antioquia", "laboral", "forma", "automatico"),
        ("Atlantico", "laboral", "forma", "automatico"),
        ("Atlantico", "domingo", "forma", "automatico"),
        ("Atlantico", "sabado", "nivel", "automatico"),
        # "no aplicar" del §8
        ("Antioquia", "domingo", "nivel", "no_aplicar"),
        ("Atlantico", "laboral", "nivel", "no_aplicar"),
        # resto: evidencia insuficiente o mixta
        ("Antioquia", "laboral", "nivel", "sugerido"),
        ("Antioquia", "sabado", "forma", "sugerido"),
        ("Antioquia", "festivo", "nivel", "sugerido"),
        ("Atlantico", "sabado", "forma", "sugerido"),
        ("Atlantico", "domingo", "nivel", "sugerido"),
        ("Atlantico", "festivo", "forma", "sugerido"),
        # UCP desconocida / no-str
        ("Desconocida", "domingo", "nivel", "sugerido"),
        ("atlantico", "laboral", "nivel", "sugerido"),
        ("", "laboral", "forma", "sugerido"),
        (None, "domingo", "forma", "sugerido"),
    ],
)
def test_recomendacion_por_segmento(ucp, tipo_dia, paso, esperado):
    assert automatico.recomendacion(ucp, tipo_dia, paso) == esperado


@pytest.mark.parametrize("tipo_dia", ["lunes", "", "Laboral", None, 5])
def test_recomendacion_tipo_invalido(tipo_dia):
    with pytest.raises(ValueError, match="tipo_dia inválido"):
        automatico.recomendacion("Antioquia", tipo_dia, "forma")


@pytest.mark.parametrize("paso", ["especiales", "base", "", "Forma", None, 1])
def test_recomendacion_paso_invalido(paso):
    with pytest.raises(ValueError, match="paso inválido"):
        automatico.recomendacion("Antioquia", "laboral", paso)


# ---------------------------------------------------------------- segmentos

@pytest.mark.parametrize(
    "ucp, seg_cambios",
    [("Antioquia", ANTIOQUIA_SEG), ("Atlantico", ATLANTICO_SEG),
     ("Desconocida", []), (None, [])],
)
def test_segmentos_completos(ucp, seg_cambios):
    seg = automatico.segmentos(ucp)
    assert list(seg) == list(automatico.TIPOS)
    for tipo in automatico.TIPOS:
        assert list(seg[tipo]) == list(automatico.PASOS)
        assert set(seg[tipo].values()) <= set(automatico.RECOMENDACIONES)
    assert seg == _esperado_segmentos(seg_cambios)


def test_segmentos_devuelve_estructura_nueva():
    primera = automatico.segmentos("Antioquia")
    primera["domingo"]["nivel"] = "tocado"
    segunda = automatico.segmentos("Antioquia")
    assert segunda["domingo"]["nivel"] == "no_aplicar"
    assert automatico.CONFIG_NO_APLICAR["Antioquia"]["domingo"] == ("nivel",)


# ---------------------------------------------------------------- pasos_automaticos

@pytest.mark.parametrize(
    "ucp, tipo_dia, esperado",
    [
        ("Antioquia", "laboral", ["forma"]),
        ("Antioquia", "sabado", []),
        ("Antioquia", "domingo", []),
        ("Antioquia", "festivo", []),
        ("Atlantico", "laboral", ["forma"]),
        ("Atlantico", "sabado", ["nivel"]),
        ("Atlantico", "domingo", ["forma"]),
        ("Atlantico", "festivo", []),
        ("Desconocida", "laboral", []),
        (None, "sabado", []),
    ],
)
def test_pasos_automaticos(ucp, tipo_dia, esperado):
    assert automatico.pasos_automaticos(ucp, tipo_dia) == esperado


def test_pasos_automaticos_tipo_invalido():
    with pytest.raises(ValueError, match="tipo_dia inválido"):
        automatico.pasos_automaticos("Antioquia", "lunes")


# ---------------------------------------------------------------- modos_paso

@pytest.mark.parametrize(
    "ucp, esperado",
    [
        ("Antioquia", {"nivel": "sugerido", "forma": "sugerido"}),
        ("Atlantico", {"nivel": "sugerido", "forma": "sugerido"}),
        ("Desconocida", {"nivel": "sugerido", "forma": "sugerido"}),
        ("atlantico", {"nivel": "sugerido", "forma": "sugerido"}),
        (None, {"nivel": "sugerido", "forma": "sugerido"}),
    ],
)
def test_modos_paso(ucp, esperado):
    assert automatico.modos_paso(ucp) == esperado


def test_modos_paso_exige_los_cuatro_tipos(monkeypatch):
    # Atlántico/nivel solo tiene regla en sábado y Antioquia/forma solo en laboral:
    # el resto de tipos cuenta como sugerido, así que el paso no es automático.
    monkeypatch.setitem(automatico.CONFIG_R2, "Completa", {
        "laboral": {"nivel": "automatico", "forma": "automatico"},
        "sabado": {"nivel": "automatico", "forma": "automatico"},
        "domingo": {"nivel": "automatico", "forma": "automatico"},
        "festivo": {"nivel": "automatico", "forma": "automatico"},
    })
    assert automatico.modos_paso("Completa") == {"nivel": "automatico", "forma": "automatico"}


def test_modos_paso_solo_pasos_r2_y_estructura_nueva():
    modos = automatico.modos_paso("Atlantico")
    assert list(modos) == list(automatico.PASOS)
    assert set(modos.values()) <= set(automatico.ESTADOS)
    modos["forma"] = "tocado"
    assert automatico.modos_paso("Atlantico")["forma"] == "sugerido"


# ---------------------------------------------------------------- import aislado

def test_import_no_muta_defaults_ni_depende_de_otros_modulos():
    antes = {m.__name__: copy.deepcopy(m.DEFAULTS) for m in _MODULOS_PASOS}
    importlib.reload(automatico)
    despues = {m.__name__: copy.deepcopy(m.DEFAULTS) for m in _MODULOS_PASOS}

    assert despues == antes
    assert not hasattr(automatico, "router")
    importados = sorted(
        v.__name__
        for v in vars(automatico).values()
        if isinstance(v, types.ModuleType) and v.__name__.startswith("src.")
    )
    assert importados == []


def test_import_aislado_no_carga_router_ni_pipeline():
    script = (
        "from src.adjustments.steps import base, referencia, nivel, forma, especiales, control\n"
        "import sys\n"
        "import src.adjustments.automatico as a\n"
        "assert 'src.adjustments.router' not in sys.modules, 'automatico importó router'\n"
        "assert 'src.adjustments.pipeline' not in sys.modules, 'automatico importó pipeline'\n"
        "assert a.estado('Antioquia', 'laboral', 'forma') == 'automatico'\n"
        "print('ok')\n"
    )
    resultado = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(_RAIZ),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert resultado.returncode == 0, resultado.stderr
    assert resultado.stdout.strip() == "ok"
