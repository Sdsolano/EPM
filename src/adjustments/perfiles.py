"""Perfiles de la Mesa de ajustes: conjuntos de params por paso que replican el flujo manual de Jamer.

`perfil` es un atajo de `RunRequest`: se expande aquí a los params de nivel/forma/control y se mezcla con los
params explícitos del usuario (el usuario gana clave a clave). El resultado expandido es lo que se guarda en
`meta.params_iniciales`, de modo que recalcular (PATCH) es determinista aunque el perfil cambie después.
Los params efectivos (g_*, peso, tramos, bandas, ventanas) quedan además en `steps[].params`, que es lo que
usa PATCH: nivel (`modo="reciente"`), forma (`fuente="reciente_clase"`) y control (claves explícitas del perfil) los
persisten aunque coincidan con el default del código.

- `jamer`: réplica pura. nivel: R1·(1+g) (`modo="reciente"`, `peso_modelo=0`); forma: forma de R1 al total del
  modelo y delta plano (`fuente="reciente_clase"`, `delta="plano"`, `tramos="off"`); control: alerta de saltos
  entre días con banda provisional de 40 MW.
- `jamer_modelo`: igual pero con `peso_modelo=0.3` (mezcla de la energía de R1·(1+g) con la del modelo).
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Optional

from src.adjustments.steps import control as _control

PERFIL_JAMER = "jamer"
PERFIL_JAMER_MODELO = "jamer_modelo"
PERFILES = (PERFIL_JAMER, PERFIL_JAMER_MODELO)

PESO_MODELO_JAMER_MODELO = 0.3  # provisional (lo decide la calibración)
BANDA_SALTO_MW = 40.0  # provisional (calibración)
_CONTROL_CONGELADOS = ("periodos_valle", "crecimiento_ventana_dias", "crecimiento_min_pares", "banda_error_pct")


def params_perfil(perfil: str) -> Dict[str, Dict[str, Any]]:
    """Params por paso del perfil (copia nueva). ValueError si el perfil no existe."""
    if perfil not in PERFILES:
        raise ValueError(f"perfil desconocido: {perfil!r}; válidos: {list(PERFILES)}")
    peso = 0.0 if perfil == PERFIL_JAMER else PESO_MODELO_JAMER_MODELO
    return {
        "nivel": {"modo": "reciente", "g_fuente": "interanual", "peso_modelo": peso},
        "forma": {"fuente": "reciente_clase", "n": 1, "delta": "plano", "tramos": "off"},
        # Los params calibrables del control van explícitos (valor vigente al expandir): quedan congelados en
        # `steps[].params` y recalibrar los defaults del código no altera corridas ya creadas.
        "control": {
            "banda_salto_mw": BANDA_SALTO_MW,
            **{k: deepcopy(_control.DEFAULTS_JAMER[k]) for k in _CONTROL_CONGELADOS},
        },
    }


def expandir(perfil: Optional[str], params: Optional[Dict[str, Dict[str, Any]]]) -> Optional[Dict[str, Dict[str, Any]]]:
    """
    Mezcla el perfil con los params del usuario (el usuario gana por clave). Sin perfil devuelve `params`
    tal cual (None incluido) para que el comportamiento previo no cambie. No muta las entradas.
    """
    if perfil is None:
        return params
    out = params_perfil(perfil)
    for paso, valores in (params or {}).items():
        if isinstance(valores, dict):
            out[paso] = {**out.get(paso, {}), **deepcopy(valores)}
        else:
            out[paso] = valores  # que lo rechace la validación del paso (422), no se oculta aquí
    return out
