"""
Configuración R2: qué pasos pueden aplicarse de forma automática por segmento.

Fuente: `reports/ajustes/backtest_modelo_20261113.md`, sección
"8. R2: qué pasos pueden ser automáticos por segmento (modelo_k1, config default)".
El backtest clasifica cada (ucp, tipo_dia, paso) como `automático`, `sugerido` o
`no aplicar` con el default actual (alpha=0.25, promedio_n=4, modelo_k1):

- `automático`: el paso ayuda en calibración Y en prueba, empeora en ≤35 % de
  las semanas de prueba y hay ≥10 días de prueba.
- `sugerido`: cualquier otro caso con evidencia insuficiente o mixta.
- `no aplicar`: empeora en calibración y en prueba.

En este módulo solo se exponen los `automático` (R2 = "automatico"); todo lo
demás queda como `sugerido`. R2 es una **clasificación de revisión para el
front**: el backend aplica siempre los pasos del pipeline y este módulo no
cambia ninguna curva ni ningún default. `modos_paso(ucp)` resume, por paso, si
todos los tipos de día con regla son automáticos (`"automatico"`) o no
(`"sugerido"`); el router lo publica siempre en `meta.extras["modos_paso"]`.
`configuracion()` y `estado()` son el detalle por tipo de día que consume el
endpoint `GET /adjustments/automatico` (ver PLAN_4C).

Sin dependencias del paquete (ni de `router`, ni de `pipeline`, ni de los
pasos): importar este módulo no lee datos, no toca `DEFAULTS` ni provoca
efectos secundarios. Las UCP se comparan de forma exacta ("Antioquia",
"Atlantico"); cualquier otra (incluida `None`) devuelve todo `sugerido`.
"""

from __future__ import annotations

from typing import Any, Dict, List

AUTOMATICO = "automatico"
SUGERIDO = "sugerido"
ESTADOS = (AUTOMATICO, SUGERIDO)

# Tipos de día tal como los devuelve `data.tipo_dia` (el festivo tiene prioridad).
TIPOS = ("laboral", "sabado", "domingo", "festivo")
# Pasos del pipeline que pueden ser automáticos (base/referencia/control no lo son).
PASOS = ("nivel", "forma")

# Configuración derivada del backtest `modelo_k1` con la config default.
# Solo aparecen los pares (tipo_dia, paso) que el reporte marcó `automático`;
# el resto se resuelve como `sugerido` en `estado`/`configuracion`.
CONFIG_R2: Dict[str, Dict[str, Dict[str, str]]] = {
    "Antioquia": {
        "laboral": {"forma": AUTOMATICO},
    },
    "Atlantico": {
        "laboral": {"forma": AUTOMATICO},
        "domingo": {"forma": AUTOMATICO},
        "sabado": {"nivel": AUTOMATICO},
    },
}

# Metadatos que acompañan a `configuracion()` para trazabilidad.
FUENTE = "reports/ajustes/backtest_modelo_20261113.md"
SECCION = "8. R2 (modelo_k1, config default)"
CRITERIO = (
    "automático si el paso ayuda en calibración y en prueba, empeora en ≤35 % "
    "de las semanas de prueba y hay ≥10 días de prueba; el resto queda sugerido"
)

UCP_CON_AUTOMATICO = tuple(CONFIG_R2)


def _validar_tipo(tipo_dia: Any) -> str:
    if tipo_dia not in TIPOS:
        raise ValueError(f"tipo_dia inválido {tipo_dia!r}; válidos: {TIPOS}")
    return tipo_dia


def _validar_paso(paso: Any) -> str:
    if paso not in PASOS:
        raise ValueError(f"paso inválido {paso!r}; válidos: {PASOS}")
    return paso


def _clave_ucp(ucp: Any) -> str:
    """Solo las UCP de texto pueden estar configuradas; el resto es desconocida."""
    return ucp if isinstance(ucp, str) else ""


def _regla(ucp: Any, tipo_dia: str, paso: str) -> str:
    """Estado configurado para el segmento; nunca devuelve un valor fuera de ESTADOS."""
    valor = CONFIG_R2.get(_clave_ucp(ucp), {}).get(tipo_dia, {}).get(paso, SUGERIDO)
    return AUTOMATICO if valor == AUTOMATICO else SUGERIDO


def estado(ucp: Any, tipo_dia: Any, paso: Any) -> str:
    """
    `"automatico"` si el backtest habilita ese (ucp, tipo_dia, paso), si no
    `"sugerido"`. UCP desconocida (o `None`) → `"sugerido"`.

    Lanza ValueError si `tipo_dia` no está en TIPOS o `paso` no está en PASOS
    (la validación ocurre antes de mirar la UCP).
    """
    return _regla(ucp, _validar_tipo(tipo_dia), _validar_paso(paso))


def pasos_automaticos(ucp: Any, tipo_dia: Any) -> List[str]:
    """Pasos automáticos del segmento, en el orden de PASOS (vacío si no hay)."""
    _validar_tipo(tipo_dia)
    return [paso for paso in PASOS if _regla(ucp, tipo_dia, paso) == AUTOMATICO]


def _estados_con_regla(ucp: Any, paso: str) -> List[str]:
    """Estados de los tipos de día con una regla explícita para el paso en CONFIG_R2."""
    por_tipo = CONFIG_R2.get(_clave_ucp(ucp), {})
    return [
        _regla(ucp, tipo, paso)
        for tipo in TIPOS
        if paso in por_tipo.get(tipo, {})
    ]


def modos_paso(ucp: Any) -> Dict[str, str]:
    """
    Modo de revisión por paso para el front: `{paso: "automatico" | "sugerido"}`.

    Un paso es `"automatico"` solo si **todos** los tipos de día con regla en
    `CONFIG_R2` lo marcan automático (y hay al menos uno); si es mixto o no
    tiene reglas → `"sugerido"`. UCP desconocida (o `None`) → todo `"sugerido"`.

    Es barato (no lee datos ni base) y no altera curvas: el backend siempre
    aplica los pasos; esto solo dice qué revisiones puede automatizar el front.
    """
    modos: Dict[str, str] = {}
    for paso in PASOS:
        estados = _estados_con_regla(ucp, paso)
        modos[paso] = AUTOMATICO if estados and all(e == AUTOMATICO for e in estados) else SUGERIDO
    return modos


def configuracion(ucp: Any) -> Dict[str, Any]:
    """
    Configuración completa por tipo de día, con ambos pasos y su estado.

    Estructura::

        {
            "ucp": <ucp tal como llegó>,
            "fuente": FUENTE,
            "seccion": SECCION,
            "criterio": CRITERIO,
            "tipos": {
                "laboral": {"nivel": "sugerido" | "automatico", "forma": ...},
                "sabado": {...}, "domingo": {...}, "festivo": {...},
            },
        }

    UCP desconocida → todos los tipos/pasos en `"sugerido"`. Devuelve
    estructuras nuevas en cada llamada (mutarlas no altera CONFIG_R2).
    """
    return {
        "ucp": ucp,
        "fuente": FUENTE,
        "seccion": SECCION,
        "criterio": CRITERIO,
        "tipos": {
            tipo: {paso: _regla(ucp, tipo, paso) for paso in PASOS} for tipo in TIPOS
        },
    }
