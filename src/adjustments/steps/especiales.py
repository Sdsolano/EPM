"""
Paso 5 — `especiales` (orden 4, entre `forma` y `control`): detecta y (solo si
se activa) corrige días especiales.

Días especiales detectados en la semana objetivo:
- Semana Santa: Domingo de Ramos .. Domingo de Resurrección (`analogos.es_semana_santa`).
- 24 y 31 de diciembre.
- Post-festivo: día laboral o sábado cuyo día anterior es festivo. El motivo se
  confirma con el descriptor real del día (`analogos.descriptor_dia`), que puede
  clasificarlo como `despues` (p. ej. martes tras lunes festivo) o como `puente`
  (p. ej. viernes tras jueves festivo, sábado tras viernes festivo).

Por día se registra un único `motivo` (`semana_santa` | `24_dic` | `31_dic` |
`post_festivo`), con prioridad en ese orden.

Elecciones sugeridas por IA (`ctx.extras["eventos"]`, 4D)
---------------------------------------------------------
Los eventos de la semana traen texto libre. Los días de la semana objetivo
cuyo texto habla de elecciones/votación/consulta popular se marcan con motivo
`elecciones_ia` SOLO en diagnóstico/alerta (`diagnostics.elecciones_ia` y
alerta info `ELECCIONES_IA`). Nunca modifican curvas: son eventos sugeridos por
IA, no verificados. Si no hay `extras["eventos"]`, no pasa nada.

Evidencia (reutiliza `analogos.seleccionar` y `steps/_analogos.py`):
- Filtros por motivo: Semana Santa → `semana_santa: true`; 24/31 dic →
  `meses: [12]`; post-festivo → la `relacion_festivo` del descriptor (`despues`
  o `puente`) y, si es `despues`, su `distancia_festivo`. Siempre
  `dia_semana: "auto"` y los años excluidos de la corrida.
- El límite de análogos es `filtros_analogos.limite` del analista cuando venga;
  si no, `LIMITE_AUTO_NIVEL`.
- `ancla="actual"` (default): E*(d) de la evidencia de la semana de referencia
  del paso nivel (homólogo/referencia); la forma previa se conserva.
- `ancla="analogos"`: E*(d) por `_analogos.evidencia_analogos` y la forma por
  `_analogos.forma_ponderada` de los análogos observados.
- Sin evidencia → ese día no cambia y se emite info `ESPECIALES_SIN_EVIDENCIA`.

Aplicación:
- `aplicar=false` (default): no toca las curvas; deja en diagnostics las
  sugerencias por día y emite info `ESPECIALES_SUGERIDOS`.
- `aplicar=true`: E(d) = (1−α)·E_previa(d) + α·E*(d). Con `ancla="analogos"`
  la curva del día toma la forma normalizada de los análogos escalada a E(d);
  con `ancla="actual"` se escala la curva previa (forma intacta). Como el
  backtest no respalda este ajuste, emite una vez el warning
  `ESPECIALES_SIN_BACKTEST` (no bloquea).
- `analogos`/`filtros_analogos` con `ancla != "analogos"` se ignoran y emiten
  info `PARAMS_ANALOGOS_IGNORADOS` (no rompe clientes).
- `alpha <= 0`: no se modifica ni la energía ni la forma; el día queda
  `aplicado=false` (α=0 no debe cambiar la curva).
- Días editados a mano en pasos previos (`fechas_editadas` de los records que
  llegan del pipeline) se saltan: se conserva la curva editada y se emite info
  `ESPECIALES_EDICION_PRESERVADA`.
- Los días de brecha (antes del lunes objetivo) nunca se modifican.
"""

from __future__ import annotations

import re
import unicodedata
from datetime import date, timedelta
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from src.adjustments import analogos as buscador
from src.adjustments.data import (
    TIPO_LABORAL,
    TIPO_SABADO,
    dias_semana,
    energia_curva,
    energia_diaria,
    iso,
    tipo_dia,
)
from src.adjustments.steps import _analogos, _params, nivel
from src.adjustments.types import Alert, Curves, RunContext, StepResult, validar_curva

MOTIVOS = ("semana_santa", "24_dic", "31_dic", "post_festivo")
MOTIVO_ELECCIONES_IA = "elecciones_ia"
DEFAULTS: Dict[str, Any] = {"aplicar": False, "alpha": 0.25, "ancla": "actual"}
_DEFAULTS_ANALOGOS: Dict[str, Any] = {
    "ancla": "actual",
    "analogos": None,
    "filtros_analogos": None,
}
FILTROS_MOTIVO: Dict[str, Dict[str, Any]] = {
    "semana_santa": {"semana_santa": True, "dia_semana": "auto"},
    "24_dic": {"meses": [12], "dia_semana": "auto"},
    "31_dic": {"meses": [12], "dia_semana": "auto"},
}
_RELACIONES_POST_FESTIVO = ("despues", "puente")
_RE_ELECCIONES = re.compile(
    r"\b(eleccion(es)?|votacion(es)?|votar|consulta popular|plebiscito|referend[oa])\b"
)


def _booleano(paso: str, clave: str, v: Any) -> bool:
    if not isinstance(v, (bool, np.bool_)):
        raise ValueError(f"{paso}.{clave} debe ser true o false, llegó {_params._corto(v)}")
    return bool(v)


def validar_params(params: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    aplicar: bool. alpha: número finito en [0, 1]. ancla: "actual" | "analogos".
    analogos / filtros_analogos: ver `_analogos.separar` (solo se devuelven si
    difieren de su default None).
    """
    resto, extra_analogos = _analogos.separar("especiales", params, _DEFAULTS_ANALOGOS)
    p = _params.completar("especiales", DEFAULTS, resto)
    p["aplicar"] = _booleano("especiales", "aplicar", p["aplicar"])
    p["alpha"] = _params.numero("especiales", "alpha", p["alpha"], minimo=0.0, maximo=1.0)
    p.update(extra_analogos)
    return p


def _motivo(d: date, festivos: set, desc: Dict[str, Any]) -> Optional[str]:
    if buscador.es_semana_santa(d):
        return "semana_santa"
    if d.month == 12 and d.day == 24:
        return "24_dic"
    if d.month == 12 and d.day == 31:
        return "31_dic"
    if (
        tipo_dia(d, festivos) in (TIPO_LABORAL, TIPO_SABADO)
        and iso(d - timedelta(days=1)) in festivos
        and desc.get("relacion_festivo") in _RELACIONES_POST_FESTIVO
    ):
        return "post_festivo"
    return None


def _normalizar_texto(texto: Any) -> str:
    s = unicodedata.normalize("NFKD", str(texto)).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"\s+", " ", s.lower()).strip()


def _eventos_elecciones(ctx: RunContext) -> Dict[str, str]:
    """{fecha ISO: evento} de la semana objetivo con texto de elecciones/votación."""
    eventos = (ctx.extras or {}).get("eventos")
    if not isinstance(eventos, dict):
        return {}
    semana = {iso(d) for d in dias_semana(ctx.semana_inicio)}
    out: Dict[str, str] = {}
    for f, texto in eventos.items():
        clave = str(f)[:10]
        if clave not in semana or not isinstance(texto, str) or not texto.strip():
            continue
        if _RE_ELECCIONES.search(_normalizar_texto(texto)):
            out[clave] = texto.strip()
    return out


def _filtros_motivo(motivo: str, desc: Dict[str, Any]) -> Dict[str, Any]:
    """Filtros por motivo; post-festivo usa la relación real del descriptor."""
    if motivo != "post_festivo":
        return dict(FILTROS_MOTIVO[motivo])
    rel = desc.get("relacion_festivo")
    if rel not in _RELACIONES_POST_FESTIVO:
        rel = "despues"
    filtros: Dict[str, Any] = {"relacion_festivo": rel, "dia_semana": "auto"}
    distance = desc.get("distancia_festivo")
    if rel == "despues" and distance is not None:
        filtros["distancia_festivo"] = int(distance)
    return filtros


def _filtros_dia(motivo: str, p: Dict[str, Any], desc: Dict[str, Any]) -> Dict[str, Any]:
    filtros = dict(p.get("filtros_analogos") or {})
    filtros.update(_filtros_motivo(motivo, desc))
    return filtros


def _limite(p: Dict[str, Any]) -> Optional[int]:
    """Límite del analista si pasó `filtros_analogos`; si no, el automático."""
    return None if p.get("filtros_analogos") is not None else _analogos.LIMITE_AUTO_NIVEL


def _evidencia_dia(
    ctx: RunContext,
    d: date,
    motivo: str,
    desc: Dict[str, Any],
    p: Dict[str, Any],
    energia: Dict[str, float],
    festivos: set,
    excluidos: set,
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]], Optional[np.ndarray], List[str], Optional[List[float]]]:
    """(evidencia, selección, forma normalizada, fechas de forma, pesos) del día."""
    filtros = _analogos.filtros_efectivos(
        _filtros_dia(motivo, p, desc), excluidos, _limite(p)
    )
    sel = _analogos.seleccion_dia(ctx, d, p.get("analogos"), filtros)
    ev = _analogos.evidencia_analogos(
        ctx, d, sel, energia, festivos, excluidos, nivel.MAX_SEMANAS_EXTRA_ANCLA
    )
    if ev is None:
        return None, sel, None, [], None
    forma, fechas, pesos = _analogos.forma_ponderada(sel["items"])
    return ev, sel, forma, fechas, pesos


def _entrada_elecciones(curva: Optional[List[float]], alpha: float, evento: str) -> Dict[str, Any]:
    e_previa = energia_curva(curva)
    return {
        "motivo": MOTIVO_ELECCIONES_IA,
        "E_previa": e_previa,
        "E_objetivo": None,
        "E_final": e_previa,
        "alpha": alpha,
        "alpha_efectivo": 0.0,
        "ratio": None,
        "ancla": None,
        "regla": "sin_evidencia",
        "analogos": [],
        "fechas_fuente": [],
        "formas_fuente": [],
        "seleccion": None,
        "pesos": None,
        "aplicado": False,
        "elecciones_ia": True,
        "evento_ia": evento,
    }


def _entrada_preservada(motivo: str, curva: Optional[List[float]], alpha: float) -> Dict[str, Any]:
    e_previa = energia_curva(curva)
    return {
        "motivo": motivo,
        "E_previa": e_previa,
        "E_objetivo": None,
        "E_final": e_previa,
        "alpha": alpha,
        "alpha_efectivo": 0.0,
        "ratio": None,
        "ancla": None,
        "regla": "edicion_preservada",
        "analogos": [],
        "fechas_fuente": [],
        "formas_fuente": [],
        "seleccion": None,
        "pesos": None,
        "aplicado": False,
    }


def step(
    ctx: RunContext,
    prev: Curves,
    params: Dict[str, Any],
    *,
    fechas_editadas: Optional[Iterable[str]] = None,
) -> StepResult:
    p = validar_params(params)
    aplicar, alpha, ancla = p["aplicar"], p["alpha"], p["ancla"]
    if ancla == "analogos":
        _analogos.validar_mapa_ctx("especiales", p.get("analogos"), ctx)

    festivos = set(ctx.festivos)
    energia = energia_diaria(ctx.historico) if ancla == "analogos" else {}
    excluidos = _analogos.anios_excluidos_ctx(ctx) if ancla == "analogos" else set()
    evidencias_nivel = nivel.evidencias(ctx) if ancla == "actual" else {}
    protegidas = {str(f)[:10] for f in (fechas_editadas or ())}
    elecciones = _eventos_elecciones(ctx)

    curves: Curves = {f: list(v) for f, v in prev.items()}
    por_dia: Dict[str, Dict[str, Any]] = {}
    detectados: Dict[str, str] = {}
    sin_evidencia: List[str] = []
    ajustados: List[str] = []
    preservados: List[str] = []
    dias_elecciones: List[str] = []

    for d in dias_semana(ctx.semana_inicio):
        f = iso(d)
        desc = buscador.descriptor_dia(d, festivos)
        motivo = _motivo(d, festivos, desc)
        if motivo is None:
            if f in elecciones:
                por_dia[f] = _entrada_elecciones(prev.get(f), alpha, elecciones[f])
                dias_elecciones.append(f)
            continue
        detectados[f] = motivo
        if f in protegidas:
            preservados.append(f)
            por_dia[f] = _entrada_preservada(motivo, prev.get(f), alpha)
            if f in elecciones:
                por_dia[f].update(elecciones_ia=True, evento_ia=elecciones[f])
                dias_elecciones.append(f)
            continue

        curva_prev = None
        if f in prev and validar_curva(prev[f]) is None:
            curva_prev = np.asarray(prev[f], dtype=float)
        e_previa = energia_curva(prev.get(f))

        sel: Optional[Dict[str, Any]] = None
        fechas_forma: List[str] = []
        pesos: Optional[List[float]] = None
        forma: Optional[np.ndarray] = None
        if ancla == "analogos":
            ev, sel, forma, fechas_forma, pesos = _evidencia_dia(
                ctx, d, motivo, desc, p, energia, festivos, excluidos
            )
        else:
            ev = evidencias_nivel.get(f)

        if ev is None:
            sin_evidencia.append(f)
        e_objetivo = ev.get("E_objetivo") if ev else None
        aplicado_dia = bool(
            aplicar
            and alpha > 0
            and ev is not None
            and e_previa is not None
            and e_previa > 0
            and e_objetivo is not None
            and e_objetivo > 0
        )
        e_final = e_previa
        if aplicado_dia:
            e_final = (1 - alpha) * e_previa + alpha * e_objetivo
            if ancla == "analogos" and forma is not None:
                curves[f] = (forma * e_final).tolist()
            else:
                curves[f] = (curva_prev * (e_final / e_previa)).tolist()
            ajustados.append(f)

        por_dia[f] = {
            "motivo": motivo,
            "E_previa": e_previa,
            "E_objetivo": e_objetivo,
            "E_final": e_final,
            "alpha": alpha,
            "alpha_efectivo": alpha if aplicado_dia else 0.0,
            "ratio": ev.get("ratio") if ev else None,
            "ancla": ev.get("ancla") if ev else None,
            "regla": ev.get("regla") if ev else "sin_evidencia",
            "analogos": ev.get("analogos") if ev and ancla == "analogos" else [],
            "fechas_fuente": fechas_forma
            or ([ev.get("dia_ref")] if ev and ev.get("dia_ref") else []),
            "formas_fuente": list(fechas_forma),
            "seleccion": sel["seleccion"] if sel else None,
            "pesos": pesos,
            "aplicado": aplicado_dia,
        }
        if f in elecciones:
            por_dia[f].update(elecciones_ia=True, evento_ia=elecciones[f])
            dias_elecciones.append(f)

    alerts: List[Alert] = []
    if aplicar:
        alerts.append(
            Alert(
                nivel="warning",
                codigo="ESPECIALES_SIN_BACKTEST",
                mensaje=(
                    "El paso especiales se aplica sin respaldo del backtest "
                    "(alpha/ancla no sostenidos); revise el resultado antes de aprobar"
                ),
                detalle={"alpha": alpha, "ancla": ancla},
            )
        )
    analogos_ignorados = ancla != "analogos" and (
        p.get("analogos") is not None or p.get("filtros_analogos") is not None
    )
    if analogos_ignorados:
        alertas_ignoradas = {
            k: v for k, v in (("analogos", p.get("analogos")),
                              ("filtros_analogos", p.get("filtros_analogos"))) if v is not None
        }
        alerts.append(
            Alert(
                nivel="info",
                codigo="PARAMS_ANALOGOS_IGNORADOS",
                mensaje=(
                    "Se recibieron analogos/filtros_analogos pero ancla no es 'analogos'; "
                    "esos params no se usan"
                ),
                detalle=alertas_ignoradas,
            )
        )
    if detectados and not aplicar:
        alerts.append(
            Alert(
                nivel="info",
                codigo="ESPECIALES_SUGERIDOS",
                mensaje=(
                    "Hay días especiales en la semana objetivo; se sugiere revisarlos "
                    "(el paso no está aplicado)"
                ),
                fechas=sorted(detectados),
                detalle={"motivos": detectados, "alpha": alpha, "ancla": ancla},
            )
        )
    if sin_evidencia:
        alerts.append(
            Alert(
                nivel="info",
                codigo="ESPECIALES_SIN_EVIDENCIA",
                mensaje="Sin análogos observados para algunos días especiales; se conserva su curva previa",
                fechas=sorted(sin_evidencia),
            )
        )
    if preservados:
        alerts.append(
            Alert(
                nivel="info",
                codigo="ESPECIALES_EDICION_PRESERVADA",
                mensaje=(
                    "Algunos días especiales tienen edición manual previa; se conserva "
                    "la curva editada y no se recalcula su forma/nivel"
                ),
                fechas=sorted(preservados),
                detalle={"motivos": {f: detectados[f] for f in preservados}},
            )
        )
    if dias_elecciones:
        alerts.append(
            Alert(
                nivel="info",
                codigo="ELECCIONES_IA",
                mensaje=(
                    "Días con elecciones/votación sugeridos por IA (no verificados); "
                    "solo diagnóstico, no se ajustan curvas"
                ),
                fechas=sorted(dias_elecciones),
                detalle={
                    "eventos": {f: elecciones[f] for f in dias_elecciones},
                    "fuente": "openai",
                    "verificado": False,
                },
            )
        )

    diagnostics: Dict[str, Any] = {
        "aplicar": aplicar,
        "alpha": alpha,
        "ancla": ancla,
        "motivos": sorted(set(detectados.values())),
        "dias_especiales": sorted(detectados),
        "dias_ajustados": sorted(ajustados),
        "dias_preservados": sorted(preservados),
        "elecciones_ia": sorted(dias_elecciones),
        "dias": por_dia,
    }
    if p.get("analogos") is not None:
        diagnostics["analogos"] = p["analogos"]
    if p.get("filtros_analogos") is not None:
        diagnostics["filtros_analogos"] = p["filtros_analogos"]
    return StepResult(curves=curves, diagnostics=diagnostics, alerts=alerts)
