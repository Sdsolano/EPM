"""
Paso 6 (orden 5) — control: no modifica curvas. Emite DOS controles de
tendencia distintos (no confundirlos) y los bloqueos de aprobación.

1) Enclavamiento — `TENDENCIA_FUERA_BANDA` (warning). Control de EDICIÓN:
   ¿alguien movió la energía semanal después de que el paso nivel la fijó?
   - `energia_nivel` = energía diaria que CALCULÓ el paso nivel (pre-edición:
     para días editados se usa `curvas_calculadas`). Si nivel se omitió, el
     pipeline pasa la energía del paso base.
   - desvio_pp = (E_semana_final / E_semana_nivel − 1)·100. |desvio_pp| >
     `banda_pp` (default 1.0) → warning con los `top_dias` días que más
     contribuyen (en la dirección del desvío).
   - Sin ediciones posteriores a nivel y con forma reconciliada, desvio_pp = 0.
   - NO ve: alpha=0, nivel omitido, base editada ni los días que ajustó
     `especiales` (en todos los casos la energía de referencia del
     enclavamiento ya incorpora esa decisión; el pipeline sustituye esos días
     por su `E_final` de especiales).

2) Coherencia con la referencia — `TENDENCIA_REFERENCIA_FUERA_BANDA` (info).
   Control de PROCESO: ¿la tendencia semanal resultante se parece a la que
   sugiere la semana de referencia, decida lo que decida el analista?
   - Tendencia de referencia (fórmula post-H1) = Σ E*(d) / E(semana previa) − 1,
     E*(d) = `E_objetivo` del paso nivel (`nivel_dias`; si no llega —nivel
     omitido o llamada directa— se recalcula con `nivel.evidencias(ctx)`). E*
     no depende de α ni de la base. Día sin E* → energía base, listado en
     `dias_sin_evidencia`.
   - Tendencia resultante = E(semana final) / E(semana previa) − 1 (semana
     previa: real si <= corte; si no, la curva estimada de la brecha).
   - diferencia_tendencia_pp = resultante − referencia. |diferencia| >
     `banda_referencia_pp` (default 2.0 pp; null la desactiva) → info con los
     días que más contribuyen. Se dispara también con alpha=0, nivel omitido
     o base editada (casos que el enclavamiento no ve). Es informativa: con
     α<1 mide en parte el desacuerdo base vs referencia, no un error (backtest
     2026-11-12), por eso no es warning ni bloqueo.

Bloqueos (`approval_blockers` del pipeline los recoge): periodos faltantes,
valores no finitos/None o negativos en la SEMANA OBJETIVO (`validar_curvas`),
y reconciliación rota (la suma diaria no coincide con la energía del paso
nivel, solo cuando el pipeline la pasa en `energia_esperada`).

Brecha (días entre fecha_corte+1 y el lunes objetivo): es estimación, no se
aprueba ni bloquea. Si algún día de la brecha falta o es inválido se emite
warning `BRECHA_INVALIDA` (afecta la semana previa de la tendencia).
"""

from __future__ import annotations

import math
from datetime import timedelta
from typing import Any, Dict, List, Optional

import numpy as np

from src.adjustments import analogos
from src.adjustments.data import dias_semana, energia_curva, energia_diaria, iso
from src.adjustments.steps import _params
from src.adjustments.steps import nivel as paso_nivel
from src.adjustments.types import MOTIVOS_CURVA, Alert, Curves, RunContext, StepResult, validar_curva

DEFAULTS: Dict[str, Any] = {"banda_pp": 1.0, "top_dias": 3, "banda_referencia_pp": 2.0}
TOLERANCIA_RECONCILIACION = 1e-6


def validar_params(params: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """banda_pp > 0; top_dias entero en [1, 7]; banda_referencia_pp None o > 0."""
    p = _params.completar("control", DEFAULTS, params)
    p["banda_pp"] = _params.numero("control", "banda_pp", p["banda_pp"], minimo=0.0, minimo_exclusivo=True)
    p["top_dias"] = _params.entero("control", "top_dias", p["top_dias"], minimo=1, maximo=7)
    if p["banda_referencia_pp"] is not None:
        p["banda_referencia_pp"] = _params.numero(
            "control", "banda_referencia_pp", p["banda_referencia_pp"], minimo=0.0, minimo_exclusivo=True
        )
    return p


def validar_curvas(curves: Curves, fechas: List[str]) -> List[Alert]:
    """Alertas de bloqueo estructurales (vía `validar_curva`) para las fechas indicadas."""
    por_motivo: Dict[str, List[str]] = {}
    for f in fechas:
        motivo = validar_curva(curves.get(f))
        if motivo is not None:
            por_motivo.setdefault(motivo, []).append(f)
    return [
        Alert("bloqueo", motivo, MOTIVOS_CURVA[motivo], por_motivo[motivo])
        for motivo in MOTIVOS_CURVA
        if motivo in por_motivo
    ]


def _finito(v: Any) -> Optional[float]:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _culpables(contrib: Dict[str, Dict[str, float]], desvio: float, top: int) -> List[str]:
    signo = 1 if desvio > 0 else -1
    return sorted(contrib, key=lambda f: signo * contrib[f]["desvio_mwh"], reverse=True)[:top]


def _enclavamiento(
    fechas: List[str],
    e_fin: Dict[str, Optional[float]],
    energia_nivel: Optional[Dict[str, float]],
    banda: float,
    top: int,
    diag: Dict[str, Any],
    alerts: List[Alert],
) -> None:
    """desvio_pp de la semana final vs la energía semanal que fijó el paso nivel."""
    diag.update(energia_semana_nivel=None, desvio_pp=None, dias={})
    if energia_nivel is None:
        alerts.append(Alert("info", "SIN_ENERGIA_NIVEL",
                            "No llegó la energía del paso nivel; no se evalúa la banda de energía semanal"))
        return
    e_niv = {f: _finito(energia_nivel.get(f)) for f in fechas}
    faltan = [f for f in fechas if e_fin[f] is None or e_niv[f] is None]
    total_niv = sum(v for v in e_niv.values() if v is not None)
    if faltan or total_niv <= 0:
        alerts.append(Alert("info", "SIN_CONTROL_TENDENCIA",
                            "No se puede comparar la energía semanal con la del paso nivel (faltan días)",
                            faltan))
        return

    total_fin = sum(e_fin.values())
    desvio_pp = (total_fin / total_niv - 1) * 100
    contrib = {
        f: {
            "energia": e_fin[f],
            "energia_nivel": e_niv[f],
            "desvio_mwh": e_fin[f] - e_niv[f],
            "desvio_pp": (e_fin[f] - e_niv[f]) / total_niv * 100,
        }
        for f in fechas
    }
    diag.update(energia_semana_nivel=total_niv, desvio_pp=desvio_pp, dias=contrib)

    if abs(desvio_pp) > banda:
        culpables = _culpables(contrib, desvio_pp, top)
        alerts.append(Alert(
            "warning", "TENDENCIA_FUERA_BANDA",
            f"La energía semanal final se desvía {desvio_pp:+.2f} % de la que fijó el paso nivel "
            f"(banda ±{banda:.2f} pp)",
            culpables,
            detalle={"desvio_pp": desvio_pp, "dias": {f: contrib[f] for f in culpables}},
        ))


def _tendencia_referencia(
    ctx: RunContext,
    prev: Curves,
    semana: list,
    fechas: List[str],
    e_fin: Dict[str, Optional[float]],
    nivel_dias: Optional[Dict[str, Dict[str, Any]]],
    banda_ref: Optional[float],
    top: int,
    diag: Dict[str, Any],
    alerts: List[Alert],
) -> None:
    """Diagnóstico: tendencia resultante vs la de referencia (Σ E*(d) vs semana previa)."""
    diag.update(
        energia_semana_previa=None, tendencia_resultante_pct=None, tendencia_esperada_pct=None,
        energia_semana_esperada=None, diferencia_tendencia_pp=None, dias_sin_evidencia=[],
        semana_ref_inicio=iso(ctx.semana_ref_inicio) if ctx.semana_ref_inicio else None,
    )
    reales = energia_diaria(ctx.historico)
    e_prev = {}
    for d, f in zip(semana, fechas):
        fp = iso(d - timedelta(days=7))
        e_prev[f] = reales[fp] if fp in reales else energia_curva(prev.get(fp))
    if any(v is None for v in e_fin.values()) or any(v is None for v in e_prev.values()):
        diag["tendencia_motivo"] = "faltan días de la semana final o de la previa"
        return
    total_fin, total_prev = sum(e_fin.values()), sum(e_prev.values())
    if total_prev <= 0:
        diag["tendencia_motivo"] = "semana previa sin energía"
        return
    t_res = total_fin / total_prev - 1
    diag.update(energia_semana_previa=total_prev, tendencia_resultante_pct=t_res * 100)

    if nivel_dias is None:
        ev = paso_nivel.evidencias(ctx)
        nivel_dias = {
            f: {"E_objetivo": ev[f]["E_objetivo"] if ev.get(f) else None, "E_base": e_fin[f]}
            for f in fechas
        }
    e_esp: Dict[str, float] = {}
    sin_evidencia: List[str] = []
    for f in fechas:
        info = nivel_dias.get(f) or {}
        e_obj = _finito(info.get("E_objetivo"))
        if e_obj is not None and e_obj > 0:
            e_esp[f] = e_obj
        else:
            sin_evidencia.append(f)
            e_base = _finito(info.get("E_base"))
            e_esp[f] = e_base if e_base is not None else e_fin[f]
    diag["dias_sin_evidencia"] = sin_evidencia
    if not ctx.semana_ref_inicio or len(sin_evidencia) == len(fechas):
        diag["tendencia_motivo"] = "sin evidencia de referencia (E*) en ningún día"
        return

    t_esp = sum(e_esp.values()) / total_prev - 1
    diferencia = (t_res - t_esp) * 100
    diag.update(
        energia_semana_esperada=sum(e_esp.values()),
        tendencia_esperada_pct=t_esp * 100,
        diferencia_tendencia_pp=diferencia,
    )
    for f in fechas:
        if f in diag["dias"]:
            diag["dias"][f].update(
                energia_esperada=e_esp[f],
                fuente_esperada="energia_base" if f in sin_evidencia else "E_objetivo_nivel",
            )

    if banda_ref is not None and abs(diferencia) > banda_ref:
        contrib = {f: {"desvio_mwh": e_fin[f] - e_esp[f]} for f in fechas}
        culpables = _culpables(contrib, diferencia, top)
        alerts.append(Alert(
            "info", "TENDENCIA_REFERENCIA_FUERA_BANDA",
            f"Tendencia resultante {t_res*100:.2f}% vs la de la semana de referencia "
            f"{t_esp*100:.2f}% (diferencia {diferencia:+.2f} pp, banda ±{banda_ref:.2f} pp)",
            culpables,
            detalle={
                "diferencia_tendencia_pp": diferencia,
                "tendencia_resultante_pct": t_res * 100,
                "tendencia_esperada_pct": t_esp * 100,
                "dias_sin_evidencia": list(sin_evidencia),
            },
        ))


def _brecha(ctx: RunContext, prev: Curves, diag: Dict[str, Any], alerts: List[Alert]) -> None:
    """Días de brecha (corte+1 .. lunes−1): solo aviso, no bloquean la aprobación."""
    fechas = [
        iso(ctx.fecha_corte + timedelta(days=i))
        for i in range(1, (ctx.semana_inicio - ctx.fecha_corte).days)
    ]
    invalidos = {f: motivo for f in fechas if (motivo := validar_curva(prev.get(f))) is not None}
    diag["dias_brecha"] = fechas
    if invalidos:
        alerts.append(Alert(
            "warning", "BRECHA_INVALIDA",
            "Días de brecha (estimados, entre el corte y la semana objetivo) faltantes o con valores "
            "inválidos; no bloquean la aprobación pero la semana previa de la tendencia queda incompleta",
            sorted(invalidos),
            detalle={"motivos": invalidos},
        ))


def _alertas_festivos(ctx: RunContext, fechas: List[str], alerts: List[Alert]) -> None:
    """Info con los festivos (de `ctx.festivos`) que caen en la semana objetivo."""
    festivos = sorted({str(f)[:10] for f in (ctx.festivos or [])} & set(fechas))
    if festivos:
        alerts.append(Alert(
            "info", "FESTIVO_EN_SEMANA",
            f"La semana objetivo contiene {len(festivos)} festivo(s)",
            festivos,
            detalle={"festivos": festivos},
        ))


def _alertas_eventos(ctx: RunContext, fechas: List[str], alerts: List[Alert]) -> None:
    """Info con los eventos sugeridos por IA (`ctx.extras["eventos"]`) de la semana objetivo."""
    eventos = (ctx.extras or {}).get("eventos")
    if not isinstance(eventos, dict) or not eventos:
        return
    semana = set(fechas)
    en_semana = {
        str(f)[:10]: texto
        for f, texto in eventos.items()
        if str(f)[:10] in semana
    }
    if not en_semana:
        return
    alerts.append(Alert(
        "info", "EVENTOS_SEMANA",
        "Eventos sugeridos por IA, no verificados",
        sorted(en_semana),
        detalle={"eventos": en_semana, "fuente": "openai", "verificado": False},
    ))


def _alertas_clima(ctx: RunContext, fechas: List[str], alerts: List[Alert]) -> None:
    """
    Info/warning con el clima observado de la semana (solo si `cargar_clima`
    cubre días de la semana; en semanas futuras no hay clima y no se emite).

    La lectura es desde el disco (cacheada en módulo por `analogos.cargar_clima`),
    no desde `ctx`: el clima observado no se persiste en la corrida (limitación
    aceptada; `ctx.extras` solo lleva eventos/festivos).

    Warning si la lluvia diaria máxima de la semana supera el p90 histórico o
    si la temperatura media de la semana se aleja más de 2σ de la histórica.
    """
    try:
        clima = analogos.cargar_clima(ctx.ucp)
    except Exception:
        return
    if not isinstance(clima, dict) or not clima:
        return

    semana = set(fechas)
    dias = {f: clima[f] for f in fechas if isinstance(clima.get(f), dict)}
    if not dias:
        return
    temps = [t for c in dias.values() if (t := _finito(c.get("temp_media"))) is not None]
    lluvias = [v for c in dias.values() if (v := _finito(c.get("lluvia_mm"))) is not None]
    if not temps and not lluvias:
        return

    hist_temps: List[float] = []
    hist_lluvias: List[float] = []
    for f, c in clima.items():
        if f in semana or not isinstance(c, dict):
            continue
        t = _finito(c.get("temp_media"))
        v = _finito(c.get("lluvia_mm"))
        if t is not None:
            hist_temps.append(t)
        if v is not None:
            hist_lluvias.append(v)

    temp_media = sum(temps) / len(temps) if temps else None
    lluvia_total = sum(lluvias) if lluvias else None
    lluvia_max = max(lluvias) if lluvias else None
    p90_lluvia = float(np.percentile(hist_lluvias, 90)) if hist_lluvias else None
    temp_hist = float(np.mean(hist_temps)) if hist_temps else None
    sigma_temp = float(np.std(hist_temps)) if len(hist_temps) > 1 else 0.0

    lluvia_extrema = p90_lluvia is not None and lluvia_max is not None and lluvia_max > p90_lluvia
    temp_extrema = (
        temp_hist is not None and temp_media is not None and sigma_temp > 0
        and abs(temp_media - temp_hist) > 2 * sigma_temp
    )
    nivel = "warning" if (lluvia_extrema or temp_extrema) else "info"
    partes = []
    if temp_media is not None:
        partes.append(f"temperatura media {temp_media:.1f} °C")
    if lluvia_total is not None:
        partes.append(f"lluvia {lluvia_total:.1f} mm")
    alerts.append(Alert(
        nivel, "CLIMA_OBSERVADO",
        "Clima observado de la semana: " + ", ".join(partes),
        list(dias),
        detalle={
            "temp_media": temp_media,
            "lluvia_mm": lluvia_total,
            "lluvia_max_diaria_mm": lluvia_max,
            "p90_lluvia_mm": p90_lluvia,
            "temp_media_historica": temp_hist,
            "sigma_temp": sigma_temp,
            "dias": {f: dict(c) for f, c in dias.items()},
        },
    ))


def step(
    ctx: RunContext,
    prev: Curves,
    params: Dict[str, Any],
    *,
    energia_esperada: Optional[Dict[str, float]] = None,
    energia_nivel: Optional[Dict[str, float]] = None,
    nivel_dias: Optional[Dict[str, Dict[str, Any]]] = None,
) -> StepResult:
    """
    `energia_esperada`: energía diaria final del paso nivel (reconciliación de la forma).
    `energia_nivel`: energía diaria de referencia para el enclavamiento (calculada por
    nivel, pre-edición; o la del paso base si nivel se omitió). None → no se evalúa.
    `nivel_dias`: `diagnostics["dias"]` del paso nivel (E*(d)) para el diagnóstico
    de tendencia; None → se recalcula E*.
    """
    p = validar_params(params)
    semana = dias_semana(ctx.semana_inicio)
    fechas = [iso(d) for d in semana]

    alerts = validar_curvas(prev, fechas)
    diag: Dict[str, Any] = {"banda_pp": p["banda_pp"], "banda_referencia_pp": p["banda_referencia_pp"]}

    # Reconciliación con el paso nivel
    rotos = {}
    for f in fechas:
        if energia_esperada and f in energia_esperada and f in prev:
            suma = energia_curva(prev[f])
            esperada = _finito(energia_esperada[f])
            if suma is None or esperada is None:
                continue  # curva inválida: ya está bloqueada por validar_curvas
            if not math.isclose(suma, esperada, rel_tol=0.0, abs_tol=TOLERANCIA_RECONCILIACION):
                rotos[f] = {"suma": suma, "esperada": esperada}
    diag["reconciliacion_verificada"] = energia_esperada is not None
    if rotos:
        alerts.append(Alert("bloqueo", "RECONCILIACION_ROTA",
                            "La suma diaria no coincide con la energía del paso nivel",
                            sorted(rotos), detalle=rotos))

    e_fin = {f: energia_curva(prev.get(f)) for f in fechas}
    total_fin = sum(e_fin.values()) if all(v is not None for v in e_fin.values()) else None
    diag["energia_semana_ajustada"] = total_fin

    _brecha(ctx, prev, diag, alerts)
    _alertas_festivos(ctx, fechas, alerts)
    _alertas_eventos(ctx, fechas, alerts)
    _alertas_clima(ctx, fechas, alerts)
    _enclavamiento(fechas, e_fin, energia_nivel, p["banda_pp"], p["top_dias"], diag, alerts)
    _tendencia_referencia(ctx, prev, semana, fechas, e_fin, nivel_dias,
                          p["banda_referencia_pp"], p["top_dias"], diag, alerts)

    return StepResult(curves={f: list(v) for f, v in prev.items()}, diagnostics=diag, alerts=alerts)
