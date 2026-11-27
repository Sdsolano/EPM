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

3) Alertas Jamer (info, no bloquean; params por defecto None = apagadas):
   - `SALTO_ENTRE_DIAS` (`banda_salto_mw`, `periodos_valle`): salto entre días
     contiguos de la semana objetivo, medido en el valle (media de `periodos_valle`),
     frente al salto de los mismos días de la referencia anual (R2). Diagnóstico `saltos`.
   - `CRECIMIENTO_FUERA_DE_TENDENCIA` (`banda_crecimiento_pp`): crecimiento de la
     semana vs R2 menos el interanual reciente Y (`crecimiento.py`). El indicador
     `diagnostics.crecimiento` se publica siempre.
   - `ACUMULADO_ANUAL_FUERA_RANGO` (`cupo_dias_anuales`, `banda_error_pct`): usa
     `acumulado.calcular_acumulado_anual` con las aprobadas del año (kw
     `aprobada_anual` o `ctx.extras["acumulado_aprobada"]`, {fecha: MWh}).

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
from datetime import date, timedelta
from typing import Any, Dict, List, Mapping, Optional

import numpy as np

from src.adjustments import acumulado, analogos, crecimiento
from src.adjustments.data import curvas_reales, dias_semana, energia_curva, iso
from src.adjustments.referencia_reciente import fecha_r2
from src.adjustments.steps import _params
from src.adjustments.steps import nivel as paso_nivel
from src.adjustments.types import MOTIVOS_CURVA, Alert, Curves, RunContext, StepResult, parse_date, validar_curva

DEFAULTS_BASE: Dict[str, Any] = {"banda_pp": 1.0, "top_dias": 3, "banda_referencia_pp": 2.0}
# Alertas Jamer (None = apagadas). `validar_params` solo las devuelve si el llamador las pasó
# (los params guardados de corridas previas no cambian); `step` completa estos defaults.
DEFAULTS_JAMER: Dict[str, Any] = {
    "banda_salto_mw": None,
    "periodos_valle": [3, 4, 5],
    "banda_crecimiento_pp": None,
    "crecimiento_ventana_dias": crecimiento.DEFAULTS_G["g_ventana_dias"],  # misma ventana que el g de nivel
    "crecimiento_min_pares": crecimiento.DEFAULTS_G["g_min_pares"],
    "cupo_dias_anuales": None,
    "banda_error_pct": 5.0,
}
DEFAULTS: Dict[str, Any] = {**DEFAULTS_BASE, **DEFAULTS_JAMER}
TOLERANCIA_RECONCILIACION = 1e-6


def validar_params(params: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    banda_pp > 0; top_dias entero en [1, 7]; banda_referencia_pp None o > 0. Las claves de
    alertas Jamer se validan y se devuelven solo si vienen en `params` (ver `DEFAULTS_JAMER`).
    """
    pedidas = {k for k in (params or {}) if k in DEFAULTS_JAMER} if isinstance(params, dict) else set()
    p = _params.completar("control", DEFAULTS, params)
    p["banda_pp"] = _params.numero("control", "banda_pp", p["banda_pp"], minimo=0.0, minimo_exclusivo=True)
    p["top_dias"] = _params.entero("control", "top_dias", p["top_dias"], minimo=1, maximo=7)
    if p["banda_referencia_pp"] is not None:
        p["banda_referencia_pp"] = _params.numero(
            "control", "banda_referencia_pp", p["banda_referencia_pp"], minimo=0.0, minimo_exclusivo=True
        )
    _validar_jamer(p)
    return {k: v for k, v in p.items() if k in DEFAULTS_BASE or k in pedidas}


def _validar_jamer(p: Dict[str, Any]) -> None:
    if p["banda_salto_mw"] is not None:
        p["banda_salto_mw"] = _params.numero(
            "control", "banda_salto_mw", p["banda_salto_mw"], minimo=0.0, minimo_exclusivo=True
        )
    valle = _params.lista_enteros("control", "periodos_valle", p["periodos_valle"], minimo=1, maximo=24, max_items=24)
    if not valle:
        raise ValueError("control.periodos_valle no puede ser vacía")
    p["periodos_valle"] = sorted(set(valle))
    if p["banda_crecimiento_pp"] is not None:
        p["banda_crecimiento_pp"] = _params.numero(
            "control", "banda_crecimiento_pp", p["banda_crecimiento_pp"], minimo=0.0, minimo_exclusivo=True
        )
    p["crecimiento_ventana_dias"] = _params.entero(
        "control", "crecimiento_ventana_dias", p["crecimiento_ventana_dias"], minimo=7, maximo=730
    )
    p["crecimiento_min_pares"] = _params.entero(
        "control", "crecimiento_min_pares", p["crecimiento_min_pares"], minimo=1, maximo=730
    )
    if p["cupo_dias_anuales"] is not None:
        p["cupo_dias_anuales"] = _params.entero("control", "cupo_dias_anuales", p["cupo_dias_anuales"], minimo=0, maximo=366)
    p["banda_error_pct"] = _params.numero(
        "control", "banda_error_pct", p["banda_error_pct"], minimo=0.0, minimo_exclusivo=True
    )


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
    reales: Mapping[str, float],
) -> None:
    """Diagnóstico: tendencia resultante vs la de referencia (Σ E*(d) vs semana previa). `reales` = energía diaria real."""
    diag.update(
        energia_semana_previa=None, tendencia_resultante_pct=None, tendencia_esperada_pct=None,
        energia_semana_esperada=None, diferencia_tendencia_pp=None, dias_sin_evidencia=[],
        semana_ref_inicio=iso(ctx.semana_ref_inicio) if ctx.semana_ref_inicio else None,
    )
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

    La lluvia se mide en periodos con lluvia (`p_i` es un código de condición del
    tiempo, no mm; ver `clima.py`): suma semanal de periodos con lluvia y fracción
    máxima diaria, solo de días con calidad suficiente.

    Warning si la fracción diaria máxima de lluvia de la semana supera el p90 histórico o
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
    fiables = {f: c for f, c in dias.items() if c.get("calidad_lluvia") == "suficiente"}
    lluvias = [v for c in fiables.values() if (v := _finito(c.get("frac_lluvia"))) is not None]
    periodos_lluvia = sum(int(c.get("lluvia_periodos") or 0) for c in fiables.values()) if lluvias else None
    periodos_conocidos = sum(int(c.get("periodos_conocidos") or 0) for c in fiables.values()) if lluvias else None
    lluvia_frac_semana = (periodos_lluvia / periodos_conocidos) if periodos_conocidos else None
    if not temps and not lluvias:
        return

    hist_temps: List[float] = []
    hist_lluvias: List[float] = []
    for f, c in clima.items():
        if f in semana or not isinstance(c, dict):
            continue
        t = _finito(c.get("temp_media"))
        v = _finito(c.get("frac_lluvia")) if c.get("calidad_lluvia") == "suficiente" else None
        if t is not None:
            hist_temps.append(t)
        if v is not None:
            hist_lluvias.append(v)

    temp_media = sum(temps) / len(temps) if temps else None
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
    if lluvia_max is not None:
        partes.append(f"{periodos_lluvia} periodos con lluvia ({(lluvia_frac_semana or 0.0) * 100:.0f} %)")
    alerts.append(Alert(
        nivel, "CLIMA_OBSERVADO",
        "Clima observado de la semana: " + ", ".join(partes),
        list(dias),
        detalle={
            "temp_media": temp_media,
            "lluvia_periodos": periodos_lluvia,
            "lluvia_frac_semana": lluvia_frac_semana,
            "lluvia_frac_max_diaria": lluvia_max,
            "p90_lluvia_frac": p90_lluvia,
            "temp_media_historica": temp_hist,
            "sigma_temp": sigma_temp,
            "dias": {f: dict(c) for f, c in dias.items()},
        },
    ))


def _r2_por_dia(
    ctx: RunContext,
    semana: list,
    fechas: List[str],
    referencias: Optional[Mapping[str, Any]],
) -> Dict[str, Optional[date]]:
    """
    Fecha de R2 de cada día objetivo (None si no hay). Con `referencias` del paso nivel (perfil Jamer) se usa
    exactamente la R2 de nivel (misma definición, con homólogo de festivos); sin ellas, el control estándar:
    `fecha_r2` sin homólogo (= semana_ref_inicio + i), None sin semana de referencia.
    """
    if referencias:
        out: Dict[str, Optional[date]] = {}
        for f in fechas:
            r2 = (referencias.get(f) or {}).get("r2")
            out[f] = parse_date(r2["fecha"]) if r2 and r2.get("fecha") else None
        return out
    return {f: fecha_r2(d, ctx.semana_inicio, ctx.semana_ref_inicio) for d, f in zip(semana, fechas)}


def _valle(curva: Any, periodos: List[int]) -> Optional[float]:
    """Media de los periodos `periodos` (1..24) de una curva válida; None si no lo es."""
    if validar_curva(curva) is not None:
        return None
    arr = np.asarray(curva, dtype=float)
    return float(np.mean([arr[p - 1] for p in periodos]))


def _curva_r2(curvas: Mapping[str, Any], r2: Optional[date]) -> Any:
    return None if r2 is None else curvas.get(iso(r2))


def _saltos(
    prev: Curves,
    fechas: List[str],
    banda: Optional[float],
    periodos: List[int],
    diag: Dict[str, Any],
    alerts: List[Alert],
    curvas: Mapping[str, Any],
    r2: Mapping[str, Optional[date]],
    hay_referencia: bool,
) -> None:
    """Salto en el valle entre días contiguos de la semana vs el salto de la referencia anual (R2)."""
    diag["saltos"] = {"banda_mw": banda, "periodos_valle": list(periodos), "pares": [], "motivo": None}
    if not hay_referencia:
        diag["saltos"]["motivo"] = "sin semana de referencia"
        return
    pares = []
    for i in range(1, len(fechas)):
        ref_a = _curva_r2(curvas, r2[fechas[i - 1]])
        ref_b = _curva_r2(curvas, r2[fechas[i]])
        v_a, v_b = _valle(prev.get(fechas[i - 1]), periodos), _valle(prev.get(fechas[i]), periodos)
        r_a, r_b = _valle(ref_a, periodos), _valle(ref_b, periodos)
        if None in (v_a, v_b, r_a, r_b):
            continue
        delta_final, delta_ref = v_b - v_a, r_b - r_a
        pares.append({
            "de": fechas[i - 1], "a": fechas[i],
            "delta_final": delta_final, "delta_ref": delta_ref, "dif_mw": delta_final - delta_ref,
        })
    diag["saltos"]["pares"] = pares
    if not pares:
        diag["saltos"]["motivo"] = "sin pares comparables (faltan curvas finales o de referencia)"
        return
    if banda is None:
        return
    fuera = [q for q in pares if abs(q["dif_mw"]) > banda]
    if fuera:
        fuera.sort(key=lambda q: abs(q["dif_mw"]), reverse=True)
        alerts.append(Alert(
            "info", "SALTO_ENTRE_DIAS",
            f"{len(fuera)} salto(s) entre días contiguos (valle P{periodos[0]}–P{periodos[-1]}) difieren "
            f"más de {banda:g} MW del salto de la referencia anual",
            sorted({q["a"] for q in fuera}),
            detalle={"banda_mw": banda, "periodos_valle": list(periodos), "pares": fuera},
        ))


def _crecimiento(
    ctx: RunContext,
    fechas: List[str],
    e_fin: Dict[str, Optional[float]],
    p: Dict[str, Any],
    top: int,
    diag: Dict[str, Any],
    alerts: List[Alert],
    curvas: Mapping[str, Any],
    energia: Mapping[str, float],
    r2: Mapping[str, Optional[date]],
    hay_referencia: bool,
) -> None:
    """Indicador (P/R2 − 1) por día y semana (siempre) y alerta si supera la tendencia reciente Y."""
    diag["crecimiento"] = {
        "semana_pct": None, "Y_pct": None, "diferencia_pp": None, "dias": {},
        "ventana": None, "n_pares": None, "fallback": None, "banda_pp": p["banda_crecimiento_pp"], "motivo": None,
    }
    d = diag["crecimiento"]
    if not hay_referencia:
        d["motivo"] = "sin semana de referencia"
        return
    e_ref = {f: (energia.get(iso(r2[f])) if r2[f] is not None else None) for f in fechas}
    dias = {}
    for f in fechas:
        if e_fin[f] is not None and e_ref[f] is not None and e_ref[f] > 0:
            dias[f] = {"energia": e_fin[f], "energia_ref": e_ref[f], "crecimiento_pct": (e_fin[f] / e_ref[f] - 1) * 100}
    d["dias"] = dias
    comparables = list(dias)
    if len(comparables) < len(fechas):
        d["motivo"] = "faltan días de la semana final o de la referencia"
        return
    tot_fin = sum(e_fin[f] for f in fechas)
    tot_ref = sum(e_ref[f] for f in fechas)
    crec_sem = (tot_fin / tot_ref - 1) * 100
    d["semana_pct"] = crec_sem

    cr = crecimiento.crecimiento_interanual(
        curvas, ctx.fecha_corte, ctx.festivos,
        ventana_dias=p["crecimiento_ventana_dias"], min_pares=p["crecimiento_min_pares"],
    )
    d.update(ventana=list(cr.ventana), n_pares=cr.n_pares, fallback=cr.fallback)
    if cr.Y is None:
        d["motivo"] = "sin pares interanuales suficientes para la tendencia reciente"
        return
    y_pct = cr.Y * 100
    dif = crec_sem - y_pct
    d.update(Y_pct=y_pct, diferencia_pp=dif)
    for f in fechas:
        dias[f]["diferencia_pp"] = dias[f]["crecimiento_pct"] - y_pct
    banda = p["banda_crecimiento_pp"]
    if banda is not None and dif > banda:
        culpables = sorted(fechas, key=lambda f: dias[f]["diferencia_pp"], reverse=True)[:top]
        alerts.append(Alert(
            "info", "CRECIMIENTO_FUERA_DE_TENDENCIA",
            f"La semana crece {crec_sem:.2f} % frente a la referencia anual y la tendencia reciente es "
            f"{y_pct:.2f} % (diferencia {dif:+.2f} pp, banda {banda:g} pp)",
            culpables,
            detalle={"semana_pct": crec_sem, "Y_pct": y_pct, "diferencia_pp": dif, "banda_pp": banda,
                     "dias": {f: dias[f] for f in culpables}},
        ))


def _acumulado(
    ctx: RunContext,
    aprobada: Optional[Mapping[str, float]],
    p: Dict[str, Any],
    diag: Dict[str, Any],
    alerts: List[Alert],
    energia: Mapping[str, float],
) -> None:
    """Alerta de acumulado anual de días fuera de banda (apagada sin `cupo_dias_anuales`)."""
    diag["acumulado_anual"] = None
    if p["cupo_dias_anuales"] is None:
        return
    if not aprobada:
        diag["acumulado_anual"] = {"estado": "sin_datos", "motivo": "sin energías aprobadas del año"}
        return
    res = acumulado.calcular_acumulado_anual(
        aprobada, energia, ctx.semana_inicio.year,
        banda_pct=p["banda_error_pct"], cupo_dias=p["cupo_dias_anuales"], hasta=iso(ctx.fecha_corte),
    )
    alerta = res.pop("alerta")
    diag["acumulado_anual"] = res
    if alerta:
        alerts.append(Alert(**alerta))


def step(
    ctx: RunContext,
    prev: Curves,
    params: Dict[str, Any],
    *,
    energia_esperada: Optional[Dict[str, float]] = None,
    energia_nivel: Optional[Dict[str, float]] = None,
    nivel_dias: Optional[Dict[str, Dict[str, Any]]] = None,
    aprobada_anual: Optional[Mapping[str, float]] = None,
    referencias: Optional[Mapping[str, Any]] = None,
) -> StepResult:
    """
    `energia_esperada`: energía diaria final del paso nivel (reconciliación de la forma).
    `energia_nivel`: energía diaria de referencia para el enclavamiento (calculada por
    nivel, pre-edición; o la del paso base si nivel se omitió). None → no se evalúa.
    `nivel_dias`: `diagnostics["dias"]` del paso nivel (E*(d)) para el diagnóstico
    de tendencia; None → se recalcula E*.
    `aprobada_anual`: {fecha: MWh aprobados} del año para `ACUMULADO_ANUAL_FUERA_RANGO`;
    None → `ctx.extras["acumulado_aprobada"]`.
    `referencias`: `diagnostics["referencias"]` del paso nivel en modo `reciente` (perfil Jamer). Si llega, R2 de
    `saltos`/`crecimiento` es la de nivel (`referencia_reciente.fecha_r2`); si no (control estándar), R2 es
    `semana_ref_inicio + i`, sin reemplazo por festivo homólogo.
    """
    p = {**DEFAULTS_JAMER, **validar_params(params)}
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

    curvas = curvas_reales(ctx.historico)  # una sola lectura del histórico para todos los diagnósticos
    energia = {f: float(v.sum()) for f, v in curvas.items()}
    r2 = _r2_por_dia(ctx, semana, fechas, referencias)
    hay_referencia = bool(referencias) or bool(ctx.semana_ref_inicio)

    _brecha(ctx, prev, diag, alerts)
    _alertas_festivos(ctx, fechas, alerts)
    _alertas_eventos(ctx, fechas, alerts)
    _alertas_clima(ctx, fechas, alerts)
    _enclavamiento(fechas, e_fin, energia_nivel, p["banda_pp"], p["top_dias"], diag, alerts)
    _tendencia_referencia(ctx, prev, semana, fechas, e_fin, nivel_dias,
                          p["banda_referencia_pp"], p["top_dias"], diag, alerts, energia)
    _saltos(prev, fechas, p["banda_salto_mw"], p["periodos_valle"], diag, alerts, curvas, r2, hay_referencia)
    _crecimiento(ctx, fechas, e_fin, p, p["top_dias"], diag, alerts, curvas, energia, r2, hay_referencia)
    _acumulado(ctx, aprobada_anual if aprobada_anual is not None else (ctx.extras or {}).get("acumulado_aprobada"),
               p, diag, alerts, energia)

    return StepResult(curves={f: list(v) for f, v in prev.items()}, diagnostics=diag, alerts=alerts)
