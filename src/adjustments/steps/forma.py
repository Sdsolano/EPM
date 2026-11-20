"""
Paso 3 — forma: reemplaza la forma horaria conservando la energía diaria del
paso anterior (suma diaria idéntica, tolerancia 1e-6).

curva(d) = [(1 − β)·forma_previa(d) + β·forma_fuente(d)] · E(d)
con formas normalizadas (suman 1) y E(d) = suma de la curva previa.

Fuente de forma por día:
- festivo → `average_holiday_hourly_profile` (festivos homólogos), usado SOLO
  como forma normalizada. Sin homólogos → domingo real más reciente.
- resto, según `params["fuente"]`:
  * "reciente": último día real <= corte con el MISMO día de semana
    y no festivo (más estricto que "mismo tipo": un lunes tiene una
    madrugada distinta a un miércoles).
  * "referencia": día equivalente de la semana de referencia (d − desfase);
    si no tiene datos o es festivo → cae a "reciente" con alerta info.
  * "promedio_n" (default, calibrado en el backtest 2026-11-12): promedio de
    las formas de los últimos n días reales del mismo día de semana, no
    festivos (n default 4).
β default 1.0 (forma de la fuente pura).

Días análogos (Fase 2b, R3; lógica compartida en `steps/_analogos.py`):
- `fuente="analogos"`: forma = promedio ponderado por score (sin suavizar
  picos) de las formas normalizadas de los análogos del día:
  * día con entrada en `params["analogos"]` → esas fechas (selección
    "analista"; pesos = score de `analogos.puntuar_fechas`; las no observadas
    se omiten con alerta info `ANALOGOS_NO_OBSERVADOS`);
  * día sin entrada → búsqueda automática `analogos.seleccionar` con
    `filtros_analogos` (o los automáticos `_analogos.FILTROS_AUTO`: mismo día
    de semana y misma relación/distancia con festivo) + años excluidos de la
    corrida y `limite = n` (selección "automatica");
  * festivo SIN entrada en el mapa → perfil homólogo (igual que las otras fuentes);
  * sin análogos observados → `promedio_n` con alerta `FORMA_FUENTE_ALTERNA`
    (motivo `sin_analogos`).
  Diagnóstico por día: `fechas_fuente`, `pesos`, `seleccion`.

Réplica de Jamer (E3, opt-in; con los defaults todo queda como antes):
- `fuente="reciente_clase"`: forma del promedio de los `n` días reales más
  recientes del MISMO tipo de día (`referencia_reciente.seleccionar_recientes`:
  un día normal salta festivos y post-festivos; festivo/post-festivo usa el
  último de su clase). Sin recientes cae a `promedio_n` (motivo
  `sin_recientes_clase`).
- `delta="plano"`: en vez de escalar proporcionalmente, la forma se aplica al
  total del MODELO (`energia_base`, S_h = forma_h·E_base) y la diferencia con la
  energía de nivel E se reparte como un delta uniforme en MW a los periodos
  "con gancho": curva_h = S_h + g_h·δ, δ = (E − E_base)/Σg. Σ = E.
- `tramos="modelo_anual"` (experimental, apagado): quita del gancho (g_h = 0) los
  tramos donde el modelo se desvía de R1 en el mismo sentido que el cambio anual
  (R2 − R3); ver `_tramos_sin_gancho`.
- Params nuevos: `analogos` (None | {"YYYY-MM-DD": ["YYYY-MM-DD", ...]}, default
  None; claves en la semana objetivo, fechas <= corte y distintas del día, a lo
  sumo 60 por día) y `filtros_analogos` (None | filtros del buscador, default
  None). Con otra fuente se aceptan (validados) pero no cambian nada; con su
  default no se guardan en los params del record.
"""

from __future__ import annotations

import math
from datetime import date, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.adjustments.data import PERIODOS, ancla_mismo_dow, curvas_reales, dias_semana, iso
from src.adjustments.referencia_reciente import fecha_r2, fecha_r3, seleccionar_recientes, unir_alertas
from src.adjustments.steps import _analogos, _params
from src.adjustments.types import N_PERIODOS, Alert, Curves, RunContext, StepResult, validar_curva
from src.prediction.holiday_homology import average_holiday_hourly_profile

FUENTES = ("reciente", "referencia", "promedio_n", "analogos", "reciente_clase")
DEFAULTS: Dict[str, Any] = {"fuente": "promedio_n", "n": 4, "beta": 1.0}
DELTAS = ("proporcional", "plano")
TRAMOS = ("off", "modelo_anual")
# Params de la réplica: default = comportamiento de hoy; solo se persisten si difieren.
DEFAULTS_JAMER: Dict[str, Any] = {
    "delta": "proporcional",
    "tramos": "off",
    "umbral_tramo_mw": 20.0,
    "min_periodos_tramo": 3,
    "suavizado_borde": 0,
    "saltar_puente": False,
}
MAX_SEMANAS_BUSQUEDA = 12
MAX_N = 60  # más de ~1 año de días del mismo día de semana no aporta y alarga la búsqueda


def validar_params(params: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    fuente en FUENTES; n entero en [1, MAX_N]; beta número finito en [0, 1].
    analogos / filtros_analogos: ver `_analogos.separar` (solo se devuelven si
    difieren de su default None).
    """
    params, extra_analogos = _analogos.separar("forma", params, _analogos.DEFAULTS_FORMA)
    # con fuente="reciente_clase" (perfil Jamer) se persisten todos los params de la réplica (ver `_separar_jamer`)
    completo = isinstance(params, dict) and params.get("fuente") == "reciente_clase"
    params, extra_jamer = _separar_jamer(params, completo=completo)
    p = _params.completar("forma", DEFAULTS, params)
    p["fuente"] = _params.opcion("forma", "fuente", p["fuente"], FUENTES)
    p["n"] = _params.entero("forma", "n", p["n"], minimo=1, maximo=MAX_N)
    p["beta"] = _params.numero("forma", "beta", p["beta"], minimo=0.0, maximo=1.0)
    p.update(extra_analogos)
    p.update(extra_jamer)
    return p


def _separar_jamer(params: Any, completo: bool = False) -> Tuple[Any, Dict[str, Any]]:
    """
    Separa y valida los params de la réplica (delta, tramos, ...) → (resto, validados).
    `validados` solo trae los que difieren de su default (idempotente, igual que
    `_analogos.separar`), salvo con `completo=True` (fuente `reciente_clase`): trae todos, para que los valores
    efectivos queden persistidos aunque después cambien los defaults del código. `tramos` distinto de "off" exige `delta="plano"`.
    """
    if not isinstance(params, dict):
        return params, {}
    resto = {k: v for k, v in params.items() if k not in DEFAULTS_JAMER}
    q = {**DEFAULTS_JAMER, **{k: v for k, v in params.items() if k in DEFAULTS_JAMER}}
    q["delta"] = _params.opcion("forma", "delta", q["delta"], DELTAS)
    q["tramos"] = _params.opcion("forma", "tramos", q["tramos"], TRAMOS)
    q["umbral_tramo_mw"] = _params.numero("forma", "umbral_tramo_mw", q["umbral_tramo_mw"],
                                          minimo=0.0, maximo=10000.0)
    q["min_periodos_tramo"] = _params.entero("forma", "min_periodos_tramo", q["min_periodos_tramo"],
                                             minimo=1, maximo=24)
    q["suavizado_borde"] = _params.entero("forma", "suavizado_borde", q["suavizado_borde"],
                                          minimo=0, maximo=12)
    q["saltar_puente"] = _params.booleano("forma", "saltar_puente", q["saltar_puente"])
    if q["tramos"] != "off" and q["delta"] != "plano":
        raise ValueError("forma: tramos distinto de 'off' requiere delta='plano'")
    return resto, q if completo else {k: v for k, v in q.items() if v != DEFAULTS_JAMER[k]}


def _normalizar(v: np.ndarray) -> Optional[np.ndarray]:
    total = float(np.sum(v))
    if not math.isfinite(total) or total <= 0:
        return None
    return np.asarray(v, dtype=float) / total


def _pico(v) -> Dict[str, float]:
    arr = np.asarray(v, dtype=float)
    i = int(np.argmax(arr))
    return {"mw": float(arr[i]), "hora": i + 1}


def _dias_mismo_dow(
    d: date, ctx: RunContext, curvas: Dict[str, np.ndarray], festivos: set, n: int
) -> Dict[str, np.ndarray]:
    """
    Hasta n días reales <= corte, mismo día de semana que d, no festivos
    (recientes primero) → {fecha: forma normalizada}.
    """
    primera = ancla_mismo_dow(d, ctx.fecha_corte)
    encontrados: Dict[str, np.ndarray] = {}
    for k in range(MAX_SEMANAS_BUSQUEDA + n):
        f = iso(primera - timedelta(days=7 * k))
        forma = _normalizar(curvas[f]) if f in curvas and f not in festivos else None
        if forma is not None:
            encontrados[f] = forma
            if len(encontrados) >= n:
                break
    return encontrados


def _forma_fuente(
    d: date,
    fuente: str,
    n: int,
    ctx: RunContext,
    curvas: Dict[str, np.ndarray],
    festivos: set,
) -> Tuple[Optional[np.ndarray], str, List[str], Optional[str]]:
    """(forma normalizada, fuente usada, fechas fuente, motivo de fallback)."""
    if iso(d) in festivos:
        perfil, meta = average_holiday_hourly_profile(
            ctx.historico, d, ctx.festivos, period_cols=PERIODOS,
            date_column="fecha", scale_with_recent_sundays=False,
        )
        forma = _normalizar(perfil) if meta.get("n_rows", 0) > 0 else None
        if forma is not None:
            return forma, f"festivo_{meta.get('match_mode')}", list(meta.get("match_dates", [])), None
        domingo = d + timedelta(days=6 - d.weekday())  # mismo día de semana que un domingo
        formas = _dias_mismo_dow(domingo, ctx, curvas, festivos, 1)
        if formas:
            return next(iter(formas.values())), "festivo_domingo_reciente", list(formas), "festivo_sin_homologo"
        return None, "sin_forma", [], "festivo_sin_homologo"

    motivo = None
    if fuente == "referencia":
        if ctx.semana_ref_inicio is not None:
            dia_ref = d - (ctx.semana_inicio - ctx.semana_ref_inicio)
            f = iso(dia_ref)
            forma_ref = _normalizar(curvas[f]) if f in curvas and f not in festivos else None
            if forma_ref is not None:
                return forma_ref, "referencia", [f], None
        motivo = "referencia_no_disponible"
        fuente = "reciente"

    cantidad = n if fuente == "promedio_n" else 1
    formas = _dias_mismo_dow(d, ctx, curvas, festivos, cantidad)
    if not formas:
        return None, "sin_forma", [], motivo or "sin_dias_comparables"
    forma = np.mean(list(formas.values()), axis=0)
    return forma / forma.sum(), fuente, list(formas), motivo


def _forma_reciente_clase(
    d: date,
    n: int,
    saltar_puente: bool,
    ctx: RunContext,
    curvas: Dict[str, np.ndarray],
    festivos: set,
) -> Tuple[Optional[np.ndarray], str, List[str], Optional[str], Dict[str, Any], List[Alert]]:
    """
    fuente="reciente_clase": (forma, fuente usada, fechas, motivo de fallback,
    diagnóstico extra, alertas de la selección).
    """
    rec = seleccionar_recientes(d, curvas, ctx.fecha_corte, festivos, n, saltar_puente=saltar_puente)
    formas = {}
    for fecha in rec.fechas:
        forma = _normalizar(curvas[fecha.isoformat()])
        if forma is not None:
            formas[fecha.isoformat()] = forma
    extra = {"clase": rec.clase, "saltados": [dict(x) for x in rec.saltados]}
    if not formas:
        forma, usada, fechas, _ = _forma_fuente(d, "promedio_n", n, ctx, curvas, festivos)
        return forma, usada, fechas, "sin_recientes_clase", extra, list(rec.alertas)
    forma = np.mean(list(formas.values()), axis=0)
    return forma / forma.sum(), "reciente_clase", list(formas), None, extra, list(rec.alertas)


def _tramos_sin_gancho(
    S: np.ndarray,
    r1: np.ndarray,
    r2: np.ndarray,
    r3: np.ndarray,
    umbral_mw: float,
    min_periodos: int,
    suavizado: int,
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """
    Pesos de gancho g_h (1 = recibe el delta, 0 = se conserva la desviación del
    modelo) y los tramos retenidos. Un periodo es candidato si la desviación del
    modelo (S − R1) y el cambio anual (R2 − R3) tienen el mismo signo (no nulo) y
    |R2 − R3| >= umbral; un tramo son >= `min_periodos` candidatos contiguos. Con
    `suavizado = k` la transición fuera del tramo es una rampa lineal j/(k+1).
    """
    desv = S - r1
    anual = r2 - r3
    cand = (np.sign(desv) * np.sign(anual) > 0) & (np.abs(anual) >= umbral_mw)
    g = np.ones(len(S))
    tramos: List[Dict[str, Any]] = []
    h = 0
    while h < len(S):
        if not cand[h]:
            h += 1
            continue
        fin = h
        while fin + 1 < len(S) and cand[fin + 1]:
            fin += 1
        if fin - h + 1 >= min_periodos:
            g[h:fin + 1] = 0.0
            for j in range(1, suavizado + 1):
                peso = j / (suavizado + 1)
                for idx in (h - j, fin + j):
                    if 0 <= idx < len(S):
                        g[idx] = min(g[idx], peso)
            media = float(np.mean(desv[h:fin + 1]))
            tramos.append({
                "desde": h + 1,
                "hasta": fin + 1,
                "desv_mw": media,
                "desv_mw_10": float(math.floor(media / 10.0 + 0.5) * 10.0),
            })
        h = fin + 1
    return g, tramos


def _g_tramos(
    d: date,
    S: np.ndarray,
    p: Dict[str, Any],
    saltar_puente: bool,
    ctx: RunContext,
    curvas: Dict[str, np.ndarray],
    festivos: set,
) -> Tuple[np.ndarray, List[Dict[str, Any]], Optional[str]]:
    """(g, tramos retenidos, motivo si no se pudo evaluar). R1 = reciente de su clase, R2 = `fecha_r2` (d − desfase de la semana de referencia; sin ella d−364), R3 = R1−364."""
    sin = np.ones(N_PERIODOS)
    if iso(d) in festivos:
        return sin, [], "festivo"
    rec = seleccionar_recientes(d, curvas, ctx.fecha_corte, festivos, 1, saltar_puente=saltar_puente)
    if not rec.fechas:
        return sin, [], "sin_r1"
    r1_fecha = rec.fechas[0]
    r1 = curvas.get(iso(r1_fecha))
    r2_fecha = fecha_r2(d, ctx.semana_inicio, ctx.semana_ref_inicio, None, anual_si_falta=True)
    r2 = curvas.get(iso(r2_fecha))
    r3 = curvas.get(iso(fecha_r3(r1_fecha)))
    if r1 is None or r2 is None or r3 is None:
        return sin, [], "sin_r2_r3"
    g, tramos = _tramos_sin_gancho(
        S, np.asarray(r1, dtype=float), np.asarray(r2, dtype=float), np.asarray(r3, dtype=float),
        float(p.get("umbral_tramo_mw", DEFAULTS_JAMER["umbral_tramo_mw"])),
        int(p.get("min_periodos_tramo", DEFAULTS_JAMER["min_periodos_tramo"])),
        int(p.get("suavizado_borde", DEFAULTS_JAMER["suavizado_borde"])),
    )
    return g, tramos, None


def _curva_plana(
    forma: np.ndarray, energia: float, energia_base: float, g: np.ndarray
) -> Tuple[np.ndarray, float]:
    """
    S_h = forma_h·E_base; delta uniforme δ = (E − E_base)/Σg sobre los periodos
    con gancho. Un periodo que quedaría < 0 se recorta a 0 y la diferencia se
    reparte proporcionalmente entre los demás (Σ = E se conserva).
    Con Σg = 0 (nadie recibe el delta) la forma se escala proporcional a E.
    """
    S = forma * energia_base
    suma_g = float(np.sum(g))
    if suma_g <= 0:
        return forma * energia, 0.0
    delta = (energia - energia_base) / suma_g
    curva = S + g * delta
    if (curva < 0).any():
        curva = np.clip(curva, 0.0, None)
        total = float(curva.sum())
        if total > 0:
            curva = curva * (energia / total)
    return curva, float(delta)


def _forma_analogos(
    d: date,
    n: int,
    ctx: RunContext,
    curvas: Dict[str, np.ndarray],
    festivos: set,
    mapa: Optional[Dict[str, List[str]]],
    filtros: Dict[str, Any],
    no_observadas: Dict[str, List[str]],
) -> Tuple[Optional[np.ndarray], str, List[str], Optional[str], Dict[str, Any]]:
    """fuente="analogos": (forma, fuente usada, fechas, motivo de fallback, diagnóstico extra)."""
    f = iso(d)
    if f in festivos and not (mapa and f in mapa):
        forma, usada, fechas, motivo = _forma_fuente(d, "promedio_n", n, ctx, curvas, festivos)
        return forma, usada, fechas, motivo, {"seleccion": "homologo_festivo", "pesos": None}
    sel = _analogos.seleccion_dia(ctx, d, mapa, filtros)
    if sel["omitidas"]:
        no_observadas[f] = sel["omitidas"]
    forma, fechas, pesos = _analogos.forma_ponderada(sel["items"])
    if forma is not None:
        return forma, "analogos", fechas, None, {"seleccion": sel["seleccion"], "pesos": pesos}
    forma, usada, fechas, _ = _forma_fuente(d, "promedio_n", n, ctx, curvas, festivos)
    return forma, usada, fechas, "sin_analogos", {"seleccion": sel["seleccion"], "pesos": None}


def step(
    ctx: RunContext,
    prev: Curves,
    params: Dict[str, Any],
    energia_base: Optional[Dict[str, float]] = None,
) -> StepResult:
    """
    `energia_base` {fecha: Σ del modelo (paso base)}: solo la usa `delta="plano"`
    (la pasa el pipeline desde `nivel.diagnostics.dias[f]["E_base"]`).
    """
    p = validar_params(params)
    fuente, n, beta = p["fuente"], p["n"], p["beta"]
    delta_modo = p.get("delta", DEFAULTS_JAMER["delta"])
    plano = delta_modo == "plano"
    tramos_modo = p.get("tramos", DEFAULTS_JAMER["tramos"])
    saltar_puente = p.get("saltar_puente", DEFAULTS_JAMER["saltar_puente"])
    alertas_sel: List[Alert] = []
    sin_energia_base: List[str] = []
    sin_gancho: List[str] = []
    if fuente == "analogos":
        _analogos.validar_mapa_ctx("forma", p.get("analogos"), ctx)
    filtros_analogos = (
        _analogos.filtros_efectivos(p.get("filtros_analogos"), _analogos.anios_excluidos_ctx(ctx), n)
        if fuente == "analogos" else None
    )
    no_observadas: Dict[str, List[str]] = {}

    curves: Curves = {f: list(v) for f, v in prev.items()}
    curvas = curvas_reales(ctx.historico)
    festivos = set(ctx.festivos)
    por_dia: Dict[str, Any] = {}
    fallbacks: List[str] = []
    sin_forma: List[str] = []

    for d in dias_semana(ctx.semana_inicio):
        f = iso(d)
        if f not in prev or validar_curva(prev[f]) is not None:
            continue  # faltante o inválida: se deja igual y el control la bloquea
        curva_prev = np.asarray(prev[f], dtype=float)
        energia = float(curva_prev.sum())
        forma_prev = _normalizar(curva_prev)
        extra_dia: Dict[str, Any] = {}
        if fuente == "analogos":
            forma_src, usada, fechas, motivo, extra_dia = _forma_analogos(
                d, n, ctx, curvas, festivos, p.get("analogos"), filtros_analogos, no_observadas
            )
        elif fuente == "reciente_clase":
            forma_src, usada, fechas, motivo, extra_dia, alertas = _forma_reciente_clase(
                d, n, saltar_puente, ctx, curvas, festivos
            )
            alertas_sel.extend(alertas)
        else:
            forma_src, usada, fechas, motivo = _forma_fuente(d, fuente, n, ctx, curvas, festivos)

        if forma_src is None or forma_prev is None:
            sin_forma.append(f)
            nueva = curva_prev
        else:
            forma = (1 - beta) * forma_prev + beta * forma_src
            forma = forma / forma.sum()
            if plano:
                e_base = (energia_base or {}).get(f)
                if e_base is None or not math.isfinite(float(e_base)) or float(e_base) <= 0:
                    sin_energia_base.append(f)
                    e_base = energia
                e_base = float(e_base)
                g = np.ones(N_PERIODOS)
                tramos_ret: List[Dict[str, Any]] = []
                motivo_tramos = None
                if tramos_modo != "off":
                    g, tramos_ret, motivo_tramos = _g_tramos(
                        d, forma * e_base, p, saltar_puente, ctx, curvas, festivos
                    )
                nueva, delta = _curva_plana(forma, energia, e_base, g)
                if float(np.sum(g)) <= 0:
                    sin_gancho.append(f)
                extra_dia = {
                    **extra_dia,
                    "energia_base": e_base,
                    "delta_plano_mw": delta,
                    "periodos_con_gancho": [h + 1 for h in range(N_PERIODOS) if g[h] > 0],
                    "tramos_retenidos": tramos_ret,
                }
                if motivo_tramos:
                    extra_dia["tramos_motivo"] = motivo_tramos
            else:
                nueva = forma * energia
            curves[f] = nueva.tolist()
        if motivo:
            fallbacks.append(f)

        por_dia[f] = {
            "fuente": usada,
            "fechas_fuente": fechas,
            "fallback": motivo,
            "energia": energia,
            "pico_antes": _pico(curva_prev),
            "pico_despues": _pico(nueva),
            "diferencia_energia": float(np.sum(nueva) - energia),
        }
        por_dia[f].update(extra_dia)

    alerts: List[Alert] = []
    if fallbacks:
        alerts.append(
            Alert(
                nivel="info",
                codigo="FORMA_FUENTE_ALTERNA",
                mensaje="Algunos días usaron una fuente de forma alterna a la pedida",
                fechas=fallbacks,
                detalle={f: por_dia[f]["fallback"] for f in fallbacks},
            )
        )
    if sin_forma:
        alerts.append(
            Alert(
                nivel="warning",
                codigo="SIN_FORMA",
                mensaje="No se encontró forma de referencia; se conserva la curva previa",
                fechas=sin_forma,
            )
        )

    alerts.extend(unir_alertas(alertas_sel))
    if sin_energia_base:
        alerts.append(
            Alert(
                nivel="info",
                codigo="FORMA_SIN_ENERGIA_BASE",
                mensaje="Sin energía del modelo para el delta plano; se escaló proporcional a la energía de nivel",
                fechas=sin_energia_base,
            )
        )
    if sin_gancho:
        alerts.append(
            Alert(
                nivel="info",
                codigo="FORMA_SIN_GANCHO",
                mensaje="Ningún periodo con gancho para el delta plano; se escaló proporcional",
                fechas=sin_gancho,
            )
        )

    alerta_no_obs = _analogos.alerta_no_observadas(no_observadas)
    if alerta_no_obs is not None:
        alerts.append(alerta_no_obs)

    diagnostics = {"fuente": fuente, "n": n, "beta": beta, "dias": por_dia}
    if plano:
        diagnostics["delta"] = delta_modo
        diagnostics["tramos"] = tramos_modo
    if filtros_analogos is not None:
        diagnostics["filtros_analogos"] = filtros_analogos
    return StepResult(curves=curves, diagnostics=diagnostics, alerts=alerts)
