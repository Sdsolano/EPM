"""B2 (noche 2): librería de experimentos (numpy puro) sobre el cache de `noche2_b2_datos.py`."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

TIPOS = ["laboral", "sabado", "domingo", "festivo"]
FRAC_CAL = 0.6


@dataclass
class DS:
    ucp: str
    comp: Dict[str, np.ndarray]          # nombre -> [W,7,24]
    real: np.ndarray                     # [W,7,24]
    tipo: np.ndarray                     # [W,7] str
    ordn: np.ndarray                     # [W,7] ordinal de fecha
    corte: np.ndarray                    # [W] ordinal del corte
    fechas: np.ndarray                   # [W,7] iso
    lunes: np.ndarray
    festivos: set
    puente: np.ndarray                   # [W,7] bool: laboral adyacente a festivo
    dow: np.ndarray                      # [W,7]
    ncal: int

    @property
    def W(self) -> int:
        return self.real.shape[0]

    def mask(self, particion: str) -> np.ndarray:
        m = np.zeros(self.W, bool)
        if particion == "cal":
            m[: self.ncal] = True
        elif particion == "test":
            m[self.ncal:] = True
        else:
            m[:] = True
        return m

    def seg(self, modo: str) -> np.ndarray:
        """Etiqueta de segmento [W,7]."""
        if modo == "global":
            return np.zeros((self.W, 7), int)
        if modo == "tipo":
            return np.vectorize(TIPOS.index)(self.tipo)
        if modo == "dow8":   # lun..vie laboral = su dow; sab; dom; festivo=7
            s = self.dow.copy()
            s[self.tipo == "festivo"] = 7
            return s
        if modo == "tipo_puente":  # 0 lab, 1 sab, 2 dom, 3 fest, 4 puente
            s = np.vectorize(TIPOS.index)(self.tipo)
            s[(self.tipo == "laboral") & self.puente] = 4
            return s
        raise ValueError(modo)


def cargar(ucp: str, directorio: Path) -> DS:
    z = np.load(Path(directorio) / f"b2_{ucp}.npz", allow_pickle=False)
    comp = {k: z[k] for k in ("base", "nivel", "final", "ana", "ref")}
    fechas = z["fechas"]
    ordn = np.vectorize(lambda f: date.fromisoformat(str(f)).toordinal())(fechas)
    corte = np.array([date.fromisoformat(str(c)).toordinal() for c in z["cortes"]])
    festivos = set(str(f) for f in z["festivos"])
    fo = {date.fromisoformat(f).toordinal() for f in festivos}
    tipo = z["tipo"].astype(str)
    puente = np.zeros(tipo.shape, bool)
    for i in range(tipo.shape[0]):
        for j in range(7):
            o = ordn[i, j]
            if tipo[i, j] == "laboral" and ((o - 1) in fo or (o + 1) in fo):
                puente[i, j] = True
    dow = np.tile(np.arange(7), (tipo.shape[0], 1))
    return DS(ucp, comp, z["real"], tipo, ordn, corte, fechas, z["lunes"], festivos, puente, dow,
              int(round(FRAC_CAL * tipo.shape[0])))


# ---------------------------------------------------------------- métricas
def metricas(pred: np.ndarray, ds: DS, particion: str, tipo: Optional[str] = None) -> Dict[str, float]:
    m = ds.mask(particion)
    p, r, t = pred[m], ds.real[m], ds.tipo[m]
    if tipo is not None:
        sel = t == tipo
    else:
        sel = np.ones_like(t, bool)
    if sel.sum() == 0:
        return dict(n=0, wape=np.nan, mape_dia=np.nan, sesgo_dia=np.nan, mape_sem=np.nan, sesgo_sem=np.nan, err_pico=np.nan)
    pp, rr = p[sel], r[sel]
    wape = 100 * np.abs(pp - rr).sum() / rr.sum()
    ed = pp.sum(1) / rr.sum(1) - 1
    pico = np.abs(pp.max(1) / rr.max(1) - 1)
    # semanal (solo semanas completas: sin filtro de tipo)
    es = p.sum((1, 2)) / r.sum((1, 2)) - 1
    return dict(n=int(sel.sum()), wape=wape, mape_dia=100 * np.abs(ed).mean(), sesgo_dia=100 * ed.mean(),
                mape_sem=100 * np.abs(es).mean(), sesgo_sem=100 * es.mean(), err_pico=100 * pico.mean())


# ---------------------------------------------------------------- corrección de sesgo
def suavizar_horas(f: np.ndarray, k: int) -> np.ndarray:
    if k <= 0:
        return f
    pad = np.pad(f, k, mode="edge")
    ker = np.ones(2 * k + 1) / (2 * k + 1)
    return np.convolve(pad, ker, mode="valid")


def corregir(pred: np.ndarray, ds: DS, seg: str = "tipo", N: int = 8, k0: float = 3.0, g: float = 1.0,
             minimo: int = 3, suav: int = 0, solo_energia: bool = False, N_fest: int = 52,
             estimador: str = "razon", conservar_energia: bool = False,
             pred_hist: Optional[np.ndarray] = None) -> np.ndarray:
    """Corrección multiplicativa por segmento×hora con ventana móvil sin fuga.

    Para la semana w solo usa días con fecha <= corte[w] (error observable). `pred_hist` = pronósticos con los que
    se miden los errores pasados (default: `pred`, o sea el mismo método; para corregir la salida de otro método se pasa).
    """
    W = ds.W
    S = ds.seg(seg)
    ph = pred if pred_hist is None else pred_hist
    flat_p = ph.reshape(-1, 24)
    flat_r = ds.real.reshape(-1, 24)
    flat_o = ds.ordn.reshape(-1)
    flat_s = S.reshape(-1)
    out = pred.copy()
    for w in range(W):
        c = ds.corte[w]
        for s in np.unique(S[w]):
            esp = (seg == "tipo" and s == 3) or (seg == "dow8" and s == 7) or (seg == "tipo_puente" and s in (3, 4))
            ventana = N_fest if esp else N
            m = (flat_o <= c) & (flat_o > c - 7 * ventana) & (flat_s == s)
            n = int(m.sum())
            if n < minimo:
                continue
            P, R = flat_p[m], flat_r[m]
            if solo_energia:
                ratio = np.full(24, R.sum() / P.sum())
            elif estimador == "mediana":
                ratio = np.median(R / np.maximum(P, 1e-9), axis=0)
            else:
                ratio = R.sum(0) / P.sum(0)
            lam = n / (n + k0) * g
            f = 1 + lam * (ratio - 1)
            f = suavizar_horas(f, suav)
            for j in np.where(S[w] == s)[0]:
                v = pred[w, j] * f
                if conservar_energia:
                    v = v * (pred[w, j].sum() / v.sum())
                out[w, j] = v
    return out


# ---------------------------------------------------------------- mezcla ponderada
def mezclar(ds: DS, comps: List[str], seg: str = "tipo", N: int = 13, p: float = 2.0, minimo: int = 3,
            N_fest: int = 52) -> np.ndarray:
    """Mezcla de curvas con pesos ∝ WAPE_i^-p por segmento en ventana móvil (solo días <= corte).

    p = 0 → promedio simple; p grande → casi "el mejor de la ventana". Segmentos sin casos suficientes → promedio simple.
    """
    W = ds.W
    S = ds.seg(seg)
    P = np.stack([ds.comp[c] for c in comps])                      # [K,W,7,24]
    K = len(comps)
    flat_o = ds.ordn.reshape(-1)
    flat_s = S.reshape(-1)
    flat_r = ds.real.reshape(-1, 24)
    err = np.stack([np.abs(P[k].reshape(-1, 24) - flat_r).sum(1) for k in range(K)])  # [K,T]
    den = flat_r.sum(1)
    out = P.mean(0).copy()
    for w in range(W):
        c = ds.corte[w]
        for s in np.unique(S[w]):
            esp = (seg == "tipo" and s == 3) or (seg == "dow8" and s == 7) or (seg == "tipo_puente" and s in (3, 4))
            ventana = N_fest if esp else N
            m = (flat_o <= c) & (flat_o > c - 7 * ventana) & (flat_s == s)
            if m.sum() < minimo:
                continue
            wp = err[:, m].sum(1) / den[m].sum()
            wt = np.power(np.maximum(wp, 1e-6), -p)
            wt = wt / wt.sum()
            for j in np.where(S[w] == s)[0]:
                out[w, j] = np.tensordot(wt, P[:, w, j], axes=1)
    return out


def mejor_en_ventana(ds: DS, comps: List[str], seg: str = "tipo", N: int = 13, minimo: int = 3, N_fest: int = 52,
                     defecto: str = "final") -> np.ndarray:
    """Elige, por segmento y semana, el método con menor WAPE en la ventana móvil (selección automática rodante)."""
    W = ds.W
    S = ds.seg(seg)
    flat_o = ds.ordn.reshape(-1)
    flat_s = S.reshape(-1)
    flat_r = ds.real.reshape(-1, 24)
    err = {c: np.abs(ds.comp[c].reshape(-1, 24) - flat_r).sum(1) for c in comps}
    den = flat_r.sum(1)
    out = ds.comp[defecto].copy()
    for w in range(W):
        c = ds.corte[w]
        for s in np.unique(S[w]):
            esp = (seg == "tipo" and s == 3)
            ventana = N_fest if esp else N
            m = (flat_o <= c) & (flat_o > c - 7 * ventana) & (flat_s == s)
            if m.sum() < minimo:
                continue
            best = min(comps, key=lambda k: err[k][m].sum() / den[m].sum())
            for j in np.where(S[w] == s)[0]:
                out[w, j] = ds.comp[best][w, j]
    return out


def fijar_energia(pred: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Reescala cada día de `pred` para que su energía diaria sea la de `ref` (conserva la forma de `pred`)."""
    return pred * (ref.sum(2, keepdims=True) / pred.sum(2, keepdims=True))


# ---------------------------------------------------------------- intervalos empíricos
def bandas(pred: np.ndarray, ds: DS, q: tuple = (0.10, 0.90), N: int = 26, minimo: int = 8, pool: int = 1,
           seg: str = "tipo", escala: float = 1.0, N_fest: int = 52, pred_hist: Optional[np.ndarray] = None):
    """Bandas empíricas [lo, hi] por segmento×hora de los errores log(real/pred) de días con fecha <= corte.

    Para cada semana w solo se usan días con fecha <= corte[w] y > corte − 7·N. `pool` junta horas vecinas (±pool) para
    tener más casos. Si el segmento tiene < `minimo` casos se usa el segmento "laboral/todos" como respaldo.
    """
    W = ds.W
    S = ds.seg(seg)
    ph = pred if pred_hist is None else pred_hist
    le = np.log(np.maximum(ds.real, 1e-9) / np.maximum(ph, 1e-9)).reshape(-1, 24)
    flat_o = ds.ordn.reshape(-1)
    flat_s = S.reshape(-1)
    lo = pred.copy()
    hi = pred.copy()
    for w in range(W):
        c = ds.corte[w]
        for s in np.unique(S[w]):
            esp = (seg == "tipo" and s == 3)
            ventana = N_fest if esp else N
            m = (flat_o <= c) & (flat_o > c - 7 * ventana) & (flat_s == s)
            if m.sum() < max(minimo // 3, 2):
                m = (flat_o <= c) & (flat_o > c - 7 * N_fest)  # respaldo: todos los segmentos
            E = le[m]
            if len(E) < 3:
                continue
            ql = np.zeros(24)
            qh = np.zeros(24)
            for h in range(24):
                hs = [x for x in range(h - pool, h + pool + 1) if 0 <= x < 24]
                v = E[:, hs].ravel()
                ql[h], qh[h] = np.quantile(v, q[0]), np.quantile(v, q[1])
            for j in np.where(S[w] == s)[0]:
                lo[w, j] = pred[w, j] * np.exp(ql * escala)
                hi[w, j] = pred[w, j] * np.exp(qh * escala)
    return lo, hi


def cobertura(lo: np.ndarray, hi: np.ndarray, ds: DS, particion: str, tipo: Optional[str] = None,
              calentamiento: int = 8) -> Dict[str, float]:
    """Cobertura real de la banda. Se excluyen las primeras `calentamiento` semanas del cache (sin historial de errores)."""
    m = ds.mask(particion)
    m[:calentamiento] = False
    r, t = ds.real[m], ds.tipo[m]
    sel = np.ones_like(t, bool) if tipo is None else (t == tipo)
    if sel.sum() == 0:
        return dict(n=0, cobertura=np.nan, ancho_rel_pct=np.nan)
    inside = (r[sel] >= lo[m][sel]) & (r[sel] <= hi[m][sel])
    ancho = (hi[m][sel] - lo[m][sel]) / np.maximum(r[sel], 1e-9)
    return dict(n=int(sel.sum()), cobertura=100 * inside.mean(), ancho_rel_pct=100 * ancho.mean())


# ---------------------------------------------------------------- clima (observado: optimista)
def anomalia_temperatura(ds: DS, clima_csv: Path, dias_previos: int = 28) -> np.ndarray:
    """dT[w,j] = T media del día objetivo (OBSERVADA) − T media de los `dias_previos` días hasta el corte. NaN sin datos."""
    import pandas as pd
    c = pd.read_csv(clima_csv)
    c["fecha"] = c["fecha"].astype(str).str[:10]
    daily = c.groupby("fecha")["p_t"].mean()
    T = {date.fromisoformat(f).toordinal(): v for f, v in daily.items()}
    dT = np.full((ds.W, 7), np.nan)
    for w in range(ds.W):
        prev = [T[o] for o in range(int(ds.corte[w]) - dias_previos + 1, int(ds.corte[w]) + 1) if o in T]
        if len(prev) < dias_previos // 2:
            continue
        mu = float(np.mean(prev))
        for j in range(7):
            o = int(ds.ordn[w, j])
            if o in T:
                dT[w, j] = T[o] - mu
    return dT


def corregir_clima(pred: np.ndarray, ds: DS, dT: np.ndarray, N: int = 26, k0: float = 5.0, minimo: int = 20,
                   solo_laboral: bool = True, g: float = 1.0) -> np.ndarray:
    """Regresión móvil de log(E_real/E_pred) del día sobre dT (anomalía de temperatura); corrige la energía del día.

    Entrena con días ya observados (fecha <= corte) con dT disponible; solo pendiente centrada (no corrige sesgo medio).
    """
    out = pred.copy()
    e = np.log(ds.real.sum(2) / pred.sum(2)).reshape(-1)
    x = dT.reshape(-1)
    flat_o = ds.ordn.reshape(-1)
    ok_tipo = (ds.tipo.reshape(-1) == "laboral") if solo_laboral else np.ones(e.shape, bool)
    for w in range(ds.W):
        c = ds.corte[w]
        m = (flat_o <= c) & (flat_o > c - 7 * N) & np.isfinite(x) & ok_tipo
        n = int(m.sum())
        if n < minimo:
            continue
        xm, em = x[m].mean(), e[m].mean()
        vx = ((x[m] - xm) ** 2).sum()
        if vx <= 1e-9:
            continue
        b = ((x[m] - xm) * (e[m] - em)).sum() / (vx + k0)
        for j in range(7):
            if not np.isfinite(dT[w, j]) or (solo_laboral and ds.tipo[w, j] != "laboral"):
                continue
            out[w, j] = pred[w, j] * np.exp(g * b * (dT[w, j] - xm))
    return out


def bandas_energia(pred: np.ndarray, ds: DS, q: tuple = (0.10, 0.90), N: int = 52, minimo_dia: int = 8,
                   minimo_sem: int = 8, escala: float = 1.0, N_fest: int = 52):
    """Bandas de energía diaria (por tipo de día) y semanal con errores log(E_real/E_pred) observables al corte.

    Devuelve (lo_dia[W,7], hi_dia[W,7], lo_sem[W], hi_sem[W]); NaN donde no hay casos suficientes.
    """
    W = ds.W
    S = ds.seg("tipo")
    e_dia = np.log(ds.real.sum(2) / pred.sum(2))
    e_sem = np.log(ds.real.sum((1, 2)) / pred.sum((1, 2)))
    flat_o, flat_s, flat_e = ds.ordn.reshape(-1), S.reshape(-1), e_dia.reshape(-1)
    lo_d = np.full((W, 7), np.nan)
    hi_d = np.full((W, 7), np.nan)
    lo_s = np.full(W, np.nan)
    hi_s = np.full(W, np.nan)
    for w in range(W):
        c = ds.corte[w]
        for s in np.unique(S[w]):
            ventana = N_fest if s == 3 else N
            m = (flat_o <= c) & (flat_o > c - 7 * ventana) & (flat_s == s)
            if m.sum() < minimo_dia:
                continue
            ql, qh = np.quantile(flat_e[m], q[0]), np.quantile(flat_e[m], q[1])
            for j in np.where(S[w] == s)[0]:
                p = pred[w, j].sum()
                lo_d[w, j], hi_d[w, j] = p * np.exp(ql * escala), p * np.exp(qh * escala)
        obs = [k for k in range(W) if k < w and ds.ordn[k, 6] <= c and ds.ordn[k, 0] > c - 7 * N]
        if len(obs) >= minimo_sem:
            ql, qh = np.quantile(e_sem[obs], q[0]), np.quantile(e_sem[obs], q[1])
            p = pred[w].sum()
            lo_s[w], hi_s[w] = p * np.exp(ql * escala), p * np.exp(qh * escala)
    return lo_d, hi_d, lo_s, hi_s
