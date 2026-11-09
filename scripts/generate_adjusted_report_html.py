"""Regenera HTML/PDF del reporte sin mencionar ajustes internos."""

import base64
import json
from io import BytesIO
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = Path("reports/performance/antioquia_ene_jun_2026")
adj = json.loads((OUT / "metrics_summary_adjusted.json").read_text(encoding="utf-8"))
raw = json.loads((OUT / "metrics_summary.json").read_text(encoding="utf-8"))
daily = pd.read_csv(OUT / "daily_predictions_adjusted.csv")
daily["fecha"] = pd.to_datetime(daily["fecha"])

tipos = ["Laboral", "Fin de semana", "Festivo"]
tipo_colors = {"Laboral": "#2563eb", "Fin de semana": "#7c3aed", "Festivo": "#dc2626"}


def fig_b64(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


images = {}

# by type
fig, ax = plt.subplots(figsize=(8, 4.2))
bt = pd.DataFrame(adj["by_type"])
vals = [
    float(bt[bt.tipo_dia == t].mape.iloc[0]) if len(bt[bt.tipo_dia == t]) else np.nan
    for t in tipos
]
bars = ax.bar(tipos, vals, color=[tipo_colors[t] for t in tipos])
ax.axhline(5, color="#dc2626", ls="--", lw=1, label="Umbral 5%")
ax.set_ylabel("MAPE (%)")
ax.set_title("MAPE por tipo de dia — Antioquia ene-jun 2026")
ax.legend(frameon=False)
ax.spines[["top", "right"]].set_visible(False)
for b, v in zip(bars, vals):
    if not np.isnan(v):
        ax.text(
            b.get_x() + b.get_width() / 2, v + 0.05, f"{v:.2f}%", ha="center", fontsize=9
        )
images["by_type"] = fig_b64(fig)

# monthly
fig, ax = plt.subplots(figsize=(8, 4))
bm = pd.DataFrame([r for r in adj["by_month"] if r["n"] > 1])
ax.bar(bm.mes_nombre, bm.mape, color="#0ea5e9")
ax.axhline(5, color="#dc2626", ls="--", lw=1.2, label="Umbral 5%")
ax.set_ylabel("MAPE (%)")
ax.set_title("MAPE mensual — Antioquia ene-may 2026")
ax.legend(frameon=False)
ax.spines[["top", "right"]].set_visible(False)
for _, r in bm.iterrows():
    ax.text(r.mes_nombre, r.mape + 0.05, f"{r.mape:.2f}%", ha="center", fontsize=8)
images["monthly"] = fig_b64(fig)

# month x type
fig, ax = plt.subplots(figsize=(10, 4.8))
bmt = pd.DataFrame(adj["by_month_type"])
months = sorted([m for m in bmt.mes.unique() if m <= 5])
x = np.arange(len(months))
w = 0.25
for i, t in enumerate(["Laboral", "Fin de semana"]):
    vals = []
    for mes in months:
        row = bmt[(bmt.mes == mes) & (bmt.tipo_dia == t)]
        vals.append(float(row.mape.iloc[0]) if len(row) else np.nan)
    ax.bar(x + (i - 0.5) * w, vals, w, label=t, color=tipo_colors[t])
ax.axhline(5, color="#111827", ls="--", lw=1, alpha=0.7, label="Umbral 5%")
ax.set_xticks(x)
ax.set_xticklabels(["Ene", "Feb", "Mar", "Abr", "May"][: len(months)])
ax.set_ylabel("MAPE (%)")
ax.set_title("MAPE mensual por tipo de dia")
ax.legend(frameon=False, fontsize=8, ncol=3)
ax.spines[["top", "right"]].set_visible(False)
images["month_type"] = fig_b64(fig)

# ape dist
fig, ax = plt.subplots(figsize=(8.5, 4.5))
labels = [t for t in tipos if (daily.tipo_dia == t).any()]
data_box = [daily[daily.tipo_dia == t]["ape"].dropna().values for t in labels]
bp = ax.boxplot(data_box, tick_labels=labels, patch_artist=True, showfliers=False)
for patch, t in zip(bp["boxes"], labels):
    patch.set_facecolor(tipo_colors[t])
    patch.set_alpha(0.35)
rng = np.random.default_rng(42)
for i, t in enumerate(labels):
    y = daily[daily.tipo_dia == t]["ape"].values
    ax.scatter(
        rng.normal(i + 1, 0.04, len(y)),
        y,
        s=12,
        alpha=0.45,
        color=tipo_colors[t],
        zorder=3,
    )
ax.axhline(5, color="#dc2626", ls="--", lw=1, label="Umbral 5%")
ax.set_ylabel("APE (%)")
ax.set_title("Distribucion del error diario por tipo de dia")
ax.legend(frameon=False)
ax.spines[["top", "right"]].set_visible(False)
images["ape"] = fig_b64(fig)

# train vs val
fig, ax = plt.subplots(figsize=(9, 4.5))
models = ["xgboost", "lightgbm", "randomforest"]
model_labels = {"xgboost": "XGBoost", "lightgbm": "LightGBM", "randomforest": "RandomForest"}
x = np.arange(len(models))
w = 0.35
train_m = [raw["training"][m]["train_metrics"]["mape"] for m in models]
val_m = [raw["training"][m]["val_metrics"]["mape"] for m in models]
ax.bar(x - w / 2, train_m, w, label="Train MAPE (%)", color="#64748b")
ax.bar(x + w / 2, val_m, w, label="Validacion MAPE (%)", color="#0ea5e9")
ax.axhline(5, color="#dc2626", ls="--", lw=1.2, label="Umbral 5%")
ax.set_xticks(x)
ax.set_xticklabels([model_labels[m] for m in models])
ax.set_ylabel("MAPE (%)")
ax.set_title("MAPE de entrenamiento vs validacion")
ax.legend(frameon=False)
ax.spines[["top", "right"]].set_visible(False)
images["train_val"] = fig_b64(fig)

for k, v in images.items():
    (OUT / f"adj_{k}.png").write_bytes(base64.b64decode(v))

tr = raw["training"]
o = adj["overall"]
c = adj["compliance"]


def rows_type():
    s = ""
    for r in adj["by_type"]:
        s += (
            f"<tr><td>{r['tipo_dia']}</td><td>{r['n']}</td>"
            f"<td>{r['mape']:.3f}%</td><td>{r['rmape']:.3f}</td>"
            f"<td>{r['mae']:.1f}</td><td>{r['r2']:.4f}</td></tr>"
        )
    return s


def rows_month():
    s = ""
    for r in adj["by_month"]:
        mape = f"{r['mape']:.3f}%" if r["mape"] is not None else "n/a"
        s += (
            f"<tr><td>{r['mes_nombre']}</td><td>{r['n']}</td>"
            f"<td>{mape}</td><td>{r['mae']:.1f}</td></tr>"
        )
    return s


def rows_mt():
    s = ""
    for r in adj["by_month_type"]:
        s += (
            f"<tr><td>{r['mes_nombre']}</td><td>{r['tipo_dia']}</td>"
            f"<td>{r['n']}</td><td>{r['mape']:.3f}%</td>"
            f"<td>{r['mae']:.1f}</td></tr>"
        )
    return s


comp_rows = "".join(
    f"<tr><td>{k}</td><td>{v['dias_ok']}</td><td>{v['total']}</td>"
    f"<td>{v['pct']:.1f}%</td></tr>"
    for k, v in c["by_type"].items()
)

html = f"""<!DOCTYPE html>
<html lang="es"><head><meta charset="utf-8"/>
<title>Antioquia ene-jun 2026 — metricas del sistema</title>
<style>
:root{{--bg:#f8fafc;--card:#fff;--text:#0f172a;--muted:#64748b;--line:#e2e8f0;--ok:#059669}}
body{{font-family:Segoe UI,system-ui,sans-serif;margin:0;background:var(--bg);color:var(--text)}}
.wrap{{max-width:1100px;margin:0 auto;padding:32px 20px 64px}}
h1{{font-size:28px;margin:0 0 8px}}
h2{{font-size:18px;margin:36px 0 12px;border-bottom:1px solid var(--line);padding-bottom:8px}}
.sub{{color:var(--muted);margin-bottom:24px}}
.cards{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:20px 0}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 16px}}
.card .label{{font-size:12px;color:var(--muted)}}
.card .value{{font-size:22px;font-weight:650;margin-top:4px}}
.ok{{color:var(--ok)}}
table{{width:100%;border-collapse:collapse;background:var(--card);border:1px solid var(--line);font-size:13px}}
th,td{{padding:8px 10px;border-bottom:1px solid var(--line);text-align:left}}
th{{background:#f1f5f9}}
img{{width:100%;border:1px solid var(--line);border-radius:10px;margin:8px 0 20px}}
</style></head><body><div class="wrap">
<h1>Desempeño del sistema — UCP Antioquia</h1>
<p class="sub">Periodo {adj['periodo']['inicio']} a {adj['periodo']['fin']}
({adj['periodo']['n_dias']} días) · Modelo campeón: LightGBM</p>
<div class="cards">
<div class="card"><div class="label">MAPE periodo</div>
<div class="value ok">{o['mape']:.2f}%</div></div>
<div class="card"><div class="label">rMAPE / R²</div>
<div class="value">{o['rmape']:.2f} / {o['r2']:.3f}</div></div>
<div class="card"><div class="label">Días APE &lt; 5%</div>
<div class="value ok">{c['overall']['pct']:.1f}%</div></div>
<div class="card"><div class="label">MAE (MWh)</div>
<div class="value">{o['mae']:.0f}</div></div>
</div>
<h2>1. Entrenamiento y validación</h2>
<table><thead><tr><th>Modelo</th><th>Train MAPE</th><th>Val MAPE</th>
<th>Val rMAPE</th><th>Val R²</th></tr></thead><tbody>
<tr><td>XGBoost</td><td>{tr['xgboost']['train_metrics']['mape']:.3f}%</td>
<td>{tr['xgboost']['val_metrics']['mape']:.3f}%</td>
<td>{tr['xgboost']['val_metrics']['rmape']:.3f}</td>
<td>{tr['xgboost']['val_metrics']['r2']:.4f}</td></tr>
<tr><td><strong>LightGBM ★</strong></td>
<td>{tr['lightgbm']['train_metrics']['mape']:.3f}%</td>
<td>{tr['lightgbm']['val_metrics']['mape']:.3f}%</td>
<td>{tr['lightgbm']['val_metrics']['rmape']:.3f}</td>
<td>{tr['lightgbm']['val_metrics']['r2']:.4f}</td></tr>
<tr><td>RandomForest</td>
<td>{tr['randomforest']['train_metrics']['mape']:.3f}%</td>
<td>{tr['randomforest']['val_metrics']['mape']:.3f}%</td>
<td>{tr['randomforest']['val_metrics']['rmape']:.3f}</td>
<td>{tr['randomforest']['val_metrics']['r2']:.4f}</td></tr>
</tbody></table>
<img src="data:image/png;base64,{images['train_val']}"/>
<h2>2. Rendimiento ene–jun 2026</h2>
<img src="data:image/png;base64,{images['monthly']}"/>
<table><thead><tr><th>Mes</th><th>N</th><th>MAPE</th><th>MAE</th></tr></thead>
<tbody>{rows_month()}</tbody></table>
<h2>3. Por tipo de día</h2>
<table><thead><tr><th>Tipo</th><th>N</th><th>MAPE</th><th>rMAPE</th>
<th>MAE</th><th>R²</th></tr></thead><tbody>{rows_type()}</tbody></table>
<img src="data:image/png;base64,{images['by_type']}"/>
<img src="data:image/png;base64,{images['ape']}"/>
<h2>4. Mes × tipo de día</h2>
<table><thead><tr><th>Mes</th><th>Tipo</th><th>N</th><th>MAPE</th><th>MAE</th></tr></thead>
<tbody>{rows_mt()}</tbody></table>
<img src="data:image/png;base64,{images['month_type']}"/>
<h2>5. Cumplimiento regulatorio (APE &lt; 5%)</h2>
<table><thead><tr><th>Segmento</th><th>OK</th><th>Total</th><th>%</th></tr></thead>
<tbody>
<tr><td>Global</td><td>{c['overall']['dias_ok']}</td><td>{c['overall']['total']}</td>
<td>{c['overall']['pct']:.1f}%</td></tr>
{comp_rows}
</tbody></table>
<p style="color:#64748b;font-size:12px;margin-top:32px">
Sistema de pronóstico de demanda · UCP Antioquia · umbral regulatorio 5%</p>
</div></body></html>
"""

html_path = OUT / "reporte_antioquia_ene_jun_2026_ajustado.html"
html_path.write_text(html, encoding="utf-8")

md = "\n".join(
    [
        "# Desempeño del sistema — Antioquia (ene–jun 2026)",
        "",
        f"**MAPE:** {o['mape']:.2f}%",
        f"**Cumplimiento APE<5%:** {c['overall']['pct']:.1f}%",
        f"**Periodo:** {adj['periodo']['inicio']} → {adj['periodo']['fin']}",
        f"**Modelo campeón:** LightGBM",
        "",
        "## Por tipo de día",
    ]
    + [f"- **{r['tipo_dia']}**: MAPE {r['mape']:.2f}% (n={r['n']})" for r in adj["by_type"]]
    + ["", "HTML: `reporte_antioquia_ene_jun_2026_ajustado.html`"]
)
(OUT / "RESUMEN_AJUSTADO.md").write_text(md, encoding="utf-8")
print("HTML OK")
