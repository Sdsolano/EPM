"""
Genera reporte de métricas Antioquia ene–jun 2026
(entrenamiento, validación y rendimiento por tipo de día).
"""

import json
import base64
from pathlib import Path
from io import BytesIO

import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("reports/performance/antioquia_ene_jun_2026")
summary = json.loads((OUT / "metrics_summary.json").read_text(encoding="utf-8"))
daily = pd.read_csv(OUT / "daily_predictions.csv")
champ_key = summary["champion_key"]
champ_daily = daily[daily.modelo == champ_key].copy()
month_names = {1: "Ene", 2: "Feb", 3: "Mar", 4: "Abr", 5: "May", 6: "Jun"}
tipos = ["Laboral", "Fin de semana", "Festivo"]
tipo_colors = {"Laboral": "#2563eb", "Fin de semana": "#7c3aed", "Festivo": "#dc2626"}
model_labels = {
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
    "randomforest": "RandomForest",
}
models = list(summary["training"].keys())


def fig_to_b64(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


images = {}

# 1) Train vs Val MAPE
fig, ax = plt.subplots(figsize=(9, 4.5))
x = np.arange(len(models))
w = 0.35
train_m = [summary["training"][m]["train_metrics"]["mape"] for m in models]
val_m = [summary["training"][m]["val_metrics"]["mape"] for m in models]
ax.bar(x - w / 2, train_m, w, label="Train MAPE (%)", color="#64748b")
ax.bar(x + w / 2, val_m, w, label="Validación MAPE (%)", color="#0ea5e9")
ax.axhline(5, color="#dc2626", ls="--", lw=1.2, label="Umbral 5%")
ax.set_xticks(x)
ax.set_xticklabels([model_labels[m] for m in models])
ax.set_ylabel("MAPE (%)")
ax.set_title("MAPE de entrenamiento vs validación — Antioquia")
ax.legend(frameon=False)
ax.spines[["top", "right"]].set_visible(False)
images["train_val"] = fig_to_b64(fig)

# 2) Period MAPE by model
fig, ax = plt.subplots(figsize=(8, 4.2))
period_mapes = [summary["period_metrics"][m]["mape"] for m in models]
bars = ax.bar(
    [model_labels[m] for m in models],
    period_mapes,
    color=["#94a3b8", "#0ea5e9", "#94a3b8"],
)
for i, m in enumerate(models):
    if m == champ_key:
        bars[i].set_color("#059669")
ax.axhline(5, color="#dc2626", ls="--", lw=1.2, label="Umbral 5%")
ax.set_ylabel("MAPE (%)")
ax.set_title(f"MAPE ene–jun {summary['periodo']['anio']} (rendimiento en periodo)")
ax.legend(frameon=False)
ax.spines[["top", "right"]].set_visible(False)
for b, v in zip(bars, period_mapes):
    ax.text(
        b.get_x() + b.get_width() / 2,
        v + 0.08,
        f"{v:.2f}%",
        ha="center",
        va="bottom",
        fontsize=9,
    )
images["period_models"] = fig_to_b64(fig)

# 3) MAPE by day type
fig, ax = plt.subplots(figsize=(9, 4.5))
by_type = pd.DataFrame(summary["by_type"])
x = np.arange(len(tipos))
w = 0.25
for i, m in enumerate(models):
    vals = []
    for t in tipos:
        row = by_type[(by_type.modelo == m) & (by_type.tipo_dia == t)]
        vals.append(float(row.mape.iloc[0]) if len(row) else 0)
    ax.bar(
        x + (i - 1) * w,
        vals,
        w,
        label=model_labels[m] + (" ★" if m == champ_key else ""),
    )
ax.axhline(5, color="#dc2626", ls="--", lw=1.0)
ax.set_xticks(x)
ax.set_xticklabels(tipos)
ax.set_ylabel("MAPE (%)")
ax.set_title("MAPE por tipo de día — ene–jun 2026")
ax.legend(frameon=False, fontsize=8)
ax.spines[["top", "right"]].set_visible(False)
images["by_type"] = fig_to_b64(fig)

# 4) Monthly MAPE by day type
fig, ax = plt.subplots(figsize=(10, 4.8))
bmt = pd.DataFrame(summary["by_month_type"])
months = sorted(bmt.mes.unique())
x = np.arange(len(months))
w = 0.25
for i, t in enumerate(tipos):
    vals = []
    for mes in months:
        row = bmt[(bmt.mes == mes) & (bmt.tipo_dia == t)]
        vals.append(float(row.mape.iloc[0]) if len(row) else np.nan)
    ax.bar(x + (i - 1) * w, vals, w, label=t, color=tipo_colors[t])
ax.axhline(5, color="#111827", ls="--", lw=1.0, alpha=0.7, label="Umbral 5%")
ax.set_xticks(x)
ax.set_xticklabels([month_names[m] for m in months])
ax.set_ylabel("MAPE (%)")
ax.set_title(f"MAPE mensual por tipo de día — campeón {summary['champion']}")
ax.legend(frameon=False, fontsize=8, ncol=4)
ax.spines[["top", "right"]].set_visible(False)
images["month_type"] = fig_to_b64(fig)

# 5) Time series
fig, ax = plt.subplots(figsize=(11, 4.5))
cd = champ_daily.sort_values("fecha")
ax.plot(pd.to_datetime(cd.fecha), cd.real, label="Real", color="#111827", lw=1.4)
ax.plot(
    pd.to_datetime(cd.fecha),
    cd.pred,
    label="Predicción",
    color="#0ea5e9",
    lw=1.4,
    alpha=0.9,
)
ax.set_ylabel("Demanda TOTAL (MWh)")
ax.set_title(f"Demanda real vs predicción — {summary['champion']} (ene–jun 2026)")
ax.legend(frameon=False)
ax.spines[["top", "right"]].set_visible(False)
fig.autofmt_xdate()
images["timeseries"] = fig_to_b64(fig)

# 6) APE distribution
fig, ax = plt.subplots(figsize=(8.5, 4.5))
data_box = [
    champ_daily[champ_daily.tipo_dia == t]["ape"].dropna().values for t in tipos
]
bp = ax.boxplot(data_box, labels=tipos, patch_artist=True, showfliers=False)
for patch, t in zip(bp["boxes"], tipos):
    patch.set_facecolor(tipo_colors[t])
    patch.set_alpha(0.35)
rng = np.random.default_rng(42)
for i, t in enumerate(tipos):
    y = champ_daily[champ_daily.tipo_dia == t]["ape"].values
    xj = rng.normal(i + 1, 0.04, size=len(y))
    ax.scatter(xj, y, s=12, alpha=0.45, color=tipo_colors[t], zorder=3)
ax.axhline(5, color="#dc2626", ls="--", lw=1.0)
ax.set_ylabel("Error absoluto % (APE)")
ax.set_title("Distribución del error diario por tipo de día")
ax.spines[["top", "right"]].set_visible(False)
images["ape_dist"] = fig_to_b64(fig)

# 7) Monthly overall
fig, ax = plt.subplots(figsize=(8, 4))
bm = pd.DataFrame(summary["by_month"])
ax.bar(bm.mes_nombre, bm.mape, color="#0ea5e9")
ax.axhline(5, color="#dc2626", ls="--", lw=1.2, label="Umbral 5%")
ax.set_ylabel("MAPE (%)")
ax.set_title(f"MAPE mensual — campeón {summary['champion']}")
ax.legend(frameon=False)
ax.spines[["top", "right"]].set_visible(False)
for _, r in bm.iterrows():
    ax.text(r.mes_nombre, r.mape + 0.05, f"{r.mape:.2f}%", ha="center", fontsize=8)
images["monthly"] = fig_to_b64(fig)

for name, b64 in images.items():
    (OUT / f"{name}.png").write_bytes(base64.b64decode(b64))

tr = summary["training"]
pm = summary["period_metrics"]
comp = summary["compliance"]


def metric_row(label, d):
    return (
        f"<tr><td>{label}</td><td>{d['mape']:.3f}%</td><td>{d['rmape']:.3f}</td>"
        f"<td>{d['mae']:.1f}</td><td>{d['rmse']:.1f}</td>"
        f"<td>{d['r2']:.4f}</td><td>{d['correlation']:.4f}</td></tr>"
    )


by_type_rows = ""
for r in summary["by_type"]:
    if r["modelo"] != champ_key:
        continue
    by_type_rows += (
        f"<tr><td>{r['tipo_dia']}</td><td>{r['n']}</td><td>{r['mape']:.3f}%</td>"
        f"<td>{r['rmape']:.3f}</td><td>{r['mae']:.1f}</td><td>{r['r2']:.4f}</td></tr>"
    )

bmt_rows = ""
for r in summary["by_month_type"]:
    bmt_rows += (
        f"<tr><td>{r['mes_nombre']}</td><td>{r['tipo_dia']}</td><td>{r['n']}</td>"
        f"<td>{r['mape']:.3f}%</td><td>{r['mae']:.1f}</td></tr>"
    )

train_rows = ""
for m in models:
    t = tr[m]["train_metrics"]
    v = tr[m]["val_metrics"]
    star = " ★ campeón" if m == champ_key else ""
    train_rows += f"""
    <tr>
      <td rowspan="2"><strong>{model_labels[m]}</strong>{star}</td>
      <td>Train</td><td>{t['mape']:.3f}%</td><td>{t['rmape']:.3f}</td>
      <td>{t['mae']:.1f}</td><td>{t['r2']:.4f}</td>
    </tr>
    <tr>
      <td>Validación</td><td>{v['mape']:.3f}%</td><td>{v['rmape']:.3f}</td>
      <td>{v['mae']:.1f}</td><td>{v['r2']:.4f}</td>
    </tr>"""

period_rows = ""
for m in models:
    d = pm[m]
    period_rows += metric_row(
        model_labels[m] + (" ★" if m == champ_key else ""), d
    )

comp_type_rows = "".join(
    f"<tr><td>{k}</td><td>{v['dias_ok']}</td><td>{v['total']}</td>"
    f"<td>{v['pct']:.1f}%</td></tr>"
    for k, v in comp["by_type"].items()
)

mape_ok = "ok" if pm[champ_key]["mape"] < 5 else "bad"

html = f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="utf-8"/>
<title>Reporte Antioquia — Ene–Jun 2026</title>
<style>
  :root {{ --bg:#f8fafc; --card:#fff; --text:#0f172a; --muted:#64748b;
           --line:#e2e8f0; --accent:#0ea5e9; --ok:#059669; --bad:#dc2626; }}
  body {{ font-family: 'Segoe UI', system-ui, sans-serif; margin:0;
         background:var(--bg); color:var(--text); }}
  .wrap {{ max-width:1100px; margin:0 auto; padding:32px 20px 64px; }}
  h1 {{ font-size:28px; margin:0 0 8px; }}
  h2 {{ font-size:18px; margin:36px 0 12px; border-bottom:1px solid var(--line);
       padding-bottom:8px; }}
  .sub {{ color:var(--muted); margin-bottom:24px; }}
  .cards {{ display:grid; grid-template-columns:repeat(4,1fr); gap:12px;
           margin:20px 0 28px; }}
  .card {{ background:var(--card); border:1px solid var(--line);
          border-radius:10px; padding:14px 16px; }}
  .card .label {{ font-size:12px; color:var(--muted); }}
  .card .value {{ font-size:22px; font-weight:650; margin-top:4px; }}
  .ok {{ color:var(--ok); }} .bad {{ color:var(--bad); }}
  table {{ width:100%; border-collapse:collapse; background:var(--card);
          border:1px solid var(--line); border-radius:8px; overflow:hidden;
          font-size:13px; }}
  th, td {{ padding:8px 10px; border-bottom:1px solid var(--line); text-align:left; }}
  th {{ background:#f1f5f9; font-weight:600; }}
  img {{ width:100%; border:1px solid var(--line); border-radius:10px;
        background:#fff; margin:8px 0 20px; }}
  .note {{ background:#fff7ed; border:1px solid #fed7aa; color:#9a3412;
          padding:12px 14px; border-radius:8px; font-size:13px; }}
  .footer {{ margin-top:40px; color:var(--muted); font-size:12px; }}
  @media (max-width:800px) {{ .cards {{ grid-template-columns:1fr 1fr; }} }}
</style>
</head>
<body>
<div class="wrap">
  <h1>Reporte de desempeño — UCP Antioquia</h1>
  <p class="sub">Métricas de entrenamiento, validación y rendimiento
  ene–jun {summary['periodo']['anio']} por tipo de día<br/>
  Periodo evaluado: <strong>{summary['periodo']['inicio']}</strong> a
  <strong>{summary['periodo']['fin']}</strong>
  ({summary['periodo']['n_dias']} días) · Modelo campeón:
  <strong>{summary['champion']}</strong></p>

  <div class="note">Nota: los features de Antioquia llegan hasta
  {summary['periodo']['fin']}; junio queda parcialmente cubierto (1 día).
  Tipos de día: Laboral (hábil no festivo), Fin de semana, Festivo (festivo en día hábil).</div>

  <div class="cards">
    <div class="card"><div class="label">MAPE periodo (campeón)</div>
      <div class="value {mape_ok}">{pm[champ_key]['mape']:.2f}%</div></div>
    <div class="card"><div class="label">MAPE validación (campeón)</div>
      <div class="value">{tr[champ_key]['val_metrics']['mape']:.2f}%</div></div>
    <div class="card"><div class="label">Días &lt; 5% APE</div>
      <div class="value">{comp['overall']['pct']:.1f}%</div></div>
    <div class="card"><div class="label">Features / muestras train</div>
      <div class="value" style="font-size:18px">
        {tr[champ_key]['n_features']} / {tr[champ_key]['n_train_samples']}
      </div></div>
  </div>

  <h2>1. Entrenamiento y validación</h2>
  <table>
    <thead><tr><th>Modelo</th><th>Conjunto</th><th>MAPE</th><th>rMAPE</th>
      <th>MAE (MWh)</th><th>R²</th></tr></thead>
    <tbody>{train_rows}</tbody>
  </table>
  <img src="data:image/png;base64,{images['train_val']}" alt="Train vs Val"/>

  <h2>2. Rendimiento ene–jun {summary['periodo']['anio']}</h2>
  <table>
    <thead><tr><th>Modelo</th><th>MAPE</th><th>rMAPE</th><th>MAE</th>
      <th>RMSE</th><th>R²</th><th>Corr</th></tr></thead>
    <tbody>{period_rows}</tbody>
  </table>
  <img src="data:image/png;base64,{images['period_models']}" alt="MAPE periodo"/>
  <img src="data:image/png;base64,{images['timeseries']}" alt="Serie temporal"/>
  <img src="data:image/png;base64,{images['monthly']}" alt="MAPE mensual"/>

  <h2>3. Rendimiento por tipo de día (campeón)</h2>
  <table>
    <thead><tr><th>Tipo de día</th><th>N días</th><th>MAPE</th><th>rMAPE</th>
      <th>MAE</th><th>R²</th></tr></thead>
    <tbody>{by_type_rows}</tbody>
  </table>
  <img src="data:image/png;base64,{images['by_type']}" alt="MAPE por tipo"/>
  <img src="data:image/png;base64,{images['ape_dist']}" alt="Distribución APE"/>

  <h2>4. MAPE mensual × tipo de día (campeón)</h2>
  <table>
    <thead><tr><th>Mes</th><th>Tipo</th><th>N</th><th>MAPE</th><th>MAE</th></tr></thead>
    <tbody>{bmt_rows}</tbody>
  </table>
  <img src="data:image/png;base64,{images['month_type']}" alt="MAPE mes x tipo"/>

  <h2>5. Cumplimiento regulatorio (APE &lt; 5%)</h2>
  <table>
    <thead><tr><th>Segmento</th><th>Días OK</th><th>Total</th>
      <th>% cumplimiento</th></tr></thead>
    <tbody>
      <tr><td>Global</td><td>{comp['overall']['dias_ok']}</td>
        <td>{comp['overall']['total']}</td>
        <td>{comp['overall']['pct']:.1f}%</td></tr>
      {comp_type_rows}
    </tbody>
  </table>

  <p class="footer">Generado desde models/Antioquia y data/features/Antioquia ·
  Umbral regulatorio MAPE 5%</p>
</div>
</body>
</html>
"""
(OUT / "reporte_antioquia_ene_jun_2026.html").write_text(html, encoding="utf-8")

md = f"""# Reporte de desempeño — Antioquia (ene–jun {summary['periodo']['anio']})

**Periodo:** {summary['periodo']['inicio']} → {summary['periodo']['fin']} ({summary['periodo']['n_dias']} días)
**Campeón:** {summary['champion']}
**Nota:** features disponibles hasta {summary['periodo']['fin']} (junio parcial).

## Entrenamiento / validación

| Modelo | Train MAPE | Val MAPE | Val rMAPE | Val R² |
|--------|------------|----------|-----------|--------|
"""
for m in models:
    t, v = tr[m]["train_metrics"], tr[m]["val_metrics"]
    star = " ★" if m == champ_key else ""
    md += (
        f"| {model_labels[m]}{star} | {t['mape']:.3f}% | {v['mape']:.3f}% | "
        f"{v['rmape']:.3f} | {v['r2']:.4f} |\n"
    )

md += f"""
## Rendimiento ene–jun {summary['periodo']['anio']}

| Modelo | MAPE | rMAPE | MAE | R² |
|--------|------|-------|-----|----|
"""
for m in models:
    d = pm[m]
    star = " ★" if m == champ_key else ""
    md += (
        f"| {model_labels[m]}{star} | {d['mape']:.3f}% | {d['rmape']:.3f} | "
        f"{d['mae']:.1f} | {d['r2']:.4f} |\n"
    )

md += """
## Por tipo de día (campeón)

| Tipo | N | MAPE | rMAPE | MAE | R² |
|------|---|------|-------|-----|----|
"""
for r in summary["by_type"]:
    if r["modelo"] != champ_key:
        continue
    md += (
        f"| {r['tipo_dia']} | {r['n']} | {r['mape']:.3f}% | {r['rmape']:.3f} | "
        f"{r['mae']:.1f} | {r['r2']:.4f} |\n"
    )

md += f"""
## Cumplimiento (APE < 5%)

- Global: **{comp['overall']['pct']:.1f}%** ({comp['overall']['dias_ok']}/{comp['overall']['total']})
"""
for k, v in comp["by_type"].items():
    md += f"- {k}: {v['pct']:.1f}% ({v['dias_ok']}/{v['total']})\n"

md += """
## Archivos

- HTML con gráficas: `reporte_antioquia_ene_jun_2026.html`
- JSON: `metrics_summary.json`
- Predicciones diarias: `daily_predictions.csv`
- PNGs en esta carpeta
"""
(OUT / "RESUMEN.md").write_text(md, encoding="utf-8")
print("OK ->", OUT)
print("Compliance", comp["overall"])
