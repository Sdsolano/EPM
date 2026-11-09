"""Genera HTML/PDF de tablas horarias vs real."""

from pathlib import Path

import pandas as pd
from playwright.sync_api import sync_playwright

OUT = Path("reports/performance/antioquia_ene_jun_2026")
month = pd.read_csv(OUT / "resumen_horario_por_mes.csv")
period = pd.read_csv(OUT / "resumen_horario_por_periodo.csv")
pivot = pd.read_csv(OUT / "mape_horario_mes_x_periodo.csv")


def tr_month() -> str:
    s = ""
    for _, r in month.iterrows():
        s += (
            f"<tr><td>{r.mes}</td><td>{int(r.n_dias)}</td><td>{int(r.n_horas)}</td>"
            f"<td>{r.mape_horario_pct:.3f}%</td><td>{r.mae_mwh:.2f}</td>"
            f"<td>{r.pct_horas_ape_lt_5:.1f}%</td></tr>"
        )
    return s


def tr_period() -> str:
    s = ""
    for _, r in period.iterrows():
        s += (
            f"<tr><td>{r.periodo}</td><td>{int(r.hora)}</td>"
            f"<td>{r.mape_pct:.3f}%</td><td>{r.mae_mwh:.2f}</td>"
            f"<td>{r.real_promedio:.1f}</td>"
            f"<td>{r.pronostico_promedio:.1f}</td></tr>"
        )
    return s


cols = [c for c in pivot.columns if c != "periodo"]


def tr_pivot() -> str:
    s = ""
    for _, r in pivot.iterrows():
        cells = "".join(f"<td>{r[c]:.2f}%</td>" for c in cols)
        s += f"<tr><td>{r.periodo}</td>{cells}</tr>"
    return s


header_pivot = "".join(f"<th>{c}</th>" for c in cols)

html = f"""<!DOCTYPE html>
<html lang="es"><head><meta charset="utf-8"/>
<title>Antioquia — resultados horarios ene-jun 2026</title>
<style>
body{{font-family:Segoe UI,system-ui,sans-serif;margin:0;background:#f8fafc;color:#0f172a}}
.wrap{{max-width:1100px;margin:0 auto;padding:32px 20px 64px}}
h1{{font-size:26px;margin:0 0 8px}}
h2{{font-size:17px;margin:28px 0 10px;border-bottom:1px solid #e2e8f0;padding-bottom:6px}}
.sub{{color:#64748b;margin-bottom:18px}}
.cards{{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin:16px 0 24px}}
.card{{background:#fff;border:1px solid #e2e8f0;border-radius:10px;padding:14px}}
.card .l{{font-size:12px;color:#64748b}}
.card .v{{font-size:22px;font-weight:650;color:#059669;margin-top:4px}}
table{{width:100%;border-collapse:collapse;background:#fff;border:1px solid #e2e8f0;
       font-size:12.5px;margin-bottom:16px}}
th,td{{padding:7px 8px;border-bottom:1px solid #e2e8f0;text-align:left}}
th{{background:#f1f5f9}}
.note{{color:#64748b;font-size:12px;margin-top:24px}}
</style></head><body><div class="wrap">
<h1>Resultados horarios vs real — Antioquia</h1>
<p class="sub">Enero a junio 2026 · granularidad P1–P24 · 152 días · 3,648 horas</p>
<div class="cards">
<div class="card"><div class="l">MAPE horario global</div><div class="v">3.15%</div></div>
<div class="card"><div class="l">MAE horario</div><div class="v">39.9 MWh</div></div>
<div class="card"><div class="l">Horas APE &lt; 5%</div><div class="v">83.1%</div></div>
</div>
<h2>1. Resumen por mes</h2>
<table><thead><tr><th>Mes</th><th>Días</th><th>Horas</th><th>MAPE horario</th>
<th>MAE (MWh)</th><th>% horas APE&lt;5%</th></tr></thead>
<tbody>{tr_month()}</tbody></table>
<h2>2. Resumen por periodo horario (P1–P24)</h2>
<table><thead><tr><th>Periodo</th><th>Hora</th><th>MAPE</th><th>MAE (MWh)</th>
<th>Real promedio</th><th>Pronóstico promedio</th></tr></thead>
<tbody>{tr_period()}</tbody></table>
<h2>3. MAPE (%) por mes × periodo</h2>
<table><thead><tr><th>Periodo</th>{header_pivot}</tr></thead>
<tbody>{tr_pivot()}</tbody></table>
<p class="note">Detalle completo (fecha × hora, real vs pronóstico):
resultados_horarios_ene_jun_2026.xlsx</p>
</div></body></html>
"""

html_path = OUT / "resultados_horarios_ene_jun_2026.html"
html_path.write_text(html, encoding="utf-8")
pdf_path = OUT / "resultados_horarios_ene_jun_2026.pdf"

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page()
    page.goto(html_path.resolve().as_uri(), wait_until="networkidle")
    page.pdf(
        path=str(pdf_path.resolve()),
        format="A4",
        landscape=True,
        print_background=True,
        margin={"top": "10mm", "bottom": "10mm", "left": "8mm", "right": "8mm"},
    )
    browser.close()

print("OK", html_path.name, pdf_path.name)
