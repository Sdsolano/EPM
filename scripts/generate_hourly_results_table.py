"""
Genera tablas de resultados horarios (P1-P24) vs real — Antioquia ene-jun 2026.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.models.metrics import calculate_mape
from src.prediction.forecaster import ForecastPipeline

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("hourly_table")
logger.setLevel(logging.INFO)

UCP = "Antioquia"
OUT = ROOT / "reports" / "performance" / "antioquia_ene_jun_2026"
OUT.mkdir(parents=True, exist_ok=True)

START = pd.Timestamp("2026-01-01")
END = pd.Timestamp("2026-06-01")
CUTOFF = START - pd.Timedelta(days=1)
PERIODS = [f"P{i}" for i in range(1, 25)]
MONTH_NAMES = {1: "Ene", 2: "Feb", 3: "Mar", 4: "Abr", 5: "May", 6: "Jun"}


def main() -> None:
    features_path = ROOT / f"data/features/{UCP}/data_with_features_latest.csv"
    model_path = ROOT / f"models/{UCP}/registry/champion_model.joblib"
    climate_path = ROOT / f"data/raw/{UCP}/clima_new.csv"

    df_full = pd.read_csv(features_path)
    df_full["FECHA"] = pd.to_datetime(df_full["FECHA"])
    actuals = df_full[
        (df_full["FECHA"] >= START) & (df_full["FECHA"] <= END)
    ][["FECHA", "TOTAL"] + PERIODS].copy()
    actuals = actuals.rename(columns={"FECHA": "fecha", "TOTAL": "total_real"})
    n_days = len(actuals)
    logger.info("Periodo: %s -> %s (%d dias)", START.date(), END.date(), n_days)

    pipeline = ForecastPipeline(
        model_path=str(model_path),
        historical_data_path=str(features_path),
        raw_climate_path=str(climate_path),
        enable_hourly_disaggregation=True,
        ucp=UCP,
    )
    pipeline.df_historico = pipeline.df_historico[
        pipeline.df_historico["fecha"] <= CUTOFF
    ].copy()

    logger.info("Prediciendo %d dias con desagregacion horaria...", n_days)
    preds = pipeline.predict_next_n_days(n_days=n_days)
    preds["fecha"] = pd.to_datetime(preds["fecha"]).dt.normalize()
    actuals["fecha"] = pd.to_datetime(actuals["fecha"]).dt.normalize()

    merged = actuals.merge(
        preds[["fecha", "demanda_predicha"] + PERIODS],
        on="fecha",
        how="inner",
        suffixes=("_real", "_pred"),
    )
    # After merge: P1_real, P1_pred, etc. and demanda_predicha, total_real
    logger.info("Dias emparejados: %d", len(merged))

    # Long format: one row per day-hour
    rows = []
    for _, r in merged.iterrows():
        fecha = r["fecha"]
        for i, p in enumerate(PERIODS, start=1):
            real = float(r[f"{p}_real"])
            pred = float(r[f"{p}_pred"])
            err = pred - real
            ape = abs(err) / real * 100 if real else np.nan
            rows.append(
                {
                    "fecha": fecha.strftime("%Y-%m-%d"),
                    "mes": int(fecha.month),
                    "mes_nombre": MONTH_NAMES[int(fecha.month)],
                    "periodo": p,
                    "hora": i,  # P1=hora 1 ... P24=hora 24
                    "real_mwh": real,
                    "pronostico_mwh": pred,
                    "error_mwh": err,
                    "ape_pct": ape,
                }
            )
    long_df = pd.DataFrame(rows)

    # Resumen por mes
    by_month = []
    for mes in range(1, 7):
        sub = long_df[long_df["mes"] == mes]
        if len(sub) == 0:
            continue
        by_month.append(
            {
                "mes": MONTH_NAMES[mes],
                "n_horas": int(len(sub)),
                "n_dias": int(sub["fecha"].nunique()),
                "mape_horario_pct": calculate_mape(sub["real_mwh"], sub["pronostico_mwh"]),
                "mae_mwh": float(np.mean(np.abs(sub["error_mwh"]))),
                "pct_horas_ape_lt_5": float((sub["ape_pct"] < 5).mean() * 100),
            }
        )
    month_df = pd.DataFrame(by_month)

    # Resumen por periodo horario (P1-P24)
    by_period = []
    for p in PERIODS:
        sub = long_df[long_df["periodo"] == p]
        by_period.append(
            {
                "periodo": p,
                "hora": int(p[1:]),
                "mape_pct": calculate_mape(sub["real_mwh"], sub["pronostico_mwh"]),
                "mae_mwh": float(np.mean(np.abs(sub["error_mwh"]))),
                "real_promedio": float(sub["real_mwh"].mean()),
                "pronostico_promedio": float(sub["pronostico_mwh"].mean()),
            }
        )
    period_df = pd.DataFrame(by_period)

    # Pivot mes x periodo (MAPE)
    pivot_mape = (
        long_df.groupby(["mes_nombre", "periodo"], sort=False)
        .apply(
            lambda g: calculate_mape(g["real_mwh"], g["pronostico_mwh"]),
            include_groups=False,
        )
        .reset_index(name="mape")
    )
    # order months
    mes_order = ["Ene", "Feb", "Mar", "Abr", "May", "Jun"]
    pivot_mape["mes_nombre"] = pd.Categorical(
        pivot_mape["mes_nombre"], categories=mes_order, ordered=True
    )
    pivot_wide = pivot_mape.pivot(
        index="periodo", columns="mes_nombre", values="mape"
    ).reindex(PERIODS)
    pivot_wide = pivot_wide.reindex(columns=[m for m in mes_order if m in pivot_wide.columns])

    # Overall
    overall_mape = calculate_mape(long_df["real_mwh"], long_df["pronostico_mwh"])
    overall_mae = float(np.mean(np.abs(long_df["error_mwh"])))
    overall_pct5 = float((long_df["ape_pct"] < 5).mean() * 100)

    # Daily totals check (sum of hours)
    day_check = (
        long_df.groupby("fecha")
        .agg(real=("real_mwh", "sum"), pred=("pronostico_mwh", "sum"))
        .reset_index()
    )
    day_mape = calculate_mape(day_check["real"], day_check["pred"])

    # Save CSV
    long_path = OUT / "resultados_horarios_ene_jun_2026.csv"
    long_df.to_csv(long_path, index=False, float_format="%.4f")
    month_df.to_csv(OUT / "resumen_horario_por_mes.csv", index=False, float_format="%.4f")
    period_df.to_csv(OUT / "resumen_horario_por_periodo.csv", index=False, float_format="%.4f")
    pivot_wide.to_csv(OUT / "mape_horario_mes_x_periodo.csv", float_format="%.4f")

    # Excel with multiple sheets
    xlsx_path = OUT / "resultados_horarios_ene_jun_2026.xlsx"
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        # Sample / full detail may be large (152*24=3648 rows) - fine for excel
        long_df.to_excel(writer, sheet_name="Detalle_horario", index=False)
        month_df.to_excel(writer, sheet_name="Resumen_por_mes", index=False)
        period_df.to_excel(writer, sheet_name="Resumen_por_periodo", index=False)
        pivot_wide.to_excel(writer, sheet_name="MAPE_mes_x_periodo")
        # Wide day view: fecha + P1_real...P24_real + P1_pred...
        wide_cols = ["fecha"]
        export_wide = merged[["fecha"]].copy()
        export_wide["fecha"] = export_wide["fecha"].dt.strftime("%Y-%m-%d")
        for p in PERIODS:
            export_wide[f"{p}_real"] = merged[f"{p}_real"].values
            export_wide[f"{p}_pronostico"] = merged[f"{p}_pred"].values
        export_wide.to_excel(writer, sheet_name="Diario_P1_P24", index=False)

        meta = pd.DataFrame(
            [
                {"metrica": "MAPE_horario_global_pct", "valor": overall_mape},
                {"metrica": "MAE_horario_global_mwh", "valor": overall_mae},
                {"metrica": "pct_horas_APE_lt_5", "valor": overall_pct5},
                {"metrica": "MAPE_diario_suma_P_pct", "valor": day_mape},
                {"metrica": "n_dias", "valor": len(merged)},
                {"metrica": "n_horas", "valor": len(long_df)},
                {"metrica": "periodo_inicio", "valor": str(merged["fecha"].min().date())},
                {"metrica": "periodo_fin", "valor": str(merged["fecha"].max().date())},
            ]
        )
        meta.to_excel(writer, sheet_name="Resumen_global", index=False)

    print("=" * 70)
    print("RESULTADOS HORARIOS vs REAL")
    print("=" * 70)
    print(f"MAPE horario global: {overall_mape:.3f}%")
    print(f"MAE horario global: {overall_mae:.2f} MWh")
    print(f"% horas APE<5%: {overall_pct5:.1f}%")
    print(f"MAPE diario (suma P): {day_mape:.3f}%")
    print(month_df.to_string(index=False))
    print("Saved:", xlsx_path)
    print("Saved:", long_path)


if __name__ == "__main__":
    main()
