"""
Backtest Antioquia ene–jun 2026 usando ForecastPipeline completo
(incluye ajustes post-predicción: festivos/navidad + régimen +5.3%).
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.models.metrics import calculate_all_metrics, calculate_mape
from src.prediction.forecaster import ForecastPipeline

logging.basicConfig(level=logging.WARNING, format="%(levelname)s - %(message)s")
logger = logging.getLogger("backtest_adj")
logger.setLevel(logging.INFO)

UCP = "Antioquia"
OUT = ROOT / "reports" / "performance" / "antioquia_ene_jun_2026"
OUT.mkdir(parents=True, exist_ok=True)

START = pd.Timestamp("2026-01-01")
# Features llegan hasta 2026-06-01; raw a julio — usamos el mismo corte de features
END = pd.Timestamp("2026-06-01")
CUTOFF = START - pd.Timedelta(days=1)  # 2025-12-31


def classify_day(is_weekend: int, is_festivo: int) -> str:
    if int(is_weekend) == 1:
        return "Fin de semana"
    if int(is_festivo) == 1:
        return "Festivo"
    return "Laboral"


def main() -> None:
    model_path = ROOT / f"models/{UCP}/registry/champion_model.joblib"
    features_path = ROOT / f"data/features/{UCP}/data_with_features_latest.csv"
    climate_path = ROOT / f"data/raw/{UCP}/clima_new.csv"

    # Actuales (ground truth)
    df_full = pd.read_csv(features_path)
    df_full["FECHA"] = pd.to_datetime(df_full["FECHA"])
    actuals = df_full[
        (df_full["FECHA"] >= START) & (df_full["FECHA"] <= END)
    ][["FECHA", "TOTAL", "is_weekend", "is_festivo"]].copy()
    actuals = actuals.rename(columns={"FECHA": "fecha", "TOTAL": "real"})
    n_days = len(actuals)
    logger.info("Periodo real: %s → %s (%d días)", START.date(), END.date(), n_days)

    # Pipeline con histórico cortado ANTES del periodo (simula producción)
    pipeline = ForecastPipeline(
        model_path=str(model_path),
        historical_data_path=str(features_path),
        raw_climate_path=str(climate_path),
        enable_hourly_disaggregation=False,  # solo TOTAL diario
        ucp=UCP,
    )
    pipeline.df_historico = pipeline.df_historico[
        pipeline.df_historico["fecha"] <= CUTOFF
    ].copy()
    logger.info(
        "Histórico cortado en %s (%d filas)",
        CUTOFF.date(),
        len(pipeline.df_historico),
    )

    # Predicción recursiva con ajustes post
    logger.info("Ejecutando predict_next_n_days(%d)...", n_days)
    preds = pipeline.predict_next_n_days(n_days=n_days)
    preds["fecha"] = pd.to_datetime(preds["fecha"]).dt.normalize()
    actuals["fecha"] = pd.to_datetime(actuals["fecha"]).dt.normalize()

    merged = actuals.merge(
        preds[["fecha", "demanda_predicha", "is_festivo", "is_weekend"]],
        on="fecha",
        how="inner",
        suffixes=("_act", "_pred"),
    )
    # Preferir flags del pipeline (API festivos) para tipo de día operativo
    merged["tipo_dia"] = [
        classify_day(w, f)
        for w, f in zip(merged["is_weekend_pred"], merged["is_festivo_pred"])
    ]
    merged["mes"] = merged["fecha"].dt.month
    merged["mes_nombre"] = merged["fecha"].dt.strftime("%b")
    merged["ape"] = (
        np.abs(merged["real"] - merged["demanda_predicha"]) / merged["real"] * 100
    )
    merged["error"] = merged["demanda_predicha"] - merged["real"]

    logger.info("Días emparejados: %d", len(merged))

    overall = calculate_all_metrics(merged["real"], merged["demanda_predicha"])
    by_type = []
    for tipo in ["Laboral", "Fin de semana", "Festivo"]:
        sub = merged[merged["tipo_dia"] == tipo]
        if len(sub) == 0:
            continue
        m = calculate_all_metrics(sub["real"], sub["demanda_predicha"])
        by_type.append({"tipo_dia": tipo, "n": int(len(sub)), **m})

    by_month = []
    for mes in range(1, 7):
        sub = merged[merged["mes"] == mes]
        if len(sub) == 0:
            continue
        m = calculate_all_metrics(sub["real"], sub["demanda_predicha"])
        by_month.append(
            {
                "mes": mes,
                "mes_nombre": ["Ene", "Feb", "Mar", "Abr", "May", "Jun"][mes - 1],
                "n": int(len(sub)),
                **m,
            }
        )

    by_month_type = []
    for mes in range(1, 7):
        for tipo in ["Laboral", "Fin de semana", "Festivo"]:
            sub = merged[(merged["mes"] == mes) & (merged["tipo_dia"] == tipo)]
            if len(sub) == 0:
                continue
            by_month_type.append(
                {
                    "mes": mes,
                    "mes_nombre": ["Ene", "Feb", "Mar", "Abr", "May", "Jun"][mes - 1],
                    "tipo_dia": tipo,
                    "n": int(len(sub)),
                    "mape": calculate_mape(sub["real"], sub["demanda_predicha"]),
                    "mae": float(np.mean(np.abs(sub["real"] - sub["demanda_predicha"]))),
                }
            )

    def compliance(sub: pd.DataFrame) -> dict:
        ok = int((sub["ape"] < 5).sum())
        total = int(len(sub))
        return {
            "dias_ok": ok,
            "total": total,
            "pct": float(ok / total * 100) if total else 0.0,
        }

    # Comparar con métricas crudas previas si existen
    raw_path = OUT / "metrics_summary.json"
    raw_mape = None
    raw_comp = None
    if raw_path.exists():
        raw = json.loads(raw_path.read_text(encoding="utf-8"))
        raw_mape = raw.get("period_metrics", {}).get("lightgbm", {}).get("mape")
        raw_comp = raw.get("compliance", {}).get("overall")

    summary = {
        "ucp": UCP,
        "modo": "forecast_pipeline_con_ajustes_post",
        "ajustes_incluidos": [
            "mezcla_historica_festivos_navidad (60-70% lag_1y)",
            "ajuste_regimen_+5.3% desde 2026-03-25",
            "prediccion_recursiva (lags de predicciones previas)",
        ],
        "periodo": {
            "inicio": str(merged["fecha"].min().date()),
            "fin": str(merged["fecha"].max().date()),
            "n_dias": int(len(merged)),
            "cutoff_historico": str(CUTOFF.date()),
            "anio": 2026,
        },
        "champion": "LightGBM",
        "overall": overall,
        "by_type": by_type,
        "by_month": by_month,
        "by_month_type": by_month_type,
        "compliance": {
            "overall": compliance(merged),
            "by_type": {
                t: compliance(merged[merged["tipo_dia"] == t])
                for t in merged["tipo_dia"].unique()
            },
        },
        "comparacion_modelo_crudo": {
            "mape_crudo": raw_mape,
            "mape_con_ajustes": overall["mape"],
            "cumplimiento_crudo_pct": raw_comp.get("pct") if raw_comp else None,
            "cumplimiento_ajustes_pct": compliance(merged)["pct"],
        },
        "distribucion_tipo": merged["tipo_dia"].value_counts().to_dict(),
    }

    # Limpiar inf/nan para JSON
    def _clean(obj):
        if isinstance(obj, dict):
            return {k: _clean(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_clean(v) for v in obj]
        if isinstance(obj, float):
            if np.isnan(obj) or np.isinf(obj):
                return None
        return obj

    summary = _clean(summary)

    (OUT / "metrics_summary_adjusted.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    merged.to_csv(OUT / "daily_predictions_adjusted.csv", index=False)

    print("=" * 70)
    print("BACKTEST CON AJUSTES POST-PREDICCIÓN")
    print("=" * 70)
    print(f"MAPE overall: {overall['mape']:.3f}%")
    print(f"rMAPE: {overall['rmape']:.3f}  R2: {overall['r2']:.4f}")
    print(f"Cumplimiento APE<5%: {summary['compliance']['overall']['pct']:.1f}%")
    if raw_mape is not None:
        print(f"vs modelo crudo MAPE: {raw_mape:.3f}% -> {overall['mape']:.3f}%")
    print("Por tipo:")
    for r in by_type:
        print(f"  {r['tipo_dia']}: MAPE={r['mape']:.3f}% n={r['n']}")
    print("Por mes:")
    for r in by_month:
        print(f"  {r['mes_nombre']}: MAPE={r['mape']:.3f}% n={r['n']}")
    print("Saved:", OUT / "metrics_summary_adjusted.json")


if __name__ == "__main__":
    main()
