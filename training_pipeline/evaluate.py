"""
evaluate.py  (MONGODB-ONLY)
-----------
Reads all metrics directly from the model_metrics collection written by
mongo_store.save_metrics() during training.  No local JSON files are read.
The final evaluation summary document is upserted into pipeline_runs.
"""

import os
import pandas as pd
import pymongo
from mongo_store import get_db, _run_id


def run_evaluation_suite():
    print("=" * 80)
    print("               MLOPS SUITE: CROSS-MODEL EVALUATION (MongoDB)              ")
    print("=" * 80)

    db = get_db()
    print("🔗 Connected to MongoDB Atlas.")

    horizons      = [24, 48, 72]
    model_names   = ["RandomForest", "XGBoost", "Ridge"]
    compiled      = {}

    for h in horizons:
        hkey = f"{h}h"
        compiled[hkey] = {"models": {}, "Winning_Model": "None"}
        best_mae   = float("inf")
        best_engine = "None"

        for model_name in model_names:
            doc = db["model_metrics"].find_one(
                {"model": model_name, "horizon_h": h},
                {"_id": 0},
            )
            if not doc:
                compiled[hkey]["models"][model_name] = {
                    "MAE": None, "R2": None, "MAPE": None,
                    "Coverage": None, "RMSE": None, "Margin": None,
                }
                continue

            mae      = doc.get("test_mae")
            r2       = doc.get("test_r2")
            mape     = doc.get("test_mape")
            coverage = doc.get("conformal_global_coverage")
            rmse     = doc.get("test_rmse")
            margin   = doc.get("conformal_margin") or doc.get("conformal_margin_width")

            compiled[hkey]["models"][model_name] = {
                "MAE":      float(mae)      if mae      is not None else None,
                "R2":       float(r2)       if r2       is not None else None,
                "MAPE":     float(mape)     if mape     is not None else None,
                "Coverage": float(coverage) if coverage is not None else None,
                "RMSE":     float(rmse)     if rmse     is not None else None,
                "Margin":   float(margin)   if margin   is not None else None,
            }

            if mae is not None and mae < best_mae:
                best_mae    = mae
                best_engine = model_name
            elif mae is not None and mae == best_mae:
                best_engine = f"{best_engine} / {model_name}"

        compiled[hkey]["Winning_Model"] = best_engine

    # ── Print dashboard ───────────────────────────────────────────────────────
    print("\n" + "=" * 95)
    print(f"{'Horizon':<8} | {'Model':<15} | {'MAE':>10} | {'MAPE':>10} | {'R²':>12} | {'Coverage':>12}")
    print("=" * 95)

    for h in horizons:
        hkey = f"{h}h"
        for model_name in model_names:
            m = compiled[hkey]["models"][model_name]
            mae_s  = f"{m['MAE']:.1f}"       if m["MAE"]      is not None else "N/A"
            mape_s = f"{m['MAPE']:.2f}%"     if m["MAPE"]     is not None else "N/A"
            r2_s   = f"{m['R2']:.3f}"        if m["R2"]       is not None else "N/A"
            cov_s  = f"{m['Coverage']*100:.1f}%" if m["Coverage"] is not None else "N/A"
            print(f"{hkey:<8} | {model_name:<15} | {mae_s:>10} | {mape_s:>10} | {r2_s:>12} | {cov_s:>12}")

        print(f"🏆 Winning Engine ({hkey}): {compiled[hkey]['Winning_Model']}")
        print("-" * 95)

    # ── Push summary to pipeline_runs ─────────────────────────────────────────
    try:
        payload = {
            "type":               "automated_pipeline_evaluation",
            "timestamp":          pd.Timestamp.now(tz="UTC").to_pydatetime(),
            "pipeline_run_status": "SUCCESS",
            "performance_summary": compiled,
        }
        db["pipeline_runs"].insert_one(payload)
        print("\n🚀 Evaluation summary pushed to pipeline_runs collection.")
    except Exception as e:
        print(f"⚠️  Could not write to pipeline_runs: {e}")

    print("\n✅ Evaluation complete. All data sourced from and stored in MongoDB Atlas.")


if __name__ == "__main__":
    run_evaluation_suite()