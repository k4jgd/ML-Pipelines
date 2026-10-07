"""Train and register the expected machine utilization engine."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient

from scripts.common.mlflow_setup import configure
from scripts.common.registry import compare_and_promote, load_champion_metric
from scripts.utilization import config, preprocess, validate
from scripts.utilization import models as model_definitions


def main(stage="initial_training"):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-path", type=Path, default=ROOT / "datasets" / "Utilization.csv"
    )
    parser.add_argument("--min-improvement", type=float, default=0)
    args = parser.parse_args()
    if args.min_improvement < 0:
        parser.error("--min-improvement must be nonnegative")
    raw = preprocess.load_data(args.data_path)
    base = preprocess.preprocess(raw)
    train, test, cutoff = preprocess.chronological_split(
        preprocess.engineer_features(base)
    )
    X_train, y_train = (
        train[config.FEATURE_COLS].astype(float),
        train[config.TARGET_COL],
    )
    X_test, y_test = test[config.FEATURE_COLS].astype(float), test[config.TARGET_COL]
    out = ROOT / "outputs" / "utilization"
    out.mkdir(parents=True, exist_ok=True)
    configure("utilization")
    client = MlflowClient()
    results, runs, predictions = [], {}, {}
    with mlflow.start_run(run_name=f"{stage}_utilization") as parent:
        mlflow.set_tag("pipeline_stage", stage)
        mlflow.log_params(
            {
                "data_path": str(args.data_path),
                "raw_rows": len(raw),
                "machines": base.MachineId.nunique(),
                "train_rows": len(train),
                "test_rows": len(test),
                "target_cutoff": str(cutoff.date()),
                "target": config.TARGET_COL,
                "capacity_minutes": 1440,
                "features": ",".join(config.FEATURE_COLS),
                "last_complete_day": str(base.Date.max().date()),
                "unknown_machine_days": int(base.UnknownDuration.sum()),
            }
        )
        for name, model in model_definitions.get_candidate_models().items():
            with mlflow.start_run(run_name=name, nested=True) as run:
                model_definitions.fit_candidate(name, model, X_train, y_train)
                preds = validate.bounded_predictions(model.predict(X_test))
                metrics = validate.evaluate(y_test, preds)
                mlflow.log_params(model.get_params())
                mlflow.log_metrics(metrics)
                mlflow.sklearn.log_model(
                    model,
                    name="model",
                    input_example=X_train.head(3),
                    signature=infer_signature(X_train, model.predict(X_train)),
                    skops_trusted_types=["sklearn.tree._tree.Tree"],
                )
                results.append({"model": name, **metrics})
                runs[name], predictions[name] = run.info.run_id, preds
        comparison = pd.DataFrame(results).sort_values("RMSE")
        comparison.to_csv(out / "model_comparison.csv", index=False)
        best = comparison.iloc[0]
        champion_metric, version = load_champion_metric(
            client,
            config.REGISTERED_MODEL_NAME,
            X_test,
            y_test,
            validate.evaluate_primary,
        )
        decision = compare_and_promote(
            client=client,
            registered_model_name=config.REGISTERED_MODEL_NAME,
            run_id=runs[best.model],
            artifact_path="model",
            challenger_metric=float(best.RMSE),
            champion_metric=champion_metric,
            higher_is_better=False,
            min_improvement=args.min_improvement,
            extra_tags={
                "source_run_id": parent.info.run_id,
                "trained_model_name": best.model,
                "unit": "%",
                "horizon": "next_calendar_day",
                "capacity_minutes": 1440,
            },
        )
        mlflow.set_tag("promotion_decision", decision.reason)
        mlflow.log_metrics(
            {
                "best_RMSE": float(best.RMSE),
                "best_MAE": float(best.MAE),
                "promoted": int(decision.promoted),
            }
        )
        report = {
            "raw_sessions": len(raw),
            "machine_days": len(base),
            "machines": int(base.MachineId.nunique()),
            "train_rows": len(train),
            "test_rows": len(test),
            "last_complete_day": str(base.Date.max().date()),
            "target_cutoff": str(cutoff.date()),
            "unknown_machine_days": int(base.UnknownDuration.sum()),
            "best_model": best.model,
            "metrics": {k: float(best[k]) for k in ["MAE", "RMSE", "R2"]},
            "persistence_baseline": validate.evaluate(y_test, test.UtilizationPct),
            "champion_compared": version,
            "champion_metric": champion_metric,
            "promoted": decision.promoted,
            "promotion_reason": decision.reason,
        }
        (out / "training_report.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        scored = test[["MachineId", "Date", "TargetDate", config.TARGET_COL]].copy()
        scored["PredictedUtilizationPct"] = predictions[best.model]
        scored.to_csv(out / "test_predictions.csv", index=False)
        fig, axes = plt.subplots(1, 3, figsize=(16, 4))
        for machine, group in base.groupby("MachineId"):
            axes[0].plot(group.Date, group.UtilizationPct, alpha=0.6)
        axes[0].set(title="Daily machine utilization", ylabel="% of 24 hours")
        axes[0].tick_params(axis="x", rotation=45)
        axes[1].bar(comparison.model, comparison.RMSE)
        axes[1].set(title="Held-out RMSE", ylabel="percentage points")
        axes[1].tick_params(axis="x", rotation=25)
        axes[2].scatter(y_test, predictions[best.model], alpha=0.6)
        axes[2].set(
            title=str(best.model),
            xlabel="Actual utilization (%)",
            ylabel="Predicted utilization (%)",
        )
        fig.tight_layout()
        fig.savefig(out / "evaluation.png", dpi=150)
        plt.close(fig)
        for filename in [
            "model_comparison.csv",
            "training_report.json",
            "test_predictions.csv",
            "evaluation.png",
        ]:
            mlflow.log_artifact(str(out / filename))
    print(comparison.to_string(index=False))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
