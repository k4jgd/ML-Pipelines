"""
Online Booking Forecast pipeline
==================================
Goal: forecast next-visit OnlineBookingCount per location so ops/staffing can plan
online-booking capacity ahead of time.

Target : NextOnlineBookingCount = OnlineBookingCount of the location's next recorded day (lead-1)
Split  : chronological (train = earliest ~75% of dates, test = most recent ~25%)
Models : Linear Regression, Ridge, Random Forest, Gradient Boosting, XGBoost
Metric : MAE, RMSE, R2 (regression - continuous demand count)
Tracking: every run + every candidate model is logged to MLflow (params, metrics, artifacts).
         The best candidate is registered in the MLflow Model Registry as the initial
         `champion` (see scripts/common/registry.py); a later retraining run (handled by the
         ETL pipeline) only replaces it if a challenger beats it.

Shares the same Demand & Operations preprocessing/feature engineering as
`scripts/demand_operations/preprocess.py` — only the target column differs.
"""

import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import pandas as pd
import seaborn as sns
from mlflow.tracking import MlflowClient

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "datasets" / "Demand&Operations.csv"
OUT = ROOT / "outputs" / "online_booking_forecast"

if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))
from scripts.common.mlflow_setup import configure
from scripts.common.registry import compare_and_promote, load_champion_metric
from scripts.online_booking_forecast import config, preprocess, validate
from scripts.online_booking_forecast import models as model_definitions


def main():
    warnings.filterwarnings("ignore")
    sns.set_theme(style="whitegrid")
    OUT.mkdir(parents=True, exist_ok=True)
    configure("online_booking_forecast")

    # 1. Load + EDA
    df = preprocess.load_data(DATA)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.histplot(df["OnlineBookingCount"], bins=15, ax=axes[0], color="#4C72B0")
    axes[0].set_title("Distribution of Daily OnlineBookingCount")

    for loc, grp in df.groupby("LocationId"):
        axes[1].plot(grp["Date"], grp["OnlineBookingCount"], marker="o", label=loc[:8])
    axes[1].set_title("OnlineBookingCount over time by Location")
    axes[1].legend(fontsize=7)
    axes[1].tick_params(axis="x", rotation=30)

    plt.tight_layout()
    plt.savefig(OUT / "01_eda_overview.png", dpi=150)
    plt.close()

    # 2-3. Cleaning / preprocessing + feature engineering
    df = preprocess.preprocess(df)
    model_df = preprocess.engineer_features(df)

    # 4. Chronological train/test split
    train, test, cutoff = preprocess.chronological_split(model_df)
    X_train, y_train = train[config.FEATURE_COLS], train[config.TARGET_COL]
    X_test, y_test = test[config.FEATURE_COLS], test[config.TARGET_COL]

    print(
        f"[OnlineBookingForecast] train={len(train)} rows, test={len(test)} rows, cutoff={cutoff.date()}"
    )

    # 5. Candidate models (each fit/eval logged as its own MLflow run)
    models = model_definitions.get_candidate_models()

    results = []
    predictions = {}
    run_ids = {}
    client = MlflowClient()
    with mlflow.start_run(run_name="online_booking_forecast_experiment") as parent_run:
        mlflow.set_tag("pipeline_stage", "initial_training")
        mlflow.log_param("n_rows_raw", len(df))
        mlflow.log_param("n_locations", df["LocationId"].nunique())
        mlflow.log_param("train_rows", len(train))
        mlflow.log_param("test_rows", len(test))
        mlflow.log_param("split_cutoff_date", str(cutoff.date()))
        mlflow.log_param("target", config.TARGET_COL)
        mlflow.log_param("features", ",".join(config.FEATURE_COLS))
        mlflow.log_artifact(str(OUT / "01_eda_overview.png"))

        for name, model in models.items():
            with mlflow.start_run(run_name=name, nested=True) as child_run:
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                predictions[name] = preds
                run_ids[name] = child_run.info.run_id
                metrics = validate.evaluate(y_test, preds)
                results.append({"model": name, **metrics})

                mlflow.log_params(model.get_params())
                for metric_name, value in metrics.items():
                    mlflow.log_metric(metric_name, value)
                if name == "XGBoost":
                    mlflow.xgboost.log_model(model, name="model")
                else:
                    mlflow.sklearn.log_model(
                        model,
                        name="model",
                        skops_trusted_types=["sklearn.tree._tree.Tree"],
                    )

        results_df = pd.DataFrame(results).sort_values("RMSE")
        print(results_df.to_string(index=False))

        best_name = results_df.iloc[0]["model"]
        best_model = models[best_name]
        best_preds = predictions[best_name]
        mlflow.log_param("best_model", best_name)
        mlflow.log_metric("best_RMSE", results_df.iloc[0]["RMSE"])
        mlflow.log_metric("best_MAE", results_df.iloc[0]["MAE"])
        mlflow.log_metric("best_R2", results_df.iloc[0]["R2"])
        results_df.to_csv(OUT / "model_comparison.csv", index=False)
        mlflow.log_artifact(str(OUT / "model_comparison.csv"))

        # Register the best candidate in the Model Registry. If a champion already
        # exists (e.g. this script was re-run), it is only replaced if this run's
        # best model actually beats it on this run's own test set.
        champion_metric, champion_version = load_champion_metric(
            client,
            config.REGISTERED_MODEL_NAME,
            X_test,
            y_test,
            validate.evaluate_primary,
        )
        decision = compare_and_promote(
            client=client,
            registered_model_name=config.REGISTERED_MODEL_NAME,
            run_id=run_ids[best_name],
            artifact_path="model",
            challenger_metric=results_df.iloc[0]["RMSE"],
            champion_metric=champion_metric,
            higher_is_better=config.HIGHER_IS_BETTER,
            extra_tags={
                "source_run_id": parent_run.info.run_id,
                "trained_model_name": best_name,
            },
        )
        mlflow.set_tag("promotion_decision", decision.reason)
        mlflow.log_metric("promoted", int(decision.promoted))
        print(
            f"Registry: {decision.reason}"
            + (
                f" -> promoted version {decision.new_version}"
                if decision.promoted
                else ""
            )
        )

        # 6. Evaluation plots
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].bar(results_df["model"], results_df["RMSE"], color="#55A868")
        axes[0].set_title("Model Comparison (RMSE, lower is better)")
        axes[0].tick_params(axis="x", rotation=20)

        axes[1].scatter(y_test, best_preds, alpha=0.7, color="#C44E52")
        lims = [
            min(y_test.min(), best_preds.min()),
            max(y_test.max(), best_preds.max()),
        ]
        axes[1].plot(lims, lims, "k--", linewidth=1)
        axes[1].set_xlabel("Actual NextOnlineBookingCount")
        axes[1].set_ylabel("Predicted NextOnlineBookingCount")
        axes[1].set_title(f"Best Model: {best_name} — Predicted vs Actual")

        plt.tight_layout()
        plt.savefig(OUT / "02_model_comparison.png", dpi=150)
        plt.close()
        mlflow.log_artifact(str(OUT / "02_model_comparison.png"))

        if hasattr(best_model, "feature_importances_"):
            imp = pd.Series(
                best_model.feature_importances_, index=config.FEATURE_COLS
            ).sort_values()
            plt.figure(figsize=(8, 6))
            imp.plot(kind="barh", color="#4C72B0")
            plt.title(f"{best_name} Feature Importance")
            plt.tight_layout()
            plt.savefig(OUT / "03_feature_importance.png", dpi=150)
            plt.close()
            mlflow.log_artifact(str(OUT / "03_feature_importance.png"))

        residuals = y_test.values - best_preds
        plt.figure(figsize=(7, 5))
        sns.histplot(residuals, bins=12, kde=True, color="#8172B2")
        plt.title(f"{best_name} Residual Distribution (Test Set)")
        plt.xlabel("Residual (Actual - Predicted)")
        plt.tight_layout()
        plt.savefig(OUT / "04_residuals.png", dpi=150)
        plt.close()
        mlflow.log_artifact(str(OUT / "04_residuals.png"))

    print(f"\nBest model: {best_name}")
    print(f"Outputs saved to: {OUT}")


if __name__ == "__main__":
    main()
