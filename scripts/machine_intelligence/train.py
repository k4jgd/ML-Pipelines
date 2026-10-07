"""
Machine Intelligence pipeline
==============================
Goal: predictive maintenance - flag machines likely to fail on their NEXT service day
so technicians can intervene before a breakdown.

Target : NextFailure = 1 if FailureCount > 0 on the machine's next recorded day, else 0
Split  : chronological (train = earliest ~75% of dates, test = most recent ~25%)
Models : Logistic Regression, Random Forest, Gradient Boosting, XGBoost (all class-weighted /
         scale-pos-weighted for the failure-minority class)
Metric : Accuracy, Precision, Recall, F1, ROC-AUC (imbalanced binary classification)
Tracking: every run + every candidate model is logged to MLflow (params, metrics, artifacts).
         The best candidate is registered in the MLflow Model Registry as the initial
         `champion` (see scripts/common/registry.py); a later retraining run (handled by the
         ETL pipeline) only replaces it if a challenger beats it.

Pipeline logic is separated into config, preprocess, models, and validate modules.
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
import numpy as np
import pandas as pd
import seaborn as sns
from mlflow.tracking import MlflowClient
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "datasets" / "MachineIntelligence.csv"
OUT = ROOT / "outputs" / "machine_intelligence"

if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))
from scripts.common.mlflow_setup import configure
from scripts.common.registry import compare_and_promote, load_champion_metric
from scripts.machine_intelligence import config, preprocess, validate
from scripts.machine_intelligence import models as model_definitions


def main():
    warnings.filterwarnings("ignore")
    sns.set_theme(style="whitegrid")
    OUT.mkdir(parents=True, exist_ok=True)
    configure("machine_intelligence")

    # 1. Load + EDA
    df = preprocess.load_data(DATA)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    sns.histplot(df["FailureCount"], bins=8, ax=axes[0, 0], color="#C44E52")
    axes[0, 0].set_title("Distribution of FailureCount per Record")

    fail_rate_by_machine = df.groupby("MachineId").apply(
        lambda x: (x["FailureCount"] > 0).mean()
    )
    fail_rate_by_machine.sort_values().plot(kind="barh", ax=axes[0, 1], color="#DD8452")
    axes[0, 1].set_title("Failure Rate by Machine")
    axes[0, 1].set_xlabel("Share of days with >=1 failure")

    sns.scatterplot(
        data=df,
        x="AvgCycleDurationMinutes",
        y="FailureCount",
        hue="OnlineObserved",
        ax=axes[1, 0],
        palette="viridis",
    )
    axes[1, 0].set_title("Cycle Duration vs Failures")

    corr_cols = [
        "SessionCount",
        "StartedSessionCount",
        "CompletedSessionCount",
        "AvgCycleDurationMinutes",
        "FailureCount",
        "OnlineObserved",
        "MaxConsecutiveFailures",
    ]
    sns.heatmap(
        df[corr_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=axes[1, 1]
    )
    axes[1, 1].set_title("Feature Correlation Heatmap")

    plt.tight_layout()
    plt.savefig(OUT / "01_eda_overview.png", dpi=150)
    plt.close()

    # 2-3. Cleaning / preprocessing + feature engineering
    df = preprocess.preprocess(df)
    model_df = preprocess.engineer_features(df)

    # 4. Chronological train/test split
    train, test, cutoff = preprocess.chronological_split(model_df)
    X_train, y_train = train[config.FEATURE_COLS], train[config.TARGET_COL].astype(int)
    X_test, y_test = test[config.FEATURE_COLS], test[config.TARGET_COL].astype(int)

    print(
        f"[MachineIntelligence] train={len(train)} rows ({y_train.mean():.1%} positive), "
        f"test={len(test)} rows ({y_test.mean():.1%} positive), cutoff={cutoff.date()}"
    )

    # 5. Candidate models
    models = model_definitions.get_candidate_models(y_train)

    results = []
    predictions, probabilities = {}, {}
    run_ids = {}
    client = MlflowClient()
    with mlflow.start_run(run_name="machine_intelligence_experiment") as parent_run:
        mlflow.set_tag("pipeline_stage", "initial_training")
        mlflow.log_param("n_rows_raw", len(df))
        mlflow.log_param("n_machines", df["MachineId"].nunique())
        mlflow.log_param("train_rows", len(train))
        mlflow.log_param("test_rows", len(test))
        mlflow.log_param("train_positive_rate", round(float(y_train.mean()), 4))
        mlflow.log_param("test_positive_rate", round(float(y_test.mean()), 4))
        mlflow.log_param("split_cutoff_date", str(cutoff.date()))
        mlflow.log_param("target", "NextFailure (FailureCount>0 on next service day)")
        mlflow.log_param("features", ",".join(config.FEATURE_COLS))
        mlflow.log_artifact(str(OUT / "01_eda_overview.png"))

        for name, model in models.items():
            with mlflow.start_run(run_name=name, nested=True) as child_run:
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                proba = model.predict_proba(X_test)[:, 1]
                predictions[name] = preds
                probabilities[name] = proba
                run_ids[name] = child_run.info.run_id

                metrics = validate.evaluate(y_test, preds, proba)
                results.append({"model": name, **metrics})

                mlflow.log_params({k: v for k, v in model.get_params().items()})
                for metric_name, value in metrics.items():
                    if not (isinstance(value, float) and np.isnan(value)):
                        mlflow.log_metric(metric_name, value)

                if name == "XGBoost":
                    mlflow.xgboost.log_model(model, name="model")
                else:
                    mlflow.sklearn.log_model(
                        model,
                        name="model",
                        skops_trusted_types=["sklearn.tree._tree.Tree"],
                    )

        results_df = pd.DataFrame(results).sort_values("F1", ascending=False)
        print(results_df.to_string(index=False))

        best_name = results_df.iloc[0]["model"]
        best_model = models[best_name]
        best_preds = predictions[best_name]

        mlflow.log_param("best_model", best_name)
        for metric_name in ["Accuracy", "Precision", "Recall", "F1", "ROC_AUC"]:
            mlflow.log_metric(
                f"best_{metric_name}", float(results_df.iloc[0][metric_name])
            )
        results_df.to_csv(OUT / "model_comparison.csv", index=False)
        mlflow.log_artifact(str(OUT / "model_comparison.csv"))

        # Register the best candidate in the Model Registry (see demand_operations.py for the
        # bootstrap-vs-challenger promotion logic shared via scripts/common/registry.py).
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
            challenger_metric=results_df.iloc[0]["F1"],
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
        metrics_plot = results_df.set_index("model")[
            ["Accuracy", "Precision", "Recall", "F1"]
        ]
        metrics_plot.plot(kind="bar", ax=axes[0])
        axes[0].set_title("Model Comparison")
        axes[0].tick_params(axis="x", rotation=20)
        axes[0].set_ylim(0, 1.05)

        ConfusionMatrixDisplay.from_predictions(
            y_test, best_preds, ax=axes[1], cmap="Blues"
        )
        axes[1].set_title(f"Confusion Matrix — {best_name}")

        plt.tight_layout()
        plt.savefig(OUT / "02_model_comparison.png", dpi=150)
        plt.close()
        mlflow.log_artifact(str(OUT / "02_model_comparison.png"))

        plt.figure(figsize=(6, 6))
        ax = plt.gca()
        for name, proba in probabilities.items():
            RocCurveDisplay.from_predictions(y_test, proba, name=name, ax=ax)
        plt.title("ROC Curves — All Candidate Models")
        plt.tight_layout()
        plt.savefig(OUT / "03_roc_curves.png", dpi=150)
        plt.close()
        mlflow.log_artifact(str(OUT / "03_roc_curves.png"))

        if hasattr(best_model, "feature_importances_"):
            imp = pd.Series(
                best_model.feature_importances_, index=config.FEATURE_COLS
            ).sort_values()
            plt.figure(figsize=(8, 6))
            imp.plot(kind="barh", color="#4C72B0")
            plt.title(f"{best_name} Feature Importance")
            plt.tight_layout()
            plt.savefig(OUT / "04_feature_importance.png", dpi=150)
            plt.close()
            mlflow.log_artifact(str(OUT / "04_feature_importance.png"))

    print(f"\nBest model: {best_name}")
    print(f"Outputs saved to: {OUT}")


if __name__ == "__main__":
    main()
