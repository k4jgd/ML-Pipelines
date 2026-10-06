"""
Cancellation pipeline
======================
Goal: flag locations at risk of cancellations on their next visit day, so
ops can follow up proactively (reminders, overbooking buffer, etc.).

Target : NextCancellationFlag = 1 if CancellationCount > 0 on the location's
         next recorded day (lead-1), else 0
Split  : chronological (train = earliest ~75% of dates, test = most recent ~25%)
Models : Logistic Regression, Random Forest, Gradient Boosting, XGBoost
Metric : Accuracy, Precision, Recall, F1, ROC_AUC (binary classification)
Tracking: every run + every candidate model is logged to MLflow (params, metrics, artifacts).
         The best candidate is registered in the MLflow Model Registry as the initial
         `champion` (see pipeline/registry.py); a later retraining run (handled by the
         ETL pipeline) only replaces it if a challenger beats it.

Shares the same Demand & Operations preprocessing/feature engineering as
`demand_operations.py` (see `pipeline/core/demand_operations_core.py` and
`pipeline/core/cancellation_core.py`) — only the target and model family differ.
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
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "Demand&Operations.csv"
OUT = ROOT / "outputs" / "cancellation"
OUT.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mlflow_setup import configure
from registry import compare_and_promote, load_champion_metric
from core import cancellation_core as core

configure("cancellation")

# 1. Load + EDA
df = core.load_data(DATA)
df["Cancelled"] = (df["CancellationCount"] > 0).astype(int)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
df["Cancelled"].value_counts().sort_index().plot(kind="bar", ax=axes[0], color="#C44E52")
axes[0].set_title("Days With vs Without Cancellations")
axes[0].set_xticklabels(["No cancellations", "Has cancellations"], rotation=0)

for loc, grp in df.groupby("LocationId"):
    axes[1].plot(grp["Date"], grp["CancellationCount"], marker="o", label=loc[:8])
axes[1].set_title("CancellationCount over time by Location")
axes[1].legend(fontsize=7)
axes[1].tick_params(axis="x", rotation=30)

plt.tight_layout()
plt.savefig(OUT / "01_eda_overview.png", dpi=150)
plt.close()

# 2-3. Cleaning / preprocessing + feature engineering
df = core.preprocess(df)
model_df = core.engineer_features(df)

# 4. Chronological train/test split
train, test, cutoff = core.chronological_split(model_df)
X_train, y_train = train[core.FEATURE_COLS], train[core.TARGET_COL]
X_test, y_test = test[core.FEATURE_COLS], test[core.TARGET_COL]

print(f"[Cancellation] train={len(train)} rows ({y_train.mean():.0%} positive), "
      f"test={len(test)} rows ({y_test.mean():.0%} positive), cutoff={cutoff.date()}")

# 5. Candidate models (each fit/eval logged as its own MLflow run)
models = core.get_candidate_models(y_train)

results = []
predictions = {}
probabilities = {}
run_ids = {}
client = MlflowClient()
with mlflow.start_run(run_name="cancellation_experiment") as parent_run:
    mlflow.set_tag("pipeline_stage", "initial_training")
    mlflow.log_param("n_rows_raw", len(df))
    mlflow.log_param("n_locations", df["LocationId"].nunique())
    mlflow.log_param("train_rows", len(train))
    mlflow.log_param("test_rows", len(test))
    mlflow.log_param("split_cutoff_date", str(cutoff.date()))
    mlflow.log_param("target", core.TARGET_COL)
    mlflow.log_param("features", ",".join(core.FEATURE_COLS))
    mlflow.log_artifact(str(OUT / "01_eda_overview.png"))

    for name, model in models.items():
        with mlflow.start_run(run_name=name, nested=True) as child_run:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            proba = model.predict_proba(X_test)[:, list(model.classes_).index(1)] \
                if hasattr(model, "predict_proba") and 1 in list(model.classes_) else None
            predictions[name] = preds
            probabilities[name] = proba
            run_ids[name] = child_run.info.run_id
            metrics = core.evaluate(y_test, preds, proba)
            results.append({"model": name, **metrics})

            mlflow.log_params(model.get_params())
            for metric_name, value in metrics.items():
                mlflow.log_metric(metric_name, value)
            if name == "XGBoost":
                mlflow.xgboost.log_model(model, name="model")
            else:
                mlflow.sklearn.log_model(
                    model, name="model",
                    skops_trusted_types=["sklearn.tree._tree.Tree"],
                )

    results_df = pd.DataFrame(results).sort_values(core.PRIMARY_METRIC, ascending=False)
    print(results_df.to_string(index=False))

    best_name = results_df.iloc[0]["model"]
    best_model = models[best_name]
    best_preds = predictions[best_name]
    best_proba = probabilities[best_name]
    mlflow.log_param("best_model", best_name)
    mlflow.log_metric("best_F1", results_df.iloc[0]["F1"])
    mlflow.log_metric("best_Accuracy", results_df.iloc[0]["Accuracy"])
    results_df.to_csv(OUT / "model_comparison.csv", index=False)
    mlflow.log_artifact(str(OUT / "model_comparison.csv"))

    # Register the best candidate in the Model Registry. If a champion already
    # exists (e.g. this script was re-run), it is only replaced if this run's
    # best model actually beats it on this run's own test set.
    champion_metric, champion_version = load_champion_metric(
        client, core.REGISTERED_MODEL_NAME, X_test, y_test, core.evaluate_primary
    )
    decision = compare_and_promote(
        client=client,
        registered_model_name=core.REGISTERED_MODEL_NAME,
        run_id=run_ids[best_name],
        artifact_path="model",
        challenger_metric=results_df.iloc[0]["F1"],
        champion_metric=champion_metric,
        higher_is_better=core.HIGHER_IS_BETTER,
        extra_tags={"source_run_id": parent_run.info.run_id, "trained_model_name": best_name},
    )
    mlflow.set_tag("promotion_decision", decision.reason)
    mlflow.log_metric("promoted", int(decision.promoted))
    print(f"Registry: {decision.reason}"
          + (f" -> promoted version {decision.new_version}" if decision.promoted else ""))

    # 6. Evaluation plots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].bar(results_df["model"], results_df["F1"], color="#55A868")
    axes[0].set_title("Model Comparison (F1, higher is better)")
    axes[0].tick_params(axis="x", rotation=20)

    ConfusionMatrixDisplay.from_predictions(y_test, best_preds, ax=axes[1], colorbar=False)
    axes[1].set_title(f"Best Model: {best_name} — Confusion Matrix")

    plt.tight_layout()
    plt.savefig(OUT / "02_model_comparison.png", dpi=150)
    plt.close()
    mlflow.log_artifact(str(OUT / "02_model_comparison.png"))

    if hasattr(best_model, "feature_importances_"):
        imp = pd.Series(best_model.feature_importances_, index=core.FEATURE_COLS).sort_values()
        plt.figure(figsize=(8, 6))
        imp.plot(kind="barh", color="#4C72B0")
        plt.title(f"{best_name} Feature Importance")
        plt.tight_layout()
        plt.savefig(OUT / "03_feature_importance.png", dpi=150)
        plt.close()
        mlflow.log_artifact(str(OUT / "03_feature_importance.png"))

    if best_proba is not None:
        plt.figure(figsize=(6, 6))
        RocCurveDisplay.from_predictions(y_test, best_proba)
        plt.title(f"{best_name} ROC Curve (Test Set)")
        plt.tight_layout()
        plt.savefig(OUT / "04_roc_curve.png", dpi=150)
        plt.close()
        mlflow.log_artifact(str(OUT / "04_roc_curve.png"))

print(f"\nBest model: {best_name}")
print(f"Outputs saved to: {OUT}")
