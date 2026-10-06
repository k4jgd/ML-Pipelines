"""
Customer Intelligence pipeline
================================
Goal: forecast each client's NEXT-visit quoted revenue so account managers can anticipate
high-value visits and prioritise retention/upsell outreach.

EDA finding: QuotedRevenuePaise is 0 for every single visit before 2026-09-19 and strictly
positive for (almost) every visit from 2026-09-19 onward, across ALL clients simultaneously.
That is a platform-wide monetization event (e.g. a pricing/quoting feature going live), not a
per-client behavioural signal. A chronological split on a binary "did this visit convert"
target is therefore degenerate (train would be 100% one class, test 100% the other) and not a
meaningful classification problem. We instead forecast the continuous revenue value, which
lets lag/trend features legitimately learn the regime change instead of being broken by it.

Target : NextQuotedRevenue = QuotedRevenuePaise on the client's next recorded visit
Split  : chronological (train = earliest ~75% of dates, test = most recent ~25%)
Models : Linear Regression, Ridge, Random Forest, Gradient Boosting, XGBoost
Metric : MAE, RMSE, R2 (regression - continuous revenue in paise)
Tracking: every run + every candidate model is logged to MLflow (params, metrics, artifacts).
         The best candidate is registered in the MLflow Model Registry as the initial
         `champion` (see pipeline/registry.py); a later retraining run (handled by the
         ETL pipeline) only replaces it if a challenger beats it.

Preprocessing / feature engineering / candidate models / evaluation live in
`pipeline/core/customer_intelligence_core.py`.
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

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "CustomerIntelligence.csv"
OUT = ROOT / "outputs" / "customer_intelligence"
OUT.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mlflow_setup import configure
from registry import compare_and_promote, load_champion_metric
from core import customer_intelligence_core as core

configure("customer_intelligence")

# 1. Load + EDA
df = core.load_data(DATA)

fig, axes = plt.subplots(2, 2, figsize=(12, 9))
sns.histplot(df["QuotedRevenuePaise"], bins=15, ax=axes[0, 0], color="#4C72B0")
axes[0, 0].set_title("Distribution of QuotedRevenuePaise")

for client_id, grp in df.groupby("ClientId"):
    axes[0, 1].plot(grp["Date"], grp["QuotedRevenuePaise"], marker="o", label=client_id[:8])
axes[0, 1].axvline(pd.Timestamp("2026-09-19"), color="red", linestyle="--", linewidth=1,
                    label="monetization switch-on")
axes[0, 1].set_title("QuotedRevenuePaise over time by Client")
axes[0, 1].legend(fontsize=7)
axes[0, 1].tick_params(axis="x", rotation=30)

daily_conv = df.groupby("Date")["Converted"].mean()
axes[1, 0].plot(daily_conv.index, daily_conv.values, marker="o", color="#DD8452")
axes[1, 0].axvline(pd.Timestamp("2026-09-19"), color="red", linestyle="--", linewidth=1)
axes[1, 0].set_title("Daily Share of Visits with QuotedRevenue > 0\n(step change = monetization event)")
axes[1, 0].tick_params(axis="x", rotation=30)

corr_cols = ["BookingCount", "WalkInCount", "OnlineBookingCount", "CancellationCount",
             "CourseCount", "MachineTypeCount", "QuotedRevenuePaise"]
sns.heatmap(df[corr_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=axes[1, 1])
axes[1, 1].set_title("Feature Correlation Heatmap")

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

print(f"[CustomerIntelligence] train={len(train)} rows, test={len(test)} rows, cutoff={cutoff.date()}")

# 5. Candidate models
models = core.get_candidate_models()

results = []
predictions = {}
run_ids = {}
client = MlflowClient()
with mlflow.start_run(run_name="customer_intelligence_experiment") as parent_run:
    mlflow.set_tag("pipeline_stage", "initial_training")
    mlflow.log_param("n_rows_raw", len(df))
    mlflow.log_param("n_clients", df["ClientId"].nunique())
    mlflow.log_param("train_rows", len(train))
    mlflow.log_param("test_rows", len(test))
    mlflow.log_param("split_cutoff_date", str(cutoff.date()))
    mlflow.log_param("target", "NextQuotedRevenue (next visit's QuotedRevenuePaise per client)")
    mlflow.log_param("features", ",".join(core.FEATURE_COLS))
    mlflow.log_param("eda_note", "monetization switched on ~2026-09-19 for all clients (see 01_eda_overview.png)")
    mlflow.log_artifact(str(OUT / "01_eda_overview.png"))

    for name, model in models.items():
        with mlflow.start_run(run_name=name, nested=True) as child_run:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            predictions[name] = preds
            run_ids[name] = child_run.info.run_id
            metrics = core.evaluate(y_test, preds)
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

    # Register the best candidate in the Model Registry (see demand_operations.py for the
    # bootstrap-vs-challenger promotion logic shared via pipeline/registry.py).
    champion_metric, champion_version = load_champion_metric(
        client, core.REGISTERED_MODEL_NAME, X_test, y_test, core.evaluate_primary
    )
    decision = compare_and_promote(
        client=client,
        registered_model_name=core.REGISTERED_MODEL_NAME,
        run_id=run_ids[best_name],
        artifact_path="model",
        challenger_metric=results_df.iloc[0]["RMSE"],
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
    axes[0].bar(results_df["model"], results_df["RMSE"], color="#55A868")
    axes[0].set_title("Model Comparison (RMSE, lower is better)")
    axes[0].tick_params(axis="x", rotation=20)

    axes[1].scatter(y_test, best_preds, alpha=0.7, color="#C44E52")
    lims = [min(y_test.min(), best_preds.min()), max(y_test.max(), best_preds.max())]
    axes[1].plot(lims, lims, "k--", linewidth=1)
    axes[1].set_xlabel("Actual NextQuotedRevenue (paise)")
    axes[1].set_ylabel("Predicted NextQuotedRevenue (paise)")
    axes[1].set_title(f"Best Model: {best_name} — Predicted vs Actual")

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

    residuals = y_test.values - best_preds
    plt.figure(figsize=(7, 5))
    sns.histplot(residuals, bins=12, kde=True, color="#8172B2")
    plt.title(f"{best_name} Residual Distribution (Test Set)")
    plt.xlabel("Residual (Actual - Predicted, paise)")
    plt.tight_layout()
    plt.savefig(OUT / "04_residuals.png", dpi=150)
    plt.close()
    mlflow.log_artifact(str(OUT / "04_residuals.png"))

print(f"\nBest model: {best_name}")
print(f"Outputs saved to: {OUT}")
