from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    precision_recall_curve,
    auc,
)

from ingest import load_dataset
from preprocess import preprocess_training_data


def train(csv_path: str = "data/gaming_health_data.csv", models_dir: str = "models"):
    """
    Runs the full ML training pipeline:
    1. Ingest CSV
    2. Preprocess data
    3. Train Random Forest
    4. Evaluate default threshold
    5. Tune classification threshold
    6. Run GridSearchCV
    7. Save best model artifacts
    """
    models_path = Path(models_dir)
    models_path.mkdir(parents=True, exist_ok=True)

    df = load_dataset(csv_path)
    X_scaled, y, feature_names, robust_scaler, category_mappings = preprocess_training_data(
        df,
        models_dir=models_dir,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    print("\nTraining set shape:", X_train.shape)
    print("Testing set shape:", X_test.shape)

    print("\nTraining target distribution:")
    print(y_train.value_counts())

    print("\nTesting target distribution:")
    print(y_test.value_counts())

    rf_model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
    )

    rf_model.fit(X_train, y_train)

    print("\nRandom Forest model trained successfully.")

    y_prob = rf_model.predict_proba(X_test)[:, 1]
    y_pred_default = (y_prob >= 0.50).astype(int)

    f1_default = f1_score(y_test, y_pred_default)
    roc_auc = roc_auc_score(y_test, y_prob)

    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    pr_auc = auc(recall, precision)

    print("\n=== Model Evaluation at Threshold = 0.50 ===")
    print(f"F1 Score: {f1_default:.4f}")
    print(f"ROC-AUC Score: {roc_auc:.4f}")
    print(f"PR-AUC Score: {pr_auc:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_default, target_names=["Healthy", "At Risk"]))

    cm_default = confusion_matrix(y_test, y_pred_default)
    print("\nConfusion Matrix at Threshold = 0.50:")
    print(cm_default)

    thresholds = np.arange(0.30, 0.71, 0.05)

    best_threshold = 0.50
    best_f1 = 0
    threshold_results = []

    for t in thresholds:
        y_pred_t = (y_prob >= t).astype(int)
        f1_t = f1_score(y_test, y_pred_t)
        threshold_results.append((float(t), float(f1_t)))

        if f1_t > best_f1:
            best_f1 = f1_t
            best_threshold = t

    threshold_df = pd.DataFrame(threshold_results, columns=["Threshold", "F1 Score"])

    print("\nThreshold tuning results:")
    print(threshold_df)

    print(f"\nBest Threshold: {best_threshold:.2f}")
    print(f"Best F1 Score: {best_f1:.4f}")

    y_pred_best = (y_prob >= best_threshold).astype(int)

    final_f1 = f1_score(y_test, y_pred_best)
    final_roc_auc = roc_auc_score(y_test, y_prob)

    precision_best, recall_best, _ = precision_recall_curve(y_test, y_prob)
    final_pr_auc = auc(recall_best, precision_best)

    print("\n=== Final Model Evaluation Before GridSearchCV ===")
    print(f"Best Threshold: {best_threshold:.2f}")
    print(f"F1 Score: {final_f1:.4f}")
    print(f"ROC-AUC Score: {final_roc_auc:.4f}")
    print(f"PR-AUC Score: {final_pr_auc:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_best, target_names=["Healthy", "At Risk"]))

    feature_importances = rf_model.feature_importances_

    importance_df = pd.DataFrame(
        {
            "Feature": feature_names,
            "Importance": feature_importances,
        }
    ).sort_values(by="Importance", ascending=False)

    print("\nFeature Importance:")
    print(importance_df)

    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    }

    grid_search = GridSearchCV(
        estimator=RandomForestClassifier(
            random_state=42,
            class_weight="balanced",
        ),
        param_grid=param_grid,
        scoring="roc_auc",
        cv=5,
        n_jobs=-1,
        verbose=1,
    )

    grid_search.fit(X_train, y_train)

    print("\nBest Parameters:")
    print(grid_search.best_params_)

    print(f"Best Cross-Validated ROC-AUC: {grid_search.best_score_:.4f}")

    best_rf_model = grid_search.best_estimator_

    y_prob_tuned = best_rf_model.predict_proba(X_test)[:, 1]

    best_threshold_tuned = 0.50
    best_f1_tuned = 0

    tuned_threshold_results = []

    for t in thresholds:
        y_pred_tuned_t = (y_prob_tuned >= t).astype(int)
        f1_tuned_t = f1_score(y_test, y_pred_tuned_t)
        tuned_threshold_results.append((float(t), float(f1_tuned_t)))

        if f1_tuned_t > best_f1_tuned:
            best_f1_tuned = f1_tuned_t
            best_threshold_tuned = t

    y_pred_tuned = (y_prob_tuned >= best_threshold_tuned).astype(int)

    roc_auc_tuned = roc_auc_score(y_test, y_prob_tuned)
    precision_tuned, recall_tuned, _ = precision_recall_curve(y_test, y_prob_tuned)
    pr_auc_tuned = auc(recall_tuned, precision_tuned)

    tuned_f1 = f1_score(y_test, y_pred_tuned)

    print("\n=== Tuned Model Results ===")
    print(f"Best Threshold: {best_threshold_tuned:.2f}")
    print(f"F1 Score: {tuned_f1:.4f}")
    print(f"ROC-AUC Score: {roc_auc_tuned:.4f}")
    print(f"PR-AUC Score: {pr_auc_tuned:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_tuned, target_names=["Healthy", "At Risk"]))

    final_metrics = {
        "default_model": {
            "threshold": 0.50,
            "f1_score": float(f1_default),
            "roc_auc": float(roc_auc),
            "pr_auc": float(pr_auc),
            "confusion_matrix": cm_default.tolist(),
        },
        "threshold_tuned_model": {
            "threshold": float(best_threshold),
            "f1_score": float(final_f1),
            "roc_auc": float(final_roc_auc),
            "pr_auc": float(final_pr_auc),
            "threshold_results": threshold_results,
        },
        "gridsearch_tuned_model": {
            "best_params": grid_search.best_params_,
            "best_cv_roc_auc": float(grid_search.best_score_),
            "threshold": float(best_threshold_tuned),
            "f1_score": float(tuned_f1),
            "roc_auc": float(roc_auc_tuned),
            "pr_auc": float(pr_auc_tuned),
            "threshold_results": tuned_threshold_results,
        },
    }

    joblib.dump(best_rf_model, models_path / "random_forest_model.pkl")
    joblib.dump(best_threshold_tuned, models_path / "best_threshold.pkl")
    joblib.dump(final_metrics, models_path / "metrics.pkl")

    with open(models_path / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(final_metrics, f, indent=4)

    importance_df.to_csv(models_path / "feature_importance.csv", index=False)

    print("\n✓ Saved final artifacts:")
    print(f"  - {models_path / 'random_forest_model.pkl'}")
    print(f"  - {models_path / 'scaler.pkl'}")
    print(f"  - {models_path / 'category_mappings.pkl'}")
    print(f"  - {models_path / 'imputation_values.pkl'}")
    print(f"  - {models_path / 'feature_columns.pkl'}")
    print(f"  - {models_path / 'best_threshold.pkl'}")
    print(f"  - {models_path / 'metrics.json'}")
    print(f"  - {models_path / 'feature_importance.csv'}")

    return final_metrics


if __name__ == "__main__":
    train()
