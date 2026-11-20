"""Train and validate classifiers using a predefined USD cost threshold.

No test set is used to select a model or threshold.
"""
from __future__ import annotations
import json
from pathlib import Path
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from .compat import CatBoostAdapter
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             roc_auc_score, precision_score, recall_score,
                             f1_score, RocCurveDisplay)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from .pipeline import SEED, HIGH_COST_USD, engineer, make_target, preprocessor, validate_raw

ROOT = Path(__file__).resolve().parents[1]

def build_models():
    specs = {
        "LogisticRegression": LogisticRegression(max_iter=1500, random_state=SEED),
        "XGBoost": XGBClassifier(n_estimators=150, max_depth=3, learning_rate=.05,
                                 subsample=.85, colsample_bytree=.85,
                                 eval_metric="logloss", random_state=SEED, n_jobs=1),
        "CatBoost": CatBoostAdapter(iterations=180, depth=4, learning_rate=.05, random_seed=SEED),
    }
    return {n: Pipeline([("preprocessor", preprocessor()), ("classifier", model)])
            for n, model in specs.items()}

def cross_validate_auc(model, x, y, folds=5):
    cv = StratifiedKFold(folds, shuffle=True, random_state=SEED)
    aucs = []
    for train_idx, val_idx in cv.split(x, y):
        fitted = clone(model).fit(x.iloc[train_idx], y.iloc[train_idx])
        prob = fitted.predict_proba(x.iloc[val_idx])[:, 1]
        aucs.append(roc_auc_score(y.iloc[val_idx], prob))
    return np.array(aucs)

def evaluate(y, prob):
    pred = (prob >= .5).astype(int)  # fixed threshold, not tuned on holdout
    return {
        "roc_auc": round(float(roc_auc_score(y, prob)), 4),
        "pr_auc": round(float(average_precision_score(y, prob)), 4),
        "brier": round(float(brier_score_loss(y, prob)), 4),
        "precision": round(float(precision_score(y, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y, pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y, pred, zero_division=0)), 4),
    }

def run_training(dataset=None, *, save_artifacts=True):
    raw_path = Path(dataset) if dataset else ROOT / "data" / "insurance.csv"
    if not raw_path.is_file():
        raise FileNotFoundError(f"Dataset missing: {raw_path}; see data/README.md")
    raw = validate_raw(pd.read_csv(raw_path), require_target=True)
    # Keep the input dataset unchanged. Duplicate exact rows may represent
    # genuine people; report rather than drop automatically.
    x = engineer(raw)
    y = make_target(raw["charges"])
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=.20, random_state=SEED, stratify=y
    )
    summaries, fitted = {}, {}
    for name, model in build_models().items():
        cv_auc = cross_validate_auc(model, x_train, y_train)
        fitted_model = clone(model).fit(x_train, y_train)
        summaries[name] = {"cv_roc_auc_mean": round(float(cv_auc.mean()),4),
                           "cv_roc_auc_std": round(float(cv_auc.std()),4)}
        fitted[name] = fitted_model
    # Model selection happens strictly within train-only CV.
    selected = max(summaries, key=lambda n: summaries[n]["cv_roc_auc_mean"])
    # Evaluate once after choosing the model (other test metrics are not used for selection).
    for name, model in fitted.items():
        prob = model.predict_proba(x_test)[:, 1]
        summaries[name]["holdout"] = evaluate(y_test, prob)
    output = {"dataset_rows": len(raw), "duplicates_reported": int(raw.duplicated().sum()),
              "high_cost_definition": f"charges >= ${HIGH_COST_USD:,.0f} (fixed illustrative cutoff)",
              "high_cost_rate": round(float(y.mean()),4),
              "selection_metric": "5-fold training CV ROC-AUC",
              "selected_model": selected, "n_train": len(x_train), "n_test": len(x_test),
              "models": summaries}
    if save_artifacts:
        (ROOT / "reports" / "figures").mkdir(parents=True,exist_ok=True)
        (ROOT / "models").mkdir(exist_ok=True)
        fig, ax = plt.subplots(figsize=(7,5))
        for name, model in fitted.items():
            p = model.predict_proba(x_test)[:,1]
            RocCurveDisplay.from_predictions(y_test,p,name=name,ax=ax)
        ax.set_title("Holdout ROC comparison (after CV selection)")
        fig.tight_layout();fig.savefig(ROOT / "reports" / "figures" / "roc_curves.png",dpi=160);plt.close(fig)
        # SHAP for selected tree classifier; explain training features on test split.
        if selected in {"CatBoost", "XGBoost"}:
            try:
                import shap
                selected_model = fitted[selected]
                transformed = selected_model.named_steps["preprocessor"].transform(x_test)
                names = selected_model.named_steps["preprocessor"].get_feature_names_out()
                values = shap.TreeExplainer(getattr(selected_model.named_steps["classifier"], "model_", selected_model.named_steps["classifier"])).shap_values(transformed)
                values = np.asarray(values)
                if values.ndim == 3: values = values[:,:,1]
                fig = plt.figure()
                shap.summary_plot(values, pd.DataFrame(transformed, columns=names), show=False)
                plt.tight_layout();plt.savefig(ROOT / "reports" / "figures" / "shap_summary.png",dpi=150,bbox_inches="tight");plt.close("all")
                output["shap_model"] = selected
            except Exception as exc:
                output["shap_note"] = f"SHAP plot unavailable: {type(exc).__name__}: {exc}"
        else:
            output["shap_note"] = "Selected baseline is not a tree model; use coefficient interpretation instead."
        # Local artifact only; ignored by git. DO NOT load untrusted joblib files.
        joblib.dump(fitted[selected], ROOT / "models" / "best_model.joblib")
        (ROOT / "reports" / "metrics.json").write_text(json.dumps(output,indent=2),encoding="utf-8")
    print(json.dumps(output,indent=2))
    return output

if __name__ == "__main__":
    run_training()
