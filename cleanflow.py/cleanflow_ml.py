"""
CleanFlow — XGBoost + CatBoost Wiper Trip Classifier
======================================================
Trains on 6 real wells (4 PERFORM, 2 ELIMINATE) and outputs:
  - Model accuracy, F1, ROC-AUC
  - SHAP feature importance
  - Per-well prediction report
  - Saved models ready for the digital twin

Install once:
    pip install xgboost catboost scikit-learn pandas numpy matplotlib shap

Run:
    python cleanflow_ml.py
"""

import pandas as pd
import numpy as np
import warnings
import os
import json
import pickle
from pathlib import Path

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, f1_score, accuracy_score)
from sklearn.pipeline import Pipeline
import xgboost as xgb
from catboost import CatBoostClassifier
import shap
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

warnings.filterwarnings("ignore")
np.random.seed(42)

# ─── Output folder ────────────────────────────────────────────────────────────
OUT = Path("cleanflow_output")
OUT.mkdir(exist_ok=True)

# ─── File registry ────────────────────────────────────────────────────────────
# label: 1 = perform wiper trip, 0 = eliminate wiper trip
FILES = {
    "WELL_A": ("WELL_A_22 INCH_Peform WIPER TRIP.csv",       1),
    "WELL_B": ("WELL_B-22 INCH_PERFORM WIPER TRIP.csv",      1),
    "WELL_C": ("WELL_C-22 INCH_ELIMINATE WIPER TRIP.csv",    0),
    "WELL_D": ("WELL_D_22 INCH_PERFORM WIPER TRIP.csv",      1),
    "WELL_E": ("WELL_E_22 INCH_PERFORM WIPER TRIP.csv",      1),
    "WELL_F": ("WELL_F_22 INCH_ELIMINATE WIPER TRIP.csv",    0),
}

# Standard column names (Well F uses WITSML names → remapped below)
COLS = ["UtcTime", "BitDepth", "BlockPos", "HookLoad",
        "PumpFlow", "ROP", "SPP", "RPM", "TRQ", "TotalDepth", "WOB"]

FEATURE_COLS = ["BitDepth", "BlockPos", "HookLoad", "PumpFlow",
                "ROP", "SPP", "RPM", "TRQ", "TotalDepth", "WOB"]


# ═══════════════════════════════════════════════════════════════════════════════
# 1. DATA LOADING & CLEANING
# ═══════════════════════════════════════════════════════════════════════════════

def load_well(path: str, label: int, well_name: str) -> pd.DataFrame:
    """
    Load one well CSV, standardise column names, clean, label.
    Row 1 = column names, Row 2 = units (skip), Row 3+ = data.
    """
    df = pd.read_csv(path, header=0, skiprows=[1], low_memory=False)

    # Well F uses different WITSML column names — remap to standard
    if "DepthMonitoring.RBD" in df.columns:
        df.columns = COLS

    df.columns = COLS  # enforce standard names

    # Parse timestamp
    df["UtcTime"] = pd.to_datetime(df["UtcTime"], utc=True, errors="coerce")
    df = df.dropna(subset=["UtcTime"]).sort_values("UtcTime").reset_index(drop=True)

    # Convert all feature columns to numeric
    for col in FEATURE_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Fill missing values: forward fill then backward fill then 0
    df[FEATURE_COLS] = df[FEATURE_COLS].ffill().bfill().fillna(0)

    # Clip extreme outliers at 1st/99th percentile per column
    for col in FEATURE_COLS:
        lo = df[col].quantile(0.01)
        hi = df[col].quantile(0.99)
        df[col] = df[col].clip(lo, hi)

    # Label and well ID
    df["label"]    = label
    df["well"]     = well_name
    df["label_str"] = "PERFORM" if label == 1 else "ELIMINATE"

    print(f"  {well_name}: {len(df):>6} rows | label={df['label_str'].iloc[0]}")
    return df


def load_all(data_dir: str = ".") -> pd.DataFrame:
    print("\n── Loading wells ──────────────────────────────────────────")
    frames = []
    for well_name, (fname, label) in FILES.items():
        path = os.path.join(data_dir, fname)
        if not os.path.exists(path):
            print(f"  WARNING: {fname} not found — skipping")
            continue
        frames.append(load_well(path, label, well_name))

    df = pd.concat(frames, ignore_index=True)
    print(f"\n  Total rows : {len(df):,}")
    print(f"  PERFORM    : {(df['label']==1).sum():,} rows "
          f"({(df['label']==1).mean()*100:.1f}%)")
    print(f"  ELIMINATE  : {(df['label']==0).sum():,} rows "
          f"({(df['label']==0).mean()*100:.1f}%)")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# 2. FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════════════════

def engineer_features(df: pd.DataFrame,
                      windows: list = [12, 60, 300]) -> pd.DataFrame:
    """
    Per-well rolling features at 5s sampling:
      windows = [12, 60, 300] → [1 min, 5 min, 25 min]

    Features per channel: rolling mean, std, min, max
    Extra: torque_rop_ratio, drag_proxy, spp_flow_ratio
    """
    print("\n── Engineering features ───────────────────────────────────")
    channels = ["HookLoad", "ROP", "SPP", "RPM", "TRQ", "WOB", "PumpFlow"]
    result_frames = []

    for well, grp in df.groupby("well"):
        grp = grp.copy().reset_index(drop=True)

        for ch in channels:
            for w in windows:
                grp[f"{ch}_mean_{w}"] = (grp[ch].rolling(w, min_periods=1)
                                                 .mean())
                grp[f"{ch}_std_{w}"]  = (grp[ch].rolling(w, min_periods=1)
                                                 .std().fillna(0))
                grp[f"{ch}_max_{w}"]  = (grp[ch].rolling(w, min_periods=1)
                                                 .max())

        # Trend: slope of last 60 samples (5 min)
        for ch in ["TRQ", "ROP", "SPP", "HookLoad"]:
            grp[f"{ch}_trend60"] = (
                grp[ch].rolling(60, min_periods=10)
                       .apply(lambda x: np.polyfit(range(len(x)), x, 1)[0]
                              if len(x) > 1 else 0, raw=True)
            )

        # Domain-specific derived features
        grp["torque_rop_ratio"] = grp["TRQ"] / (grp["ROP"] + 1e-3)
        grp["spp_flow_ratio"]   = grp["SPP"] / (grp["PumpFlow"] + 1e-3)
        grp["drag_proxy"]       = grp["HookLoad"].diff().abs().fillna(0)
        grp["depth_progress"]   = grp["BitDepth"].diff().fillna(0)
        grp["on_bottom"]        = (
            (grp["BitDepth"] >= grp["TotalDepth"] - 5).astype(int)
        )

        result_frames.append(grp)

    out = pd.concat(result_frames, ignore_index=True)

    # Drop rows with NaN from rolling (warm-up period)
    out = out.dropna().reset_index(drop=True)

    eng_cols = [c for c in out.columns
                if c not in ["UtcTime", "label", "well", "label_str"]
                and c not in FEATURE_COLS]
    print(f"  Base features    : {len(FEATURE_COLS)}")
    print(f"  Engineered feats : {len(eng_cols)}")
    print(f"  Total features   : {len(FEATURE_COLS) + len(eng_cols)}")
    print(f"  Rows after clean : {len(out):,}")
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# 3. TRAIN / TEST SPLIT  (leave-one-well-out)
# ═══════════════════════════════════════════════════════════════════════════════

def get_feature_cols(df: pd.DataFrame) -> list:
    exclude = {"UtcTime", "label", "well", "label_str"}
    return [c for c in df.columns if c not in exclude]


def train_test_split_by_well(df: pd.DataFrame,
                              test_well: str = "WELL_A") -> tuple:
    """
    Leave-one-well-out split.
    Default: test on WELL_A (largest PERFORM well).
    """
    train = df[df["well"] != test_well].copy()
    test  = df[df["well"] == test_well].copy()
    feat_cols = get_feature_cols(df)
    X_train = train[feat_cols].values
    y_train = train["label"].values
    X_test  = test[feat_cols].values
    y_test  = test["label"].values
    return X_train, y_train, X_test, y_test, feat_cols


# ═══════════════════════════════════════════════════════════════════════════════
# 4. MODELS
# ═══════════════════════════════════════════════════════════════════════════════

def build_xgboost() -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        gamma=0.1,
        scale_pos_weight=1,        # adjust if class imbalance
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )


def build_catboost() -> CatBoostClassifier:
    return CatBoostClassifier(
        iterations=400,
        depth=6,
        learning_rate=0.05,
        l2_leaf_reg=3,
        border_count=128,
        random_seed=42,
        verbose=0,
        task_type="CPU",
        eval_metric="AUC",
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 5. EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate(model, X_test, y_test, model_name: str,
             feat_cols: list, results: dict):
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc   = accuracy_score(y_test, y_pred)
    f1    = f1_score(y_test, y_pred)
    auc   = roc_auc_score(y_test, y_proba)
    cm    = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred,
                                   target_names=["ELIMINATE", "PERFORM"])

    results[model_name] = {"accuracy": acc, "f1": f1, "auc": auc}

    print(f"\n── {model_name} Results ────────────────────────────────────")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  F1 Score : {f1:.4f}")
    print(f"  ROC-AUC  : {auc:.4f}")
    print(f"\n{report}")

    return y_pred, y_proba, cm


# ═══════════════════════════════════════════════════════════════════════════════
# 6. SHAP FEATURE IMPORTANCE
# ═══════════════════════════════════════════════════════════════════════════════

def compute_shap(model, X_test, feat_cols: list,
                 model_name: str, n_samples: int = 1000):
    """Compute and plot SHAP values."""
    print(f"\n── SHAP ({model_name}) ────────────────────────────────────")

    # Sample for speed
    idx = np.random.choice(len(X_test), min(n_samples, len(X_test)),
                           replace=False)
    X_sample = X_test[idx]

    if "XGBoost" in model_name:
        explainer = shap.TreeExplainer(model)
    else:
        explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(X_sample)

    # Mean absolute SHAP per feature
    mean_shap = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame({
        "feature": feat_cols,
        "shap_importance": mean_shap
    }).sort_values("shap_importance", ascending=False)

    print(f"\n  Top 10 features ({model_name}):")
    print(importance_df.head(10).to_string(index=False))

    # Save top features for agent JSON packet
    top10 = importance_df.head(10).to_dict(orient="records")
    with open(OUT / f"shap_top10_{model_name.replace(' ','_')}.json", "w") as f:
        json.dump(top10, f, indent=2)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    top = importance_df.head(15)
    colors = ["#D85A30" if i < 3 else "#534AB7" if i < 7 else "#1D9E75"
              for i in range(len(top))]
    ax.barh(top["feature"][::-1], top["shap_importance"][::-1],
            color=colors[::-1])
    ax.set_xlabel("Mean |SHAP value|", fontsize=11)
    ax.set_title(f"Top 15 Features — {model_name}", fontsize=13, fontweight="bold")
    ax.axvline(0, color="black", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(OUT / f"shap_{model_name.replace(' ','_')}.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: shap_{model_name.replace(' ','_')}.png")

    return importance_df


# ═══════════════════════════════════════════════════════════════════════════════
# 7. CONFUSION MATRIX PLOT
# ═══════════════════════════════════════════════════════════════════════════════

def plot_confusion_matrices(cms: dict, results: dict):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("CleanFlow — Confusion Matrices", fontsize=14, fontweight="bold")

    for ax, (model_name, cm) in zip(axes, cms.items()):
        sns.heatmap(cm, annot=True, fmt="d", ax=ax,
                    cmap="RdYlGn",
                    xticklabels=["ELIMINATE", "PERFORM"],
                    yticklabels=["ELIMINATE", "PERFORM"])
        ax.set_xlabel("Predicted", fontsize=10)
        ax.set_ylabel("Actual", fontsize=10)
        r = results[model_name]
        ax.set_title(
            f"{model_name}\nAcc={r['accuracy']:.3f}  "
            f"F1={r['f1']:.3f}  AUC={r['auc']:.3f}",
            fontsize=10
        )

    plt.tight_layout()
    plt.savefig(OUT / "confusion_matrices.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("\n  Saved: confusion_matrices.png")


# ═══════════════════════════════════════════════════════════════════════════════
# 8. PER-WELL PREDICTION REPORT
# ═══════════════════════════════════════════════════════════════════════════════

def per_well_report(df_eng: pd.DataFrame,
                    xgb_model, cat_model,
                    feat_cols: list):
    """Predict on each well and show how confident each model is."""
    print("\n── Per-Well Prediction Report ─────────────────────────────")
    print(f"  {'Well':<10} {'True Label':<12} {'XGB Proba':<12} "
          f"{'XGB Call':<12} {'CAT Proba':<12} {'CAT Call':<12}")
    print("  " + "-" * 68)

    report_rows = []
    for well, grp in df_eng.groupby("well"):
        X = grp[feat_cols].values
        y_true = grp["label"].iloc[0]
        label_str = grp["label_str"].iloc[0]

        xgb_prob = xgb_model.predict_proba(X)[:, 1].mean()
        cat_prob = cat_model.predict_proba(X)[:, 1].mean()

        xgb_call = "PERFORM" if xgb_prob > 0.5 else "ELIMINATE"
        cat_call = "PERFORM" if cat_prob > 0.5 else "ELIMINATE"

        xgb_ok = "✓" if xgb_call == label_str else "✗"
        cat_ok = "✓" if cat_call == label_str else "✗"

        print(f"  {well:<10} {label_str:<12} "
              f"{xgb_prob:.3f} {xgb_ok:<7}  {xgb_call:<10} "
              f"{cat_prob:.3f} {cat_ok:<7}  {cat_call:<10}")

        report_rows.append({
            "well": well,
            "true_label": label_str,
            "xgb_probability": round(xgb_prob, 4),
            "xgb_prediction": xgb_call,
            "xgb_correct": xgb_call == label_str,
            "cat_probability": round(cat_prob, 4),
            "cat_prediction": cat_call,
            "cat_correct": cat_call == label_str,
        })

    report_df = pd.DataFrame(report_rows)
    report_df.to_csv(OUT / "per_well_report.csv", index=False)
    print(f"\n  Saved: per_well_report.csv")
    return report_df


# ═══════════════════════════════════════════════════════════════════════════════
# 9. SAVE MODELS
# ═══════════════════════════════════════════════════════════════════════════════

def save_models(xgb_model, cat_model, feat_cols: list, scaler):
    """Save trained models for use in CleanFlow digital twin."""

    # XGBoost — native format
    xgb_model.save_model(str(OUT / "cleanflow_xgboost.json"))

    # CatBoost — native format
    cat_model.save_model(str(OUT / "cleanflow_catboost.cbm"))

    # Feature column list (needed to reconstruct inputs)
    with open(OUT / "feature_cols.json", "w") as f:
        json.dump(feat_cols, f, indent=2)

    # Scaler
    with open(OUT / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    print("\n── Saved models ───────────────────────────────────────────")
    print(f"  cleanflow_xgboost.json  (XGBoost native)")
    print(f"  cleanflow_catboost.cbm  (CatBoost native)")
    print(f"  feature_cols.json       (feature list)")
    print(f"  scaler.pkl              (StandardScaler)")


# ═══════════════════════════════════════════════════════════════════════════════
# 10. INFERENCE FUNCTION (plug into digital twin)
# ═══════════════════════════════════════════════════════════════════════════════

def predict_wiper_trip(sensor_row: dict,
                       xgb_model, cat_model,
                       feat_cols: list,
                       scaler) -> dict:
    """
    Single-row inference. Plug into your digital twin loop.

    Args:
        sensor_row: dict with engineered feature values
        xgb_model, cat_model: loaded trained models
        feat_cols: list from feature_cols.json
        scaler: fitted StandardScaler

    Returns:
        dict with probabilities, ensemble vote, and recommendation
    """
    X = np.array([[sensor_row.get(f, 0.0) for f in feat_cols]])
    X_scaled = scaler.transform(X)

    xgb_prob = xgb_model.predict_proba(X_scaled)[0, 1]
    cat_prob  = cat_model.predict_proba(X_scaled)[0, 1]

    # Weighted ensemble: CatBoost slightly higher weight
    ensemble_prob = 0.45 * xgb_prob + 0.55 * cat_prob

    if ensemble_prob > 0.75:
        recommendation = "PERFORM_WIPER_TRIP"
        confidence     = "high"
    elif ensemble_prob > 0.50:
        recommendation = "MONITOR_CLOSELY"
        confidence     = "medium"
    else:
        recommendation = "CONTINUE_DRILLING"
        confidence     = "low"

    return {
        "xgb_probability":       round(float(xgb_prob), 4),
        "catboost_probability":  round(float(cat_prob), 4),
        "ensemble_probability":  round(float(ensemble_prob), 4),
        "recommendation":        recommendation,
        "confidence":            confidence,
        "drill_state_predicted": 2 if ensemble_prob > 0.75 else 0,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    print("=" * 60)
    print("CleanFlow — XGBoost + CatBoost Training Pipeline")
    print("=" * 60)

    # ── 1. Load data ──────────────────────────────────────────────────────
    # If CSV files are in a different folder, change data_dir:
    df_raw = load_all(data_dir=".")

    # ── 2. Feature engineering ────────────────────────────────────────────
    df_eng = engineer_features(df_raw, windows=[12, 60, 300])

    feat_cols = get_feature_cols(df_eng)

    # ── 3. Train/test split (leave WELL_A out for testing) ───────────────
    X_train, y_train, X_test, y_test, feat_cols = \
        train_test_split_by_well(df_eng, test_well="WELL_A")

    print(f"\n── Train/Test Split ───────────────────────────────────────")
    print(f"  Train: {X_train.shape[0]:,} rows from "
          f"{df_eng[df_eng.well != 'WELL_A'].well.nunique()} wells")
    print(f"  Test : {X_test.shape[0]:,} rows from WELL_A")

    # Scale
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    # ── 4. Train XGBoost ──────────────────────────────────────────────────
    print("\n── Training XGBoost ───────────────────────────────────────")
    xgb_model = build_xgboost()
    xgb_model.fit(
        X_train_s, y_train,
        eval_set=[(X_test_s, y_test)],
        verbose=False,
    )
    print("  Done.")

    # ── 5. Train CatBoost ─────────────────────────────────────────────────
    print("\n── Training CatBoost ──────────────────────────────────────")
    cat_model = build_catboost()
    cat_model.fit(
        X_train_s, y_train,
        eval_set=(X_test_s, y_test),
        verbose=False,
    )
    print("  Done.")

    # ── 6. Evaluate ───────────────────────────────────────────────────────
    results = {}
    cms     = {}

    y_pred_xgb, y_proba_xgb, cm_xgb = evaluate(
        xgb_model, X_test_s, y_test, "XGBoost", feat_cols, results)
    cms["XGBoost"] = cm_xgb

    y_pred_cat, y_proba_cat, cm_cat = evaluate(
        cat_model, X_test_s, y_test, "CatBoost", feat_cols, results)
    cms["CatBoost"] = cm_cat

    # Ensemble
    ensemble_proba = 0.45 * y_proba_xgb + 0.55 * y_proba_cat
    ensemble_pred  = (ensemble_proba > 0.5).astype(int)
    ens_acc  = accuracy_score(y_test, ensemble_pred)
    ens_f1   = f1_score(y_test, ensemble_pred)
    ens_auc  = roc_auc_score(y_test, ensemble_proba)
    results["Ensemble"] = {"accuracy": ens_acc, "f1": ens_f1, "auc": ens_auc}
    print(f"\n── Ensemble Results ───────────────────────────────────────")
    print(f"  Accuracy : {ens_acc:.4f}")
    print(f"  F1 Score : {ens_f1:.4f}")
    print(f"  ROC-AUC  : {ens_auc:.4f}")

    # ── 7. Confusion matrix plot ──────────────────────────────────────────
    plot_confusion_matrices(cms, results)

    # ── 8. SHAP ───────────────────────────────────────────────────────────
    shap_xgb = compute_shap(xgb_model, X_test_s, feat_cols, "XGBoost")
    shap_cat = compute_shap(cat_model,  X_test_s, feat_cols, "CatBoost")

    # ── 9. Per-well report ────────────────────────────────────────────────
    per_well_report(df_eng, xgb_model, cat_model, feat_cols)

    # ── 10. Save models ───────────────────────────────────────────────────
    save_models(xgb_model, cat_model, feat_cols, scaler)

    # ── 11. Summary ───────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    for model_name, r in results.items():
        print(f"  {model_name:<12} | "
              f"Acc={r['accuracy']:.4f} | "
              f"F1={r['f1']:.4f} | "
              f"AUC={r['auc']:.4f}")

    print(f"\n  All outputs saved to: {OUT.resolve()}/")
    print("\nExample inference (plug into your digital twin):")
    print("-" * 60)
    sample_row = {f: float(X_test[0][i]) for i, f in enumerate(feat_cols)}
    result = predict_wiper_trip(sample_row, xgb_model, cat_model,
                                feat_cols, scaler)
    print(json.dumps(result, indent=2))
