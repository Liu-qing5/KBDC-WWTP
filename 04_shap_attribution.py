#!/usr/bin/env python3
"""
04_shap_attribution.py
======================

Reproducible SHAP attribution for the KBDC study.

Pipeline position
-----------------
    01_preprocessing.py
        -> 02_process_features.py
        -> 03_model_benchmark.py
        -> 04_shap_attribution.py

This script consolidates the original Figure 5 / Supplementary Figure S4-S5
analysis scripts into one reproducible module.

Study definitions implemented here
----------------------------------
Ori (7 variables)
    TREAT_FLOW, IN_COD, IN_TN, IN_NH4, IN_TP, ANO_MLSS, ANO_SV30

PI (17 variables)
    Ori 7
    + EFF_TN_lag1, EFF_NH4_lag1
    + LOW_FLOW_P, COD/TN, COD_DEFICIT,
      R_TN_lag1, R_NH4_lag1, R_MAX_lag1, R_MEAN_lag1, TN_PER_VSS

The fitted Ori-GBR and PI-GBR are explained with TreeSHAP. By default, SHAP
is evaluated on the retained historical training segment, matching the
original figure-generation scripts.

Outputs
-------
Main-text Figure 5 source figures:
    Fig5a_ori_pcc_vs_shap
    Fig5b_pi_pcc_vs_shap
    Fig5c_pi_shap_importance_and_process_groups
    Fig5d_pi_shap_beeswarm
    Fig5e_pi_shap_heatmap

Supplementary figures:
    FigS4a_ori_shap_importance
    FigS4b_ori_shap_beeswarm
    FigS4c_ori_shap_heatmap
    FigS5_pi_dependence

Auditable source data:
    Ori/PI SHAP matrices
    model-space feature matrices
    PCC-vs-SHAP tables
    feature-level SHAP statistics
    overlapping process-group attribution scores
    SHAP additivity checks
    dependence-plot source data

Model handling
--------------
03_model_benchmark.py saves models/pi_gbr_model.joblib. If an Ori model is
not present, this script reconstructs it from the full training segment with
the reported GBR configuration and saves models/ori_gbr_model.joblib. If the
PI model is also absent, it is reconstructed in the same way. No held-out
test data are used by this script.

Process-group note
------------------
The manuscript's process-group summary is intentionally overlapping:
    - COD/TN and COD_DEFICIT enter both C/N and IN-N.
    - R_MAX_lag1 and R_MEAN_lag1 enter both LAG-TN and LAG-NH4.

Therefore the donut/group percentages are normalized overlapping group
scores, not a mutually exclusive decomposition of total SHAP importance.
This reproduces the logic used in the original Figure 5c script.
"""

from __future__ import annotations

import argparse
import json
import pickle
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec

import shap
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import pdist
from scipy.stats import pearsonr

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


# -----------------------------------------------------------------------------
# Project paths
# -----------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name.lower() == "code" else SCRIPT_DIR

DEFAULT_TRAIN = PROJECT_ROOT / "data" / "processed" / "train_model_ready.csv"
DEFAULT_RESULTS = PROJECT_ROOT / "results" / "shap_attribution"
DEFAULT_MODELS = PROJECT_ROOT / "models"

TARGET = "CARBON_DOS"
RANDOM_SEED = 42


# -----------------------------------------------------------------------------
# Feature-space definitions
# -----------------------------------------------------------------------------

ORI_FEATURES: List[str] = [
    "TREAT_FLOW",
    "IN_COD",
    "IN_TN",
    "IN_NH4",
    "IN_TP",
    "ANO_MLSS",
    "ANO_SV30",
]

PI_ADDED_FEATURES: List[str] = [
    "EFF_TN_lag1",
    "EFF_NH4_lag1",
    "LOW_FLOW_P",
    "COD/TN",
    "COD_DEFICIT",
    "R_TN_lag1",
    "R_NH4_lag1",
    "R_MAX_lag1",
    "R_MEAN_lag1",
    "TN_PER_VSS",
]

PI_FEATURES: List[str] = ORI_FEATURES + PI_ADDED_FEATURES

FEATURE_SPACES: Dict[str, List[str]] = {
    "Ori": ORI_FEATURES,
    "PI": PI_FEATURES,
}

# Reported final GBR configuration (same selected configuration for Ori and PI).
GBR_CONFIG: Dict[str, Any] = {
    "subsample": 0.6,
    "n_estimators": 500,
    "min_samples_split": 5,
    "min_samples_leaf": 1,
    "max_features": "sqrt",
    "max_depth": 6,
    "learning_rate": 0.03,
}

# Historical analysis names -> manuscript/public-code names.
COLUMN_ALIASES: Dict[str, Sequence[str]] = {
    "TREAT_FLOW": ("TREAT_FLOW", "TREAT_FLOW_t(t/d)"),
    "IN_COD": ("IN_COD", "IN_COD_t(mg/L)"),
    "IN_TN": ("IN_TN", "IN_TN_t(mg/L)"),
    "IN_NH4": ("IN_NH4", "IN_NH4_t(mg/L)"),
    "IN_TP": ("IN_TP", "IN_TP_t(mg/L)"),
    "ANO_MLSS": ("ANO_MLSS", "ANO_MLSS(mg/L)"),
    "ANO_SV30": ("ANO_SV30", "ANO_SV30(%)"),
    "EFF_TN_lag1": ("EFF_TN_lag1", "EFF_TN_lag1(mg/L)", "EFF_TN_LAG1"),
    "EFF_NH4_lag1": ("EFF_NH4_lag1", "EFF_NH4_lag1(mg/L)", "EFF_NH4_LAG1"),
    "LOW_FLOW_P": ("LOW_FLOW_P", "LOW_FLOW_PRESSURE"),
    "COD/TN": ("COD/TN", "COD_TN", "COD_TN_RATIO"),
    "COD_DEFICIT": ("COD_DEFICIT",),
    "R_TN_lag1": ("R_TN_lag1", "RISK_TN_lag1"),
    "R_NH4_lag1": ("R_NH4_lag1", "RISK_NH4_lag1"),
    "R_MAX_lag1": ("R_MAX_lag1", "RISK_MAX_lag1"),
    "R_MEAN_lag1": ("R_MEAN_lag1", "RISK_MEAN_lag1"),
    "TN_PER_VSS": ("TN_PER_VSS",),
    TARGET: (TARGET, "CARBON_DOS(g/t.water)"),
}

DISPLAY_NAMES: Dict[str, str] = {
    "TREAT_FLOW": "TREAT_FLOW",
    "IN_COD": "IN_COD",
    "IN_TN": "IN_TN",
    "IN_NH4": "IN_NH4",
    "IN_TP": "IN_TP",
    "ANO_MLSS": "ANO_MLSS",
    "ANO_SV30": "ANO_SV30",
    "EFF_TN_lag1": "EFF_TN_lag1",
    "EFF_NH4_lag1": "EFF_NH4_lag1",
    "LOW_FLOW_P": "LOW_FLOW_P",
    "COD/TN": "COD/TN",
    "COD_DEFICIT": "COD_DEFICIT",
    "R_TN_lag1": "R_TN_lag1",
    "R_NH4_lag1": "R_NH4_lag1",
    "R_MAX_lag1": "R_MAX_lag1",
    "R_MEAN_lag1": "R_MEAN_lag1",
    "TN_PER_VSS": "TN_PER_VSS",
}

# Overlapping process groups reproduced from the original Figure 5c code.
GROUP_DEFINITIONS: Dict[str, List[str]] = {
    "FLOW": [
        "TREAT_FLOW",
        "LOW_FLOW_P",
    ],
    "C/N": [
        "IN_COD",
        "COD/TN",
        "COD_DEFICIT",
    ],
    "IN-N": [
        "IN_TN",
        "COD/TN",
        "COD_DEFICIT",
        "TN_PER_VSS",
    ],
    "LAG-TN": [
        "EFF_TN_lag1",
        "R_TN_lag1",
        "R_MAX_lag1",
        "R_MEAN_lag1",
    ],
    "LAG-NH4": [
        "EFF_NH4_lag1",
        "R_NH4_lag1",
        "R_MAX_lag1",
        "R_MEAN_lag1",
    ],
}
GROUP_ORDER = ["FLOW", "C/N", "IN-N", "LAG-TN", "LAG-NH4", "Other"]

DEPENDENCE_FEATURES = [
    "TREAT_FLOW",
    "LOW_FLOW_P",
    "IN_TN",
    "COD_DEFICIT",
    "EFF_TN_lag1",
]

# Palette retained from the original figure scripts.
ORI_COLOR = "#E07A5F"
PI_COLOR = "#2F6F8F"
ORI_LIGHT = "#F4CFC6"
PI_LIGHT = "#C9DCE7"
PI_DARK = "#1F4E66"
NEUTRAL = "#BDBDBD"

GROUP_COLORS = {
    "FLOW": PI_COLOR,
    "C/N": ORI_COLOR,
    "IN-N": PI_LIGHT,
    "LAG-TN": ORI_LIGHT,
    "LAG-NH4": PI_DARK,
    "Other": NEUTRAL,
}


# -----------------------------------------------------------------------------
# Data/model helpers
# -----------------------------------------------------------------------------

def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported table type: {path}")


def canonicalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = out.columns.astype(str).str.strip()

    rename_map: Dict[str, str] = {}
    for canonical, aliases in COLUMN_ALIASES.items():
        if canonical in out.columns:
            continue
        hits = [a for a in aliases if a in out.columns]
        if len(hits) == 1:
            rename_map[hits[0]] = canonical
        elif len(hits) > 1:
            rename_map[hits[0]] = canonical
            warnings.warn(
                f"Multiple aliases found for {canonical}: {hits}. Using {hits[0]}.",
                RuntimeWarning,
            )
    if rename_map:
        out = out.rename(columns=rename_map)
    return out


def validate_training_frame(df: pd.DataFrame) -> None:
    required = set(PI_FEATURES + [TARGET])
    missing = sorted(required.difference(df.columns))
    if missing:
        raise KeyError(
            "Training data are missing required columns:\n"
            f"{missing}\nAvailable columns:\n{list(df.columns)}"
        )

    if df[TARGET].isna().any():
        raise ValueError(f"Target {TARGET} contains missing values.")

    if len(df) != 512:
        warnings.warn(
            f"Study pipeline retained 512 training records; found {len(df)}.",
            RuntimeWarning,
        )


def numeric_frame(df: pd.DataFrame, features: Sequence[str]) -> pd.DataFrame:
    X = df[list(features)].copy()
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    return X


def build_gbr_pipeline(seed: int = RANDOM_SEED) -> Pipeline:
    model = GradientBoostingRegressor(random_state=seed, **GBR_CONFIG)
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("model", model),
        ]
    )


def load_model_any(path: Path):
    try:
        return joblib.load(path)
    except Exception:
        with path.open("rb") as f:
            return pickle.load(f)


def load_or_fit_gbr(
    representation: str,
    model_path: Path,
    train_df: pd.DataFrame,
    seed: int,
    save_fallback: bool = True,
):
    features = FEATURE_SPACES[representation]
    X = numeric_frame(train_df, features)
    y = pd.to_numeric(train_df[TARGET], errors="raise").to_numpy(dtype=float)

    if model_path.exists():
        model = load_model_any(model_path)
        try:
            _ = np.asarray(model.predict(X.iloc[: min(5, len(X))])).reshape(-1)
        except Exception as exc:
            raise RuntimeError(
                f"Existing {representation} GBR at {model_path} could not predict "
                "from the current canonical feature matrix."
            ) from exc
        return model, False

    model = build_gbr_pipeline(seed)
    model.fit(X, y)

    if save_fallback:
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_path)

        meta = {
            "model": "GBR",
            "representation": representation,
            "target": TARGET,
            "features": features,
            "selected_config": GBR_CONFIG,
            "seed": seed,
            "training_records": int(len(train_df)),
            "created_by": "04_shap_attribution.py fallback reconstruction",
            "note": (
                "Model reconstructed from the retained historical training segment "
                "because the corresponding model file was absent."
            ),
        }
        with model_path.with_suffix(".metadata.json").open("w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

    return model, True


def extract_pipeline_parts(model, X: pd.DataFrame) -> Tuple[Any, np.ndarray, np.ndarray]:
    """
    Return final tree estimator, model-space matrix, and full-model predictions.

    For the public 03 pipeline, preprocessing is a median imputer. The function
    also supports a bare fitted GradientBoostingRegressor for compatibility with
    historical model files.
    """
    if isinstance(model, Pipeline):
        final_estimator = model.steps[-1][1]
        if len(model.steps) > 1:
            preprocesser = Pipeline(model.steps[:-1])
            X_model = preprocesser.transform(X)
        else:
            X_model = X.to_numpy(dtype=float)
        if hasattr(X_model, "toarray"):
            X_model = X_model.toarray()
        X_model = np.asarray(X_model, dtype=float)
        pred_pipeline = np.asarray(model.predict(X), dtype=float).reshape(-1)
        return final_estimator, X_model, pred_pipeline

    # Compatibility path for historical bare GBR objects.
    X_num = X.copy()
    if X_num.isna().any().any():
        X_num = X_num.fillna(X_num.median(numeric_only=True))
    X_model = X_num.to_numpy(dtype=float)
    pred = np.asarray(model.predict(X_num), dtype=float).reshape(-1)
    return model, X_model, pred


def safe_pearson(x: pd.Series, y: pd.Series) -> Tuple[float, float]:
    pair = pd.concat([x, y], axis=1).dropna()
    if len(pair) < 3:
        return np.nan, np.nan
    xv = pair.iloc[:, 0].to_numpy(dtype=float)
    yv = pair.iloc[:, 1].to_numpy(dtype=float)
    if np.nanstd(xv) == 0 or np.nanstd(yv) == 0:
        return np.nan, np.nan
    r, p = pearsonr(xv, yv)
    return float(r), float(p)


@dataclass
class ShapBundle:
    representation: str
    features: List[str]
    raw_X: pd.DataFrame
    model_X: np.ndarray
    y: np.ndarray
    shap_values: np.ndarray
    expected_value: float
    prediction_final: np.ndarray
    prediction_pipeline: np.ndarray
    importance: pd.DataFrame
    additivity: pd.DataFrame


def compute_shap_bundle(
    representation: str,
    model,
    train_df: pd.DataFrame,
) -> ShapBundle:
    features = FEATURE_SPACES[representation]
    raw_X = numeric_frame(train_df, features)
    y = pd.to_numeric(train_df[TARGET], errors="raise").to_numpy(dtype=float)

    final_estimator, model_X, pred_pipeline = extract_pipeline_parts(model, raw_X)

    explainer = shap.TreeExplainer(final_estimator)
    shap_values = explainer.shap_values(model_X)
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    shap_values = np.asarray(shap_values, dtype=float)

    if shap_values.ndim != 2:
        raise RuntimeError(
            f"Unexpected SHAP array shape for {representation}: {shap_values.shape}"
        )
    if shap_values.shape != model_X.shape:
        raise RuntimeError(
            f"SHAP/model matrix shape mismatch for {representation}: "
            f"{shap_values.shape} vs {model_X.shape}"
        )

    expected = np.asarray(explainer.expected_value).reshape(-1)
    expected_value = float(expected[0])

    pred_final = np.asarray(final_estimator.predict(model_X), dtype=float).reshape(-1)
    reconstructed = expected_value + shap_values.sum(axis=1)

    additivity = pd.DataFrame(
        {
            "row_id": np.arange(len(raw_X), dtype=int),
            "prediction_final_estimator": pred_final,
            "prediction_full_pipeline": pred_pipeline,
            "expected_value": expected_value,
            "sum_SHAP": shap_values.sum(axis=1),
            "expected_plus_sum_SHAP": reconstructed,
            "final_minus_reconstructed": pred_final - reconstructed,
            "pipeline_minus_final": pred_pipeline - pred_final,
        }
    )

    mean_abs = np.abs(shap_values).mean(axis=0)
    mean_signed = shap_values.mean(axis=0)

    rows: List[Dict[str, Any]] = []
    y_series = pd.Series(y, index=raw_X.index)
    for j, feature in enumerate(features):
        r, p = safe_pearson(raw_X[feature], y_series)
        vals = shap_values[:, j]
        rows.append(
            {
                "feature": feature,
                "display_feature": DISPLAY_NAMES.get(feature, feature),
                "mean_abs_SHAP": float(mean_abs[j]),
                "mean_SHAP": float(mean_signed[j]),
                "PCC_with_CARBON_DOS": r,
                "PCC_p_value": p,
                "abs_PCC": np.nan if np.isnan(r) else abs(r),
                "positive_SHAP_ratio": float(np.mean(vals > 0)),
                "negative_SHAP_ratio": float(np.mean(vals < 0)),
                "SHAP_min": float(np.min(vals)),
                "SHAP_max": float(np.max(vals)),
                "SHAP_p10": float(np.percentile(vals, 10)),
                "SHAP_p90": float(np.percentile(vals, 90)),
            }
        )

    importance = (
        pd.DataFrame(rows)
        .sort_values("mean_abs_SHAP", ascending=False)
        .reset_index(drop=True)
    )
    importance.insert(0, "rank", np.arange(1, len(importance) + 1))

    return ShapBundle(
        representation=representation,
        features=list(features),
        raw_X=raw_X,
        model_X=model_X,
        y=y,
        shap_values=shap_values,
        expected_value=expected_value,
        prediction_final=pred_final,
        prediction_pipeline=pred_pipeline,
        importance=importance,
        additivity=additivity,
    )


# -----------------------------------------------------------------------------
# Process-group summary
# -----------------------------------------------------------------------------

def compute_overlapping_group_summary(pi_bundle: ShapBundle) -> pd.DataFrame:
    imp = dict(
        zip(
            pi_bundle.importance["feature"],
            pi_bundle.importance["mean_abs_SHAP"],
        )
    )

    grouped_union: set[str] = set()
    for features in GROUP_DEFINITIONS.values():
        grouped_union.update(features)

    other_features = [f for f in pi_bundle.features if f not in grouped_union]

    rows: List[Dict[str, Any]] = []
    for group in GROUP_ORDER:
        if group == "Other":
            features = other_features
        else:
            features = GROUP_DEFINITIONS[group]

        score = float(sum(float(imp.get(f, 0.0)) for f in features))
        rows.append(
            {
                "group": group,
                "overlapping_group_score": score,
                "features": "; ".join(features),
                "n_feature_entries": len(features),
            }
        )

    out = pd.DataFrame(rows)
    total = float(out["overlapping_group_score"].sum())
    if total > 0:
        out["normalized_group_percent"] = (
            out["overlapping_group_score"] / total * 100.0
        )
    else:
        out["normalized_group_percent"] = 0.0

    out["grouping_note"] = (
        "Overlapping process groups; shared features can contribute to more than "
        "one group. Percentages are normalized group scores, not a mutually "
        "exclusive SHAP decomposition."
    )
    return out


# -----------------------------------------------------------------------------
# Source-data export
# -----------------------------------------------------------------------------

def export_bundle(bundle: ShapBundle, source_dir: Path) -> None:
    source_dir.mkdir(parents=True, exist_ok=True)
    tag = bundle.representation.lower()

    bundle.importance.to_csv(
        source_dir / f"{tag}_feature_shap_statistics.csv", index=False
    )
    bundle.additivity.to_csv(
        source_dir / f"{tag}_shap_additivity_check.csv", index=False
    )

    shap_df = pd.DataFrame(
        bundle.shap_values,
        columns=[f"SHAP__{f}" for f in bundle.features],
    )
    shap_df.insert(0, "row_id", np.arange(len(shap_df), dtype=int))
    shap_df.to_csv(source_dir / f"{tag}_shap_values.csv", index=False)

    model_x_df = pd.DataFrame(bundle.model_X, columns=bundle.features)
    model_x_df.insert(0, "row_id", np.arange(len(model_x_df), dtype=int))
    model_x_df.to_csv(
        source_dir / f"{tag}_model_space_feature_values.csv", index=False
    )

    raw_x_df = bundle.raw_X.reset_index(drop=True).copy()
    raw_x_df.insert(0, "row_id", np.arange(len(raw_x_df), dtype=int))
    raw_x_df.to_csv(
        source_dir / f"{tag}_raw_feature_values.csv", index=False
    )

    pred_df = pd.DataFrame(
        {
            "row_id": np.arange(len(bundle.y), dtype=int),
            "CARBON_DOS": bundle.y,
            "GBR_prediction": bundle.prediction_pipeline,
        }
    )
    pred_df.to_csv(source_dir / f"{tag}_training_predictions.csv", index=False)


def export_dependence_source(pi_bundle: ShapBundle, source_dir: Path) -> None:
    long_rows: List[Dict[str, Any]] = []
    for feature in DEPENDENCE_FEATURES:
        j = pi_bundle.features.index(feature)
        for i in range(len(pi_bundle.raw_X)):
            long_rows.append(
                {
                    "row_id": i,
                    "feature": feature,
                    "feature_value_raw": pi_bundle.raw_X.iloc[i][feature],
                    "feature_value_model_space": pi_bundle.model_X[i, j],
                    "SHAP_value": pi_bundle.shap_values[i, j],
                    "GBR_prediction": pi_bundle.prediction_pipeline[i],
                }
            )
    pd.DataFrame(long_rows).to_csv(
        source_dir / "pi_dependence_source_data.csv", index=False
    )


# -----------------------------------------------------------------------------
# Plotting helpers
# -----------------------------------------------------------------------------

def configure_matplotlib() -> None:
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42


def save_figure(fig, out_dir: Path, stem: str, dpi: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        out_dir / f"{stem}.png",
        dpi=dpi,
        bbox_inches="tight",
        facecolor="white",
    )
    fig.savefig(
        out_dir / f"{stem}.pdf",
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)


def plot_pcc_vs_shap(
    bundle: ShapBundle,
    out_dir: Path,
    stem: str,
    dpi: int,
    distinguish_pi_added: bool,
) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 7.0), dpi=220)

    for _, row in bundle.importance.iterrows():
        feature = str(row["feature"])
        x = float(row["PCC_with_CARBON_DOS"])
        y = float(row["mean_abs_SHAP"])
        if not np.isfinite(x) or not np.isfinite(y):
            continue

        if distinguish_pi_added and feature in PI_ADDED_FEATURES:
            marker = "*"
            color = PI_COLOR
            size = 230
        else:
            marker = "o"
            color = ORI_COLOR
            size = 150

        ax.scatter(
            x,
            y,
            marker=marker,
            s=size,
            color=color,
            edgecolors="none",
            alpha=0.93,
            zorder=3,
        )
        ax.annotate(
            DISPLAY_NAMES.get(feature, feature),
            (x, y),
            xytext=(6, 5),
            textcoords="offset points",
            fontsize=9.5,
            color=color,
            zorder=4,
        )

    ax.axvline(0.0, color="#BFBFBF", linestyle="--", linewidth=1.2, zorder=1)
    ax.set_xlabel("Pearson Correlation Coefficient (PCC)", fontsize=12)
    ax.set_ylabel("mean |SHAP value|", fontsize=12)
    ax.tick_params(labelsize=10)
    ax.grid(False)

    if distinguish_pi_added:
        handles = [
            Line2D(
                [0], [0], marker="o", color="w", markerfacecolor=ORI_COLOR,
                markersize=9, label="Ori variables"
            ),
            Line2D(
                [0], [0], marker="*", color="w", markerfacecolor=PI_COLOR,
                markersize=13, label="PI-added state features"
            ),
        ]
        ax.legend(handles=handles, frameon=False, fontsize=9, loc="best")

    save_figure(fig, out_dir, stem, dpi)


def plot_importance_bar(
    bundle: ShapBundle,
    out_dir: Path,
    stem: str,
    dpi: int,
    color: str,
) -> None:
    df = bundle.importance.sort_values("mean_abs_SHAP", ascending=True)
    fig_h = max(5.5, 0.42 * len(df) + 1.8)
    fig, ax = plt.subplots(figsize=(8.3, fig_h), dpi=220)

    ax.barh(
        df["display_feature"],
        df["mean_abs_SHAP"],
        color=color,
        edgecolor="none",
        alpha=0.92,
    )
    ax.set_xlabel("mean |SHAP value|", fontsize=12)
    ax.tick_params(axis="x", labelsize=10)
    ax.tick_params(axis="y", labelsize=9)
    ax.grid(axis="x", linestyle="--", linewidth=0.6, alpha=0.25)
    ax.set_axisbelow(True)

    save_figure(fig, out_dir, stem, dpi)


def plot_pi_importance_and_groups(
    pi_bundle: ShapBundle,
    group_df: pd.DataFrame,
    out_dir: Path,
    stem: str,
    dpi: int,
) -> None:
    df = pi_bundle.importance.sort_values("mean_abs_SHAP", ascending=True)

    fig = plt.figure(figsize=(11.5, 9.5), dpi=220)
    ax = fig.add_axes([0.20, 0.10, 0.54, 0.83])

    ax.barh(
        df["display_feature"],
        df["mean_abs_SHAP"],
        color=PI_COLOR,
        edgecolor="none",
        alpha=0.94,
    )
    ax.set_xlabel("mean |SHAP value|", fontsize=12)
    ax.tick_params(axis="x", labelsize=10)
    ax.tick_params(axis="y", labelsize=9)
    ax.grid(axis="x", linestyle="--", linewidth=0.6, alpha=0.22)
    ax.set_axisbelow(True)

    inset = fig.add_axes([0.69, 0.21, 0.29, 0.42])
    sizes = [
        float(group_df.loc[group_df["group"] == g, "normalized_group_percent"].iloc[0])
        for g in GROUP_ORDER
    ]
    colors = [GROUP_COLORS[g] for g in GROUP_ORDER]

    wedges, texts, autotexts = inset.pie(
        sizes,
        colors=colors,
        startangle=90,
        counterclock=False,
        autopct=lambda p: f"{p:.1f}%" if p >= 3 else "",
        pctdistance=0.77,
        wedgeprops=dict(width=0.46, edgecolor="white", linewidth=0.8),
        textprops=dict(fontsize=8),
    )
    inset.text(0, 0, "Groups", ha="center", va="center", fontsize=10)
    inset.set_title("Process groups", fontsize=10, pad=4)
    inset.legend(
        wedges,
        GROUP_ORDER,
        loc="center left",
        bbox_to_anchor=(0.98, 0.5),
        frameon=False,
        fontsize=8,
    )
    inset.set_aspect("equal")

    fig.text(
        0.70,
        0.10,
        "Overlapping group scores\n(shared descriptors can enter >1 group)",
        fontsize=8,
        ha="left",
        va="bottom",
    )

    save_figure(fig, out_dir, stem, dpi)


def plot_beeswarm(
    bundle: ShapBundle,
    out_dir: Path,
    stem: str,
    dpi: int,
    max_display: int | None,
) -> None:
    display_cols = [DISPLAY_NAMES.get(f, f) for f in bundle.features]
    X_plot = pd.DataFrame(bundle.model_X, columns=display_cols)

    n_display = len(bundle.features) if max_display is None else min(
        max_display, len(bundle.features)
    )

    plt.figure(figsize=(10.2, max(6.0, 0.48 * n_display + 2.0)), dpi=220)
    shap.summary_plot(
        bundle.shap_values,
        X_plot,
        plot_type="dot",
        max_display=n_display,
        show=False,
        plot_size=None,
    )
    fig = plt.gcf()
    ax = fig.axes[0]
    ax.set_title("")
    ax.set_xlabel("SHAP value (impact on model output)", fontsize=12)
    ax.set_ylabel("")
    ax.tick_params(axis="x", labelsize=10)
    ax.tick_params(axis="y", labelsize=9)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(axis="y", linestyle=(0, (1, 4)), linewidth=0.6, color="#CFCFCF")

    if len(fig.axes) > 1:
        cbar_ax = fig.axes[-1]
        cbar_ax.set_ylabel("Feature value", fontsize=11)
        cbar_ax.tick_params(labelsize=9)

    save_figure(fig, out_dir, stem, dpi)


def sample_order(
    bundle: ShapBundle,
    feature_indices: Sequence[int],
    mode: str,
    center_positive_block: bool,
) -> np.ndarray:
    n = len(bundle.prediction_pipeline)
    if n <= 1 or mode == "original":
        order = np.arange(n)
    elif mode == "pred":
        order = np.argsort(bundle.prediction_pipeline)
    elif mode == "cluster":
        mat = bundle.shap_values[:, list(feature_indices)].copy()
        mean = mat.mean(axis=0, keepdims=True)
        std = mat.std(axis=0, keepdims=True)
        std[std == 0] = 1.0
        mat = (mat - mean) / std
        mat = np.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0)
        if len(mat) < 3:
            order = np.arange(n)
        else:
            dist = pdist(mat, metric="euclidean")
            Z = linkage(dist, method="ward")
            order = leaves_list(Z)
    else:
        raise ValueError(f"Unknown heatmap sort mode: {mode}")

    if not center_positive_block or len(order) < 10:
        return np.asarray(order, dtype=int)

    shap_sub = bundle.shap_values[:, list(feature_indices)]
    positive_score = np.sum(np.maximum(shap_sub, 0.0), axis=1)
    ordered_score = positive_score[order]

    window = max(5, int(round(len(order) * 0.18)))
    window = min(window, len(order))
    if window >= len(order):
        return np.asarray(order, dtype=int)

    rolling = np.convolve(ordered_score, np.ones(window), mode="valid")
    best_start = int(np.argmax(rolling))
    best_center = best_start + window // 2
    target_center = len(order) // 2
    shift = target_center - best_center

    return np.roll(np.asarray(order, dtype=int), shift)


def plot_heatmap(
    bundle: ShapBundle,
    out_dir: Path,
    stem: str,
    dpi: int,
    sort_mode: str,
    center_positive_block: bool,
) -> None:
    feature_order = (
        bundle.importance.sort_values("mean_abs_SHAP", ascending=False)["feature"]
        .tolist()
    )
    feature_indices = [bundle.features.index(f) for f in feature_order]
    order = sample_order(
        bundle,
        feature_indices,
        mode=sort_mode,
        center_positive_block=center_positive_block,
    )

    heat = bundle.shap_values[order][:, feature_indices].T
    pred = bundle.prediction_pipeline[order]
    mean_abs = np.mean(np.abs(bundle.shap_values[:, feature_indices]), axis=0)

    vmax = float(np.percentile(np.abs(heat), 99))
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0

    fig = plt.figure(figsize=(14.0, max(6.0, 0.30 * len(feature_order) + 3.0)), dpi=220)
    gs = GridSpec(
        2,
        2,
        figure=fig,
        height_ratios=[1.2, 8.8],
        width_ratios=[30, 2.4],
        hspace=0.05,
        wspace=0.12,
    )

    ax_top = fig.add_subplot(gs[0, 0])
    ax_heat = fig.add_subplot(gs[1, 0])
    ax_bar = fig.add_subplot(gs[1, 1])

    x = np.arange(len(order))
    ax_top.plot(x, pred, color="black", linewidth=1.2)
    ax_top.set_xlim(0, max(1, len(order) - 1))
    ax_top.set_xticks([])
    ax_top.set_ylabel("GBR\nprediction", fontsize=9)
    ax_top.tick_params(axis="y", labelsize=8)
    ax_top.spines["right"].set_visible(False)
    ax_top.spines["top"].set_visible(False)

    im = ax_heat.imshow(
        heat,
        aspect="auto",
        cmap="coolwarm",
        vmin=-vmax,
        vmax=vmax,
        interpolation="nearest",
    )
    ax_heat.set_yticks(np.arange(len(feature_order)))
    ax_heat.set_yticklabels(
        [DISPLAY_NAMES.get(f, f) for f in feature_order],
        fontsize=8.5,
    )
    ax_heat.set_xlabel("Instances", fontsize=11)
    ax_heat.tick_params(axis="x", labelsize=8)

    ax_bar.barh(
        np.arange(len(feature_order)),
        mean_abs,
        color="black",
        height=0.65,
    )
    ax_bar.set_ylim(len(feature_order) - 0.5, -0.5)
    ax_bar.set_yticks([])
    ax_bar.set_xlabel("mean\n|SHAP|", fontsize=8)
    ax_bar.tick_params(axis="x", labelsize=7)
    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)

    cbar = fig.colorbar(
        im,
        ax=[ax_top, ax_heat, ax_bar],
        fraction=0.018,
        pad=0.02,
    )
    cbar.set_label("SHAP value", fontsize=10)
    cbar.ax.tick_params(labelsize=8)

    save_figure(fig, out_dir, stem, dpi)


def plot_dependence_grid(
    pi_bundle: ShapBundle,
    out_dir: Path,
    stem: str,
    dpi: int,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(13.0, 8.0), dpi=220)
    axes_flat = axes.ravel()

    for ax, feature in zip(axes_flat, DEPENDENCE_FEATURES):
        j = pi_bundle.features.index(feature)
        x = pi_bundle.model_X[:, j]
        y = pi_bundle.shap_values[:, j]

        ax.scatter(
            x,
            y,
            s=24,
            alpha=0.70,
            color=PI_COLOR,
            edgecolors="none",
        )
        ax.axhline(0.0, color="#BFBFBF", linestyle="--", linewidth=0.9)
        ax.set_xlabel(DISPLAY_NAMES.get(feature, feature), fontsize=10)
        ax.set_ylabel("SHAP value", fontsize=10)
        ax.tick_params(labelsize=8.5)
        ax.grid(False)

    # Sixth panel is intentionally unused because Fig. S5 contains five panels.
    axes_flat[-1].axis("off")
    fig.tight_layout()
    save_figure(fig, out_dir, stem, dpi)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="KBDC Ori/PI GBR SHAP attribution and Figure 5 / Fig. S4-S5 source data."
    )
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS)
    parser.add_argument("--ori-model", type=Path, default=None)
    parser.add_argument("--pi-model", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument(
        "--max-display",
        type=int,
        default=None,
        help="Maximum features shown in beeswarm plots; default shows all.",
    )
    parser.add_argument(
        "--heatmap-sort",
        choices=["cluster", "pred", "original"],
        default="cluster",
    )
    parser.add_argument(
        "--no-center-positive-block",
        action="store_true",
        help="Disable the visual rotation used to center the strongest positive-SHAP block.",
    )
    parser.add_argument(
        "--do-not-save-fallback-models",
        action="store_true",
        help="Do not save reconstructed Ori/PI GBR files when model files are missing.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    configure_matplotlib()

    args.results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = args.results_dir / "figures"
    source_dir = args.results_dir / "source_data"
    figures_dir.mkdir(parents=True, exist_ok=True)
    source_dir.mkdir(parents=True, exist_ok=True)
    args.models_dir.mkdir(parents=True, exist_ok=True)

    ori_model_path = (
        args.ori_model
        if args.ori_model is not None
        else args.models_dir / "ori_gbr_model.joblib"
    )
    pi_model_path = (
        args.pi_model
        if args.pi_model is not None
        else args.models_dir / "pi_gbr_model.joblib"
    )

    print("KBDC SHAP attribution")
    print(f"  training data: {args.train}")
    print(f"  Ori model    : {ori_model_path}")
    print(f"  PI model     : {pi_model_path}")
    print(f"  results      : {args.results_dir}")

    train = canonicalize_columns(read_table(args.train))
    validate_training_frame(train)

    save_fallback = not args.do_not_save_fallback_models

    ori_model, ori_rebuilt = load_or_fit_gbr(
        "Ori",
        ori_model_path,
        train,
        seed=args.seed,
        save_fallback=save_fallback,
    )
    pi_model, pi_rebuilt = load_or_fit_gbr(
        "PI",
        pi_model_path,
        train,
        seed=args.seed,
        save_fallback=save_fallback,
    )

    if ori_rebuilt:
        print("  Ori-GBR model file was absent; reconstructed from training data.")
    if pi_rebuilt:
        print("  PI-GBR model file was absent; reconstructed from training data.")

    print("\nComputing TreeSHAP ...")
    ori = compute_shap_bundle("Ori", ori_model, train)
    pi = compute_shap_bundle("PI", pi_model, train)

    print(
        "  Ori max additivity error:",
        f"{ori.additivity['final_minus_reconstructed'].abs().max():.10f}",
    )
    print(
        "  PI  max additivity error:",
        f"{pi.additivity['final_minus_reconstructed'].abs().max():.10f}",
    )

    # Source data
    export_bundle(ori, source_dir)
    export_bundle(pi, source_dir)

    group_df = compute_overlapping_group_summary(pi)
    group_df.to_csv(
        source_dir / "pi_overlapping_process_group_attribution.csv",
        index=False,
    )
    export_dependence_source(pi, source_dir)

    # Direct PCC-vs-SHAP tables for Fig. 5a/b.
    ori.importance[
        ["rank", "feature", "mean_abs_SHAP", "PCC_with_CARBON_DOS", "PCC_p_value"]
    ].to_csv(source_dir / "fig5a_ori_pcc_vs_shap.csv", index=False)

    pi.importance[
        ["rank", "feature", "mean_abs_SHAP", "PCC_with_CARBON_DOS", "PCC_p_value"]
    ].to_csv(source_dir / "fig5b_pi_pcc_vs_shap.csv", index=False)

    # Main Figure 5
    plot_pcc_vs_shap(
        ori,
        figures_dir,
        "Fig5a_ori_pcc_vs_shap",
        args.dpi,
        distinguish_pi_added=False,
    )
    plot_pcc_vs_shap(
        pi,
        figures_dir,
        "Fig5b_pi_pcc_vs_shap",
        args.dpi,
        distinguish_pi_added=True,
    )
    plot_pi_importance_and_groups(
        pi,
        group_df,
        figures_dir,
        "Fig5c_pi_shap_importance_and_process_groups",
        args.dpi,
    )
    plot_beeswarm(
        pi,
        figures_dir,
        "Fig5d_pi_shap_beeswarm",
        args.dpi,
        args.max_display,
    )
    plot_heatmap(
        pi,
        figures_dir,
        "Fig5e_pi_shap_heatmap",
        args.dpi,
        args.heatmap_sort,
        center_positive_block=not args.no_center_positive_block,
    )

    # Supplementary Figure S4: Ori-GBR attribution
    plot_importance_bar(
        ori,
        figures_dir,
        "FigS4a_ori_shap_importance",
        args.dpi,
        ORI_COLOR,
    )
    plot_beeswarm(
        ori,
        figures_dir,
        "FigS4b_ori_shap_beeswarm",
        args.dpi,
        args.max_display,
    )
    plot_heatmap(
        ori,
        figures_dir,
        "FigS4c_ori_shap_heatmap",
        args.dpi,
        args.heatmap_sort,
        center_positive_block=not args.no_center_positive_block,
    )

    # Supplementary Figure S5: five PI dependence panels
    plot_dependence_grid(
        pi,
        figures_dir,
        "FigS5_pi_dependence",
        args.dpi,
    )

    run_metadata = {
        "target": TARGET,
        "training_file": str(args.train),
        "training_records": int(len(train)),
        "ori_model_path": str(ori_model_path),
        "pi_model_path": str(pi_model_path),
        "ori_model_reconstructed_here": bool(ori_rebuilt),
        "pi_model_reconstructed_here": bool(pi_rebuilt),
        "ori_features": ORI_FEATURES,
        "pi_features": PI_FEATURES,
        "pi_added_features": PI_ADDED_FEATURES,
        "gbr_config_if_reconstructed": GBR_CONFIG,
        "shap_evaluation_segment": "retained historical training segment",
        "heatmap_sort": args.heatmap_sort,
        "heatmap_positive_block_centered": not args.no_center_positive_block,
        "group_summary_type": "overlapping normalized process-group scores",
        "group_definitions": GROUP_DEFINITIONS,
        "dependence_features": DEPENDENCE_FEATURES,
        "ori_max_additivity_error": float(
            ori.additivity["final_minus_reconstructed"].abs().max()
        ),
        "pi_max_additivity_error": float(
            pi.additivity["final_minus_reconstructed"].abs().max()
        ),
        "shap_version": getattr(shap, "__version__", "unknown"),
    }
    with (args.results_dir / "run_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(run_metadata, f, indent=2, ensure_ascii=False)

    print("\nCompleted.")
    print(f"  figures    : {figures_dir}")
    print(f"  source data: {source_dir}")
    print(f"  metadata   : {args.results_dir / 'run_metadata.json'}")


if __name__ == "__main__":
    main()
