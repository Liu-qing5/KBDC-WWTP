#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
06_quantile_gbr.py
==================

Part 6 of the KBDC reproducibility pipeline.

Purpose
-------
Train P20, P50, and P80 Quantile-GBR models in the same PI feature space as
the retained PI-GBR baseline, and use the P20-P80 interval only as a
historical operator-response envelope / plausibility diagnostic.

This module does NOT determine the final KBDC dose and does NOT apply an
uncertainty-derived dose penalty.

Expected upstream files
-----------------------
data/processed/train_model_ready.csv
data/processed/test_model_ready.csv
models/pi_gbr_model.joblib

Optional
--------
A field-validation table can be supplied with --validation. It does not need
CARBON_DOS because the envelope is used diagnostically during deployment.

Outputs
-------
models/quantile_gbr_p20.joblib
models/quantile_gbr_p50.joblib
models/quantile_gbr_p80.joblib
models/quantile_gbr_preprocess.joblib

results/quantile_gbr/
    source_data/
    figures/
    quantile_gbr_metadata.json
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent if HERE.name.lower() == "code" else HERE

DEFAULT_TRAIN = ROOT / "data" / "processed" / "train_model_ready.csv"
DEFAULT_TEST = ROOT / "data" / "processed" / "test_model_ready.csv"
DEFAULT_PI_GBR = ROOT / "models" / "pi_gbr_model.joblib"
DEFAULT_MODELS_DIR = ROOT / "models"
DEFAULT_RESULTS_DIR = ROOT / "results" / "quantile_gbr"

TARGET_ALIASES = ("CARBON_DOS", "CARBON_DOS(g/t.water)")
EPS = 1e-8
RANDOM_STATE = 42


# ---------------------------------------------------------------------
# IO / model helpers
# ---------------------------------------------------------------------

def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".csv", ".txt"}:
        df = pd.read_csv(path)
    elif suffix in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
    elif suffix in {".parquet", ".pq"}:
        df = pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported table type: {path}")
    df.columns = df.columns.astype(str).str.strip()
    return df


def find_target(df: pd.DataFrame, required: bool = True) -> Optional[str]:
    for c in TARGET_ALIASES:
        if c in df.columns:
            return c
    if required:
        raise ValueError(
            f"Target column not found. Expected one of: {TARGET_ALIASES}"
        )
    return None


def get_model_features(model) -> List[str]:
    if hasattr(model, "feature_names_in_"):
        return [str(x) for x in model.feature_names_in_]

    if hasattr(model, "named_steps"):
        # Prefer pipeline-level feature names, then inspect steps.
        if hasattr(model, "feature_names_in_"):
            return [str(x) for x in model.feature_names_in_]
        for _, step in model.named_steps.items():
            if hasattr(step, "feature_names_in_"):
                return [str(x) for x in step.feature_names_in_]

    raise AttributeError(
        "Could not recover feature names from the saved PI-GBR model."
    )


def get_underlying_gbr(model) -> Optional[GradientBoostingRegressor]:
    if isinstance(model, GradientBoostingRegressor):
        return model

    if hasattr(model, "named_steps"):
        # Common case: pipeline(..., ("model", GradientBoostingRegressor(...))).
        if "model" in model.named_steps:
            candidate = model.named_steps["model"]
            if isinstance(candidate, GradientBoostingRegressor):
                return candidate

        for _, step in reversed(list(model.named_steps.items())):
            if isinstance(step, GradientBoostingRegressor):
                return step

    return None


def predict_pi_gbr(model, df: pd.DataFrame, feature_cols: Sequence[str]) -> np.ndarray:
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Input table is missing PI-GBR features: {missing}")

    X = df[list(feature_cols)].copy()
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")

    # If the saved object is a Pipeline, let its fitted preprocessing handle X.
    if hasattr(model, "named_steps"):
        return np.asarray(model.predict(X), dtype=float).reshape(-1)

    # For a plain estimator, use training medians later via predict_quantiles().
    return np.asarray(model.predict(X), dtype=float).reshape(-1)


def make_quantile_base_params(pi_model) -> Dict:
    """
    Reuse the retained GBR's tree/boosting settings when available, changing
    only loss/alpha for quantile training. This mirrors the original analysis.
    """
    gbr = get_underlying_gbr(pi_model)
    if gbr is None:
        # Historical fallback present in the original analysis script.
        return {
            "n_estimators": 300,
            "learning_rate": 0.03,
            "max_depth": 3,
            "subsample": 0.8,
            "random_state": RANDOM_STATE,
        }

    allowed = set(GradientBoostingRegressor().get_params().keys())
    params = {k: v for k, v in gbr.get_params().items() if k in allowed}

    # These are replaced explicitly for quantile training.
    params.pop("loss", None)
    params.pop("alpha", None)
    params["random_state"] = RANDOM_STATE
    return params


# ---------------------------------------------------------------------
# Leakage-safe quantile preprocessing
# ---------------------------------------------------------------------

def prepare_X(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    train_median: Optional[pd.Series] = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Input table is missing required PI features: {missing}")

    X = df[list(feature_cols)].copy()
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")

    if train_median is None:
        train_median = X.median(numeric_only=True)

    X = X.fillna(train_median)
    if X.isna().any().any():
        bad = X.columns[X.isna().any()].tolist()
        raise ValueError(f"Unresolved missing values after train-median fill: {bad}")

    return X, train_median


def train_quantile_gbr(
    alpha: float,
    base_params: Dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> GradientBoostingRegressor:
    params = dict(base_params)
    params["loss"] = "quantile"
    params["alpha"] = float(alpha)

    model = GradientBoostingRegressor(**params)
    model.fit(X_train, y_train)
    return model


def fix_quantile_crossing(
    p20: np.ndarray,
    p50: np.ndarray,
    p80: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    arr = np.vstack([p20, p50, p80]).T
    arr = np.sort(arr, axis=1)
    return arr[:, 0], arr[:, 1], arr[:, 2]


def predict_quantiles(
    models: Dict[str, GradientBoostingRegressor],
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    train_median: pd.Series,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    X, _ = prepare_X(df, feature_cols, train_median=train_median)
    p20 = np.asarray(models["p20"].predict(X), dtype=float)
    p50 = np.asarray(models["p50"].predict(X), dtype=float)
    p80 = np.asarray(models["p80"].predict(X), dtype=float)
    return fix_quantile_crossing(p20, p50, p80)


# ---------------------------------------------------------------------
# Metrics / outputs
# ---------------------------------------------------------------------

def quantile_metrics(
    y_true: pd.Series,
    p20: np.ndarray,
    p50: np.ndarray,
    p80: np.ndarray,
) -> Dict[str, float]:
    y = pd.to_numeric(y_true, errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(y)

    y = y[valid]
    q20 = np.asarray(p20, dtype=float)[valid]
    q50 = np.asarray(p50, dtype=float)[valid]
    q80 = np.asarray(p80, dtype=float)[valid]

    if len(y) == 0:
        return {}

    mape = np.mean(
        np.abs((y - q50) / np.where(np.abs(y) > EPS, np.abs(y), np.nan))
    ) * 100.0

    return {
        "N": int(len(y)),
        "P50_MAE": float(mean_absolute_error(y, q50)),
        "P50_MAPE_percent": float(mape),
        "P50_RMSE": float(np.sqrt(mean_squared_error(y, q50))),
        "P50_R2": float(r2_score(y, q50)),
        "P20_P80_PICP_percent": float(np.mean((y >= q20) & (y <= q80)) * 100.0),
        "P20_P80_MPIW": float(np.mean(q80 - q20)),
        "P20_P80_NMPIW_percent": float(
            np.mean((q80 - q20) / (np.abs(q50) + EPS)) * 100.0
        ),
    }


def build_result_table(
    df: pd.DataFrame,
    pi_baseline: np.ndarray,
    p20: np.ndarray,
    p50: np.ndarray,
    p80: np.ndarray,
) -> pd.DataFrame:
    out = df.copy()
    out["D_GBR"] = pi_baseline
    out["P20"] = p20
    out["P50"] = p50
    out["P80"] = p80
    out["P20_P80_width"] = p80 - p20
    out["D_GBR_inside_P20_P80"] = (pi_baseline >= p20) & (pi_baseline <= p80)
    return out


def save_envelope_plot(
    result: pd.DataFrame,
    out_path: Path,
    title: str,
    dpi: int = 600,
) -> None:
    x = np.arange(1, len(result) + 1)

    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    ax.fill_between(
        x,
        result["P20"].to_numpy(dtype=float),
        result["P80"].to_numpy(dtype=float),
        alpha=0.25,
        label="P20-P80 baseline core envelope",
    )
    ax.plot(
        x,
        result["D_GBR"].to_numpy(dtype=float),
        linewidth=1.8,
        label="PI-GBR baseline",
    )
    ax.set_xlabel("Sample index")
    ax.set_ylabel("Carbon-source dosage intensity (g/t water)")
    ax.set_title(title)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train P20/P50/P80 Quantile-GBR historical-domain envelope."
    )
    p.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    p.add_argument("--test", type=Path, default=DEFAULT_TEST)
    p.add_argument(
        "--validation",
        type=Path,
        default=None,
        help="Optional field-validation table (CSV/XLSX/Parquet).",
    )
    p.add_argument("--pi-gbr-model", type=Path, default=DEFAULT_PI_GBR)
    p.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    p.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    p.add_argument("--dpi", type=int, default=600)
    return p


def main() -> None:
    args = build_parser().parse_args()

    args.models_dir.mkdir(parents=True, exist_ok=True)
    source_dir = args.results_dir / "source_data"
    figure_dir = args.results_dir / "figures"
    source_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    if not args.pi_gbr_model.exists():
        raise FileNotFoundError(
            f"PI-GBR model not found: {args.pi_gbr_model}\n"
            "Run 03_model_benchmark.py first."
        )

    train_df = read_table(args.train)
    test_df = read_table(args.test)
    valid_df = read_table(args.validation) if args.validation is not None else None

    pi_model = joblib.load(args.pi_gbr_model)
    feature_cols = get_model_features(pi_model)

    target_train = find_target(train_df, required=True)
    target_test = find_target(test_df, required=True)

    X_train, train_median = prepare_X(train_df, feature_cols, train_median=None)
    y_train = pd.to_numeric(train_df[target_train], errors="coerce")
    valid_train = y_train.notna()

    X_train = X_train.loc[valid_train].reset_index(drop=True)
    y_train = y_train.loc[valid_train].reset_index(drop=True)

    base_params = make_quantile_base_params(pi_model)

    models = {
        "p20": train_quantile_gbr(0.20, base_params, X_train, y_train),
        "p50": train_quantile_gbr(0.50, base_params, X_train, y_train),
        "p80": train_quantile_gbr(0.80, base_params, X_train, y_train),
    }

    # Save models and the exact train-fitted preprocessing state.
    joblib.dump(models["p20"], args.models_dir / "quantile_gbr_p20.joblib")
    joblib.dump(models["p50"], args.models_dir / "quantile_gbr_p50.joblib")
    joblib.dump(models["p80"], args.models_dir / "quantile_gbr_p80.joblib")
    joblib.dump(
        {
            "feature_names": list(feature_cols),
            "train_median": train_median.to_dict(),
        },
        args.models_dir / "quantile_gbr_preprocess.joblib",
    )

    # Historical held-out test.
    p20_t, p50_t, p80_t = predict_quantiles(
        models, test_df, feature_cols, train_median
    )
    baseline_t = predict_pi_gbr(pi_model, test_df, feature_cols)
    test_result = build_result_table(
        test_df, baseline_t, p20_t, p50_t, p80_t
    )
    test_result.to_csv(
        source_dir / "historical_test_quantile_envelope.csv",
        index=False,
        encoding="utf-8-sig",
    )

    test_metrics = quantile_metrics(
        test_df[target_test], p20_t, p50_t, p80_t
    )
    test_metrics["D_GBR_inside_P20_P80_percent"] = float(
        test_result["D_GBR_inside_P20_P80"].mean() * 100.0
    )

    save_envelope_plot(
        test_result,
        figure_dir / "historical_test_p20_p80_envelope.png",
        title="Historical held-out test: P20-P80 baseline core envelope",
        dpi=args.dpi,
    )

    metadata = {
        "feature_names": list(feature_cols),
        "quantiles": [0.20, 0.50, 0.80],
        "base_gbr_params": base_params,
        "test_metrics": test_metrics,
        "interpretation": (
            "P20-P80 is a historical operator-response plausibility envelope. "
            "It does not calculate D_KBDC or determine adjustment magnitude."
        ),
    }

    # Optional prospective validation table.
    if valid_df is not None:
        p20_v, p50_v, p80_v = predict_quantiles(
            models, valid_df, feature_cols, train_median
        )
        baseline_v = predict_pi_gbr(pi_model, valid_df, feature_cols)

        valid_result = build_result_table(
            valid_df, baseline_v, p20_v, p50_v, p80_v
        )
        valid_result.to_csv(
            source_dir / "field_validation_quantile_envelope.csv",
            index=False,
            encoding="utf-8-sig",
        )
        valid_inside = float(
            valid_result["D_GBR_inside_P20_P80"].mean() * 100.0
        )
        metadata["field_validation_D_GBR_inside_P20_P80_percent"] = valid_inside

        save_envelope_plot(
            valid_result,
            figure_dir / "field_validation_p20_p80_envelope.png",
            title="Field validation: P20-P80 baseline core envelope",
            dpi=args.dpi,
        )

    (args.results_dir / "quantile_gbr_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    print("Completed Quantile-GBR envelope analysis.")
    print("Models:", args.models_dir)
    print("Results:", args.results_dir)
    print(
        "Historical-test D_GBR inside P20-P80:",
        f"{test_metrics['D_GBR_inside_P20_P80_percent']:.2f}%",
    )


if __name__ == "__main__":
    main()
