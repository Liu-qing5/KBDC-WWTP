#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
08_knn_validation.py
====================

Part 8 of the KBDC reproducibility pipeline.

Purpose
-------
Construct the validation-excluded KNN-matched historical engineering reference
reported in Text S9, Table S6, Algorithm S3, Table S7, Fig. 7, and Fig. S10.


Matching design
---------------
K = 5 historical neighbors are selected from the modelling-period historical
library using nine leakage-safe variables available before dosing:

    TREAT_FLOW
    IN_COD
    IN_TN
    IN_NH4
    IN_TP
    EFF_TN_lag1
    EFF_NH4_lag1
    ANO_MLSS
    ANO_SV30

Historical-library medians are used for missing-value filling, StandardScaler
is fitted on the historical library only, and weighted Euclidean distance is
calculated using the fixed Table S6 weights. Similarity is 1 / (1 + distance).

CARBON_DOS, D_GBR, D_KBDC, risk scores, same-day effluent outcomes, and other
strategy outputs are excluded from the matching vector.

Reference roles
---------------
- K = 5 mean/median/P25/P75: historical dosing reference.
- The same K = 5 set: historical TN reference.
- Nearest-1 only: historical NH4 diagnostic used for Fig. S10.

The KNN reference is descriptive of historical operator practice under similar
recorded states; it is not a causal counterfactual and not an alternative
control policy.

Typical use
-----------
python 08_knn_validation.py \
    --history data/processed/historical_modeling_library.csv \
    --validation data/field/validation_28d.csv

Outputs
-------
results/knn_validation/
    source_data/
        knn_daily_summary.csv
        knn_neighbor_detail.csv
        knn_weekly_summary.csv
        knn_28day_summary.csv
        fig7_reference_source.csv
        figs10_nearest1_nh4_source.csv
        knn_settings.csv
        knn_validation_reference.xlsx
    figures/
        FigS10_nearest1_historical_NH4.png
    run_metadata.json
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent if HERE.name.lower() == "code" else HERE
DEFAULT_RESULTS = ROOT / "results" / "knn_validation"

K_NEIGHBORS = 5
EPS = 1e-12


# ---------------------------------------------------------------------
# Table S6 settings
# ---------------------------------------------------------------------

KNN_FEATURES = [
    "TREAT_FLOW",
    "IN_COD",
    "IN_TN",
    "IN_NH4",
    "IN_TP",
    "EFF_TN_lag1",
    "EFF_NH4_lag1",
    "ANO_MLSS",
    "ANO_SV30",
]

FEATURE_WEIGHTS = {
    "TREAT_FLOW": 0.20,
    "IN_COD": 0.08,
    "IN_TN": 0.12,
    "IN_NH4": 0.10,
    "IN_TP": 0.05,
    "EFF_TN_lag1": 0.15,
    "EFF_NH4_lag1": 0.10,
    "ANO_MLSS": 0.12,
    "ANO_SV30": 0.08,
}

COLUMN_ALIASES: Dict[str, Sequence[str]] = {
    "TREAT_FLOW": ("TREAT_FLOW", "TREAT_FLOW_t(t/d)"),
    "IN_COD": ("IN_COD", "IN_COD_t(mg/L)"),
    "IN_TN": ("IN_TN", "IN_TN_t(mg/L)"),
    "IN_NH4": ("IN_NH4", "IN_NH4_t(mg/L)"),
    "IN_TP": ("IN_TP", "IN_TP_t(mg/L)"),
    "EFF_TN_lag1": ("EFF_TN_lag1", "EFF_TN_lag1(mg/L)", "EFF_TN_LAG1"),
    "EFF_NH4_lag1": (
        "EFF_NH4_lag1",
        "EFF_NH4_lag1(mg/L)",
        "EFF_NH4_LAG1",
    ),
    "ANO_MLSS": ("ANO_MLSS", "ANO_MLSS(mg/L)"),
    "ANO_SV30": ("ANO_SV30", "ANO_SV30(%)"),

    "CARBON_DOS": ("CARBON_DOS", "CARBON_DOS(g/t.water)"),
    "EFF_TN": ("EFF_TN", "EFF_TN(mg/L)"),
    "EFF_NH4": ("EFF_NH4", "EFF_NH4(mg/L)"),

    # Validation strategy/baseline aliases used only for reference summaries.
    "D_GBR": (
        "D_GBR",
        "GBR_Base_Dose(g/t.water)",
        "GBR_Base_Dose",
    ),
    "D_KBDC": (
        "D_KBDC",
        "Strategy_CARBON_DOS(g/t.water)",
        "Strategy_CARBON_DOS",
    ),
}

DATE_CANDIDATES = [
    "DATE", "Date", "date", "日期", "采样日期", "时间", "datetime", "Datetime"
]


# ---------------------------------------------------------------------
# Generic helpers
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
        raise ValueError(f"Unsupported file type: {path}")

    df.columns = df.columns.astype(str).str.strip()
    return canonicalize_columns(df)


def canonicalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    rename = {}

    for canonical, aliases in COLUMN_ALIASES.items():
        if canonical in out.columns:
            continue
        hit = next((a for a in aliases if a in out.columns), None)
        if hit is not None:
            rename[hit] = canonical

    return out.rename(columns=rename)


def require_columns(df: pd.DataFrame, cols: Sequence[str], label: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{label} missing required columns: {missing}")


def to_numeric(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def first_existing_column(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    return next((c for c in candidates if c in df.columns), None)


def weighted_distance_one_to_many(
    x: np.ndarray,
    Y: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    diff = Y - x
    return np.sqrt(np.sum(weights * diff * diff, axis=1))


def quantile_safe(s: pd.Series, q: float) -> float:
    x = pd.to_numeric(s, errors="coerce").dropna()
    return float(x.quantile(q)) if len(x) else float("nan")


# ---------------------------------------------------------------------
# Validation-excluded historical library
# ---------------------------------------------------------------------

def exclude_validation_dates(
    history: pd.DataFrame,
    validation: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """
    Explicitly exclude validation dates before imputation/scaling/matching when
    a common date field is available. If no common date field exists, the
    caller-provided historical table is assumed to already be the modelling
    library described in the SI.
    """
    h_date = first_existing_column(history, DATE_CANDIDATES)
    v_date = first_existing_column(validation, DATE_CANDIDATES)

    audit = {
        "history_date_column": h_date,
        "validation_date_column": v_date,
        "history_rows_before": int(len(history)),
        "overlap_rows_removed": 0,
        "history_rows_after": int(len(history)),
        "date_based_exclusion_performed": False,
    }

    if h_date is None or v_date is None:
        warnings.warn(
            "No common date field was detected. The supplied historical table "
            "is therefore assumed to already exclude all prospective-validation "
            "records before preprocessing and matching.",
            RuntimeWarning,
        )
        return history.copy().reset_index(drop=True), audit

    hist_dates = pd.to_datetime(history[h_date], errors="coerce").dt.normalize()
    valid_dates = pd.to_datetime(validation[v_date], errors="coerce").dt.normalize()

    valid_set = set(valid_dates.dropna().tolist())
    overlap = hist_dates.isin(valid_set)

    filtered = history.loc[~overlap].copy().reset_index(drop=True)

    audit.update(
        {
            "overlap_rows_removed": int(overlap.sum()),
            "history_rows_after": int(len(filtered)),
            "date_based_exclusion_performed": True,
        }
    )
    return filtered, audit


# ---------------------------------------------------------------------
# KNN construction
# ---------------------------------------------------------------------

def fit_knn_reference(
    history: pd.DataFrame,
    validation: pd.DataFrame,
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    Dict[str, object],
]:
    require_columns(history, KNN_FEATURES, "historical library")
    require_columns(validation, KNN_FEATURES, "validation data")
    require_columns(history, ["CARBON_DOS"], "historical library")

    history, exclusion_audit = exclude_validation_dates(history, validation)
    if len(history) < K_NEIGHBORS:
        raise ValueError(
            f"Historical library contains only {len(history)} records after "
            f"validation exclusion; at least {K_NEIGHBORS} are required."
        )

    numeric_cols = list(
        dict.fromkeys(
            KNN_FEATURES
            + ["CARBON_DOS", "EFF_TN", "EFF_NH4", "D_GBR", "D_KBDC"]
        )
    )
    hist_num = to_numeric(history, numeric_cols)
    valid_num = to_numeric(validation, numeric_cols)

    # Historical-library-only missing-value fill.
    medians = hist_num[KNN_FEATURES].median(numeric_only=True)
    hist_num[KNN_FEATURES] = hist_num[KNN_FEATURES].fillna(medians)
    valid_num[KNN_FEATURES] = valid_num[KNN_FEATURES].fillna(medians)

    if hist_num[KNN_FEATURES].isna().any().any():
        bad = hist_num[KNN_FEATURES].columns[
            hist_num[KNN_FEATURES].isna().any()
        ].tolist()
        raise ValueError(
            f"Historical-library medians could not resolve missing values: {bad}"
        )

    if valid_num[KNN_FEATURES].isna().any().any():
        bad = valid_num[KNN_FEATURES].columns[
            valid_num[KNN_FEATURES].isna().any()
        ].tolist()
        raise ValueError(
            f"Validation inputs remain missing after historical-median fill: {bad}"
        )

    # Historical-library-only standardization.
    scaler = StandardScaler()
    X_hist = scaler.fit_transform(hist_num[KNN_FEATURES].to_numpy(dtype=float))
    X_valid = scaler.transform(valid_num[KNN_FEATURES].to_numpy(dtype=float))

    weights = np.array(
        [FEATURE_WEIGHTS[f] for f in KNN_FEATURES],
        dtype=float,
    )
    weights = weights / weights.sum()

    daily_rows: List[Dict[str, object]] = []
    neighbor_rows: List[Dict[str, object]] = []
    figs10_rows: List[Dict[str, object]] = []

    date_col = first_existing_column(validation, DATE_CANDIDATES)

    for i in range(len(validation)):
        distances = weighted_distance_one_to_many(
            X_valid[i],
            X_hist,
            weights,
        )

        order = np.argsort(distances)
        nearest_idx = order[:K_NEIGHBORS]
        nearest_dist = distances[nearest_idx]
        nearest_sim = 1.0 / (1.0 + nearest_dist)

        hist_neighbors = hist_num.iloc[nearest_idx].copy()
        hist_neighbors_raw = history.iloc[nearest_idx].copy()

        hist_dose = pd.to_numeric(
            hist_neighbors["CARBON_DOS"],
            errors="coerce",
        )

        if "TREAT_FLOW" in hist_neighbors.columns:
            hist_flow = pd.to_numeric(
                hist_neighbors["TREAT_FLOW"],
                errors="coerce",
            )
            hist_mass = hist_dose * hist_flow / 1000.0
        else:
            hist_mass = pd.Series(np.nan, index=hist_neighbors.index)

        day = {
            "validation_day": i + 1,
            "week": i // 7 + 1,
            "nearest1_historical_row": int(nearest_idx[0] + 1),
            "nearest1_distance": float(nearest_dist[0]),
            "nearest1_similarity": float(nearest_sim[0]),
            "k5_mean_distance": float(np.mean(nearest_dist)),
            "k5_mean_similarity": float(np.mean(nearest_sim)),

            "KNN_CARBON_DOS_mean_g_t": float(hist_dose.mean()),
            "KNN_CARBON_DOS_median_g_t": float(hist_dose.median()),
            "KNN_CARBON_DOS_P25_g_t": quantile_safe(hist_dose, 0.25),
            "KNN_CARBON_DOS_P75_g_t": quantile_safe(hist_dose, 0.75),

            "KNN_dose_mass_mean_kg_d": float(hist_mass.mean()),
            "KNN_dose_mass_median_kg_d": float(hist_mass.median()),
            "KNN_dose_mass_P25_kg_d": quantile_safe(hist_mass, 0.25),
            "KNN_dose_mass_P75_kg_d": quantile_safe(hist_mass, 0.75),
        }

        if date_col is not None:
            day["validation_date"] = validation.iloc[i][date_col]

        # Same K=5 set supplies historical TN reference.
        if "EFF_TN" in hist_neighbors.columns:
            tn = pd.to_numeric(hist_neighbors["EFF_TN"], errors="coerce")
            day.update(
                {
                    "KNN_historical_TN_mean_mg_L": float(tn.mean()),
                    "KNN_historical_TN_median_mg_L": float(tn.median()),
                    "KNN_historical_TN_P25_mg_L": quantile_safe(tn, 0.25),
                    "KNN_historical_TN_P75_mg_L": quantile_safe(tn, 0.75),
                }
            )

        # Nearest-1 only supplies historical NH4 diagnostic.
        nearest1_nh4 = float("nan")
        if "EFF_NH4" in hist_neighbors.columns:
            nearest1_nh4 = float(
                pd.to_numeric(
                    pd.Series([hist_neighbors.iloc[0]["EFF_NH4"]]),
                    errors="coerce",
                ).iloc[0]
            )
            day["nearest1_historical_NH4_mg_L"] = nearest1_nh4

        # Validation posterior outcomes are carried only for evaluation.
        if "EFF_TN" in valid_num.columns:
            day["validation_EFF_TN_mg_L"] = float(valid_num.iloc[i]["EFF_TN"])
        if "EFF_NH4" in valid_num.columns:
            day["validation_EFF_NH4_mg_L"] = float(valid_num.iloc[i]["EFF_NH4"])

        # Optional already-computed KBDC / GBR references from validation.
        val_flow = float(valid_num.iloc[i]["TREAT_FLOW"])
        if "D_GBR" in valid_num.columns and np.isfinite(valid_num.iloc[i]["D_GBR"]):
            d_gbr = float(valid_num.iloc[i]["D_GBR"])
            day["D_GBR_g_t"] = d_gbr
            day["GBR_mass_kg_d"] = d_gbr * val_flow / 1000.0

        if "D_KBDC" in valid_num.columns and np.isfinite(valid_num.iloc[i]["D_KBDC"]):
            d_kbdc = float(valid_num.iloc[i]["D_KBDC"])
            day["D_KBDC_g_t"] = d_kbdc
            day["KBDC_mass_kg_d"] = d_kbdc * val_flow / 1000.0

        daily_rows.append(day)

        # Full K=5 neighbor detail.
        for rank, (hist_idx, dist, sim) in enumerate(
            zip(nearest_idx, nearest_dist, nearest_sim),
            start=1,
        ):
            hnum = hist_num.iloc[hist_idx]
            hraw = history.iloc[hist_idx]

            row = {
                "validation_day": i + 1,
                "week": i // 7 + 1,
                "neighbor_rank": rank,
                "historical_row": int(hist_idx + 1),
                "distance": float(dist),
                "similarity": float(sim),
                "historical_CARBON_DOS_g_t": float(hnum["CARBON_DOS"]),
                "historical_TREAT_FLOW_t_d": float(hnum["TREAT_FLOW"]),
                "historical_dose_mass_kg_d": (
                    float(hnum["CARBON_DOS"]) * float(hnum["TREAT_FLOW"]) / 1000.0
                ),
            }

            if date_col is not None:
                row["validation_date"] = validation.iloc[i][date_col]

            h_date_col = first_existing_column(history, DATE_CANDIDATES)
            if h_date_col is not None:
                row["historical_date"] = hraw[h_date_col]

            if "EFF_TN" in hnum.index:
                row["historical_EFF_TN_mg_L"] = float(hnum["EFF_TN"])
            if "EFF_NH4" in hnum.index:
                row["historical_EFF_NH4_mg_L"] = float(hnum["EFF_NH4"])

            for f in KNN_FEATURES:
                row[f"validation_{f}"] = float(valid_num.iloc[i][f])
                row[f"historical_{f}"] = float(hnum[f])
                row[f"difference_historical_minus_validation_{f}"] = (
                    float(hnum[f]) - float(valid_num.iloc[i][f])
                )

            neighbor_rows.append(row)

        if "EFF_NH4" in valid_num.columns:
            figs10_rows.append(
                {
                    "validation_day": i + 1,
                    "week": i // 7 + 1,
                    "validation_NH4_mg_L": float(valid_num.iloc[i]["EFF_NH4"]),
                    "nearest1_historical_NH4_mg_L": nearest1_nh4,
                    "nearest1_similarity": float(nearest_sim[0]),
                }
            )

    daily = pd.DataFrame(daily_rows)
    neighbors = pd.DataFrame(neighbor_rows)
    figs10 = pd.DataFrame(figs10_rows)

    metadata = {
        "K": K_NEIGHBORS,
        "matching_features": KNN_FEATURES,
        "feature_weights": FEATURE_WEIGHTS,
        "historical_medians": {
            k: (None if pd.isna(v) else float(v))
            for k, v in medians.items()
        },
        "scaler_mean": {
            f: float(v) for f, v in zip(KNN_FEATURES, scaler.mean_)
        },
        "scaler_scale": {
            f: float(v) for f, v in zip(KNN_FEATURES, scaler.scale_)
        },
        "validation_exclusion_audit": exclusion_audit,
    }
    return daily, neighbors, figs10, metadata


# ---------------------------------------------------------------------
# Weekly / 28-day summaries
# ---------------------------------------------------------------------

def weekly_summary(daily: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for week, sub in daily.groupby("week", sort=True):
        row = {
            "week": int(week),
            "days": int(len(sub)),
            "KNN_reference_dose_kg_week": float(
                sub["KNN_dose_mass_mean_kg_d"].sum()
            ),
            "KNN_mean_similarity": float(sub["k5_mean_similarity"].mean()),
            "nearest1_mean_similarity": float(
                sub["nearest1_similarity"].mean()
            ),
        }

        if "KNN_historical_TN_mean_mg_L" in sub.columns:
            row["KNN_historical_TN_mean_mg_L"] = float(
                sub["KNN_historical_TN_mean_mg_L"].mean()
            )
            row["KNN_historical_TN_SD_mg_L"] = float(
                sub["KNN_historical_TN_mean_mg_L"].std(ddof=1)
            )

        if "validation_EFF_TN_mg_L" in sub.columns:
            row["validation_TN_mean_mg_L"] = float(
                sub["validation_EFF_TN_mg_L"].mean()
            )
            row["validation_TN_SD_mg_L"] = float(
                sub["validation_EFF_TN_mg_L"].std(ddof=1)
            )

        if "validation_EFF_NH4_mg_L" in sub.columns:
            row["validation_NH4_mean_mg_L"] = float(
                sub["validation_EFF_NH4_mg_L"].mean()
            )
            row["validation_NH4_SD_mg_L"] = float(
                sub["validation_EFF_NH4_mg_L"].std(ddof=1)
            )

        if "nearest1_historical_NH4_mg_L" in sub.columns:
            row["nearest1_historical_NH4_mean_mg_L"] = float(
                sub["nearest1_historical_NH4_mg_L"].mean()
            )
            row["nearest1_historical_NH4_SD_mg_L"] = float(
                sub["nearest1_historical_NH4_mg_L"].std(ddof=1)
            )

        if "GBR_mass_kg_d" in sub.columns:
            row["GBR_baseline_kg_week"] = float(sub["GBR_mass_kg_d"].sum())

        if "KBDC_mass_kg_d" in sub.columns:
            row["KBDC_kg_week"] = float(sub["KBDC_mass_kg_d"].sum())

        if (
            "GBR_mass_kg_d" in sub.columns
            and "KBDC_mass_kg_d" in sub.columns
        ):
            g = float(sub["GBR_mass_kg_d"].sum())
            k = float(sub["KBDC_mass_kg_d"].sum())
            row["reduction_vs_GBR_percent"] = (
                (g - k) / g * 100.0 if abs(g) > EPS else np.nan
            )

        rows.append(row)

    return pd.DataFrame(rows)


def aggregate_summary(daily: pd.DataFrame) -> pd.DataFrame:
    knn_total = float(daily["KNN_dose_mass_mean_kg_d"].sum())

    row = {
        "period_days": int(len(daily)),
        "KNN_reference_total_kg": knn_total,
        "nearest1_mean_similarity": float(
            daily["nearest1_similarity"].mean()
        ),
        "K5_mean_similarity": float(
            daily["k5_mean_similarity"].mean()
        ),
    }

    if "GBR_mass_kg_d" in daily.columns:
        gbr_total = float(daily["GBR_mass_kg_d"].sum())
        row["GBR_baseline_total_kg"] = gbr_total

    if "KBDC_mass_kg_d" in daily.columns:
        kbdc_total = float(daily["KBDC_mass_kg_d"].sum())
        row["KBDC_total_kg"] = kbdc_total
        row["reduction_vs_KNN_percent"] = (
            (knn_total - kbdc_total) / knn_total * 100.0
            if abs(knn_total) > EPS
            else np.nan
        )

        if "GBR_mass_kg_d" in daily.columns:
            gbr_total = float(daily["GBR_mass_kg_d"].sum())
            row["reduction_vs_GBR_percent"] = (
                (gbr_total - kbdc_total) / gbr_total * 100.0
                if abs(gbr_total) > EPS
                else np.nan
            )

    return pd.DataFrame([row])


# ---------------------------------------------------------------------
# Source tables / figures
# ---------------------------------------------------------------------

def fig7_source(daily: pd.DataFrame) -> pd.DataFrame:
    keep = [
        c for c in [
            "validation_day",
            "week",
            "validation_date",
            "KNN_CARBON_DOS_mean_g_t",
            "KNN_CARBON_DOS_median_g_t",
            "KNN_CARBON_DOS_P25_g_t",
            "KNN_CARBON_DOS_P75_g_t",
            "KNN_dose_mass_mean_kg_d",
            "KNN_historical_TN_mean_mg_L",
            "validation_EFF_TN_mg_L",
            "D_GBR_g_t",
            "D_KBDC_g_t",
            "GBR_mass_kg_d",
            "KBDC_mass_kg_d",
        ]
        if c in daily.columns
    ]
    return daily[keep].copy()


def save_figs10(
    source: pd.DataFrame,
    out_dir: Path,
    dpi: int,
) -> None:
    if source.empty:
        return

    weekly = (
        source.groupby("week", as_index=False)
        .agg(
            validation_NH4_mean_mg_L=("validation_NH4_mg_L", "mean"),
            nearest1_historical_NH4_mean_mg_L=(
                "nearest1_historical_NH4_mg_L",
                "mean",
            ),
            validation_NH4_SD_mg_L=("validation_NH4_mg_L", "std"),
            nearest1_historical_NH4_SD_mg_L=(
                "nearest1_historical_NH4_mg_L",
                "std",
            ),
        )
    )

    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    x = weekly["week"].to_numpy(dtype=int)

    ax.errorbar(
        x,
        weekly["validation_NH4_mean_mg_L"],
        yerr=weekly["validation_NH4_SD_mg_L"],
        marker="o",
        capsize=4,
        linewidth=1.6,
        label="KBDC validation",
    )
    ax.errorbar(
        x,
        weekly["nearest1_historical_NH4_mean_mg_L"],
        yerr=weekly["nearest1_historical_NH4_SD_mg_L"],
        marker="s",
        capsize=4,
        linewidth=1.6,
        label="Nearest-1 historical reference",
    )
    ax.axhline(5.0, linestyle="--", linewidth=1.1, label="NH4-N control limit")
    ax.set_xticks(x)
    ax.set_xlabel("Validation week")
    ax.set_ylabel("Effluent NH4-N (mg/L)")
    ax.legend(frameon=False)
    fig.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        out_dir / "FigS10_nearest1_historical_NH4.png",
        dpi=dpi,
        bbox_inches="tight",
        facecolor="white",
    )
    fig.savefig(
        out_dir / "FigS10_nearest1_historical_NH4.pdf",
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)


def settings_table() -> pd.DataFrame:
    rows = [
        ("Purpose", "KNN-matched historical operator dosing reference; engineering reference only"),
        ("K", K_NEIGHBORS),
        ("Matching variables", "; ".join(KNN_FEATURES)),
        (
            "Feature weights",
            "; ".join(f"{f}={FEATURE_WEIGHTS[f]:.2f}" for f in KNN_FEATURES),
        ),
        (
            "Standardization",
            "StandardScaler fitted only on validation-excluded historical library",
        ),
        (
            "Missing values",
            "Filled by validation-excluded historical-library medians",
        ),
        (
            "Distance",
            "Weighted Euclidean distance in standardized feature space",
        ),
        ("Similarity", "1 / (1 + distance)"),
        (
            "Excluded from matching",
            "CARBON_DOS; D_GBR; D_KBDC; risk score; same-day effluent outcomes; derived strategy outputs",
        ),
        (
            "K=5 role",
            "Mean/median/P25/P75 historical dosing reference and historical TN reference",
        ),
        (
            "Nearest-1 role",
            "Supplementary historical effluent NH4 diagnostic only",
        ),
    ]
    return pd.DataFrame(rows, columns=["item", "setting"])


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Construct the validation-excluded KNN historical reference."
    )
    p.add_argument(
        "--history",
        type=Path,
        required=True,
        help="Modelling-period historical library; validation records must be excluded.",
    )
    p.add_argument(
        "--validation",
        type=Path,
        required=True,
        help="28-day prospective validation table.",
    )
    p.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    p.add_argument("--dpi", type=int, default=600)
    return p


def main() -> None:
    args = build_parser().parse_args()

    source_dir = args.results_dir / "source_data"
    figure_dir = args.results_dir / "figures"
    source_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    history = read_table(args.history)
    validation = read_table(args.validation)

    daily, neighbors, figs10, metadata = fit_knn_reference(
        history,
        validation,
    )

    weekly = weekly_summary(daily)
    aggregate = aggregate_summary(daily)
    fig7 = fig7_source(daily)
    settings = settings_table()

    daily.to_csv(
        source_dir / "knn_daily_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    neighbors.to_csv(
        source_dir / "knn_neighbor_detail.csv",
        index=False,
        encoding="utf-8-sig",
    )
    weekly.to_csv(
        source_dir / "knn_weekly_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    aggregate.to_csv(
        source_dir / "knn_28day_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    fig7.to_csv(
        source_dir / "fig7_reference_source.csv",
        index=False,
        encoding="utf-8-sig",
    )
    figs10.to_csv(
        source_dir / "figs10_nearest1_nh4_source.csv",
        index=False,
        encoding="utf-8-sig",
    )
    settings.to_csv(
        source_dir / "knn_settings.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # One consolidated workbook mirrors the user's historical Excel audit flow
    # as a consolidated audit output.
    xlsx_path = source_dir / "knn_validation_reference.xlsx"
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        daily.to_excel(writer, sheet_name="daily_summary", index=False)
        weekly.to_excel(writer, sheet_name="weekly_summary", index=False)
        aggregate.to_excel(writer, sheet_name="28day_summary", index=False)
        neighbors.to_excel(writer, sheet_name="K5_neighbors", index=False)
        fig7.to_excel(writer, sheet_name="Fig7_source", index=False)
        figs10.to_excel(writer, sheet_name="FigS10_source", index=False)
        settings.to_excel(writer, sheet_name="settings", index=False)

    save_figs10(figs10, figure_dir, args.dpi)

    metadata.update(
        {
            "history_file": str(args.history),
            "validation_file": str(args.validation),
            "validation_rows": int(len(validation)),
            "daily_output_rows": int(len(daily)),
            "neighbor_output_rows": int(len(neighbors)),
            "reference_interpretation": (
                "State-matched descriptive historical operator reference; "
                "not a causal counterfactual or alternative control policy."
            ),
        }
    )

    (args.results_dir / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    print("Completed KNN historical-reference analysis.")
    print("Daily source:", source_dir / "knn_daily_summary.csv")
    print("Weekly source:", source_dir / "knn_weekly_summary.csv")
    print("28-day source:", source_dir / "knn_28day_summary.csv")
    print("Workbook:", xlsx_path)
    print("Figures:", figure_dir)


if __name__ == "__main__":
    main()
