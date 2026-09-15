#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
01_preprocessing.py
===================

Leakage-aware preprocessing pipeline for the KBDC study.

This script consolidates the logic from the original data-splitting,
missing-value handling, training-only outlier screening, and PCC screening
scripts into one auditable pipeline.

It reproduces the study preprocessing sequence:
    chronological split (523 train / 129 held-out test)
    -> training-only / past-only missing-value handling
    -> training-only outlier screening
    -> PCC redundancy audit (flagging only, not automatic deletion)
    -> export of cleaned base data and audit tables

Important design choices
------------------------
1. The held-out test set is never used to estimate preprocessing statistics.
2. Outlier exclusion is applied only to the training set.
3. PCC > 0.8 is treated as a redundancy flag, not an automatic deletion rule.
4. IN_FLOW and ANO_MLVSS are retained as auxiliary variables because they are
   needed later for VFC and TN_PER_VSS, respectively.
5. ANO_SVI is kept in the cleaned base export for auditability, but is marked
   as excluded from the final model representation.

Default repository layout
-------------------------
KBDC-WWTP/
├── code/
│   └── 01_preprocessing.py
└── data/
    ├── raw/
    │   ├── train_raw.xlsx
    │   └── test_raw.xlsx
    └── processed/
        └── 01_preprocessing/

You may alternatively provide a single chronological 652-row file with
--full-data; the script will split its first 523 rows as training and the
remaining 129 rows as held-out test data.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Study constants
# ---------------------------------------------------------------------------

TRAIN_SIZE = 523
TEST_SIZE = 129
PCC_THRESHOLD = 0.80
OUTLIER_FEATURE_THRESHOLD = 4

TARGET_COL = "CARBON_DOS(g/t.water)"

CONTINUOUS_VARS = [
    "TREAT_FLOW_t(t/d)",
    "IN_FLOW(t/d)",
    "IN_COD_t(mg/L)",
    "IN_TN_t(mg/L)",
    "IN_NH4_t(mg/L)",
    "IN_TP_t(mg/L)",
]

BIO_STATE_VARS = [
    "ANO_MLSS(mg/L)",
    "ANO_MLVSS(mg/L)",
    "ANO_SV30(%)",
    "ANO_SVI(L/mg)",
]

CONTROL_VARS = [TARGET_COL]

ANALYSIS_VARS = CONTINUOUS_VARS + BIO_STATE_VARS + CONTROL_VARS

# These variables are physically expected to remain positive in the historical
# dataset used for the paper. Restricting the zero check to this explicit list
# avoids accidentally deleting rows because of unrelated numeric metadata.
POSITIVE_ONLY_VARS = ANALYSIS_VARS.copy()

FEATURE_ROLE_ROWS = [
    {
        "variable": "TREAT_FLOW_t(t/d)",
        "role": "Direct model input; source for LOW_FLOW_P",
        "final_model_input": "Yes",
    },
    {
        "variable": "IN_FLOW(t/d)",
        "role": "Auxiliary variable retained for VFC flow-reliability assessment",
        "final_model_input": "No",
    },
    {
        "variable": "IN_COD_t(mg/L)",
        "role": "Direct model input; source for COD/TN and COD_DEFICIT",
        "final_model_input": "Yes",
    },
    {
        "variable": "IN_TN_t(mg/L)",
        "role": "Direct model input; source for COD/TN, COD_DEFICIT, TN_PER_VSS",
        "final_model_input": "Yes",
    },
    {
        "variable": "IN_NH4_t(mg/L)",
        "role": "Direct model input",
        "final_model_input": "Yes",
    },
    {
        "variable": "IN_TP_t(mg/L)",
        "role": "Direct model input",
        "final_model_input": "Yes",
    },
    {
        "variable": "ANO_MLSS(mg/L)",
        "role": "Direct sludge-state model input",
        "final_model_input": "Yes",
    },
    {
        "variable": "ANO_MLVSS(mg/L)",
        "role": "Auxiliary variable retained to construct TN_PER_VSS",
        "final_model_input": "No (used for engineered feature)",
    },
    {
        "variable": "ANO_SV30(%)",
        "role": "Direct sludge-settling model input",
        "final_model_input": "Yes",
    },
    {
        "variable": "ANO_SVI(L/mg)",
        "role": "Screened out of final model representation",
        "final_model_input": "No",
    },
    {
        "variable": TARGET_COL,
        "role": "Supervised target; excluded from PCC calculations",
        "final_model_input": "Target",
    },
]


# ---------------------------------------------------------------------------
# I/O utilities
# ---------------------------------------------------------------------------

def read_table(path: Path) -> pd.DataFrame:
    """Read .xlsx/.xls or .csv and normalize column whitespace."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
    elif suffix == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported input format: {path.suffix}")

    df.columns = df.columns.astype(str).str.strip()
    return df


def save_df_pair(df: pd.DataFrame, base_path: Path) -> None:
    """Save the same table as XLSX and UTF-8-SIG CSV."""
    base_path = Path(base_path)
    base_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(base_path.with_suffix(".xlsx"), index=False)
    df.to_csv(base_path.with_suffix(".csv"), index=False, encoding="utf-8-sig")


def require_columns(df: pd.DataFrame, cols: Iterable[str], dataset_name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"{dataset_name} is missing required columns:\n  - "
            + "\n  - ".join(missing)
        )


# ---------------------------------------------------------------------------
# Step 1: chronological split
# ---------------------------------------------------------------------------

def chronological_split(
    full_df: pd.DataFrame,
    train_size: int = TRAIN_SIZE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split strictly by row order; never shuffle."""
    if len(full_df) < train_size + 1:
        raise ValueError(
            f"Full dataset has {len(full_df)} rows, fewer than train_size={train_size}."
        )

    train_df = full_df.iloc[:train_size].copy().reset_index(drop=True)
    test_df = full_df.iloc[train_size:].copy().reset_index(drop=True)
    return train_df, test_df


# ---------------------------------------------------------------------------
# Step 2: missing-value handling
# ---------------------------------------------------------------------------

def missing_summary(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "dataset": dataset_name,
            "variable": df.columns,
            "n_rows": len(df),
            "n_missing": df.isna().sum().values,
            "missing_percent": (df.isna().mean() * 100).values,
        }
    )


def impute_training(train_df: pd.DataFrame) -> pd.DataFrame:
    """
    Training-only imputation:
      - hydraulic/influent variables: linear interpolation;
      - sludge-state variables: 3-day centered rolling mean, then ffill/bfill;
      - control target: ffill/bfill (no effect for the supplied study data,
        where CARBON_DOS contains no missing values).
    """
    require_columns(train_df, ANALYSIS_VARS, "training data")
    out = train_df.copy()

    for col in CONTINUOUS_VARS:
        s = pd.to_numeric(out[col], errors="coerce")
        out[col] = s.interpolate(method="linear", limit_direction="both")

    for col in BIO_STATE_VARS:
        s = pd.to_numeric(out[col], errors="coerce")
        rolling_mean = s.rolling(window=3, min_periods=1, center=True).mean()
        out[col] = s.fillna(rolling_mean).ffill().bfill()

    for col in CONTROL_VARS:
        s = pd.to_numeric(out[col], errors="coerce")
        out[col] = s.ffill().bfill()

    return out


def impute_heldout(
    train_filled: pd.DataFrame,
    test_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Held-out imputation uses historical information only.

    The last filled training row is prepended so that the first test day may
    use the immediately preceding historical value. Each study variable is
    forward-filled for at most one day. No backward fill, centered rolling
    mean, or interpolation is used in held-out data.
    """
    require_columns(test_df, ANALYSIS_VARS, "held-out test data")

    test_out = test_df.copy()
    history_last = train_filled.tail(1).copy()
    combined = pd.concat([history_last, test_out], ignore_index=True)

    for col in ANALYSIS_VARS:
        s = pd.to_numeric(combined[col], errors="coerce")
        combined[col] = s.ffill(limit=1)

    return combined.iloc[1:].reset_index(drop=True)


def distribution_stability(
    before: pd.DataFrame,
    after: pd.DataFrame,
    dataset_name: str,
) -> pd.DataFrame:
    """
    Compare mean and standard deviation before vs after actual filling.

    Relative change (%) = |after - before| / |before| * 100.
    """
    rows = []
    for col in ANALYSIS_VARS:
        b = pd.to_numeric(before[col], errors="coerce")
        a = pd.to_numeric(after[col], errors="coerce")

        mean_before = float(b.mean())
        mean_after = float(a.mean())
        std_before = float(b.std(ddof=1))
        std_after = float(a.std(ddof=1))

        mean_rel = (
            abs(mean_after - mean_before) / abs(mean_before) * 100
            if mean_before != 0
            else np.nan
        )
        std_rel = (
            abs(std_after - std_before) / abs(std_before) * 100
            if std_before != 0
            else np.nan
        )

        rows.append(
            {
                "dataset": dataset_name,
                "variable": col,
                "mean_before": mean_before,
                "mean_after": mean_after,
                "mean_relative_change_percent": mean_rel,
                "std_before": std_before,
                "std_after": std_after,
                "std_relative_change_percent": std_rel,
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Step 3: training-only outlier audit and exclusion
# ---------------------------------------------------------------------------

def training_outlier_screen(
    train_filled: pd.DataFrame,
    iqr_feature_threshold: int = OUTLIER_FEATURE_THRESHOLD,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Apply the study's training-only exclusion rule.

    A training record is excluded when:
      (a) any variable explicitly expected to remain positive is zero; OR
      (b) at least `iqr_feature_threshold` variables are outside their own
          training-derived [Q1 - 1.5*IQR, Q3 + 1.5*IQR] interval.

    Returns
    -------
    clean_train
    variable_iqr_audit
    row_audit
    removed_rows
    retained_row_map
    """
    require_columns(train_filled, ANALYSIS_VARS, "filled training data")

    numeric = train_filled[ANALYSIS_VARS].apply(pd.to_numeric, errors="coerce")
    iqr_flags = pd.DataFrame(index=train_filled.index)
    variable_rows = []

    for col in ANALYSIS_VARS:
        s = numeric[col].dropna()
        q1 = float(s.quantile(0.25))
        q3 = float(s.quantile(0.75))
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        mask = (numeric[col] < lower) | (numeric[col] > upper)
        iqr_flags[col] = mask

        variable_rows.append(
            {
                "variable": col,
                "Q1": q1,
                "Q3": q3,
                "IQR": iqr,
                "lower_bound": lower,
                "upper_bound": upper,
                "n_iqr_outliers": int(mask.sum()),
                "iqr_outlier_percent": float(mask.mean() * 100),
            }
        )

    iqr_count = iqr_flags.sum(axis=1)

    positive_cols = [c for c in POSITIVE_ONLY_VARS if c in numeric.columns]
    zero_flags = numeric[positive_cols].eq(0)
    zero_count = zero_flags.sum(axis=1)
    has_invalid_zero = zero_count.gt(0)

    remove_mask = has_invalid_zero | iqr_count.ge(iqr_feature_threshold)

    reasons = np.select(
        [
            has_invalid_zero & iqr_count.ge(iqr_feature_threshold),
            has_invalid_zero,
            iqr_count.ge(iqr_feature_threshold),
        ],
        [
            f"zero in positive-only variable + >= {iqr_feature_threshold} IQR flags",
            "zero in positive-only variable",
            f">= {iqr_feature_threshold} IQR flags",
        ],
        default="retained",
    )

    row_audit = pd.DataFrame(
        {
            "training_row_0based": train_filled.index,
            "training_row_1based": train_filled.index + 1,
            "n_zero_positive_only": zero_count.values,
            "n_iqr_flags": iqr_count.values,
            "excluded": remove_mask.values,
            "reason": reasons,
        }
    )

    removed_rows = train_filled.loc[remove_mask].copy()
    removed_rows.insert(0, "training_row_0based", removed_rows.index)
    removed_rows.insert(1, "training_row_1based", removed_rows.index + 1)

    retained_indices = train_filled.index[~remove_mask]
    retained_row_map = pd.DataFrame(
        {
            "clean_train_row_0based": np.arange(len(retained_indices)),
            "original_training_row_0based": retained_indices.to_numpy(),
            "original_training_row_1based": retained_indices.to_numpy() + 1,
        }
    )

    clean_train = train_filled.loc[~remove_mask].copy().reset_index(drop=True)
    variable_iqr_audit = pd.DataFrame(variable_rows)

    return (
        clean_train,
        variable_iqr_audit,
        row_audit,
        removed_rows,
        retained_row_map,
    )


# ---------------------------------------------------------------------------
# Step 4: PCC audit (flag only; no automatic feature deletion)
# ---------------------------------------------------------------------------

def pcc_audit(
    clean_train: pd.DataFrame,
    threshold: float = PCC_THRESHOLD,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute Pearson correlations among training predictors only.

    The target is excluded. |PCC| > threshold is a redundancy flag only.
    No feature is automatically deleted here.
    """
    predictor_cols = [c for c in ANALYSIS_VARS if c != TARGET_COL]
    require_columns(clean_train, predictor_cols, "clean training data")

    numeric = clean_train[predictor_cols].apply(pd.to_numeric, errors="coerce")
    nonconstant = [c for c in predictor_cols if numeric[c].nunique(dropna=True) > 1]
    corr = numeric[nonconstant].corr(method="pearson")

    pairs = []
    for i, col1 in enumerate(nonconstant):
        for col2 in nonconstant[i + 1 :]:
            r = corr.loc[col1, col2]
            if pd.notna(r) and abs(r) > threshold:
                pairs.append(
                    {
                        "feature_1": col1,
                        "feature_2": col2,
                        "PCC": float(r),
                        "abs_PCC": float(abs(r)),
                        "flag": f"|PCC| > {threshold}",
                        "action": "Manual/process-informed review; no automatic deletion",
                    }
                )

    high_corr_pairs = pd.DataFrame(pairs)
    if not high_corr_pairs.empty:
        high_corr_pairs = high_corr_pairs.sort_values(
            "abs_PCC", ascending=False
        ).reset_index(drop=True)

    return corr, high_corr_pairs


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    output_dir: Path,
    full_data: Path | None = None,
    train_data: Path | None = None,
    test_data: Path | None = None,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- load / split
    if full_data is not None:
        full_df = read_table(full_data)
        train_raw, test_raw = chronological_split(full_df, TRAIN_SIZE)
        source_mode = "single chronological full dataset"
    else:
        if train_data is None or test_data is None:
            raise ValueError(
                "Provide either --full-data, or both --train-data and --test-data."
            )
        train_raw = read_table(train_data)
        test_raw = read_table(test_data)
        source_mode = "pre-split train/test datasets"

    require_columns(train_raw, ANALYSIS_VARS, "raw training data")
    require_columns(test_raw, ANALYSIS_VARS, "raw held-out test data")

    # Validate the study split when the supplied files are the paper dataset.
    if len(train_raw) != TRAIN_SIZE:
        print(
            f"WARNING: training data contain {len(train_raw)} rows; "
            f"the paper dataset contains {TRAIN_SIZE}."
        )
    if len(test_raw) != TEST_SIZE:
        print(
            f"WARNING: held-out test data contain {len(test_raw)} rows; "
            f"the paper dataset contains {TEST_SIZE}."
        )

    # --- missingness before
    missing_before = pd.concat(
        [
            missing_summary(train_raw, "train_before"),
            missing_summary(test_raw, "test_before"),
        ],
        ignore_index=True,
    )

    # --- imputation
    train_filled = impute_training(train_raw)
    test_filled = impute_heldout(train_filled, test_raw)

    missing_after = pd.concat(
        [
            missing_summary(train_filled, "train_after"),
            missing_summary(test_filled, "test_after"),
        ],
        ignore_index=True,
    )

    stability = pd.concat(
        [
            distribution_stability(train_raw, train_filled, "train"),
            distribution_stability(test_raw, test_filled, "test"),
        ],
        ignore_index=True,
    )

    # --- training-only outlier screen
    (
        train_clean,
        iqr_audit,
        row_audit,
        removed_rows,
        retained_row_map,
    ) = training_outlier_screen(train_filled)

    # Held-out test records are not excluded.
    test_clean = test_filled.copy().reset_index(drop=True)

    # --- PCC audit and explicit process roles
    corr_matrix, high_corr_pairs = pcc_audit(train_clean)
    feature_roles = pd.DataFrame(FEATURE_ROLE_ROWS)

    # --- save core datasets
    save_df_pair(train_clean, output_dir / "train_clean")
    save_df_pair(test_clean, output_dir / "test_clean")
    retained_row_map.to_csv(
        output_dir / "train_retained_rows.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # --- audit workbook
    with pd.ExcelWriter(
        output_dir / "preprocessing_audit.xlsx",
        engine="openpyxl",
    ) as writer:
        missing_before.to_excel(writer, sheet_name="missing_before", index=False)
        missing_after.to_excel(writer, sheet_name="missing_after", index=False)
        stability.to_excel(writer, sheet_name="distribution_stability", index=False)
        iqr_audit.to_excel(writer, sheet_name="training_IQR_audit", index=False)
        row_audit.to_excel(writer, sheet_name="training_row_audit", index=False)
        removed_rows.to_excel(writer, sheet_name="removed_training_rows", index=False)
        corr_matrix.to_excel(writer, sheet_name="training_PCC_matrix")
        high_corr_pairs.to_excel(writer, sheet_name="high_PCC_pairs", index=False)
        feature_roles.to_excel(writer, sheet_name="feature_roles", index=False)

    # --- compact machine-readable summary
    summary = {
        "source_mode": source_mode,
        "raw_train_rows": int(len(train_raw)),
        "raw_test_rows": int(len(test_raw)),
        "train_missing_cells_before": int(
            train_raw[ANALYSIS_VARS].isna().sum().sum()
        ),
        "test_missing_cells_before": int(
            test_raw[ANALYSIS_VARS].isna().sum().sum()
        ),
        "train_missing_cells_after": int(
            train_filled[ANALYSIS_VARS].isna().sum().sum()
        ),
        "test_missing_cells_after": int(
            test_filled[ANALYSIS_VARS].isna().sum().sum()
        ),
        "training_rows_removed": int(len(removed_rows)),
        "training_rows_retained": int(len(train_clean)),
        "heldout_rows_retained": int(len(test_clean)),
        "outlier_feature_threshold": OUTLIER_FEATURE_THRESHOLD,
        "pcc_flag_threshold": PCC_THRESHOLD,
        "pcc_is_flag_only": True,
    }

    with open(output_dir / "preprocessing_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\nKBDC preprocessing completed.")
    print(f"Raw training rows:        {len(train_raw)}")
    print(f"Raw held-out rows:        {len(test_raw)}")
    print(f"Training rows removed:    {len(removed_rows)}")
    print(f"Training rows retained:   {len(train_clean)}")
    print(f"Held-out rows retained:   {len(test_clean)}")
    print(
        "Missing cells after fill (train/test): "
        f"{summary['train_missing_cells_after']}/"
        f"{summary['test_missing_cells_after']}"
    )
    print(f"Outputs: {output_dir.resolve()}")


def parse_args() -> argparse.Namespace:
    script_path = Path(__file__).resolve()
    project_root = script_path.parent.parent if script_path.parent.name == "code" else script_path.parent

    parser = argparse.ArgumentParser(
        description="Leakage-aware preprocessing for the KBDC paper dataset."
    )
    parser.add_argument(
        "--full-data",
        type=Path,
        default=None,
        help="Optional single chronological dataset. If given, first 523 rows are training.",
    )
    parser.add_argument(
        "--train-data",
        type=Path,
        default=project_root / "data" / "raw" / "train_raw.xlsx",
        help="Pre-split raw training file (default: data/raw/train_raw.xlsx).",
    )
    parser.add_argument(
        "--test-data",
        type=Path,
        default=project_root / "data" / "raw" / "test_raw.xlsx",
        help="Pre-split raw held-out file (default: data/raw/test_raw.xlsx).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "data" / "processed" / "01_preprocessing",
        help="Output directory.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(
        output_dir=args.output_dir,
        full_data=args.full_data,
        train_data=args.train_data,
        test_data=args.test_data,
    )
