#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
02_process_features.py
======================

Process-informed feature construction for the KBDC study.

This script converts the cleaned base variables produced by
01_preprocessing.py into the model-ready representation used in the paper.

The calculations reproduce the spreadsheet logic supplied with the study:
    LOW_FLOW_PRESSURE = max(0, Q25_train - TREAT_FLOW)
    COD/TN            = IN_COD / IN_TN
    COD_DEFICIT       = max(0, 4.12*IN_TN - 0.555*IN_COD)
    TN_PER_VSS        = IN_TN / ANO_MLVSS
    RISK_TN_lag1      = EFF_TN_lag1 / 15
    RISK_NH4_lag1     = EFF_NH4_lag1 / 5
    RISK_MAX_lag1     = max(RISK_TN_lag1, RISK_NH4_lag1)
    RISK_MEAN_lag1    = mean(RISK_TN_lag1, RISK_NH4_lag1)

Name mapping to the Supplementary Information:
    LOW_FLOW_PRESSURE -> LOW_FLOW_P
    RISK_TN_lag1      -> R_TN_lag1
    RISK_NH4_lag1     -> R_NH4_lag1
    RISK_MAX_lag1     -> R_MAX_lag1
    RISK_MEAN_lag1    -> R_MEAN_lag1

Lagged effluent inputs
----------------------
The original spreadsheets used EFF_TN_lag1 and EFF_NH4_lag1 created by
shifting the confirmed effluent measurements by one day. Because the supplied
clean base files do not contain same-day effluent TN/NH4, this script accepts
the lagged columns in either of two honest ways:

1. They are already present in the train/test input files; OR
2. They are supplied through --train-lag-source and --test-lag-source.
   The lag-source files must be row-aligned with the cleaned train/test data.

This avoids pretending that unavailable same-day effluent measurements can be
reconstructed from the base predictors.

Risk-ratio definition
---------------------
For PI-GBR feature construction, the boundary-relative descriptors are direct
ratios to the plant limits and are NOT clipped:

    RISK_TN_lag1  = EFF_TN_lag1 / 15
    RISK_NH4_lag1 = EFF_NH4_lag1 / 5

Values above 1 are intentionally retained because they preserve exceedance
magnitude for baseline learning. Clipping to [0, 1] belongs only to the later
KBDC risk-scoring layer, not to PI feature construction.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Study constants
# ---------------------------------------------------------------------------

TN_LIMIT_MG_L = 15.0
NH4_LIMIT_MG_L = 5.0
COD_DEFICIT_TN_COEF = 4.12
COD_DEFICIT_COD_COEF = 0.555

TARGET_COL = "CARBON_DOS(g/t.water)"

TREAT_FLOW = "TREAT_FLOW_t(t/d)"
IN_FLOW = "IN_FLOW(t/d)"
IN_COD = "IN_COD_t(mg/L)"
IN_TN = "IN_TN_t(mg/L)"
IN_NH4 = "IN_NH4_t(mg/L)"
IN_TP = "IN_TP_t(mg/L)"
ANO_MLSS = "ANO_MLSS(mg/L)"
ANO_MLVSS = "ANO_MLVSS(mg/L)"
ANO_SV30 = "ANO_SV30(%)"
ANO_SVI = "ANO_SVI(L/mg)"

EFF_TN_LAG = "EFF_TN_lag1(mg/L)"
EFF_NH4_LAG = "EFF_NH4_lag1(mg/L)"

LOW_FLOW = "LOW_FLOW_PRESSURE"
COD_TN = "COD/TN"
COD_DEFICIT = "COD_DEFICIT"
R_TN = "RISK_TN_lag1"
R_NH4 = "RISK_NH4_lag1"
R_MAX = "RISK_MAX_lag1"
R_MEAN = "RISK_MEAN_lag1"
TN_PER_VSS = "TN_PER_VSS"

MODEL_READY_COLUMNS = [
    TREAT_FLOW,
    LOW_FLOW,
    IN_COD,
    IN_TN,
    COD_TN,
    COD_DEFICIT,
    IN_NH4,
    IN_TP,
    EFF_TN_LAG,
    EFF_NH4_LAG,
    R_TN,
    R_NH4,
    R_MAX,
    R_MEAN,
    ANO_MLSS,
    ANO_SV30,
    TN_PER_VSS,
    TARGET_COL,
]

PAPER_NAME_MAPPING = {
    LOW_FLOW: "LOW_FLOW_P",
    R_TN: "R_TN_lag1",
    R_NH4: "R_NH4_lag1",
    R_MAX: "R_MAX_lag1",
    R_MEAN: "R_MEAN_lag1",
}


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def read_table(path: Path) -> pd.DataFrame:
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
    base_path = Path(base_path)
    base_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(base_path.with_suffix(".xlsx"), index=False)
    df.to_csv(base_path.with_suffix(".csv"), index=False, encoding="utf-8-sig")


def require_columns(df: pd.DataFrame, cols: list[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"{name} is missing required columns:\n  - "
            + "\n  - ".join(missing)
        )


# ---------------------------------------------------------------------------
# Lagged effluent handling
# ---------------------------------------------------------------------------

def attach_lagged_effluent(
    df: pd.DataFrame,
    dataset_name: str,
    lag_source: Path | None = None,
) -> pd.DataFrame:
    """
    Ensure EFF_TN_lag1 and EFF_NH4_lag1 are present.

    If they are not already in `df`, read them from a row-aligned lag source.
    """
    out = df.copy()

    if EFF_TN_LAG in out.columns and EFF_NH4_LAG in out.columns:
        return out

    if lag_source is None:
        raise ValueError(
            f"{dataset_name} does not contain {EFF_TN_LAG} and {EFF_NH4_LAG}.\n"
            f"Provide a row-aligned lag source with --{dataset_name}-lag-source "
            "or include these lagged columns in the input data."
        )

    lag_df = read_table(lag_source)
    require_columns(lag_df, [EFF_TN_LAG, EFF_NH4_LAG], f"{dataset_name} lag source")

    if len(lag_df) != len(out):
        raise ValueError(
            f"{dataset_name} lag source has {len(lag_df)} rows, "
            f"but {dataset_name} input has {len(out)} rows. "
            "Lag-source rows must be aligned one-to-one with the cleaned data."
        )

    out[EFF_TN_LAG] = pd.to_numeric(lag_df[EFF_TN_LAG], errors="coerce").to_numpy()
    out[EFF_NH4_LAG] = pd.to_numeric(lag_df[EFF_NH4_LAG], errors="coerce").to_numpy()
    return out


# ---------------------------------------------------------------------------
# Feature construction
# ---------------------------------------------------------------------------

def add_process_features(
    df: pd.DataFrame,
    q25_train: float,
) -> pd.DataFrame:
    required = [
        TREAT_FLOW,
        IN_COD,
        IN_TN,
        IN_NH4,
        IN_TP,
        ANO_MLSS,
        ANO_SV30,
        TARGET_COL,
        EFF_TN_LAG,
        EFF_NH4_LAG,
    ]
    require_columns(df, required, "feature-construction input")

    out = df.copy()

    # Convert required numerical columns explicitly.
    for col in required:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    # 1) Hydraulic scale / low-flow pressure.
    out[LOW_FLOW] = np.maximum(
        0.0,
        float(q25_train) - out[TREAT_FLOW],
    )

    # 2) Apparent carbon-nitrogen limitation.
    if (out[IN_TN] == 0).any():
        raise ValueError("IN_TN contains zero; COD/TN would be undefined.")
    out[COD_TN] = out[IN_COD] / out[IN_TN]
    out[COD_DEFICIT] = np.maximum(
        0.0,
        COD_DEFICIT_TN_COEF * out[IN_TN]
        - COD_DEFICIT_COD_COEF * out[IN_COD],
    )

    # 3) Delayed nitrogen-safety feedback.
    r_tn = out[EFF_TN_LAG] / TN_LIMIT_MG_L
    r_nh4 = out[EFF_NH4_LAG] / NH4_LIMIT_MG_L

    out[R_TN] = r_tn
    out[R_NH4] = r_nh4
    out[R_MAX] = np.maximum(out[R_TN], out[R_NH4])
    out[R_MEAN] = (out[R_TN] + out[R_NH4]) / 2.0

    # 4) Sludge-supported nitrogen pressure.
    if ANO_MLVSS in out.columns:
        out[ANO_MLVSS] = pd.to_numeric(out[ANO_MLVSS], errors="coerce")
        if (out[ANO_MLVSS] == 0).any():
            raise ValueError("ANO_MLVSS contains zero; TN_PER_VSS would be undefined.")
        out[TN_PER_VSS] = out[IN_TN] / out[ANO_MLVSS]
    elif TN_PER_VSS in out.columns:
        # Compatibility mode for an already-engineered spreadsheet.
        out[TN_PER_VSS] = pd.to_numeric(out[TN_PER_VSS], errors="coerce")
    else:
        raise ValueError(
            f"Need either {ANO_MLVSS} to calculate {TN_PER_VSS}, "
            f"or an existing {TN_PER_VSS} column."
        )

    return out


def to_model_ready(df: pd.DataFrame) -> pd.DataFrame:
    require_columns(df, MODEL_READY_COLUMNS, "engineered dataset")
    return df[MODEL_READY_COLUMNS].copy()


def make_formula_table(q25_train: float) -> pd.DataFrame:
    risk_formula_tn = f"{EFF_TN_LAG} / {TN_LIMIT_MG_L:g}"
    risk_formula_nh4 = f"{EFF_NH4_LAG} / {NH4_LIMIT_MG_L:g}"

    return pd.DataFrame(
        [
            {
                "feature": LOW_FLOW,
                "SI_name": PAPER_NAME_MAPPING[LOW_FLOW],
                "formula": f"max(0, {q25_train:.12g} - {TREAT_FLOW})",
            },
            {
                "feature": COD_TN,
                "SI_name": "COD/TN",
                "formula": f"{IN_COD} / {IN_TN}",
            },
            {
                "feature": COD_DEFICIT,
                "SI_name": COD_DEFICIT,
                "formula": (
                    f"max(0, {COD_DEFICIT_TN_COEF}*{IN_TN} - "
                    f"{COD_DEFICIT_COD_COEF}*{IN_COD})"
                ),
            },
            {
                "feature": R_TN,
                "SI_name": PAPER_NAME_MAPPING[R_TN],
                "formula": risk_formula_tn,
            },
            {
                "feature": R_NH4,
                "SI_name": PAPER_NAME_MAPPING[R_NH4],
                "formula": risk_formula_nh4,
            },
            {
                "feature": R_MAX,
                "SI_name": PAPER_NAME_MAPPING[R_MAX],
                "formula": f"max({R_TN}, {R_NH4})",
            },
            {
                "feature": R_MEAN,
                "SI_name": PAPER_NAME_MAPPING[R_MEAN],
                "formula": f"mean({R_TN}, {R_NH4})",
            },
            {
                "feature": TN_PER_VSS,
                "SI_name": TN_PER_VSS,
                "formula": f"{IN_TN} / {ANO_MLVSS}",
            },
        ]
    )


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    train_input: Path,
    test_input: Path,
    output_dir: Path,
    train_lag_source: Path | None = None,
    test_lag_source: Path | None = None,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train = read_table(train_input)
    test = read_table(test_input)

    train = attach_lagged_effluent(
        train,
        dataset_name="train",
        lag_source=train_lag_source,
    )
    test = attach_lagged_effluent(
        test,
        dataset_name="test",
        lag_source=test_lag_source,
    )

    require_columns(train, [TREAT_FLOW], "training input")
    q25_train = float(pd.to_numeric(train[TREAT_FLOW], errors="coerce").quantile(0.25))

    train_eng = add_process_features(
        train,
        q25_train=q25_train,
    )
    test_eng = add_process_features(
        test,
        q25_train=q25_train,
    )

    train_model = to_model_ready(train_eng)
    test_model = to_model_ready(test_eng)

    # Export both the augmented audit data and the exact model-ready matrix.
    save_df_pair(train_eng, output_dir / "train_augmented")
    save_df_pair(test_eng, output_dir / "test_augmented")
    save_df_pair(train_model, output_dir / "train_model_ready")
    save_df_pair(test_model, output_dir / "test_model_ready")

    formula_table = make_formula_table(q25_train)
    formula_table.to_csv(
        output_dir / "feature_formulas.csv",
        index=False,
        encoding="utf-8-sig",
    )

    parameters = {
        "Q25_train_TREAT_FLOW_t_per_d": q25_train,
        "TN_limit_mg_L": TN_LIMIT_MG_L,
        "NH4_limit_mg_L": NH4_LIMIT_MG_L,
        "COD_DEFICIT_TN_coefficient": COD_DEFICIT_TN_COEF,
        "COD_DEFICIT_COD_coefficient": COD_DEFICIT_COD_COEF,
        "PI_risk_ratio_clipping": "none",
        "spreadsheet_to_SI_name_mapping": PAPER_NAME_MAPPING,
        "train_rows": int(len(train_model)),
        "test_rows": int(len(test_model)),
    }

    with open(output_dir / "feature_parameters.json", "w", encoding="utf-8") as f:
        json.dump(parameters, f, ensure_ascii=False, indent=2)

    print("\nKBDC process-informed feature construction completed.")
    print(f"Training rows: {len(train_model)}")
    print(f"Test rows:     {len(test_model)}")
    print(f"Q25_train:     {q25_train:.12f} t/d")
    print("PI risk ratios: direct division by plant limits; no clipping")
    print(f"Outputs:       {output_dir.resolve()}")


def parse_args() -> argparse.Namespace:
    script_path = Path(__file__).resolve()
    project_root = script_path.parent.parent if script_path.parent.name == "code" else script_path.parent

    parser = argparse.ArgumentParser(
        description="Construct process-informed features for the KBDC study."
    )
    parser.add_argument(
        "--train-input",
        type=Path,
        default=project_root
        / "data"
        / "processed"
        / "01_preprocessing"
        / "train_clean.xlsx",
        help="Clean training data from 01_preprocessing.py.",
    )
    parser.add_argument(
        "--test-input",
        type=Path,
        default=project_root
        / "data"
        / "processed"
        / "01_preprocessing"
        / "test_clean.xlsx",
        help="Clean held-out test data from 01_preprocessing.py.",
    )
    parser.add_argument(
        "--train-lag-source",
        type=Path,
        default=None,
        help=(
            "Optional row-aligned file containing EFF_TN_lag1(mg/L) and "
            "EFF_NH4_lag1(mg/L) for the cleaned training rows."
        ),
    )
    parser.add_argument(
        "--test-lag-source",
        type=Path,
        default=None,
        help=(
            "Optional row-aligned file containing EFF_TN_lag1(mg/L) and "
            "EFF_NH4_lag1(mg/L) for the held-out test rows."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "data" / "processed" / "02_process_features",
        help="Output directory.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(
        train_input=args.train_input,
        test_input=args.test_input,
        output_dir=args.output_dir,
        train_lag_source=args.train_lag_source,
        test_lag_source=args.test_lag_source,
    )
