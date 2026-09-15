#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
07_kbdc_sequential_control.py
=============================

Part 7 of the KBDC reproducibility pipeline.

This module joins the deployment-stage pieces into one sequential workflow:

    field pre-dosing inputs
        -> retained 1-day VFC + two-layer QC gate
        -> confirmed/corrected hydraulic input
        -> process-informed state
        -> PI-GBR conservative baseline D_GBR
        -> P20-P80 historical-domain plausibility check
        -> six-factor nitrogen-risk score
        -> Reduce / Retain / Protect decision
        -> daily audit log
        -> posterior weekly TN/NH4 feedback after week completion
        -> next-week alpha / beta / gamma

Important boundaries
--------------------
1. Measured TREAT_FLOW remains the primary input.
2. A VFC/QC flag does not by itself activate protective addition.
3. If a flagged flow is confirmed or corrected, the confirmed/corrected value
   is used. If reliability remains unresolved, low-risk reduction is blocked.
4. Same-day true effluent TN/NH4 are never used for the current-day dosing
   decision. They are read only by the completed-week feedback routine.
5. The P20-P80 envelope is a plausibility / review gate, not a dose calculator.
6. Weekly feedback affects the next week only; past recommendations are never
   rewritten.
7. PI-GBR boundary-relative descriptors are not clipped and may exceed 1;
   [0, 1] clipping is applied only inside the KBDC risk-scoring layer.

Expected upstream models
------------------------
models/pi_gbr_model.joblib
models/vfc_rf_1day.joblib
models/quantile_gbr_p20.joblib
models/quantile_gbr_p50.joblib
models/quantile_gbr_p80.joblib
models/quantile_gbr_preprocess.joblib

Typical use
-----------
A) Run one current week of recommendations without using posterior outcomes:
   python 07_kbdc_sequential_control.py recommend --input week1.xlsx

B) After that week is completed and true posterior EFF_TN/EFF_NH4 are known:
   python 07_kbdc_sequential_control.py close-week --week-file week1.xlsx

C) Reproduce a 28-day sequential validation in one pass:
   python 07_kbdc_sequential_control.py replay --input validation_28d.xlsx \
       --reset-state --week-size 7

Optional field-QC columns
-------------------------
FLOW_QC_STATUS:
    confirmed / resolved / pass     -> flagged measured flow was confirmed
    unresolved / pending            -> unresolved; reduction remains blocked

CORRECTED_TREAT_FLOW:
    numeric corrected flow supplied after field/operator QC

FIELD_QC_FLAG:
    optional flag for obvious inconsistency with IN_FLOW / pump / flow-meter /
    pressure information

ENVELOPE_REVIEW_REQUIRED:
    optional operator/engineering flag that the observed envelope departure is
    substantial enough to require conservative review.

ENVELOPE_REVIEW_STATUS:
    confirmed / acceptable / resolved / pass
    can release a manually flagged envelope-review hold.

The script always records whether D_GBR is inside P20-P80, but it does not
invent a numerical definition of "substantially outside" because the
manuscript/SI do not define one.

These audit columns are implementation inputs only; they do not change the
scientific risk-factor definitions.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Paths / fixed study settings
# ---------------------------------------------------------------------

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent if HERE.name.lower() == "code" else HERE

DEFAULT_TRAIN = ROOT / "data" / "processed" / "train_model_ready.csv"

DEFAULT_PI_GBR = ROOT / "models" / "pi_gbr_model.joblib"
DEFAULT_VFC = ROOT / "models" / "vfc_rf_1day.joblib"
DEFAULT_Q20 = ROOT / "models" / "quantile_gbr_p20.joblib"
DEFAULT_Q50 = ROOT / "models" / "quantile_gbr_p50.joblib"
DEFAULT_Q80 = ROOT / "models" / "quantile_gbr_p80.joblib"
DEFAULT_QPRE = ROOT / "models" / "quantile_gbr_preprocess.joblib"

DEFAULT_STATE = ROOT / "state" / "kbdc_control_state.json"
DEFAULT_DAILY_LOG = ROOT / "logs" / "kbdc_daily_control_log.csv"
DEFAULT_WEEKLY_LOG = ROOT / "logs" / "kbdc_weekly_feedback_log.csv"
DEFAULT_RESULTS = ROOT / "results" / "kbdc_control"

# PI feature definitions / fixed process constants.
FLOW_Q25 = 3287.9097
TN_LIMIT = 15.0
NH4_LIMIT = 5.0
TN_NEAR = 13.5
NH4_NEAR = 4.0

# VFC/QC gate reported in SI.
FLOW_Q01 = 737.16
FLOW_Q99 = 12270.13
VFC_DISCREPANCY_THRESHOLD = 18.96

# KBDC decision settings.
T1 = 0.65
T2 = 0.75
R_MIN = 0.05
R_MAX = 0.25
G_MIN = 0.05
G_MAX = 0.20
CONTINUITY_SCALE_D = 0.15
CONTINUITY_MIN = 0.95
CONTINUITY_MAX = 1.08

# Weekly parameter bounds.
ALPHA_MIN, ALPHA_MAX = 0.5, 2.0
BETA_MIN, BETA_MAX = 0.5, 2.0
GAMMA_MIN, GAMMA_MAX = 0.01, 0.15

EPS = 1e-8


# ---------------------------------------------------------------------
# Column harmonization
# ---------------------------------------------------------------------

ALIASES: Dict[str, Sequence[str]] = {
    "CARBON_DOS": ("CARBON_DOS", "CARBON_DOS(g/t.water)"),

    "IN_FLOW": ("IN_FLOW", "IN_FLOW(t/d)", "IN_FLOW_t(t/d)"),
    "TREAT_FLOW": ("TREAT_FLOW", "TREAT_FLOW_t(t/d)"),
    "TREAT_FLOW_lag1": (
        "TREAT_FLOW_lag1",
        "TREAT_FLOW_t(t/d)_lag1",
        "TREAT_FLOW_LAG1",
    ),

    "IN_COD": ("IN_COD", "IN_COD_t(mg/L)"),
    "IN_TN": ("IN_TN", "IN_TN_t(mg/L)"),
    "IN_NH4": ("IN_NH4", "IN_NH4_t(mg/L)"),
    "IN_TP": ("IN_TP", "IN_TP_t(mg/L)"),
    "ANO_MLSS": ("ANO_MLSS", "ANO_MLSS(mg/L)"),
    "ANO_MLVSS": ("ANO_MLVSS", "ANO_MLVSS(mg/L)"),
    "ANO_SV30": ("ANO_SV30", "ANO_SV30(%)"),

    "EFF_TN_lag1": ("EFF_TN_lag1", "EFF_TN_lag1(mg/L)", "EFF_TN_LAG1"),
    "EFF_NH4_lag1": (
        "EFF_NH4_lag1",
        "EFF_NH4_lag1(mg/L)",
        "EFF_NH4_LAG1",
    ),

    "EFF_TN": ("EFF_TN", "EFF_TN(mg/L)"),
    "EFF_NH4": ("EFF_NH4", "EFF_NH4(mg/L)"),

    "LOW_FLOW_P": ("LOW_FLOW_P", "LOW_FLOW_PRESSURE"),
    "COD/TN": ("COD/TN", "COD_TN", "COD_TN_RATIO"),
    "COD_DEFICIT": ("COD_DEFICIT",),
    "R_TN_lag1": ("R_TN_lag1", "RISK_TN_lag1"),
    "R_NH4_lag1": ("R_NH4_lag1", "RISK_NH4_lag1"),
    "R_MAX_lag1": ("R_MAX_lag1", "RISK_MAX_lag1"),
    "R_MEAN_lag1": ("R_MEAN_lag1", "RISK_MEAN_lag1"),
    "TN_PER_VSS": ("TN_PER_VSS",),

    # Audit / human-in-the-loop QC fields.
    "FLOW_QC_STATUS": (
        "FLOW_QC_STATUS",
        "flow_qc_status",
        "QC_STATUS",
    ),
    "CORRECTED_TREAT_FLOW": (
        "CORRECTED_TREAT_FLOW",
        "corrected_treat_flow",
        "TREAT_FLOW_corrected",
    ),
    "FIELD_QC_FLAG": (
        "FIELD_QC_FLAG",
        "field_qc_flag",
    ),
    "ENVELOPE_REVIEW_REQUIRED": (
        "ENVELOPE_REVIEW_REQUIRED",
        "envelope_review_required",
    ),
    "ENVELOPE_REVIEW_STATUS": (
        "ENVELOPE_REVIEW_STATUS",
        "envelope_review_status",
    ),
}

ALIAS_TO_CANONICAL = {
    alias: canonical
    for canonical, aliases in ALIASES.items()
    for alias in aliases
}


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
        raise ValueError(f"Unsupported table type: {path}")

    df.columns = df.columns.astype(str).str.strip()
    return canonicalize_columns(df)


def canonicalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = out.columns.astype(str).str.strip()

    rename = {}
    for canonical, aliases in ALIASES.items():
        if canonical in out.columns:
            continue
        hit = next((a for a in aliases if a in out.columns), None)
        if hit is not None:
            rename[hit] = canonical

    return out.rename(columns=rename)


def numeric_value(value) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return x


def as_bool(value) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return False
    s = str(value).strip().lower()
    return s in {"1", "true", "t", "yes", "y", "是", "flag", "flagged"}


def normalized_status(value) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return str(value).strip().lower()


def clip(x, lo: float, hi: float):
    return np.minimum(np.maximum(x, lo), hi)


def safe_div(num: float, den: float) -> float:
    if not np.isfinite(den) or abs(den) <= EPS:
        return 0.0
    value = num / den
    return float(value) if np.isfinite(value) else 0.0


def append_csv(row_df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    row_df.to_csv(
        path,
        mode="a" if exists else "w",
        header=not exists,
        index=False,
        encoding="utf-8-sig",
    )


def get_model_features(model) -> List[str]:
    if hasattr(model, "feature_names_in_"):
        return [str(x) for x in model.feature_names_in_]

    if hasattr(model, "named_steps"):
        if hasattr(model, "feature_names_in_"):
            return [str(x) for x in model.feature_names_in_]
        for _, step in model.named_steps.items():
            if hasattr(step, "feature_names_in_"):
                return [str(x) for x in step.feature_names_in_]

    raise AttributeError("Could not recover fitted model feature names.")


def expected_frame(row_or_df: pd.DataFrame, expected: Sequence[str]) -> pd.DataFrame:
    data = {}
    for wanted in expected:
        if wanted in row_or_df.columns:
            data[wanted] = row_or_df[wanted]
            continue

        canonical = ALIAS_TO_CANONICAL.get(wanted)
        if canonical is not None and canonical in row_or_df.columns:
            data[wanted] = row_or_df[canonical]
            continue

        raise ValueError(f"Required model feature not found: {wanted}")

    out = pd.DataFrame(data, index=row_or_df.index)
    for c in out.columns:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def predict_model(model, df: pd.DataFrame) -> np.ndarray:
    expected = get_model_features(model)
    X = expected_frame(df, expected)
    return np.asarray(model.predict(X), dtype=float).reshape(-1)


def get_underlying_tree_model(model):
    if hasattr(model, "feature_importances_"):
        return model, get_model_features(model)

    if hasattr(model, "named_steps"):
        if "model" in model.named_steps:
            step = model.named_steps["model"]
            if hasattr(step, "feature_importances_"):
                return step, get_model_features(model)

        for _, step in reversed(list(model.named_steps.items())):
            if hasattr(step, "feature_importances_"):
                return step, get_model_features(model)

    raise AttributeError(
        "PI-GBR model does not expose feature_importances_."
    )


# ---------------------------------------------------------------------
# PI descriptors
# ---------------------------------------------------------------------

def rebuild_pi_row(row: pd.Series, flow_used: float) -> pd.DataFrame:
    """
    Rebuild the dependent PI descriptors after the QC-confirmed flow is chosen.

    TN_PER_VSS is computed from ANO_MLVSS when available; otherwise an existing
    TN_PER_VSS field is retained because ANO_MLVSS is an auxiliary construction
    variable rather than an independent PI model input.
    """
    r = row.copy()

    for c in [
        "IN_COD", "IN_TN", "IN_NH4", "IN_TP",
        "ANO_MLSS", "ANO_MLVSS", "ANO_SV30",
        "EFF_TN_lag1", "EFF_NH4_lag1",
        "TN_PER_VSS",
    ]:
        if c in r.index:
            r[c] = numeric_value(r[c])

    r["TREAT_FLOW"] = float(flow_used)

    required = ["IN_COD", "IN_TN", "EFF_TN_lag1", "EFF_NH4_lag1"]
    missing = [c for c in required if c not in r.index or not np.isfinite(numeric_value(r[c]))]
    if missing:
        raise ValueError(f"Cannot construct PI state; missing/non-numeric: {missing}")

    cod = float(r["IN_COD"])
    tn = float(r["IN_TN"])
    eff_tn = float(r["EFF_TN_lag1"])
    eff_nh4 = float(r["EFF_NH4_lag1"])

    r["LOW_FLOW_P"] = max(0.0, FLOW_Q25 - float(flow_used))
    r["COD/TN"] = safe_div(cod, tn)
    r["COD_DEFICIT"] = max(0.0, 4.12 * tn - 0.555 * cod)

    # PI-GBR boundary-relative descriptors retain exceedance magnitude.
    # Do NOT clip here. [0, 1] clipping is applied only in risk_score().
    r_tn = float(eff_tn / TN_LIMIT)
    r_nh4 = float(eff_nh4 / NH4_LIMIT)
    r["R_TN_lag1"] = r_tn
    r["R_NH4_lag1"] = r_nh4
    r["R_MAX_lag1"] = max(r_tn, r_nh4)
    r["R_MEAN_lag1"] = 0.5 * (r_tn + r_nh4)

    mlvss = numeric_value(r.get("ANO_MLVSS", np.nan))
    if np.isfinite(mlvss) and abs(mlvss) > EPS:
        r["TN_PER_VSS"] = safe_div(tn, mlvss)
    elif not np.isfinite(numeric_value(r.get("TN_PER_VSS", np.nan))):
        raise ValueError(
            "TN_PER_VSS is required by the PI model but neither TN_PER_VSS "
            "nor auxiliary ANO_MLVSS is available."
        )

    return pd.DataFrame([r])


def ensure_train_pi(train: pd.DataFrame) -> pd.DataFrame:
    out = train.copy()

    if "LOW_FLOW_P" not in out.columns and "TREAT_FLOW" in out.columns:
        out["LOW_FLOW_P"] = np.maximum(
            0.0,
            FLOW_Q25 - pd.to_numeric(out["TREAT_FLOW"], errors="coerce"),
        )

    if "COD_DEFICIT" not in out.columns:
        out["COD_DEFICIT"] = np.maximum(
            0.0,
            4.12 * pd.to_numeric(out["IN_TN"], errors="coerce")
            - 0.555 * pd.to_numeric(out["IN_COD"], errors="coerce"),
        )

    return out


# ---------------------------------------------------------------------
# Risk normalization and fused weights
# ---------------------------------------------------------------------

def training_risk_norm(train_df: pd.DataFrame) -> Dict[str, float]:
    train = ensure_train_pi(train_df)

    required = ["IN_TN", "COD_DEFICIT", "LOW_FLOW_P"]
    missing = [c for c in required if c not in train.columns]
    if missing:
        raise ValueError(
            f"Training table missing risk-normalization variables: {missing}"
        )

    for c in required:
        train[c] = pd.to_numeric(train[c], errors="coerce")

    return {
        "q25_in_tn": float(train["IN_TN"].quantile(0.25)),
        "q90_in_tn": float(train["IN_TN"].quantile(0.90)),
        "q90_cd": float(max(train["COD_DEFICIT"].quantile(0.90), EPS)),
        "q90_lf": float(max(train["LOW_FLOW_P"].quantile(0.90), EPS)),
    }


RISK_DOMAIN_WEIGHTS = {
    "EFF_TN_lag1": 0.28,
    "EFF_NH4_lag1": 0.07,
    "R_MAX_lag1": 0.19,
    "IN_TN": 0.17,
    "COD_DEFICIT": 0.15,
    "LOW_FLOW_P": 0.14,
}


def build_risk_weights(pi_model, blend_ratio: float = 0.15) -> Dict[str, float]:
    tree_model, feature_names = get_underlying_tree_model(pi_model)
    importances = np.asarray(tree_model.feature_importances_, dtype=float)

    if len(importances) != len(feature_names):
        raise ValueError(
            "Feature-importance length does not match fitted feature-name length."
        )

    importance_by_canonical: Dict[str, float] = {}
    for name, value in zip(feature_names, importances):
        canonical = ALIAS_TO_CANONICAL.get(str(name), str(name))
        importance_by_canonical[canonical] = float(value)

    imp_sum = sum(
        importance_by_canonical.get(f, 0.0)
        for f in RISK_DOMAIN_WEIGHTS
    )
    if imp_sum <= 0:
        imp_sum = 1.0

    weights = {}
    for f, domain_w in RISK_DOMAIN_WEIGHTS.items():
        model_share = importance_by_canonical.get(f, 0.0) / imp_sum
        weights[f] = (
            (1.0 - blend_ratio) * domain_w
            + blend_ratio * model_share
        )

    total = sum(weights.values()) or 1.0
    return {k: float(v / total) for k, v in weights.items()}


def risk_score(
    pi_row: pd.Series,
    norm: Mapping[str, float],
    weights: Mapping[str, float],
    beta: float,
) -> Tuple[Dict[str, float], float, float]:
    eff_tn = float(clip(float(pi_row["EFF_TN_lag1"]) / TN_LIMIT, 0, 1))
    eff_nh4 = float(clip(float(pi_row["EFF_NH4_lag1"]) / NH4_LIMIT, 0, 1))
    risk_max = float(clip(float(pi_row["R_MAX_lag1"]), 0, 1))

    q25 = float(norm["q25_in_tn"])
    q90 = float(norm["q90_in_tn"])
    if q90 <= q25:
        in_tn_norm = 0.0
    else:
        in_tn_norm = float(
            clip((float(pi_row["IN_TN"]) - q25) / (q90 - q25), 0, 1)
        )

    cd_norm = float(
        clip(float(pi_row["COD_DEFICIT"]) / float(norm["q90_cd"]), 0, 1)
    )
    lf_norm = float(
        clip(float(pi_row["LOW_FLOW_P"]) / float(norm["q90_lf"]), 0, 1)
    )

    factors = {
        "EFF_TN_lag1": eff_tn,
        "EFF_NH4_lag1": eff_nh4,
        "R_MAX_lag1": risk_max,
        "IN_TN": in_tn_norm,
        "COD_DEFICIT": cd_norm,
        "LOW_FLOW_P": lf_norm,
    }

    risk = float(
        clip(
            sum(weights[k] * factors[k] for k in weights),
            0.0,
            1.0,
        )
    )
    adjusted = float(clip(beta * risk, 0.0, 1.0))
    return factors, risk, adjusted


# ---------------------------------------------------------------------
# State
# ---------------------------------------------------------------------

def initial_state() -> Dict[str, Any]:
    return {
        "week_number": 1,
        "alpha": 1.0,
        "beta": 1.0,
        "gamma": 0.06,
        "previous_valid_risk": None,
        "previous_flow_for_vfc": None,
        "previous_week_near_ratio": None,
        "previous_week_tn_std": None,
        "previous_week_nh4_std": None,
        "completed_days": 0,
    }


def load_state(path: Path, reset: bool = False) -> Dict[str, Any]:
    if reset or not path.exists():
        return initial_state()

    state = json.loads(path.read_text(encoding="utf-8"))
    base = initial_state()
    base.update(state)
    return base


def save_state(path: Path, state: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(state), ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------
# VFC/QC gate
# ---------------------------------------------------------------------

RESOLVED_MEASURED = {"confirmed", "resolved", "pass", "measured_confirmed"}
RESOLVED_ENVELOPE = {"confirmed", "acceptable", "resolved", "pass"}


def prepare_vfc_row(row: pd.Series, state: Mapping[str, Any], vfc_model) -> pd.DataFrame:
    r = row.copy()

    if "TREAT_FLOW_lag1" not in r.index or not np.isfinite(
        numeric_value(r.get("TREAT_FLOW_lag1", np.nan))
    ):
        prev = numeric_value(state.get("previous_flow_for_vfc"))
        if np.isfinite(prev):
            r["TREAT_FLOW_lag1"] = prev

    expected = get_model_features(vfc_model)
    frame = pd.DataFrame([r])
    X = expected_frame(frame, expected)

    if X.isna().any().any():
        bad = X.columns[X.isna().any()].tolist()
        raise ValueError(
            "VFC input contains missing values. The retained 1-day VFC needs "
            f"the following unresolved fields: {bad}"
        )

    return X


def vfc_qc_gate(
    row: pd.Series,
    state: Mapping[str, Any],
    vfc_model,
) -> Dict[str, Any]:
    X_vfc = prepare_vfc_row(row, state, vfc_model)
    vfc_estimate = float(vfc_model.predict(X_vfc)[0])

    measured = numeric_value(row.get("TREAT_FLOW", np.nan))
    corrected = numeric_value(row.get("CORRECTED_TREAT_FLOW", np.nan))
    qc_status = normalized_status(row.get("FLOW_QC_STATUS", ""))
    field_qc_flag = as_bool(row.get("FIELD_QC_FLAG", False))

    missing = not np.isfinite(measured)
    out_of_range = (
        np.isfinite(measured)
        and (measured < FLOW_Q01 or measured > FLOW_Q99)
    )
    first_layer = bool(missing or out_of_range)

    if np.isfinite(measured) and not first_layer:
        discrepancy = (
            abs(vfc_estimate - measured)
            / max(abs(measured), EPS)
            * 100.0
        )
        second_layer = discrepancy > VFC_DISCREPANCY_THRESHOLD
    else:
        discrepancy = float("nan")
        second_layer = False

    triggered = bool(first_layer or second_layer or field_qc_flag)

    if not triggered:
        flow_used = measured
        source = "measured"
        resolved = True
    elif np.isfinite(corrected):
        flow_used = corrected
        source = "field_corrected"
        resolved = True
    elif qc_status in RESOLVED_MEASURED and np.isfinite(measured):
        flow_used = measured
        source = "measured_confirmed"
        resolved = True
    else:
        # Condition-triggered VFC fallback keeps the calculation executable,
        # but does not count as field confirmation and cannot release reduction.
        flow_used = vfc_estimate
        source = "VFC_pending_QC"
        resolved = False

    return {
        "TREAT_FLOW_measured": measured,
        "TREAT_FLOW_vfc": vfc_estimate,
        "first_layer_flag": first_layer,
        "second_layer_flag": bool(second_layer),
        "field_qc_flag": field_qc_flag,
        "same_day_vfc_discrepancy_percent": discrepancy,
        "qc_triggered": triggered,
        "qc_resolved": resolved,
        "flow_source_used": source,
        "TREAT_FLOW_used": float(flow_used),
        "reduction_allowed_by_qc": bool(resolved),
    }


# ---------------------------------------------------------------------
# Quantile envelope
# ---------------------------------------------------------------------

def load_quantile_bundle(
    p20_path: Path,
    p50_path: Path,
    p80_path: Path,
    preprocess_path: Path,
):
    models = {
        "p20": joblib.load(p20_path),
        "p50": joblib.load(p50_path),
        "p80": joblib.load(p80_path),
    }
    prep = joblib.load(preprocess_path)

    feature_names = [str(x) for x in prep["feature_names"]]
    medians = {str(k): float(v) for k, v in prep["train_median"].items()}
    return models, feature_names, medians


def quantile_predict_row(
    pi_df: pd.DataFrame,
    models,
    feature_names: Sequence[str],
    medians: Mapping[str, float],
) -> Tuple[float, float, float]:
    X = expected_frame(pi_df, feature_names)

    for c in X.columns:
        if X[c].isna().any():
            if c not in medians:
                raise ValueError(f"No train median stored for quantile feature: {c}")
            X[c] = X[c].fillna(medians[c])

    vals = [
        float(models["p20"].predict(X)[0]),
        float(models["p50"].predict(X)[0]),
        float(models["p80"].predict(X)[0]),
    ]
    vals.sort()
    return vals[0], vals[1], vals[2]


def envelope_gate(
    d_gbr: float,
    p20: float,
    p80: float,
    row: pd.Series,
) -> Dict[str, Any]:
    """
    P20-P80 is recorded automatically. The source materials do not define a
    numerical threshold for "substantially outside", so the decision to require
    conservative review is supplied as an auditable operator/engineering flag.
    """
    inside = bool(p20 <= d_gbr <= p80)
    outside = not inside

    review_required = as_bool(
        row.get("ENVELOPE_REVIEW_REQUIRED", False)
    )
    status = normalized_status(row.get("ENVELOPE_REVIEW_STATUS", ""))
    confirmed = status in RESOLVED_ENVELOPE

    reduction_allowed = (not review_required) or confirmed

    return {
        "P20": float(p20),
        "P80": float(p80),
        "D_GBR_inside_P20_P80": inside,
        "envelope_outside_P20_P80": outside,
        "envelope_review_required": bool(review_required),
        "envelope_review_confirmed": bool(confirmed),
        "reduction_allowed_by_envelope": bool(reduction_allowed),
    }


# ---------------------------------------------------------------------
# Daily KBDC recommendation
# ---------------------------------------------------------------------

def recommend_one_day(
    row: pd.Series,
    state: Dict[str, Any],
    pi_model,
    vfc_model,
    quantile_models,
    quantile_features: Sequence[str],
    quantile_medians: Mapping[str, float],
    norm_params: Mapping[str, float],
    risk_weights: Mapping[str, float],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    qc = vfc_qc_gate(row, state, vfc_model)

    pi_df = rebuild_pi_row(row, qc["TREAT_FLOW_used"])
    pi_row = pi_df.iloc[0]

    d_gbr = float(predict_model(pi_model, pi_df)[0])
    p20, p50, p80 = quantile_predict_row(
        pi_df,
        quantile_models,
        quantile_features,
        quantile_medians,
    )
    env = envelope_gate(d_gbr, p20, p80, row)

    alpha = float(state["alpha"])
    beta = float(state["beta"])
    gamma = float(state["gamma"])

    factors, risk, adjusted = risk_score(
        pi_row,
        norm_params,
        risk_weights,
        beta,
    )

    # Low-risk reduction candidate.
    base_reduction = alpha * (
        R_MIN + (R_MAX - R_MIN) * (1.0 - adjusted)
    )

    prev_risk = numeric_value(state.get("previous_valid_risk"))
    if np.isfinite(prev_risk):
        delta_r = float(
            clip((prev_risk - risk) / CONTINUITY_SCALE_D, -1.0, 1.0)
        )
    else:
        delta_r = 0.0

    continuity_factor = float(
        clip(1.0 + gamma * delta_r, CONTINUITY_MIN, CONTINUITY_MAX)
    )
    reduction_candidate = float(
        clip(base_reduction * continuity_factor, 0.0, R_MAX)
    )

    low_risk = adjusted < T1
    high_risk = adjusted >= T2

    reduction_allowed = (
        qc["reduction_allowed_by_qc"]
        and env["reduction_allowed_by_envelope"]
    )

    reduction_pct = 0.0
    addition_pct = 0.0

    if low_risk:
        if reduction_allowed:
            reduction_pct = reduction_candidate
            action = "Reduce"
        elif not qc["reduction_allowed_by_qc"]:
            action = "QC hold"
        else:
            action = "Retain - envelope review"

    elif high_risk:
        addition_pct = float(
            clip(
                G_MIN
                + (G_MAX - G_MIN)
                * ((adjusted - T2) / (1.0 - T2)),
                G_MIN,
                G_MAX,
            )
        )
        action = "Protect"

    else:
        action = "Retain"

    d_kbdc = float(d_gbr * (1.0 + addition_pct - reduction_pct))

    result = {
        "week_number": int(state["week_number"]),
        "day_sequence": int(state.get("completed_days", 0)) + 1,

        **qc,
        **env,

        "D_GBR": d_gbr,
        "P50": float(p50),

        "alpha": alpha,
        "beta": beta,
        "gamma": gamma,

        "risk_EFF_TN_lag1": factors["EFF_TN_lag1"],
        "risk_EFF_NH4_lag1": factors["EFF_NH4_lag1"],
        "risk_R_MAX_lag1": factors["R_MAX_lag1"],
        "risk_IN_TN": factors["IN_TN"],
        "risk_COD_DEFICIT": factors["COD_DEFICIT"],
        "risk_LOW_FLOW_P": factors["LOW_FLOW_P"],

        "R_t": risk,
        "R_adj_t": adjusted,

        "base_reduction_ratio": float(base_reduction),
        "continuity_delta": delta_r,
        "continuity_factor": continuity_factor,

        "reduction_ratio": reduction_pct,
        "addition_ratio": addition_pct,
        "action": action,
        "D_KBDC": d_kbdc,

        "reduction_allowed": bool(reduction_allowed),
    }

    # Preserve actual field dose as an audit field only; it is never used to
    # calculate today's recommendation.
    actual = numeric_value(row.get("CARBON_DOS", np.nan))
    if np.isfinite(actual):
        result["CARBON_DOS_observed"] = actual

    new_state = dict(state)
    new_state["completed_days"] = int(state.get("completed_days", 0)) + 1
    new_state["previous_flow_for_vfc"] = float(qc["TREAT_FLOW_used"])

    # Continuity uses the previous valid daily risk. Unresolved hydraulic QC
    # does not overwrite that valid-risk memory.
    if qc["qc_resolved"]:
        new_state["previous_valid_risk"] = float(risk)

    return result, new_state


# ---------------------------------------------------------------------
# Weekly posterior feedback
# ---------------------------------------------------------------------

def posterior_week_stats(
    week_df: pd.DataFrame,
    previous_near_ratio: Optional[float],
    previous_tn_std: Optional[float],
    previous_nh4_std: Optional[float],
) -> Dict[str, Any]:
    if "EFF_TN" not in week_df.columns or "EFF_NH4" not in week_df.columns:
        raise ValueError(
            "Completed-week feedback requires true posterior EFF_TN and EFF_NH4."
        )

    tn = pd.to_numeric(week_df["EFF_TN"], errors="coerce")
    nh4 = pd.to_numeric(week_df["EFF_NH4"], errors="coerce")
    valid = tn.notna() & nh4.notna()

    tn = tn[valid]
    nh4 = nh4[valid]
    if len(tn) == 0:
        raise ValueError("No valid true posterior TN/NH4 values in completed week.")

    near = (tn > TN_NEAR) | (nh4 > NH4_NEAR)
    exceed = (tn > TN_LIMIT) | (nh4 > NH4_LIMIT)

    tn_std = float(tn.std(ddof=1)) if len(tn) > 1 else 0.0
    nh4_std = float(nh4.std(ddof=1)) if len(nh4) > 1 else 0.0
    near_ratio = float(near.mean())

    near_up = (
        False
        if previous_near_ratio is None
        else near_ratio > float(previous_near_ratio)
    )

    # Audit only: does not enter A/B/C classification.
    if previous_tn_std is None:
        tn_vol_up = False
    elif float(previous_tn_std) == 0:
        tn_vol_up = tn_std > 0
    else:
        tn_vol_up = tn_std > float(previous_tn_std) * 1.20

    if previous_nh4_std is None:
        nh4_vol_up = False
    elif float(previous_nh4_std) == 0:
        nh4_vol_up = nh4_std > 0
    else:
        nh4_vol_up = nh4_std > float(previous_nh4_std) * 1.20

    return {
        "sample_count": int(len(tn)),
        "mean_tn": float(tn.mean()),
        "mean_nh4": float(nh4.mean()),
        "max_tn": float(tn.max()),
        "max_nh4": float(nh4.max()),
        "tn_std": tn_std,
        "nh4_std": nh4_std,
        "near_threshold_count": int(near.sum()),
        "near_threshold_ratio": near_ratio,
        "near_threshold_freq_up": bool(near_up),
        "volatility_up": bool(tn_vol_up or nh4_vol_up),
        "exceed_count": int(exceed.sum()),
        "exceed_ratio": float(exceed.mean()),
        "exceed": bool(exceed.any()),
    }


def weekly_update(
    state: Dict[str, Any],
    stats: Mapping[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    alpha_before = float(state["alpha"])
    beta_before = float(state["beta"])
    gamma_before = float(state["gamma"])

    mean_tn = float(stats["mean_tn"])
    mean_nh4 = float(stats["mean_nh4"])
    near_up = bool(stats["near_threshold_freq_up"])
    exceed = bool(stats["exceed"])

    if exceed or mean_tn > 10.0 or mean_nh4 > 3.5:
        scenario = "C_risk_rising"
        a_mult, b_mult, g_mult = 0.80, 1.25, 0.85
    elif (
        mean_tn <= 8.5
        and mean_nh4 <= 3.0
        and not near_up
        and not exceed
    ):
        scenario = "A_stable_with_margin"
        a_mult, b_mult, g_mult = 1.15, 0.90, 1.05
    else:
        scenario = "B_near_boundary"
        a_mult, b_mult, g_mult = 0.90, 1.15, 1.00

    alpha_after = float(
        clip(alpha_before * a_mult, ALPHA_MIN, ALPHA_MAX)
    )
    beta_after = float(
        clip(beta_before * b_mult, BETA_MIN, BETA_MAX)
    )
    gamma_after = float(
        clip(gamma_before * g_mult, GAMMA_MIN, GAMMA_MAX)
    )

    updated = dict(state)
    updated["alpha"] = alpha_after
    updated["beta"] = beta_after
    updated["gamma"] = gamma_after
    updated["week_number"] = int(state["week_number"]) + 1

    updated["previous_week_near_ratio"] = float(
        stats["near_threshold_ratio"]
    )
    updated["previous_week_tn_std"] = float(stats["tn_std"])
    updated["previous_week_nh4_std"] = float(stats["nh4_std"])

    audit = {
        "completed_week": int(state["week_number"]),
        "scenario": scenario,
        **dict(stats),

        "alpha_before": alpha_before,
        "beta_before": beta_before,
        "gamma_before": gamma_before,

        "alpha_multiplier": a_mult,
        "beta_multiplier": b_mult,
        "gamma_multiplier": g_mult,

        "alpha_next_week": alpha_after,
        "beta_next_week": beta_after,
        "gamma_next_week": gamma_after,
    }
    return updated, audit


def close_week_from_df(
    week_df: pd.DataFrame,
    state: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    stats = posterior_week_stats(
        week_df,
        state.get("previous_week_near_ratio"),
        state.get("previous_week_tn_std"),
        state.get("previous_week_nh4_std"),
    )
    return weekly_update(state, stats)


# ---------------------------------------------------------------------
# Bundle loading
# ---------------------------------------------------------------------

def require_file(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")


def load_runtime(args):
    require_file(args.pi_gbr_model, "PI-GBR model")
    require_file(args.vfc_model, "1-day VFC model")
    require_file(args.q20_model, "Quantile-GBR P20 model")
    require_file(args.q50_model, "Quantile-GBR P50 model")
    require_file(args.q80_model, "Quantile-GBR P80 model")
    require_file(args.qpreprocess, "Quantile preprocessing state")
    require_file(args.train, "Training model-ready table")

    pi_model = joblib.load(args.pi_gbr_model)
    vfc_model = joblib.load(args.vfc_model)
    qmodels, qfeatures, qmedians = load_quantile_bundle(
        args.q20_model,
        args.q50_model,
        args.q80_model,
        args.qpreprocess,
    )

    train_df = read_table(args.train)
    norm = training_risk_norm(train_df)
    weights = build_risk_weights(pi_model, blend_ratio=0.15)

    return {
        "pi_model": pi_model,
        "vfc_model": vfc_model,
        "qmodels": qmodels,
        "qfeatures": qfeatures,
        "qmedians": qmedians,
        "norm": norm,
        "weights": weights,
    }


# ---------------------------------------------------------------------
# Command implementations
# ---------------------------------------------------------------------

def run_recommend(args) -> None:
    runtime = load_runtime(args)
    state = load_state(args.state_file, reset=args.reset_state)

    df = read_table(args.input)
    records = []

    for _, row in df.iterrows():
        result, state = recommend_one_day(
            row=row,
            state=state,
            pi_model=runtime["pi_model"],
            vfc_model=runtime["vfc_model"],
            quantile_models=runtime["qmodels"],
            quantile_features=runtime["qfeatures"],
            quantile_medians=runtime["qmedians"],
            norm_params=runtime["norm"],
            risk_weights=runtime["weights"],
        )
        records.append(result)

    out = pd.DataFrame(records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False, encoding="utf-8-sig")
    append_csv(out, args.daily_log)

    save_state(args.state_file, state)

    summary = {
        "rows_processed": len(out),
        "current_week_number": state["week_number"],
        "current_alpha": state["alpha"],
        "current_beta": state["beta"],
        "current_gamma": state["gamma"],
        "state_file": str(args.state_file),
        "daily_log": str(args.daily_log),
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("Recommendations written:", args.output)
    print("State written:", args.state_file)


def run_close_week(args) -> None:
    state = load_state(args.state_file, reset=False)
    week_df = read_table(args.week_file)

    updated, audit = close_week_from_df(week_df, state)
    save_state(args.state_file, updated)

    audit_df = pd.DataFrame([audit])
    append_csv(audit_df, args.weekly_log)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("Completed-week feedback:", audit["scenario"])
    print(
        "Next-week alpha/beta/gamma:",
        audit["alpha_next_week"],
        audit["beta_next_week"],
        audit["gamma_next_week"],
    )
    print("State written:", args.state_file)


def run_replay(args) -> None:
    runtime = load_runtime(args)
    state = load_state(args.state_file, reset=args.reset_state)

    df = read_table(args.input)
    all_daily = []
    weekly_audits = []

    for i, (_, row) in enumerate(df.iterrows(), start=1):
        result, state = recommend_one_day(
            row=row,
            state=state,
            pi_model=runtime["pi_model"],
            vfc_model=runtime["vfc_model"],
            quantile_models=runtime["qmodels"],
            quantile_features=runtime["qfeatures"],
            quantile_medians=runtime["qmedians"],
            norm_params=runtime["norm"],
            risk_weights=runtime["weights"],
        )
        all_daily.append(result)

        # Posterior data are read only after this entire week has been completed.
        if i % args.week_size == 0:
            week_slice = df.iloc[i - args.week_size:i].copy()
            if "EFF_TN" in week_slice.columns and "EFF_NH4" in week_slice.columns:
                state, audit = close_week_from_df(week_slice, state)
                weekly_audits.append(audit)

    daily_df = pd.DataFrame(all_daily)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    daily_df.to_csv(args.output, index=False, encoding="utf-8-sig")
    append_csv(daily_df, args.daily_log)

    if weekly_audits:
        weekly_df = pd.DataFrame(weekly_audits)
        append_csv(weekly_df, args.weekly_log)
        args.weekly_output.parent.mkdir(parents=True, exist_ok=True)
        weekly_df.to_csv(
            args.weekly_output,
            index=False,
            encoding="utf-8-sig",
        )

    save_state(args.state_file, state)

    summary = {
        "days_processed": len(daily_df),
        "weeks_closed": len(weekly_audits),
        "final_week_number": state["week_number"],
        "final_alpha": state["alpha"],
        "final_beta": state["beta"],
        "final_gamma": state["gamma"],
        "state_file": str(args.state_file),
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("Replay daily output:", args.output)
    if weekly_audits:
        print("Replay weekly output:", args.weekly_output)
    print("Final state:", args.state_file)


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def add_runtime_args(p):
    p.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    p.add_argument("--pi-gbr-model", type=Path, default=DEFAULT_PI_GBR)
    p.add_argument("--vfc-model", type=Path, default=DEFAULT_VFC)
    p.add_argument("--q20-model", type=Path, default=DEFAULT_Q20)
    p.add_argument("--q50-model", type=Path, default=DEFAULT_Q50)
    p.add_argument("--q80-model", type=Path, default=DEFAULT_Q80)
    p.add_argument("--qpreprocess", type=Path, default=DEFAULT_QPRE)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Sequential VFC/QC-gated KBDC control and weekly feedback."
    )
    sub = p.add_subparsers(dest="command", required=True)

    # Daily / current-week recommendation mode.
    r = sub.add_parser(
        "recommend",
        help="Generate daily KBDC recommendations without using same-day posterior outcomes.",
    )
    add_runtime_args(r)
    r.add_argument("--input", type=Path, required=True)
    r.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_RESULTS / "latest_recommendations.csv",
    )
    r.add_argument(
        "--summary",
        type=Path,
        default=DEFAULT_RESULTS / "latest_recommendation_summary.json",
    )
    r.add_argument("--state-file", type=Path, default=DEFAULT_STATE)
    r.add_argument("--daily-log", type=Path, default=DEFAULT_DAILY_LOG)
    r.add_argument("--reset-state", action="store_true")
    r.set_defaults(func=run_recommend)

    # Completed-week posterior feedback.
    w = sub.add_parser(
        "close-week",
        help="Use true completed-week posterior TN/NH4 to update next-week alpha/beta/gamma.",
    )
    w.add_argument("--week-file", type=Path, required=True)
    w.add_argument("--state-file", type=Path, default=DEFAULT_STATE)
    w.add_argument("--weekly-log", type=Path, default=DEFAULT_WEEKLY_LOG)
    w.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_RESULTS / "latest_weekly_feedback.json",
    )
    w.set_defaults(func=run_close_week)

    # Sequential 28-day / multiweek replay.
    q = sub.add_parser(
        "replay",
        help="Sequentially replay a multiweek validation file; weekly posterior feedback is applied only after each completed week.",
    )
    add_runtime_args(q)
    q.add_argument("--input", type=Path, required=True)
    q.add_argument("--week-size", type=int, default=7)
    q.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_RESULTS / "replay_daily_recommendations.csv",
    )
    q.add_argument(
        "--weekly-output",
        type=Path,
        default=DEFAULT_RESULTS / "replay_weekly_feedback.csv",
    )
    q.add_argument(
        "--summary",
        type=Path,
        default=DEFAULT_RESULTS / "replay_summary.json",
    )
    q.add_argument("--state-file", type=Path, default=DEFAULT_STATE)
    q.add_argument("--daily-log", type=Path, default=DEFAULT_DAILY_LOG)
    q.add_argument("--weekly-log", type=Path, default=DEFAULT_WEEKLY_LOG)
    q.add_argument("--reset-state", action="store_true")
    q.set_defaults(func=run_replay)

    return p


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
