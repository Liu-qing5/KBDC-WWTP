#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
05_robustness_vfc.py

KBDC Part 5: input robustness and virtual-flow reliability.

Covers the analyses reported in main-text Section 3.3 / Fig. 6 and SI Text S6:
1) grouped single-chain zeroing robustness;
2) global random bidirectional multivariable perturbation;
3) 1-, 2-, and 3-day RF virtual-flow models;
4) isolated TREAT_FLOW zeroing with/without VFC;
5) positive flow-amplitude crossover and activation-threshold surface;
6) first- and second-layer VFC/QC flags.

PI descriptors rebuilt after perturbation use the same definition as the
training PI-GBR features: EFF_TN_lag1 / 15 and EFF_NH4_lag1 / 5 are not
clipped, so values above 1 are retained.

Run after 03_model_benchmark.py so models/pi_gbr_model.joblib is available.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ---------------------------------------------------------------------
# Paths / fixed study settings
# ---------------------------------------------------------------------

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent if HERE.name.lower() == "code" else HERE

DEFAULT_TRAIN = ROOT / "data" / "processed" / "train_model_ready.csv"
DEFAULT_TEST = ROOT / "data" / "processed" / "test_model_ready.csv"
DEFAULT_GBR = ROOT / "models" / "pi_gbr_model.joblib"
DEFAULT_MODELS = ROOT / "models"
DEFAULT_OUT = ROOT / "results" / "robustness_vfc"

TARGET = "CARBON_DOS"
FLOW_Q25 = 3287.9097
TN_LIMIT = 15.0
NH4_LIMIT = 5.0
QC_THRESHOLD = 18.96
SEED = 42
EPS = 1e-8

GROUP_ZERO_RATES = [0, .05, .10, .15, .20, .25]

FIXED_AMPLITUDES = [.05, .10, .15, .20, .25, .30]
RATIO_SWEEP = [0, .05, .10, .15, .20, .25, .30, .35, .40, .45, .50]
FIXED_RATIOS = [.05, .10, .15, .20, .25, .30]
AMPLITUDE_SWEEP = [0, .05, .10, .15, .20, .25, .30, .35, .40, .45, .50]

POS_AMP = np.round(np.arange(0, 50.0001, .5), 2)
TAU_SWEEP = np.round(np.arange(0, 50.0001, .5), 2)


# ---------------------------------------------------------------------
# Public names + legacy spreadsheet aliases
# ---------------------------------------------------------------------

ALIASES = {
    TARGET: [TARGET, "CARBON_DOS(g/t.water)"],
    "IN_FLOW": ["IN_FLOW", "IN_FLOW(t/d)", "IN_FLOW_t(t/d)"],
    "TREAT_FLOW": ["TREAT_FLOW", "TREAT_FLOW_t(t/d)"],
    "IN_COD": ["IN_COD", "IN_COD_t(mg/L)"],
    "IN_TN": ["IN_TN", "IN_TN_t(mg/L)"],
    "IN_NH4": ["IN_NH4", "IN_NH4_t(mg/L)"],
    "IN_TP": ["IN_TP", "IN_TP_t(mg/L)"],
    "ANO_MLSS": ["ANO_MLSS", "ANO_MLSS(mg/L)"],
    "ANO_MLVSS": ["ANO_MLVSS", "ANO_MLVSS(mg/L)"],
    "ANO_SV30": ["ANO_SV30", "ANO_SV30(%)"],
    "EFF_TN_lag1": ["EFF_TN_lag1", "EFF_TN_lag1(mg/L)", "EFF_TN_LAG1"],
    "EFF_NH4_lag1": ["EFF_NH4_lag1", "EFF_NH4_lag1(mg/L)", "EFF_NH4_LAG1"],
    "LOW_FLOW_P": ["LOW_FLOW_P", "LOW_FLOW_PRESSURE"],
    "COD/TN": ["COD/TN", "COD_TN", "COD_TN_RATIO"],
    "COD_DEFICIT": ["COD_DEFICIT"],
    "R_TN_lag1": ["R_TN_lag1", "RISK_TN_lag1"],
    "R_NH4_lag1": ["R_NH4_lag1", "RISK_NH4_lag1"],
    "R_MAX_lag1": ["R_MAX_lag1", "RISK_MAX_lag1"],
    "R_MEAN_lag1": ["R_MEAN_lag1", "RISK_MEAN_lag1"],
    "TN_PER_VSS": ["TN_PER_VSS"],
    "TREAT_FLOW_lag1": ["TREAT_FLOW_lag1", "TREAT_FLOW_t(t/d)_lag1"],
    "TREAT_FLOW_lag2": ["TREAT_FLOW_lag2", "TREAT_FLOW_t(t/d)_lag2"],
    "TREAT_FLOW_lag3": ["TREAT_FLOW_lag3", "TREAT_FLOW_t(t/d)_lag3"],
}
ALIAS_TO_CANON = {a: c for c, aa in ALIASES.items() for a in aa}

PI_FEATURES = [
    "TREAT_FLOW", "IN_COD", "IN_TN", "IN_NH4", "IN_TP",
    "ANO_MLSS", "ANO_SV30",
    "EFF_TN_lag1", "EFF_NH4_lag1",
    "LOW_FLOW_P", "COD/TN", "COD_DEFICIT",
    "R_TN_lag1", "R_NH4_lag1", "R_MAX_lag1", "R_MEAN_lag1",
    "TN_PER_VSS",
]

GROUP_DRIVERS = {
    "FLOW": "TREAT_FLOW",
    "C/N": "IN_COD",
    "IN-N": "IN_TN",
    "LAG-TN": "EFF_TN_lag1",
    "LAG-NH4": "EFF_NH4_lag1",
}

GLOBAL_RAW = [
    "TREAT_FLOW", "IN_COD", "IN_TN", "IN_NH4", "IN_TP",
    "EFF_TN_lag1", "EFF_NH4_lag1", "ANO_MLSS", "ANO_SV30",
]

VFC_CONFIG = {
    1: dict(n_estimators=30, max_depth=8, min_samples_split=20,
            min_samples_leaf=8, max_features=.5, bootstrap=True),
    2: dict(n_estimators=50, max_depth=8, min_samples_split=5,
            min_samples_leaf=2, max_features=.5, bootstrap=True),
    3: dict(n_estimators=50, max_depth=8, min_samples_split=5,
            min_samples_leaf=2, max_features=.5, bootstrap=True),
}


# ---------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------

def read_table(path: Path) -> pd.DataFrame:
    s = path.suffix.lower()
    if s in {".csv", ".txt"}:
        return pd.read_csv(path)
    if s in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if s in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported file type: {path}")


def canonicalize(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = out.columns.astype(str).str.strip()
    ren = {}
    for canon, names in ALIASES.items():
        if canon in out.columns:
            continue
        hit = next((x for x in names if x in out.columns), None)
        if hit is not None:
            ren[hit] = canon
    return out.rename(columns=ren)


def require(df: pd.DataFrame, cols: Sequence[str], label: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{label} missing columns: {missing}")


def numeric(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def calc_metrics(y, pred) -> Dict[str, float]:
    y = np.asarray(y, float)
    pred = np.asarray(pred, float)
    return {
        "R2": float(r2_score(y, pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y, pred))),
        "MAE": float(mean_absolute_error(y, pred)),
        "MAPE(%)": float(np.mean(np.abs((y - pred) /
                         np.maximum(np.abs(y), EPS))) * 100),
    }


def model_features(model) -> List[str]:
    if hasattr(model, "feature_names_in_"):
        return [str(x) for x in model.feature_names_in_]
    if hasattr(model, "named_steps"):
        for _, step in model.named_steps.items():
            if hasattr(step, "feature_names_in_"):
                return [str(x) for x in step.feature_names_in_]
    raise AttributeError("Cannot identify model feature names.")


def frame_for_model(df: pd.DataFrame, model) -> pd.DataFrame:
    out = {}
    for wanted in model_features(model):
        if wanted in df.columns:
            out[wanted] = df[wanted]
        else:
            canon = ALIAS_TO_CANON.get(wanted)
            if canon is None or canon not in df.columns:
                raise ValueError(f"Model feature not found: {wanted}")
            out[wanted] = df[canon]
    return pd.DataFrame(out, index=df.index)


def predict_gbr(model, df: pd.DataFrame) -> np.ndarray:
    return np.asarray(model.predict(frame_for_model(df, model)), float)


def safe_div(num, den) -> np.ndarray:
    num, den = np.asarray(num, float), np.asarray(den, float)
    out = np.zeros_like(num)
    ok = np.isfinite(den) & (np.abs(den) > EPS)
    out[ok] = num[ok] / den[ok]
    out[~np.isfinite(out)] = 0
    return out


# ---------------------------------------------------------------------
# PI descriptor reconstruction after raw-input perturbation
# ---------------------------------------------------------------------

def auxiliary_vss(reference: pd.DataFrame) -> np.ndarray:
    if "ANO_MLVSS" in reference.columns:
        a = pd.to_numeric(reference["ANO_MLVSS"], errors="coerce").to_numpy(float)
        good = np.isfinite(a) & (a > EPS)
        if good.any():
            fill = float(np.nanmedian(a[good]))
            return np.where(good, a, fill)

    require(reference, ["IN_TN", "TN_PER_VSS"], "reference")
    tn = pd.to_numeric(reference["IN_TN"], errors="coerce").to_numpy(float)
    ratio = pd.to_numeric(reference["TN_PER_VSS"], errors="coerce").to_numpy(float)
    a = safe_div(tn, ratio)
    good = np.isfinite(a) & (a > EPS)
    fill = float(np.nanmedian(a[good])) if good.any() else 1.0
    return np.where(good, a, fill)


def rebuild_pi(df: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    out = numeric(df, [
        "TREAT_FLOW", "IN_COD", "IN_TN", "EFF_TN_lag1", "EFF_NH4_lag1"
    ])
    require(out, ["TREAT_FLOW", "IN_COD", "IN_TN",
                  "EFF_TN_lag1", "EFF_NH4_lag1"], "PI rebuild")

    flow = out["TREAT_FLOW"].to_numpy(float)
    cod = out["IN_COD"].to_numpy(float)
    tn = out["IN_TN"].to_numpy(float)
    eff_tn = out["EFF_TN_lag1"].to_numpy(float)
    eff_nh4 = out["EFF_NH4_lag1"].to_numpy(float)

    out["LOW_FLOW_P"] = np.maximum(0, FLOW_Q25 - flow)
    out["COD/TN"] = safe_div(cod, tn)
    out["COD_DEFICIT"] = np.maximum(0, 4.12 * tn - 0.555 * cod)

    # Reconstruct the same PI-GBR boundary-relative descriptors used in training.
    # Do NOT clip here: values > 1 retain the magnitude of limit exceedance.
    # [0, 1] clipping belongs only to the later KBDC risk-scoring layer.
    r_tn = eff_tn / TN_LIMIT
    r_nh4 = eff_nh4 / NH4_LIMIT
    out["R_TN_lag1"] = r_tn
    out["R_NH4_lag1"] = r_nh4
    out["R_MAX_lag1"] = np.maximum(r_tn, r_nh4)
    out["R_MEAN_lag1"] = .5 * (r_tn + r_nh4)

    vss = auxiliary_vss(reference)
    if len(vss) != len(out):
        raise ValueError("Reference must be row-aligned with perturbed data.")
    out["TN_PER_VSS"] = safe_div(tn, vss)
    return out


# ---------------------------------------------------------------------
# VFC 1/2/3-day models
# ---------------------------------------------------------------------

def add_missing_flow_lags(train: pd.DataFrame, test: pd.DataFrame):
    needed = [f"TREAT_FLOW_lag{i}" for i in (1, 2, 3)]
    if all(c in train.columns and c in test.columns for c in needed):
        return train.copy(), test.copy()

    full = pd.concat([
        train.assign(__split="train", __row=np.arange(len(train))),
        test.assign(__split="test", __row=np.arange(len(test))),
    ], ignore_index=True)

    for i in (1, 2, 3):
        col = f"TREAT_FLOW_lag{i}"
        if col not in full.columns:
            full[col] = full["TREAT_FLOW"].shift(i)

    tr = full[full.__split.eq("train")].sort_values("__row").drop(
        columns=["__split", "__row"]).reset_index(drop=True)
    te = full[full.__split.eq("test")].sort_values("__row").drop(
        columns=["__split", "__row"]).reset_index(drop=True)
    warnings.warn("TREAT_FLOW lag columns were reconstructed from chronology.")
    return tr, te


def vfc_cols(days: int) -> List[str]:
    return [
        "IN_FLOW",
        *[f"TREAT_FLOW_lag{i}" for i in range(1, days + 1)],
        "IN_COD", "IN_TN", "IN_NH4", "IN_TP", "ANO_MLSS", "ANO_SV30",
    ]


def fit_vfc(train: pd.DataFrame, test: pd.DataFrame, model_dir: Path):
    train, test = add_missing_flow_lags(train, test)
    models, rows, predictions = {}, [], []
    model_dir.mkdir(parents=True, exist_ok=True)

    for days in (1, 2, 3):
        cols = vfc_cols(days)
        require(train, cols + ["TREAT_FLOW"], f"VFC {days}-day train")
        require(test, cols + ["TREAT_FLOW"], f"VFC {days}-day test")

        tr = numeric(train[cols + ["TREAT_FLOW"]], cols + ["TREAT_FLOW"]).dropna()
        te = numeric(test[cols + ["TREAT_FLOW"]], cols + ["TREAT_FLOW"]).dropna()

        rf = RandomForestRegressor(
            random_state=SEED, n_jobs=-1, **VFC_CONFIG[days]
        )
        rf.fit(tr[cols], tr["TREAT_FLOW"])
        pred = rf.predict(te[cols])

        rows.append({
            "lag_days": days, "train_n": len(tr), "test_n": len(te),
            **VFC_CONFIG[days], **calc_metrics(te["TREAT_FLOW"], pred),
        })
        predictions.append(pd.DataFrame({
            "lag_days": days,
            "row_id": np.arange(len(te)),
            "TREAT_FLOW_true": te["TREAT_FLOW"].to_numpy(),
            "TREAT_FLOW_vfc": pred,
        }))

        joblib.dump(rf, model_dir / f"vfc_rf_{days}day.joblib")
        models[days] = rf

    return models, pd.DataFrame(rows), pd.concat(predictions, ignore_index=True)


def predict_vfc(model, df: pd.DataFrame) -> np.ndarray:
    cols = [str(x) for x in model.feature_names_in_]
    require(df, cols, "VFC input")
    X = numeric(df[cols], cols)
    if X.isna().any().any():
        raise ValueError("Missing value in VFC input.")
    return np.maximum(np.asarray(model.predict(X), float), 0)


# ---------------------------------------------------------------------
# A. Grouped single-chain zeroing
# ---------------------------------------------------------------------

def dropout_mask(n: int, rate: float, seed: int) -> np.ndarray:
    if rate <= 0:
        return np.zeros(n, bool)
    rng = np.random.default_rng(seed)
    k = max(1, int(round(n * rate)))
    idx = rng.choice(n, k, replace=False)
    mask = np.zeros(n, bool)
    mask[idx] = True
    return mask


def grouped_zeroing(test: pd.DataFrame, gbr) -> pd.DataFrame:
    y = test[TARGET].to_numpy(float)
    rows = []

    for gi, (group, raw) in enumerate(GROUP_DRIVERS.items()):
        for rate in GROUP_ZERO_RATES:
            mask = dropout_mask(len(test), rate, SEED + gi * 100 + int(rate * 100))
            x = test.copy()
            x.loc[mask, raw] = 0
            x = rebuild_pi(x, test)
            rows.append({
                "group": group,
                "raw_driver": raw,
                "zeroing_rate_percent": int(round(rate * 100)),
                "abnormal_count": int(mask.sum()),
                **calc_metrics(y, predict_gbr(gbr, x)),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# B. Global random bidirectional perturbation
# ---------------------------------------------------------------------

def sample_mask(n: int, ratio: float, seed: int) -> np.ndarray:
    if ratio <= 0:
        return np.zeros(n, bool)
    rng = np.random.default_rng(seed)
    k = max(1, int(round(n * ratio)))
    idx = rng.choice(n, k, replace=False)
    mask = np.zeros(n, bool)
    mask[idx] = True
    return mask


def perturb_global(df: pd.DataFrame, mask: np.ndarray, amp: float):
    out = df.copy()
    idx = np.where(mask)[0]
    signs_full = np.zeros(len(df), int)
    if len(idx) == 0 or amp == 0:
        return out, signs_full

    rng = np.random.default_rng(SEED + int(round(amp * 1000)) + len(idx))
    signs = rng.choice([-1, 1], size=len(idx), p=[.5, .5])
    factor = np.maximum(1 + signs * amp, 0)

    for col in GLOBAL_RAW:
        vals = pd.to_numeric(out[col], errors="coerce").to_numpy(float)
        vals[idx] *= factor
        out[col] = np.maximum(vals, 0)

    signs_full[idx] = signs
    return out, signs_full


def vfc_correct_selected(x: pd.DataFrame, reference: pd.DataFrame, vfc, mask):
    out = x.copy()
    qv = predict_vfc(vfc, out)
    out.loc[mask, "TREAT_FLOW"] = qv[mask]
    return rebuild_pi(out, reference)


def global_random(test: pd.DataFrame, gbr, vfc):
    y = test[TARGET].to_numpy(float)
    a_rows, b_rows = [], []

    # Fixed amplitude; anomaly ratio 0-50%.
    for amp in FIXED_AMPLITUDES:
        for j, ratio in enumerate(RATIO_SWEEP):
            mask = sample_mask(len(test), ratio, SEED + 80000 + j)
            x, signs = perturb_global(test, mask, amp)
            x = rebuild_pi(x, test)
            no = calc_metrics(y, predict_gbr(gbr, x))
            xv = vfc_correct_selected(x, test, vfc, mask)
            yes = calc_metrics(y, predict_gbr(gbr, xv))
            a_rows.append({
                "fixed_amplitude_percent": int(round(amp * 100)),
                "anomaly_ratio_percent": int(round(ratio * 100)),
                "abnormal_count": int(mask.sum()),
                "negative_count": int((signs == -1).sum()),
                "positive_count": int((signs == 1).sum()),
                "MAPE_without_VFC(%)": no["MAPE(%)"],
                "MAPE_with_VFC(%)": yes["MAPE(%)"],
                "R2_without_VFC": no["R2"], "R2_with_VFC": yes["R2"],
            })

    # Fixed anomaly ratio; amplitude 0-50%.
    for ratio in FIXED_RATIOS:
        rp = int(round(ratio * 100))
        mask = sample_mask(len(test), ratio, SEED + 90000 + rp)
        for amp in AMPLITUDE_SWEEP:
            x, signs = perturb_global(test, mask, amp)
            x = rebuild_pi(x, test)
            no = calc_metrics(y, predict_gbr(gbr, x))
            xv = vfc_correct_selected(x, test, vfc, mask)
            yes = calc_metrics(y, predict_gbr(gbr, xv))
            b_rows.append({
                "fixed_anomaly_ratio_percent": rp,
                "amplitude_percent": int(round(amp * 100)),
                "abnormal_count": int(mask.sum()),
                "negative_count": int((signs == -1).sum()),
                "positive_count": int((signs == 1).sum()),
                "MAPE_without_VFC(%)": no["MAPE(%)"],
                "MAPE_with_VFC(%)": yes["MAPE(%)"],
                "R2_without_VFC": no["R2"], "R2_with_VFC": yes["R2"],
            })

    return pd.DataFrame(a_rows), pd.DataFrame(b_rows)


# ---------------------------------------------------------------------
# Isolated TREAT_FLOW zeroing with / without VFC
# ---------------------------------------------------------------------

def isolated_flow_zeroing(test: pd.DataFrame, gbr, vfc) -> pd.DataFrame:
    y = test[TARGET].to_numpy(float)
    base = calc_metrics(y, predict_gbr(gbr, rebuild_pi(test, test)))
    rows = [{
        "zeroing_rate_percent": 0, "zeroed_count": 0,
        "MAPE_without_VFC(%)": base["MAPE(%)"],
        "MAPE_with_VFC(%)": base["MAPE(%)"],
        "R2_without_VFC": base["R2"], "R2_with_VFC": base["R2"],
    }]

    n = len(test)
    for rate in GROUP_ZERO_RATES[1:]:
        rng = np.random.RandomState(SEED)
        idx = rng.choice(n, size=int(n * rate), replace=False)
        mask = np.zeros(n, bool)
        mask[idx] = True

        x = test.copy()
        x.loc[mask, "TREAT_FLOW"] = 0
        x = rebuild_pi(x, test)
        no = calc_metrics(y, predict_gbr(gbr, x))

        xv = vfc_correct_selected(x, test, vfc, mask)
        yes = calc_metrics(y, predict_gbr(gbr, xv))

        rows.append({
            "zeroing_rate_percent": int(round(rate * 100)),
            "zeroed_count": int(mask.sum()),
            "MAPE_without_VFC(%)": no["MAPE(%)"],
            "MAPE_with_VFC(%)": yes["MAPE(%)"],
            "R2_without_VFC": no["R2"], "R2_with_VFC": yes["R2"],
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Positive flow-amplitude crossover + threshold surface
# ---------------------------------------------------------------------

def crossover_from_curve(df: pd.DataFrame) -> float:
    sub = df[df["Amplitude (%)"] > 0].sort_values("Amplitude (%)")
    x = sub["Amplitude (%)"].to_numpy(float)
    d = (sub["MAPE base (%)"] - sub["MAPE always-VFC (%)"]).to_numpy(float)
    if len(x) == 0:
        return np.nan
    if d[0] >= 0:
        return float(x[0])
    for i in range(1, len(x)):
        if d[i] == 0:
            return float(x[i])
        if d[i - 1] < 0 < d[i]:
            return float(x[i - 1] - d[i - 1] * (x[i] - x[i - 1]) /
                         (d[i] - d[i - 1]))
    return np.nan


def flow_amplitude(test: pd.DataFrame, gbr, vfc, make_grid=True):
    y = test[TARGET].to_numpy(float)
    q_true = test["TREAT_FLOW"].to_numpy(float)
    q_vfc = predict_vfc(vfc, test)

    clean = calc_metrics(y, predict_gbr(gbr, rebuild_pi(test, test)))

    always = test.copy()
    always["TREAT_FLOW"] = q_vfc
    always = rebuild_pi(always, test)
    m_always = calc_metrics(y, predict_gbr(gbr, always))

    curve, heat = [], []

    for amp in POS_AMP:
        q_abn = q_true * (1 + amp / 100)
        x = test.copy()
        x["TREAT_FLOW"] = q_abn
        x = rebuild_pi(x, test)
        mb = calc_metrics(y, predict_gbr(gbr, x))
        mv = clean if amp == 0 else m_always

        curve.append({
            "Amplitude (%)": amp,
            "MAPE base (%)": mb["MAPE(%)"],
            "MAPE always-VFC (%)": mv["MAPE(%)"],
            "R2 base": mb["R2"], "R2 always-VFC": mv["R2"],
        })

        if not make_grid:
            continue

        discrepancy = np.abs(q_abn - q_vfc) / np.maximum(np.abs(q_abn), EPS) * 100
        for tau in TAU_SWEEP:
            trigger = np.zeros(len(test), bool) if amp == 0 else discrepancy > tau
            xc = x.copy()
            xc.loc[trigger, "TREAT_FLOW"] = q_vfc[trigger]
            xc = rebuild_pi(xc, test)
            mc = calc_metrics(y, predict_gbr(gbr, xc))
            heat.append({
                "Amplitude (%)": amp, "Tau (%)": tau,
                "Triggered count": int(trigger.sum()),
                "Trigger rate (%)": float(trigger.mean() * 100),
                "MAPE base (%)": mb["MAPE(%)"],
                "MAPE threshold correction (%)": mc["MAPE(%)"],
                "MAPE improvement vs base (%)": mb["MAPE(%)"] - mc["MAPE(%)"],
                "R2 threshold correction": mc["R2"],
            })

    curve = pd.DataFrame(curve)
    heat = pd.DataFrame(heat)
    return curve, heat, crossover_from_curve(curve)


# ---------------------------------------------------------------------
# Two-layer VFC/QC flags
# ---------------------------------------------------------------------

def qc_flags(train: pd.DataFrame, test: pd.DataFrame, vfc):
    q01 = float(train["TREAT_FLOW"].quantile(.01))
    q99 = float(train["TREAT_FLOW"].quantile(.99))
    measured = test["TREAT_FLOW"].to_numpy(float)
    virtual = predict_vfc(vfc, test)

    first = (~np.isfinite(measured)) | (measured < q01) | (measured > q99)
    discrepancy = np.abs(virtual - measured) / np.maximum(np.abs(measured), EPS) * 100
    second = (~first) & (discrepancy > QC_THRESHOLD)

    df = pd.DataFrame({
        "row_id": np.arange(len(test)),
        "TREAT_FLOW_measured": measured,
        "TREAT_FLOW_vfc": virtual,
        "first_layer_flag": first,
        "same_day_absolute_discrepancy_percent": discrepancy,
        "second_layer_flag": second,
        "VFC_QC_flag": first | second,
    })
    return df, {"train_q01": q01, "train_q99": q99,
                "second_layer_threshold_percent": QC_THRESHOLD}


# ---------------------------------------------------------------------
# Compact figure generation
# ---------------------------------------------------------------------

def style():
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["pdf.fonttype"] = 42


def savefig(fig, out: Path, name: str, dpi: int):
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.png", dpi=dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_lag(perf, out, dpi):
    d = perf.sort_values("lag_days")
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    x = np.arange(len(d))
    ax.bar(x, d["R2"], width=.55, alpha=.8)
    ax.set_xticks(x, [f"{i}-day" for i in d["lag_days"]])
    ax.set_ylabel("R²")
    ax2 = ax.twinx()
    ax2.plot(x, d["MAPE(%)"], marker="o")
    ax2.set_ylabel("MAPE (%)")
    ax.set_xlabel("Historical-flow lag setting")
    fig.tight_layout()
    savefig(fig, out, "Fig6c_vfc_lag_comparison", dpi)


def plot_flow_zero(d, out, dpi):
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    ax.plot(d["zeroing_rate_percent"], d["MAPE_without_VFC(%)"],
            marker="o", label="Without VFC")
    ax.plot(d["zeroing_rate_percent"], d["MAPE_with_VFC(%)"],
            marker="s", label="With VFC")
    ax.set_xlabel("TREAT_FLOW zeroing rate (%)")
    ax.set_ylabel("Test MAPE (%)")
    ax.grid(ls="--", alpha=.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    savefig(fig, out, "Fig6d_flow_zeroing_with_without_vfc", dpi)


def plot_crossover(curve, cross, out, dpi):
    fig, ax = plt.subplots(figsize=(7.6, 5.4))
    ax.plot(curve["Amplitude (%)"], curve["MAPE base (%)"], label="Base model")
    ax.plot(curve["Amplitude (%)"], curve["MAPE always-VFC (%)"], label="Always-on VFC")
    if np.isfinite(cross):
        ax.axvline(cross, ls="--", c="black")
    ax.set_xlabel("Flow amplitude anomaly (% of Q_true)")
    ax.set_ylabel("Test MAPE (%)")
    ax.legend(frameon=False)
    ax.grid(ls="--", alpha=.3)
    fig.tight_layout()
    savefig(fig, out, "Fig6e_flow_amplitude_crossover", dpi)


def plot_threshold(heat, cross, out, dpi):
    if heat.empty:
        return
    p = heat.pivot_table(index="Tau (%)", columns="Amplitude (%)",
                         values="MAPE improvement vs base (%)").sort_index().sort_index(axis=1)
    x, y, z = p.columns.to_numpy(float), p.index.to_numpy(float), p.to_numpy(float)
    m = max(float(np.nanmax(np.abs(z))), 1e-8)
    cmap = LinearSegmentedColormap.from_list(
        "robust", ["#ad5867", "#f3d9df", "#f5f5f5", "#dcdcf2", "#6663ad"])
    norm = TwoSlopeNorm(vmin=-m, vcenter=0, vmax=m)

    fig, ax = plt.subplots(figsize=(8.6, 6.2))
    cf = ax.contourf(x, y, z, levels=np.linspace(-m, m, 81),
                     cmap=cmap, norm=norm, extend="both")
    if np.nanmin(z) <= 0 <= np.nanmax(z):
        ax.contour(x, y, z, levels=[0], colors="black", linewidths=1.2)
    if np.isfinite(cross):
        ax.axvline(cross, ls="--", c="black")
    ax.axhline(QC_THRESHOLD, ls=":", c="black")
    ax.set_xlabel("Flow amplitude anomaly (% of Q_true)")
    ax.set_ylabel("VFC activation threshold (%)")
    fig.colorbar(cf, ax=ax, label="MAPE improvement vs base (%)")
    fig.tight_layout()
    savefig(fig, out, "Fig6f_vfc_threshold_surface", dpi)


def plot_global(a, b, out, dpi):
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.2))
    for amp in sorted(a["fixed_amplitude_percent"].unique()):
        s = a[a.fixed_amplitude_percent.eq(amp)].sort_values("anomaly_ratio_percent")
        axes[0,0].plot(s.anomaly_ratio_percent, s["MAPE_without_VFC(%)"],
                       marker="o", label=f"±{amp}%")
        axes[1,0].plot(s.anomaly_ratio_percent, s["MAPE_with_VFC(%)"],
                       marker="o", label=f"±{amp}%")
    for ratio in sorted(b["fixed_anomaly_ratio_percent"].unique()):
        s = b[b.fixed_anomaly_ratio_percent.eq(ratio)].sort_values("amplitude_percent")
        axes[0,1].plot(s.amplitude_percent, s["MAPE_without_VFC(%)"],
                       marker="o", label=f"{ratio}%")
        axes[1,1].plot(s.amplitude_percent, s["MAPE_with_VFC(%)"],
                       marker="o", label=f"{ratio}%")
    titles = [
        "Global perturbation without VFC", "Global perturbation without VFC",
        "Global perturbation with VFC", "Global perturbation with VFC"
    ]
    for ax, title in zip(axes.ravel(), titles):
        ax.set_title(title)
        ax.set_ylabel("Test MAPE (%)")
        ax.grid(ls="--", alpha=.25)
        ax.legend(frameon=False, fontsize=7, ncol=2)
    axes[0,0].set_xlabel("Anomaly ratio (%)")
    axes[1,0].set_xlabel("Anomaly ratio (%)")
    axes[0,1].set_xlabel("Random bidirectional amplitude (%)")
    axes[1,1].set_xlabel("Random bidirectional amplitude (%)")
    fig.tight_layout()
    savefig(fig, out, "FigS8_global_random_perturbation", dpi)


def plot_vfc_qc(train, vfc, bounds, out, dpi):
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.5))
    q = np.sort(train["TREAT_FLOW"].dropna().to_numpy(float))
    axes[0].plot(np.arange(len(q)), q)
    axes[0].axhline(bounds["train_q01"], ls="--")
    axes[0].axhline(bounds["train_q99"], ls="--")
    axes[0].set_xlabel("Sorted training-sample index")
    axes[0].set_ylabel("TREAT_FLOW (t/d)")

    imp = pd.DataFrame({"feature": vfc.feature_names_in_,
                        "importance": vfc.feature_importances_}).sort_values("importance")
    axes[1].barh(imp.feature, imp.importance)
    axes[1].set_xlabel("Feature importance")
    axes[1].tick_params(axis="y", labelsize=8)
    fig.tight_layout()
    savefig(fig, out, "FigS6_vfc_range_and_feature_importance", dpi)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def parser():
    p = argparse.ArgumentParser()
    p.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    p.add_argument("--test", type=Path, default=DEFAULT_TEST)
    p.add_argument("--gbr-model", type=Path, default=DEFAULT_GBR)
    p.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS)
    p.add_argument("--results-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--dpi", type=int, default=600)
    p.add_argument("--skip-threshold-grid", action="store_true")
    return p


def main():
    args = parser().parse_args()
    style()

    src = args.results_dir / "source_data"
    figs = args.results_dir / "figures"
    src.mkdir(parents=True, exist_ok=True)
    figs.mkdir(parents=True, exist_ok=True)

    train = canonicalize(read_table(args.train))
    test = canonicalize(read_table(args.test))

    num = list(ALIASES.keys())
    train, test = numeric(train, num), numeric(test, num)

    require(test, PI_FEATURES + [TARGET], "held-out test")
    require(train, ["TREAT_FLOW"], "training data")

    # Use the final SI definitions as the common reconstruction boundary.
    train_ref, test_ref = train.copy(), test.copy()
    if all(c in train.columns for c in
           ["IN_COD", "IN_TN", "EFF_TN_lag1", "EFF_NH4_lag1", "TN_PER_VSS"]):
        train = rebuild_pi(train, train_ref)
    test = rebuild_pi(test, test_ref)

    if not args.gbr_model.exists():
        raise FileNotFoundError(f"Missing PI-GBR model: {args.gbr_model}")
    gbr = joblib.load(args.gbr_model)

    # VFC lag models.
    vfc_models, vfc_perf, vfc_pred = fit_vfc(train, test, args.models_dir)
    vfc1 = vfc_models[1]
    vfc_perf.to_csv(src / "vfc_lag_performance.csv", index=False)
    vfc_pred.to_csv(src / "vfc_lag_predictions.csv", index=False)

    # VFC/QC flags.
    flags, bounds = qc_flags(train, test, vfc1)
    flags.to_csv(src / "vfc_qc_flags.csv", index=False)
    (src / "vfc_qc_settings.json").write_text(
        json.dumps(bounds, indent=2), encoding="utf-8")

    # Robustness experiments.
    grouped = grouped_zeroing(test, gbr)
    grouped.to_csv(src / "grouped_zeroing_robustness.csv", index=False)

    glob_a, glob_b = global_random(test, gbr, vfc1)
    glob_a.to_csv(src / "global_random_fixed_amplitude.csv", index=False)
    glob_b.to_csv(src / "global_random_fixed_ratio.csv", index=False)

    flow_zero = isolated_flow_zeroing(test, gbr, vfc1)
    flow_zero.to_csv(src / "isolated_flow_zeroing_with_without_vfc.csv", index=False)

    curve, heat, cross = flow_amplitude(
        test, gbr, vfc1, make_grid=not args.skip_threshold_grid)
    curve.to_csv(src / "flow_amplitude_crossover_curve.csv", index=False)
    if not heat.empty:
        heat.to_csv(src / "vfc_activation_threshold_surface.csv", index=False)

    summary = {
        "FLOW_Q25": FLOW_Q25,
        "first_layer_train_q01": bounds["train_q01"],
        "first_layer_train_q99": bounds["train_q99"],
        "second_layer_threshold_percent": QC_THRESHOLD,
        "computed_flow_amplitude_crossover_percent": (
            None if not np.isfinite(cross) else float(cross)
        ),
        "VFC_configs": VFC_CONFIG,
    }
    (args.results_dir / "run_metadata.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")

    # Key figures/source-figure checks.
    plot_lag(vfc_perf, figs, args.dpi)
    plot_flow_zero(flow_zero, figs, args.dpi)
    plot_crossover(curve, cross, figs, args.dpi)
    plot_threshold(heat, cross, figs, args.dpi)
    plot_global(glob_a, glob_b, figs, args.dpi)
    plot_vfc_qc(train, vfc1, bounds, figs, args.dpi)

    print("Completed.")
    print("Source data:", src)
    print("Figures:", figs)
    if np.isfinite(cross):
        print(f"Computed flow-amplitude crossover: {cross:.4f}%")
    print(f"Operational VFC/QC discrepancy threshold: {QC_THRESHOLD:.2f}%")


if __name__ == "__main__":
    main()
