#!/usr/bin/env python3
"""
03_model_benchmark.py
=====================

Reproducible Ori-vs-PI model benchmark for the KBDC study.

This script is the third step of the public analysis pipeline:

    01_preprocessing.py
        -> 02_process_features.py
        -> 03_model_benchmark.py

Scientific design implemented here
----------------------------------
1. The held-out test set is never used for model selection.
2. Hyperparameter evaluation uses five-fold expanding-window TimeSeriesSplit.
3. Ori and PI are evaluated independently under the same chronological protocol.
4. Preprocessing is fitted only on the training portion of each CV fold.
   This includes the DNN/LSTM imputer and scaler (fold-wise fitting).
5. The final selected configuration is refitted on the complete 512-record
   training set and evaluated once on the 129-record held-out set.

Feature spaces
--------------
Ori (7 variables):
    TREAT_FLOW, IN_COD, IN_TN, IN_NH4, IN_TP, ANO_MLSS, ANO_SV30

PI (17 variables):
    the 7 Ori variables
    + EFF_TN_lag1, EFF_NH4_lag1
    + LOW_FLOW_P, COD/TN, COD_DEFICIT,
      R_TN_lag1, R_NH4_lag1, R_MAX_lag1, R_MEAN_lag1, TN_PER_VSS

Two selection modes are available:

    --selection-mode reported   (default)
        Refit/evaluate the final configurations reported in Table S3 and
        calculate their fold-wise validation R2. This is the deterministic
        manuscript-reproduction path.

    --selection-mode search
        Re-run hyperparameter selection within the training set. Grid search
        is used for DT/KNN; randomized search for SVR/RF/GBR; random sampling
        for DNN/LSTM. The exact number of randomized candidates is controlled
        by command-line arguments because the SI reports the candidate domains
        but not the number of sampled configurations.

Default inputs (when this file is placed in code/):
    data/processed/train_model_ready.csv
    data/processed/test_model_ready.csv

Main outputs:
    results/model_benchmark/benchmark_metrics_long.csv
    results/model_benchmark/table_s3_panel_c_performance.csv
    results/model_benchmark/cv_fold_metrics.csv
    results/model_benchmark/heldout_predictions_long.csv
    results/model_benchmark/selected_configurations.json
    results/model_benchmark/feature_spaces.json
    models/pi_gbr_model.joblib
    models/pi_gbr_metadata.json

The code accepts the historical spreadsheet/code aliases used during analysis
(e.g. LOW_FLOW_PRESSURE and RISK_TN_lag1) and internally harmonizes them to
manuscript names (LOW_FLOW_P and R_TN_lag1).
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import random
import sys
import time
import warnings
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, TimeSeriesSplit
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except Exception as exc:  # pragma: no cover - import failure handled at runtime
    torch = None
    nn = None
    DataLoader = None
    TensorDataset = None
    TORCH_IMPORT_ERROR = exc
else:
    TORCH_IMPORT_ERROR = None


# -----------------------------------------------------------------------------
# Project paths and study constants
# -----------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name.lower() == "code" else SCRIPT_DIR

DEFAULT_TRAIN = PROJECT_ROOT / "data" / "processed" / "train_model_ready.csv"
DEFAULT_TEST = PROJECT_ROOT / "data" / "processed" / "test_model_ready.csv"
DEFAULT_RESULTS = PROJECT_ROOT / "results" / "model_benchmark"
DEFAULT_MODELS = PROJECT_ROOT / "models"

RANDOM_SEED = 42
TARGET = "CARBON_DOS"
N_SPLITS = 5

ORI_FEATURES: List[str] = [
    "TREAT_FLOW",
    "IN_COD",
    "IN_TN",
    "IN_NH4",
    "IN_TP",
    "ANO_MLSS",
    "ANO_SV30",
]

PI_FEATURES: List[str] = ORI_FEATURES + [
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

FEATURE_SPACES: Dict[str, List[str]] = {
    "Ori": ORI_FEATURES,
    "PI": PI_FEATURES,
}

# Historical code/spreadsheet names -> manuscript names.
COLUMN_ALIASES: Dict[str, Sequence[str]] = {
    "TREAT_FLOW": ("TREAT_FLOW", "TREAT_FLOW_t(t/d)"),
    "IN_COD": ("IN_COD", "IN_COD_t(mg/L)"),
    "IN_TN": ("IN_TN", "IN_TN_t(mg/L)"),
    "IN_NH4": ("IN_NH4", "IN_NH4_t(mg/L)"),
    "IN_TP": ("IN_TP", "IN_TP_t(mg/L)"),
    "ANO_MLSS": ("ANO_MLSS", "ANO_MLSS(mg/L)"),
    "ANO_SV30": ("ANO_SV30", "ANO_SV30(%)"),
    "EFF_TN_lag1": ("EFF_TN_lag1", "EFF_TN_LAG1", "EFF_TN_lag1(mg/L)"),
    "EFF_NH4_lag1": ("EFF_NH4_lag1", "EFF_NH4_LAG1", "EFF_NH4_lag1(mg/L)"),
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

MODEL_ORDER = ["LIN", "DT", "SVR", "KNN", "RF", "GBR", "DNN", "LSTM"]
REP_ORDER = ["Ori", "PI"]

# Final configurations reported in current Table S3, Panel B.
REPORTED_CONFIGS: Dict[str, Dict[str, Dict[str, Any]]] = {
    "LIN": {
        "Ori": {},
        "PI": {},
    },
    "DT": {
        "Ori": {
            "max_depth": 3,
            "min_samples_split": 2,
            "min_samples_leaf": 3,
            "max_features": None,
        },
        "PI": {
            "max_depth": None,
            "max_features": None,
            "min_samples_leaf": 3,
            "min_samples_split": 10,
        },
    },
    "SVR": {
        "Ori": {
            "kernel": "rbf",
            "C": 200,
            "gamma": "auto",
            "epsilon": 0.2,
            "degree": 3,
        },
        "PI": {
            "kernel": "rbf",
            "gamma": 0.05,
            "epsilon": 0.001,
            "degree": 4,
            "C": 200,
        },
    },
    "KNN": {
        "Ori": {"n_neighbors": 5, "weights": "distance", "p": 2},
        "PI": {"n_neighbors": 5, "p": 2, "weights": "distance"},
    },
    "RF": {
        "Ori": {
            "n_estimators": 100,
            "max_depth": 12,
            "min_samples_split": 3,
            "min_samples_leaf": 1,
            "max_features": "sqrt",
            "bootstrap": True,
        },
        "PI": {
            "n_estimators": 500,
            "max_depth": 10,
            "min_samples_split": 5,
            "min_samples_leaf": 1,
            "max_features": "sqrt",
            "bootstrap": False,
        },
    },
    "GBR": {
        "Ori": {
            "subsample": 0.6,
            "n_estimators": 500,
            "min_samples_split": 5,
            "min_samples_leaf": 1,
            "max_features": "sqrt",
            "max_depth": 6,
            "learning_rate": 0.03,
        },
        "PI": {
            "subsample": 0.6,
            "n_estimators": 500,
            "min_samples_split": 5,
            "min_samples_leaf": 1,
            "max_features": "sqrt",
            "max_depth": 6,
            "learning_rate": 0.03,
        },
    },
    "DNN": {
        "Ori": {
            "hidden_dims": (128, 64),
            "dropout": 0.1,
            "learning_rate": 0.01,
            "batch_size": 16,
            "epochs": 150,
        },
        "PI": {
            "hidden_dims": (128, 64),
            "dropout": 0.3,
            "learning_rate": 0.01,
            "batch_size": 32,
            "epochs": 150,
        },
    },
    "LSTM": {
        "Ori": {
            "hidden_size": 128,
            "num_layers": 1,
            "dropout": 0.2,
            "bidirectional": True,
            "learning_rate": 0.01,
            "batch_size": 16,
            "epochs": 200,
        },
        "PI": {
            "hidden_size": 128,
            "num_layers": 1,
            "dropout": 0.2,
            "bidirectional": True,
            "learning_rate": 0.01,
            "batch_size": 16,
            "epochs": 200,
        },
    },
}

# Candidate domains reported in Table S3, Panel A.
SEARCH_SPACES: Dict[str, Dict[str, Sequence[Any]]] = {
    "DT": {
        "max_depth": [None, 3, 5, 8, 10, 15, 20],
        "min_samples_split": [2, 3, 5, 10, 15],
        "min_samples_leaf": [1, 2, 3, 5, 8],
        "max_features": [None, "sqrt", "log2"],
    },
    "KNN": {
        "n_neighbors": [3, 5, 7, 9, 11, 15],
        "weights": ["uniform", "distance"],
        "p": [1, 2],
    },
    "SVR": {
        "kernel": ["rbf", "poly", "sigmoid"],
        "C": [0.1, 1, 5, 10, 50, 100, 200],
        "epsilon": [0.001, 0.01, 0.05, 0.1, 0.2],
        "gamma": ["scale", "auto", 0.001, 0.01, 0.05, 0.1],
        "degree": [2, 3, 4],
    },
    "RF": {
        "n_estimators": [100, 200, 300, 500, 800],
        "max_depth": [None, 5, 8, 10, 12, 15, 20],
        "min_samples_split": [2, 3, 5, 8, 10],
        "min_samples_leaf": [1, 2, 3, 4, 5],
        "max_features": ["sqrt", "log2", 0.6, 0.8, 1.0, None],
        "bootstrap": [True, False],
    },
    "GBR": {
        "n_estimators": [100, 200, 300, 500],
        "learning_rate": [0.005, 0.01, 0.03, 0.05, 0.08, 0.1],
        "max_depth": [2, 3, 4, 5, 6, 8],
        "min_samples_split": [2, 3, 5, 8, 10],
        "min_samples_leaf": [1, 2, 3, 4, 5],
        "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
        "max_features": [None, "sqrt", "log2", 0.6, 0.8, 1.0],
    },
    "DNN": {
        "hidden_dims": [(64, 32), (128, 64), (256, 128), (128, 64, 32)],
        "dropout": [0.1, 0.2, 0.3, 0.4],
        "learning_rate": [0.01, 0.005, 0.001, 0.0005, 0.0001],
        "batch_size": [16, 32, 64, 128],
        "epochs": [150, 200, 300],
    },
    "LSTM": {
        "hidden_size": [32, 64, 128, 256],
        "num_layers": [1, 2, 3],
        "dropout": [0.1, 0.2, 0.3, 0.4],
        "bidirectional": [False, True],
        "learning_rate": [0.01, 0.005, 0.001, 0.0005, 0.0001],
        "batch_size": [16, 32, 64, 128],
        "epochs": [150, 200, 300],
    },
}


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def set_global_seed(seed: int) -> None:
    """Set Python/NumPy/PyTorch seeds for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # Deterministic flags reduce avoidable run-to-run variation.
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def json_ready(obj: Any) -> Any:
    """Convert NumPy/tuple/path objects into JSON-serializable Python objects."""
    if isinstance(obj, dict):
        return {str(k): json_ready(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_ready(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, Path):
        return str(obj)
    return obj


def read_table(path: Path) -> pd.DataFrame:
    """Read CSV or Excel input while preserving row order."""
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise ValueError(f"Unsupported input format: {path.suffix}; use CSV/XLSX/XLS")


def canonicalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Harmonize historical code/spreadsheet names to manuscript variable names."""
    out = df.copy()
    rename_map: Dict[str, str] = {}
    for canonical, aliases in COLUMN_ALIASES.items():
        if canonical in out.columns:
            continue
        hits = [name for name in aliases if name in out.columns]
        if len(hits) == 1:
            rename_map[hits[0]] = canonical
        elif len(hits) > 1:
            # Prefer first alias deterministically; this situation should be rare.
            rename_map[hits[0]] = canonical
            warnings.warn(
                f"Multiple aliases found for {canonical}: {hits}. Using {hits[0]}.",
                RuntimeWarning,
            )
    if rename_map:
        out = out.rename(columns=rename_map)
    return out


def validate_input_frames(train: pd.DataFrame, test: pd.DataFrame) -> None:
    required = set(PI_FEATURES + [TARGET])
    for label, df in [("training", train), ("held-out test", test)]:
        missing = sorted(required.difference(df.columns))
        if missing:
            raise KeyError(
                f"{label} data are missing required model columns: {missing}\n"
                f"Available columns: {list(df.columns)}"
            )

    if len(train) != 512:
        warnings.warn(
            f"Expected 512 retained training records from the study pipeline; found {len(train)}.",
            RuntimeWarning,
        )
    if len(test) != 129:
        warnings.warn(
            f"Expected 129 held-out test records from the study pipeline; found {len(test)}.",
            RuntimeWarning,
        )

    if train[TARGET].isna().any() or test[TARGET].isna().any():
        raise ValueError(f"Target {TARGET} contains missing values; target imputation is not allowed.")

    if (train[TARGET] <= 0).any() or (test[TARGET] <= 0).any():
        warnings.warn(
            "CARBON_DOS contains non-positive values. MAPE will require special interpretation.",
            RuntimeWarning,
        )


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    return {
        "R2": float(r2_score(y_true, y_pred)),
        "MAPE_pct": float(mean_absolute_percentage_error(y_true, y_pred) * 100.0),
        "RMSE": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
    }


def prefix_params(params: Mapping[str, Any], prefix: str = "model__") -> Dict[str, Any]:
    return {f"{prefix}{k}": v for k, v in params.items()}


def strip_model_prefix(params: Mapping[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in params.items():
        out[k[len("model__") :] if k.startswith("model__") else k] = v
    return out


def iter_product(space: Mapping[str, Sequence[Any]]) -> List[Dict[str, Any]]:
    keys = list(space.keys())
    return [dict(zip(keys, values)) for values in itertools.product(*(space[k] for k in keys))]


def sample_configs(
    space: Mapping[str, Sequence[Any]], n_samples: int, seed: int
) -> List[Dict[str, Any]]:
    all_configs = iter_product(space)
    if n_samples <= 0:
        raise ValueError("Random neural-search sample count must be >= 1.")
    if n_samples >= len(all_configs):
        return all_configs
    rng = random.Random(seed)
    return rng.sample(all_configs, n_samples)


# -----------------------------------------------------------------------------
# Conventional sklearn models
# -----------------------------------------------------------------------------

def build_sklearn_pipeline(model_name: str, params: Mapping[str, Any], seed: int) -> Pipeline:
    """Create a leakage-safe sklearn pipeline for one model/configuration."""
    model_name = model_name.upper()
    params = dict(params)

    if model_name == "LIN":
        model = LinearRegression(**params)
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", model),
            ]
        )

    if model_name == "DT":
        model = DecisionTreeRegressor(random_state=seed, **params)
        return Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", model)])

    if model_name == "SVR":
        model = SVR(**params)
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", model),
            ]
        )

    if model_name == "KNN":
        model = KNeighborsRegressor(**params)
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", model),
            ]
        )

    if model_name == "RF":
        model = RandomForestRegressor(random_state=seed, n_jobs=1, **params)
        return Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", model)])

    if model_name == "GBR":
        model = GradientBoostingRegressor(random_state=seed, **params)
        return Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", model)])

    raise ValueError(f"Unsupported sklearn model: {model_name}")


def evaluate_sklearn_cv(
    model_name: str,
    params: Mapping[str, Any],
    X: pd.DataFrame,
    y: np.ndarray,
    tscv: TimeSeriesSplit,
    seed: int,
) -> Tuple[List[Dict[str, Any]], float]:
    rows: List[Dict[str, Any]] = []
    scores: List[float] = []
    for fold, (tr_idx, va_idx) in enumerate(tscv.split(X), start=1):
        estimator = build_sklearn_pipeline(model_name, params, seed)
        estimator.fit(X.iloc[tr_idx], y[tr_idx])
        pred = estimator.predict(X.iloc[va_idx])
        score = float(r2_score(y[va_idx], pred))
        scores.append(score)
        rows.append(
            {
                "fold": fold,
                "train_n": int(len(tr_idx)),
                "validation_n": int(len(va_idx)),
                "validation_R2": score,
            }
        )
    return rows, float(np.mean(scores))


def search_sklearn_model(
    model_name: str,
    X: pd.DataFrame,
    y: np.ndarray,
    tscv: TimeSeriesSplit,
    seed: int,
    n_jobs: int,
    random_iterations: Mapping[str, int],
) -> Tuple[Dict[str, Any], float]:
    """Select conventional-model hyperparameters using training-only TimeSeriesSplit."""
    if model_name == "LIN":
        _, mean_r2 = evaluate_sklearn_cv("LIN", {}, X, y, tscv, seed)
        return {}, mean_r2

    base = build_sklearn_pipeline(model_name, {}, seed)
    param_space = prefix_params(SEARCH_SPACES[model_name])

    if model_name in {"DT", "KNN"}:
        search = GridSearchCV(
            estimator=base,
            param_grid=param_space,
            scoring="r2",
            cv=tscv,
            n_jobs=n_jobs,
            refit=False,
            error_score="raise",
            return_train_score=False,
        )
    elif model_name in {"SVR", "RF", "GBR"}:
        requested = int(random_iterations[model_name])
        total = int(np.prod([len(v) for v in SEARCH_SPACES[model_name].values()]))
        n_iter = min(requested, total)
        search = RandomizedSearchCV(
            estimator=base,
            param_distributions=param_space,
            n_iter=n_iter,
            scoring="r2",
            cv=tscv,
            random_state=seed,
            n_jobs=n_jobs,
            refit=False,
            error_score="raise",
            return_train_score=False,
        )
    else:
        raise ValueError(f"Search is not implemented for {model_name}")

    search.fit(X, y)
    return strip_model_prefix(search.best_params_), float(search.best_score_)


# -----------------------------------------------------------------------------
# PyTorch DNN/LSTM models and fold-wise preprocessing
# -----------------------------------------------------------------------------

if nn is not None:

    class DNNRegressor(nn.Module):
        def __init__(self, input_dim: int, hidden_dims: Sequence[int], dropout: float):
            super().__init__()
            layers: List[nn.Module] = []
            prev = input_dim
            for width in hidden_dims:
                layers.append(nn.Linear(prev, int(width)))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(float(dropout)))
                prev = int(width)
            layers.append(nn.Linear(prev, 1))
            self.network = nn.Sequential(*layers)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.network(x).squeeze(-1)


    class LSTMRegressor(nn.Module):
        """Single-step LSTM comparator using the daily pre-dosing state representation."""

        def __init__(
            self,
            input_dim: int,
            hidden_size: int,
            num_layers: int,
            dropout: float,
            bidirectional: bool,
        ):
            super().__init__()
            internal_dropout = float(dropout) if int(num_layers) > 1 else 0.0
            self.lstm = nn.LSTM(
                input_size=input_dim,
                hidden_size=int(hidden_size),
                num_layers=int(num_layers),
                batch_first=True,
                dropout=internal_dropout,
                bidirectional=bool(bidirectional),
            )
            out_dim = int(hidden_size) * (2 if bidirectional else 1)
            self.output = nn.Linear(out_dim, 1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x shape: [batch, sequence_length=1, features]
            out, _ = self.lstm(x)
            last = out[:, -1, :]
            return self.output(last).squeeze(-1)


def require_torch() -> None:
    if torch is None:
        raise RuntimeError(
            "PyTorch is required for DNN/LSTM models but could not be imported. "
            f"Original import error: {TORCH_IMPORT_ERROR}"
        )


def resolve_device(name: str) -> str:
    require_torch()
    name = name.lower()
    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        warnings.warn("CUDA requested but unavailable; falling back to CPU.", RuntimeWarning)
        return "cpu"
    if name not in {"cpu", "cuda"}:
        raise ValueError("--device must be one of: cpu, cuda, auto")
    return name


def build_torch_model(model_name: str, input_dim: int, config: Mapping[str, Any]):
    require_torch()
    if model_name == "DNN":
        return DNNRegressor(
            input_dim=input_dim,
            hidden_dims=config["hidden_dims"],
            dropout=float(config["dropout"]),
        )
    if model_name == "LSTM":
        return LSTMRegressor(
            input_dim=input_dim,
            hidden_size=int(config["hidden_size"]),
            num_layers=int(config["num_layers"]),
            dropout=float(config["dropout"]),
            bidirectional=bool(config["bidirectional"]),
        )
    raise ValueError(model_name)


def train_torch_model(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    config: Mapping[str, Any],
    seed: int,
    device: str,
):
    require_torch()
    set_global_seed(seed)

    X_tensor = torch.as_tensor(X_train, dtype=torch.float32)
    y_tensor = torch.as_tensor(y_train, dtype=torch.float32)
    if model_name == "LSTM":
        X_tensor = X_tensor.unsqueeze(1)  # single-step sequence

    dataset = TensorDataset(X_tensor, y_tensor)
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        generator=generator,
    )

    model = build_torch_model(model_name, X_train.shape[1], config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["learning_rate"]))
    criterion = nn.MSELoss()

    model.train()
    for _ in range(int(config["epochs"])):
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()

    return model


def predict_torch_model(model_name: str, model, X: np.ndarray, device: str) -> np.ndarray:
    require_torch()
    model.eval()
    x = torch.as_tensor(X, dtype=torch.float32)
    if model_name == "LSTM":
        x = x.unsqueeze(1)
    with torch.no_grad():
        pred = model(x.to(device)).detach().cpu().numpy().reshape(-1)
    return pred


def foldwise_preprocess(
    X_train: pd.DataFrame, X_valid: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray, SimpleImputer, StandardScaler]:
    """Fit imputer and scaler only on the current fold's training rows."""
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    train_imp = imputer.fit_transform(X_train)
    valid_imp = imputer.transform(X_valid)
    train_scaled = scaler.fit_transform(train_imp)
    valid_scaled = scaler.transform(valid_imp)
    return train_scaled, valid_scaled, imputer, scaler


def evaluate_torch_cv(
    model_name: str,
    config: Mapping[str, Any],
    X: pd.DataFrame,
    y: np.ndarray,
    tscv: TimeSeriesSplit,
    seed: int,
    device: str,
) -> Tuple[List[Dict[str, Any]], float]:
    rows: List[Dict[str, Any]] = []
    scores: List[float] = []
    for fold, (tr_idx, va_idx) in enumerate(tscv.split(X), start=1):
        Xtr, Xva, _, _ = foldwise_preprocess(X.iloc[tr_idx], X.iloc[va_idx])
        model = train_torch_model(
            model_name,
            Xtr,
            y[tr_idx],
            config,
            seed=seed + fold,
            device=device,
        )
        pred = predict_torch_model(model_name, model, Xva, device)
        score = float(r2_score(y[va_idx], pred))
        scores.append(score)
        rows.append(
            {
                "fold": fold,
                "train_n": int(len(tr_idx)),
                "validation_n": int(len(va_idx)),
                "validation_R2": score,
            }
        )
        del model
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
    return rows, float(np.mean(scores))


def search_torch_model(
    model_name: str,
    X: pd.DataFrame,
    y: np.ndarray,
    tscv: TimeSeriesSplit,
    seed: int,
    device: str,
    n_samples: int,
) -> Tuple[Dict[str, Any], float]:
    candidates = sample_configs(SEARCH_SPACES[model_name], n_samples, seed)
    best_config: Dict[str, Any] | None = None
    best_score = -np.inf

    for idx, config in enumerate(candidates, start=1):
        _, mean_r2 = evaluate_torch_cv(
            model_name,
            config,
            X,
            y,
            tscv,
            seed=seed,
            device=device,
        )
        if mean_r2 > best_score:
            best_score = mean_r2
            best_config = deepcopy(config)

    assert best_config is not None
    return best_config, float(best_score)


def fit_evaluate_torch_final(
    model_name: str,
    config: Mapping[str, Any],
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    seed: int,
    device: str,
) -> np.ndarray:
    """Fit final imputer/scaler on all training data, never on held-out test data."""
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(imputer.fit_transform(X_train))
    Xte = scaler.transform(imputer.transform(X_test))
    model = train_torch_model(model_name, Xtr, y_train, config, seed, device)
    pred = predict_torch_model(model_name, model, Xte, device)
    return pred


# -----------------------------------------------------------------------------
# Benchmark orchestration
# -----------------------------------------------------------------------------

def select_config(
    model_name: str,
    representation: str,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    tscv: TimeSeriesSplit,
    args: argparse.Namespace,
    device: str,
) -> Tuple[Dict[str, Any], float | None]:
    if args.selection_mode == "reported":
        return deepcopy(REPORTED_CONFIGS[model_name][representation]), None

    if model_name in {"LIN", "DT", "SVR", "KNN", "RF", "GBR"}:
        random_iterations = {
            "SVR": args.svr_n_iter,
            "RF": args.rf_n_iter,
            "GBR": args.gbr_n_iter,
        }
        return search_sklearn_model(
            model_name,
            X_train,
            y_train,
            tscv,
            seed=args.seed,
            n_jobs=args.n_jobs,
            random_iterations=random_iterations,
        )

    n_samples = args.dnn_samples if model_name == "DNN" else args.lstm_samples
    return search_torch_model(
        model_name,
        X_train,
        y_train,
        tscv,
        seed=args.seed,
        device=device,
        n_samples=n_samples,
    )


def benchmark_one(
    model_name: str,
    representation: str,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    tscv: TimeSeriesSplit,
    args: argparse.Namespace,
    device: str,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], np.ndarray, Any | None]:
    print(f"\n[{representation} | {model_name}] selecting/evaluating ...", flush=True)
    config, search_best_cv = select_config(
        model_name, representation, X_train, y_train, tscv, args, device
    )

    if model_name in {"LIN", "DT", "SVR", "KNN", "RF", "GBR"}:
        fold_rows, mean_cv_r2 = evaluate_sklearn_cv(
            model_name, config, X_train, y_train, tscv, args.seed
        )
        final_estimator = build_sklearn_pipeline(model_name, config, args.seed)
        final_estimator.fit(X_train, y_train)
        y_pred = final_estimator.predict(X_test)
    else:
        fold_rows, mean_cv_r2 = evaluate_torch_cv(
            model_name, config, X_train, y_train, tscv, args.seed, device
        )
        y_pred = fit_evaluate_torch_final(
            model_name,
            config,
            X_train,
            y_train,
            X_test,
            seed=args.seed,
            device=device,
        )
        final_estimator = None

    metrics = regression_metrics(y_test, y_pred)
    result = {
        "model": model_name,
        "representation": representation,
        "n_features": int(X_train.shape[1]),
        "CV_mean_R2": mean_cv_r2,
        "search_best_CV_R2": search_best_cv,
        **metrics,
        "selected_config": json.dumps(json_ready(config), ensure_ascii=False, sort_keys=True),
    }

    for row in fold_rows:
        row["model"] = model_name
        row["representation"] = representation

    print(
        f"    CV mean R2={mean_cv_r2:.6f} | "
        f"test R2={metrics['R2']:.6f} | MAPE={metrics['MAPE_pct']:.2f}% | "
        f"RMSE={metrics['RMSE']:.2f} | MAE={metrics['MAE']:.2f}",
        flush=True,
    )
    return result, fold_rows, np.asarray(y_pred).reshape(-1), final_estimator


def make_table_s3_panel_c(metrics_long: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for model in MODEL_ORDER:
        sub = metrics_long[metrics_long["model"] == model].set_index("representation")
        if not {"Ori", "PI"}.issubset(sub.index):
            continue
        ori = sub.loc["Ori"]
        pi = sub.loc["PI"]
        rows.append(
            {
                "Model": model,
                "Ori_R2": ori["R2"],
                "PI_R2": pi["R2"],
                "Delta_R2": pi["R2"] - ori["R2"],
                "Ori_MAPE_pct": ori["MAPE_pct"],
                "PI_MAPE_pct": pi["MAPE_pct"],
                "Ori_RMSE": ori["RMSE"],
                "PI_RMSE": pi["RMSE"],
                "Ori_MAE": ori["MAE"],
                "PI_MAE": pi["MAE"],
            }
        )
    return pd.DataFrame(rows)


def parse_models(value: str) -> List[str]:
    requested = [x.strip().upper() for x in value.split(",") if x.strip()]
    invalid = [x for x in requested if x not in MODEL_ORDER]
    if invalid:
        raise argparse.ArgumentTypeError(
            f"Unknown models: {invalid}. Allowed: {','.join(MODEL_ORDER)}"
        )
    # Preserve canonical manuscript order.
    return [m for m in MODEL_ORDER if m in requested]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="KBDC Ori-vs-PI benchmark across eight regression model families."
    )
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--test", type=Path, default=DEFAULT_TEST)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS)
    parser.add_argument(
        "--selection-mode",
        choices=["reported", "search"],
        default="reported",
        help=(
            "reported = evaluate Table S3 final configurations; "
            "search = rerun training-only hyperparameter selection."
        ),
    )
    parser.add_argument(
        "--models",
        type=parse_models,
        default=MODEL_ORDER,
        help="Comma-separated subset, e.g. LIN,DT,SVR,KNN,RF,GBR,DNN,LSTM",
    )
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")

    # Search-mode controls. The SI gives candidate domains but not sampled counts.
    parser.add_argument("--svr-n-iter", type=int, default=100)
    parser.add_argument("--rf-n-iter", type=int, default=100)
    parser.add_argument("--gbr-n-iter", type=int, default=100)
    parser.add_argument("--dnn-samples", type=int, default=40)
    parser.add_argument("--lstm-samples", type=int, default=40)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    set_global_seed(args.seed)

    selected_models: List[str] = args.models
    if any(m in {"DNN", "LSTM"} for m in selected_models):
        device = resolve_device(args.device)
    else:
        device = "cpu"

    args.results_dir.mkdir(parents=True, exist_ok=True)
    args.models_dir.mkdir(parents=True, exist_ok=True)

    print("KBDC model benchmark", flush=True)
    print(f"  train: {args.train}", flush=True)
    print(f"  test : {args.test}", flush=True)
    print(f"  selection mode: {args.selection_mode}", flush=True)
    print(f"  models: {selected_models}", flush=True)
    print(f"  neural device: {device}", flush=True)

    train = canonicalize_columns(read_table(args.train))
    test = canonicalize_columns(read_table(args.test))
    validate_input_frames(train, test)

    y_train = train[TARGET].to_numpy(dtype=float)
    y_test = test[TARGET].to_numpy(dtype=float)
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)

    metrics_rows: List[Dict[str, Any]] = []
    cv_rows: List[Dict[str, Any]] = []
    pred_rows: List[Dict[str, Any]] = []
    selected_configs: Dict[str, Dict[str, Dict[str, Any]]] = {}
    pi_gbr_estimator = None
    pi_gbr_config: Dict[str, Any] | None = None

    start = time.time()
    for representation in REP_ORDER:
        features = FEATURE_SPACES[representation]
        X_train = train[features].copy()
        X_test = test[features].copy()
        selected_configs[representation] = {}

        for model_name in selected_models:
            result, folds, y_pred, estimator = benchmark_one(
                model_name,
                representation,
                X_train,
                y_train,
                X_test,
                y_test,
                tscv,
                args,
                device,
            )
            metrics_rows.append(result)
            cv_rows.extend(folds)
            selected_configs[representation][model_name] = json.loads(result["selected_config"])

            for i, (truth, pred) in enumerate(zip(y_test, y_pred)):
                pred_rows.append(
                    {
                        "test_row": int(i),
                        "model": model_name,
                        "representation": representation,
                        "y_true": float(truth),
                        "y_pred": float(pred),
                    }
                )

            if representation == "PI" and model_name == "GBR":
                pi_gbr_estimator = estimator
                pi_gbr_config = selected_configs[representation][model_name]

    elapsed = time.time() - start

    metrics_df = pd.DataFrame(metrics_rows)
    if not metrics_df.empty:
        metrics_df["model"] = pd.Categorical(
            metrics_df["model"], categories=MODEL_ORDER, ordered=True
        )
        metrics_df["representation"] = pd.Categorical(
            metrics_df["representation"], categories=REP_ORDER, ordered=True
        )
        metrics_df = metrics_df.sort_values(["model", "representation"]).reset_index(drop=True)
        metrics_df["model"] = metrics_df["model"].astype(str)
        metrics_df["representation"] = metrics_df["representation"].astype(str)

    cv_df = pd.DataFrame(cv_rows)
    pred_df = pd.DataFrame(pred_rows)
    panel_c_df = make_table_s3_panel_c(metrics_df)

    panel_b_rows = []
    for rep in REP_ORDER:
        for model in MODEL_ORDER:
            if rep in selected_configs and model in selected_configs[rep]:
                panel_b_rows.append(
                    {
                        "Model": model,
                        "Representation": rep,
                        "Selected_configuration": json.dumps(
                            json_ready(selected_configs[rep][model]),
                            ensure_ascii=False,
                            sort_keys=True,
                        ),
                    }
                )
    panel_b_df = pd.DataFrame(panel_b_rows)
    fig_s3_df = cv_df[cv_df["representation"] == "PI"].copy() if not cv_df.empty else cv_df.copy()
    fig4_df = panel_c_df.copy()

    metrics_path = args.results_dir / "benchmark_metrics_long.csv"
    cv_path = args.results_dir / "cv_fold_metrics.csv"
    pred_path = args.results_dir / "heldout_predictions_long.csv"
    panel_b_path = args.results_dir / "table_s3_panel_b_selected_configs.csv"
    panel_c_path = args.results_dir / "table_s3_panel_c_performance.csv"
    fig_s3_path = args.results_dir / "fig_s3_pi_cv_r2.csv"
    fig4_path = args.results_dir / "fig4_model_comparison.csv"
    configs_path = args.results_dir / "selected_configurations.json"
    feature_path = args.results_dir / "feature_spaces.json"
    metadata_path = args.results_dir / "run_metadata.json"

    metrics_df.to_csv(metrics_path, index=False)
    cv_df.to_csv(cv_path, index=False)
    pred_df.to_csv(pred_path, index=False)
    panel_b_df.to_csv(panel_b_path, index=False)
    panel_c_df.to_csv(panel_c_path, index=False)
    fig_s3_df.to_csv(fig_s3_path, index=False)
    fig4_df.to_csv(fig4_path, index=False)

    with configs_path.open("w", encoding="utf-8") as f:
        json.dump(json_ready(selected_configs), f, indent=2, ensure_ascii=False)

    with feature_path.open("w", encoding="utf-8") as f:
        json.dump(json_ready(FEATURE_SPACES), f, indent=2, ensure_ascii=False)

    run_meta = {
        "selection_mode": args.selection_mode,
        "seed": args.seed,
        "n_splits": N_SPLITS,
        "train_records": len(train),
        "test_records": len(test),
        "models": selected_models,
        "ori_feature_count": len(ORI_FEATURES),
        "pi_feature_count": len(PI_FEATURES),
        "elapsed_seconds": elapsed,
        "python": sys.version,
        "torch": None if torch is None else torch.__version__,
    }
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(json_ready(run_meta), f, indent=2, ensure_ascii=False)

    # Save the selected PI-GBR as the conservative operational baseline learner.
    if pi_gbr_estimator is not None:
        model_path = args.models_dir / "pi_gbr_model.joblib"
        joblib.dump(pi_gbr_estimator, model_path)
        pi_meta = {
            "model": "GBR",
            "representation": "PI",
            "target": TARGET,
            "features": PI_FEATURES,
            "selected_config": pi_gbr_config,
            "selection_mode": args.selection_mode,
            "seed": args.seed,
            "training_records": len(train),
            "heldout_records": len(test),
            "note": (
                "Model fitted on the full retained historical training segment only. "
                "The held-out test set was used only for final evaluation."
            ),
        }
        with (args.models_dir / "pi_gbr_metadata.json").open("w", encoding="utf-8") as f:
            json.dump(json_ready(pi_meta), f, indent=2, ensure_ascii=False)

    print("\nCompleted.", flush=True)
    print(f"  metrics : {metrics_path}", flush=True)
    print(f"  Table S3 configs: {panel_b_path}", flush=True)
    print(f"  Table S3 metrics: {panel_c_path}", flush=True)
    print(f"  Fig. S3 data: {fig_s3_path}", flush=True)
    print(f"  Fig. 4 data : {fig4_path}", flush=True)
    print(f"  CV folds : {cv_path}", flush=True)
    print(f"  predictions: {pred_path}", flush=True)
    if pi_gbr_estimator is not None:
        print(f"  PI-GBR  : {args.models_dir / 'pi_gbr_model.joblib'}", flush=True)
    print(f"  elapsed : {elapsed/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
