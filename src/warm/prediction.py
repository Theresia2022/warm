from __future__ import annotations

from pathlib import Path
from typing import Protocol
import copy
import random

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .config import PredictorConfig

EPS = 1e-8


class Predictor(Protocol):
    name: str
    def fit(self, train_df: pd.DataFrame, val_df: pd.DataFrame | None = None): ...
    def predict(self, df: pd.DataFrame) -> pd.DataFrame: ...


def _augment_features(df: pd.DataFrame) -> pd.DataFrame:
    """Memory-conscious feature augmentation.

    Formal library experiments may contain hundreds of thousands of candidate
    rows, so avoid a deep copy followed by a whole-frame ``replace`` operation.
    """
    out = df.copy(deep=False)
    for c in ["encoding_protocol_source", "encoding_protocol_replica"]:
        if c in out.columns:
            out[c] = out[c].fillna("unknown").astype(str).str.lower()

    def safe_ratio(a: str, b: str, name: str):
        if a not in out.columns or b not in out.columns:
            return
        num = pd.to_numeric(out[a], errors="coerce")
        den = pd.to_numeric(out[b], errors="coerce").mask(lambda x: x == 0)
        values = (num / den).mask(lambda x: ~np.isfinite(x))
        out[name] = values.astype(np.float32)

    safe_ratio("weight_replica", "weight_source", "rel_w")
    safe_ratio("height_replica", "height_source", "rel_h")
    safe_ratio("frame_rate_replica", "frame_rate_source", "rel_f")
    safe_ratio("gop_length_replica", "gop_length_source", "rel_g")

    if all(c in out.columns for c in ["weight_replica", "height_replica", "weight_source", "height_source"]):
        numerator = pd.to_numeric(out["weight_replica"], errors="coerce") * pd.to_numeric(out["height_replica"], errors="coerce")
        denominator = (
            pd.to_numeric(out["weight_source"], errors="coerce") * pd.to_numeric(out["height_source"], errors="coerce")
        ).mask(lambda x: x == 0)
        values = (numerator / denominator).mask(lambda x: ~np.isfinite(x))
        out["rel_pixels"] = values.astype(np.float32)
    return out


FULL_NUMERIC_CANDIDATES = [
    # Match the historical predictor feature contract. Ground-truth outcome
    # columns serving_cost_replica/storage_cost_replica/serving_speedup_value/storage_overhead_radio are intentionally excluded.
    "height_source", "weight_source", "gop_length_source", "frame_rate_source", "storage_cost_source", "bit_rate_key_frame_source", "bit_rate_non-key_frame_source", "encoding_params",
    "height_replica", "weight_replica", "gop_length_replica", "frame_rate_replica",
    # Leakage-free derived source/candidate specification ratios.
    "rel_w", "rel_h", "rel_f", "rel_g", "rel_pixels",
]
FULL_CATEGORICAL_CANDIDATES = ["encoding_protocol_source", "encoding_protocol_replica"]
CONFIG_ONLY_NUMERIC = ["height_replica", "weight_replica", "frame_rate_replica", "gop_length_replica", "encoding_params"]
CONFIG_ONLY_CATEGORICAL = ["encoding_protocol_replica"]


def _feature_columns_from_aug(aug: pd.DataFrame, profile: str = "full") -> tuple[list[str], list[str]]:
    if profile == "full":
        numeric_pool = FULL_NUMERIC_CANDIDATES
        categorical_pool = FULL_CATEGORICAL_CANDIDATES
    elif profile == "config_only":
        numeric_pool = CONFIG_ONLY_NUMERIC
        categorical_pool = CONFIG_ONLY_CATEGORICAL
    else:
        raise ValueError(f"unknown feature profile: {profile}")
    numeric = [c for c in numeric_pool if c in aug.columns and not aug[c].isna().all()]
    categorical = [c for c in categorical_pool if c in aug.columns]
    if not numeric and not categorical:
        raise ValueError(f"no usable predictor features for profile={profile}")
    return numeric, categorical


def _feature_columns(df: pd.DataFrame, profile: str = "full") -> tuple[list[str], list[str]]:
    return _feature_columns_from_aug(_augment_features(df), profile)


class _SharedFNN(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: tuple[int, ...], head_dim: int, dropout: float):
        super().__init__()
        layers: list[nn.Module] = []
        d = input_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)])
            d = h
        self.backbone = nn.Sequential(*layers)
        self.storage_head = nn.Sequential(nn.Linear(d, head_dim), nn.ReLU(), nn.Linear(head_dim, 1))
        self.decode_head = nn.Sequential(nn.Linear(d, head_dim), nn.ReLU(), nn.Linear(head_dim, 1))

    def forward(self, x):
        h = self.backbone(x)
        return torch.cat([self.storage_head(h), self.decode_head(h)], dim=1)


class MultiTaskFNNPredictor:
    name = "multi_task_fnn"

    def __init__(
        self,
        cfg: PredictorConfig | None = None,
        device: str | None = None,
        feature_profile: str = "full",
    ):
        self.cfg = cfg or PredictorConfig()
        self.feature_profile = feature_profile
        self.name = "multi_task_fnn" if feature_profile == "full" else f"multi_task_fnn_{feature_profile}"
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.numeric_cols: list[str] = []
        self.categorical_cols: list[str] = []
        self.feature_columns_: list[str] = []
        self.num_mean: dict[str, float] = {}
        self.num_std: dict[str, float] = {}
        self.target_mu = np.zeros(2, dtype=np.float64)
        self.target_sigma = np.ones(2, dtype=np.float64)
        self.calibration_a = np.ones(2, dtype=np.float64)
        self.calibration_b = np.zeros(2, dtype=np.float64)
        self.residual_std = np.zeros(2, dtype=np.float64)
        self.model: _SharedFNN | None = None

    def _seed(self):
        random.seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)
        torch.manual_seed(self.cfg.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.cfg.seed)

    def _fit_feature_transform(self, df: pd.DataFrame) -> np.ndarray:
        aug = _augment_features(df)
        self.numeric_cols, self.categorical_cols = _feature_columns_from_aug(aug, self.feature_profile)
        x = pd.DataFrame(index=aug.index)
        for c in self.numeric_cols:
            vals = pd.to_numeric(aug[c], errors="coerce").astype(np.float32)
            mean = float(vals.mean()) if np.isfinite(vals.mean()) else 0.0
            std = float(vals.std(ddof=0)) if np.isfinite(vals.std(ddof=0)) else 1.0
            if std <= EPS:
                std = 1.0
            self.num_mean[c], self.num_std[c] = mean, std
            x[c] = vals.fillna(mean).astype(np.float32)
        if self.categorical_cols:
            cats = pd.get_dummies(
                aug[self.categorical_cols].fillna("unknown").astype(str),
                prefix=self.categorical_cols,
                dtype=np.float32,
            )
            x = pd.concat([x, cats], axis=1)
        self.feature_columns_ = list(x.columns)
        for c in self.numeric_cols:
            if c in x.columns:
                x[c] = (x[c] - self.num_mean[c]) / self.num_std[c]
        return x.to_numpy(np.float32)

    def _transform_features(self, df: pd.DataFrame) -> np.ndarray:
        aug = _augment_features(df)
        x = pd.DataFrame(index=aug.index)
        for c in self.numeric_cols:
            vals = pd.to_numeric(aug[c], errors="coerce") if c in aug.columns else pd.Series(np.nan, index=aug.index)
            x[c] = vals.fillna(self.num_mean[c]).astype(np.float32)
        if self.categorical_cols:
            existing = [c for c in self.categorical_cols if c in aug.columns]
            cat_frame = pd.DataFrame(index=aug.index)
            for c in self.categorical_cols:
                cat_frame[c] = aug[c].fillna("unknown").astype(str) if c in existing else "unknown"
            cats = pd.get_dummies(cat_frame, prefix=self.categorical_cols, dtype=np.float32)
            x = pd.concat([x, cats], axis=1)
        x = x.reindex(columns=self.feature_columns_, fill_value=0.0)
        for c in self.numeric_cols:
            if c in x.columns:
                x[c] = (x[c] - self.num_mean[c]) / self.num_std[c]
        return x.to_numpy(np.float32)

    def _targets(self, df: pd.DataFrame, fit: bool = False) -> np.ndarray:
        y = np.column_stack([
            np.log(np.maximum(pd.to_numeric(df["storage_overhead_radio"], errors="coerce").to_numpy(float), EPS)),
            np.log(np.maximum(pd.to_numeric(df["serving_speedup_value"], errors="coerce").to_numpy(float), EPS)),
        ])
        if fit:
            self.target_mu = np.nanmean(y, axis=0)
            self.target_sigma = np.nanstd(y, axis=0)
            self.target_sigma = np.where(self.target_sigma > EPS, self.target_sigma, 1.0)
        return ((y - self.target_mu) / self.target_sigma).astype(np.float32)

    def fit(self, train_df: pd.DataFrame, val_df: pd.DataFrame | None = None):
        self._seed()
        train = train_df.dropna(subset=["storage_overhead_radio", "serving_speedup_value"]).reset_index(drop=True)
        if len(train) == 0:
            raise ValueError("no valid training rows")
        X = self._fit_feature_transform(train)
        y = self._targets(train, fit=True)

        if val_df is None or len(val_df) == 0:
            split = max(1, int(len(train) * 0.85))
            X_train, X_val = X[:split], X[split:]
            y_train, y_val = y[:split], y[split:]
            if len(X_val) == 0:
                X_val, y_val = X_train, y_train
            calibration_df = None
        else:
            X_train, y_train = X, y
            X_val = self._transform_features(val_df)
            y_val = self._targets(val_df, fit=False)
            calibration_df = val_df

        self.model = _SharedFNN(
            X_train.shape[1], self.cfg.hidden_dims, self.cfg.head_dim, self.cfg.dropout
        ).to(self.device)
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.cfg.learning_rate, weight_decay=self.cfg.weight_decay
        )
        loss_fn = nn.HuberLoss(delta=self.cfg.huber_delta, reduction="none")
        ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
        loader = DataLoader(ds, batch_size=min(self.cfg.batch_size, len(ds)), shuffle=True)
        Xv = torch.from_numpy(X_val).to(self.device)
        yv = torch.from_numpy(y_val).to(self.device)

        best_state = copy.deepcopy(self.model.state_dict())
        best_val = float("inf")
        stale = 0
        for _epoch in range(self.cfg.epochs):
            self.model.train()
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad(set_to_none=True)
                pred = self.model(xb)
                raw = loss_fn(pred, yb)
                loss = (
                    self.cfg.storage_loss_weight * raw[:, 0].mean()
                    + self.cfg.decode_loss_weight * raw[:, 1].mean()
                )
                loss.backward()
                optimizer.step()
            self.model.eval()
            with torch.no_grad():
                pv = self.model(Xv)
                raw = loss_fn(pv, yv)
                val_loss = float((
                    self.cfg.storage_loss_weight * raw[:, 0].mean()
                    + self.cfg.decode_loss_weight * raw[:, 1].mean()
                ).cpu())
            if val_loss + 1e-8 < best_val:
                best_val = val_loss
                best_state = copy.deepcopy(self.model.state_dict())
                stale = 0
            else:
                stale += 1
                if stale >= self.cfg.patience:
                    break
        self.model.load_state_dict(best_state)

        # Calibration is fit ONLY on the calibration subset of train.json.
        if calibration_df is not None and len(calibration_df) > 1:
            z_true = self._targets(calibration_df, fit=False).astype(float)
            z_pred = self._predict_standardized(calibration_df)
            for j in range(2):
                if np.std(z_pred[:, j]) > EPS:
                    a, b = np.polyfit(z_pred[:, j], z_true[:, j], deg=1)
                else:
                    a, b = 1.0, float(np.mean(z_true[:, j] - z_pred[:, j]))
                self.calibration_a[j], self.calibration_b[j] = float(a), float(b)
            z_cal = z_pred * self.calibration_a + self.calibration_b
            raw_true = np.exp(z_true * self.target_sigma + self.target_mu)
            raw_pred = np.exp(z_cal * self.target_sigma + self.target_mu)
            self.residual_std = np.nanstd(raw_true - raw_pred, axis=0)
        return self

    def _predict_standardized(self, df: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("predictor not fitted")
        X_np = self._transform_features(df)
        self.model.eval()
        chunks = []
        with torch.no_grad():
            for start in range(0, len(X_np), max(self.cfg.batch_size, 1)):
                xb = torch.from_numpy(X_np[start:start + self.cfg.batch_size]).to(self.device)
                chunks.append(self.model(xb).cpu().numpy())
        return np.concatenate(chunks, axis=0) if chunks else np.empty((0, 2))

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        z = self._predict_standardized(df).astype(float)
        z = z * self.calibration_a + self.calibration_b
        raw = np.exp(z * self.target_sigma + self.target_mu)
        out = df.copy()
        out["storage_overhead_radio_pred"] = np.clip(raw[:, 0], EPS, None)
        out["serving_speedup_value_pred"] = np.clip(raw[:, 1], EPS, None)
        out["storage_overhead_radio_std"] = float(self.residual_std[0])
        out["serving_speedup_value_std"] = float(self.residual_std[1])
        return out

    def save(self, path: str | Path) -> None:
        if self.model is None:
            raise RuntimeError("predictor not fitted")
        payload = {
            "cfg": self.cfg.__dict__,
            "feature_profile": self.feature_profile,
            "numeric_cols": self.numeric_cols,
            "categorical_cols": self.categorical_cols,
            "feature_columns": self.feature_columns_,
            "num_mean": self.num_mean,
            "num_std": self.num_std,
            "target_mu": self.target_mu,
            "target_sigma": self.target_sigma,
            "calibration_a": self.calibration_a,
            "calibration_b": self.calibration_b,
            "residual_std": self.residual_std,
            "state_dict": self.model.state_dict(),
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, path)



def make_predictor(kind: str = "fnn", cfg: PredictorConfig | None = None, seed: int = 42) -> Predictor:
    """Construct the single WARM cost estimator; no predictor-baseline alternatives."""
    if kind.lower() not in {"fnn", "warm", "multi_task_fnn", "fnn_full"}:
        raise ValueError("Public WARM includes only the FNN cost estimator")
    pcfg = copy.deepcopy(cfg or PredictorConfig(seed=seed))
    pcfg.seed = seed
    return MultiTaskFNNPredictor(pcfg, feature_profile="full")
