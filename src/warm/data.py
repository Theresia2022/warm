"""JSON statistical-data loaders for WARM.

Published semantic field names are used unchanged throughout the codebase.
VID and CID are persistent string identifiers; cand_idx is an array index.
Catalog observations are kept separate from predicted planning costs.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Iterable

import numpy as np
import pandas as pd

KB_TO_MB = 1.0 / 1024.0
US_TO_S = 1e-6

# Deliberately preserve the public 'weight' and 'radio' spellings.
PUBLIC_FIELDS = frozenset({
    "VID", "CID",
    "height_source", "weight_source", "frame_rate_source",
    "gop_length_source", "bit_rate_key_frame_source", "bit_rate_non-key_frame_source",
    "storage_cost_source", "serving_cost_source", "frame_nums_source",
    "encoding_protocol_source", "height_replica", "weight_replica",
    "frame_rate_replica", "gop_length_replica", "encoding_protocol_replica",
    "encoding_params", "serving_speedup_value", "storage_overhead_radio",
    "storage_cost_replica", "serving_cost_replica",
})

ORIGINAL_REQUIRED = (
    "VID", "height_source", "weight_source", "frame_rate_source", "gop_length_source", "bit_rate_key_frame_source", "bit_rate_non-key_frame_source",
    "encoding_protocol_source", "storage_cost_source", "serving_cost_source", "frame_nums_source",
)
COPIES_REQUIRED = (
    "VID", "CID", "height_replica", "weight_replica", "frame_rate_replica", "gop_length_replica", "encoding_protocol_replica",
    "encoding_params", "serving_speedup_value", "storage_overhead_radio", "frame_nums_source",
)
PREDICTOR_FEATURE_REQUIRED = (
    "height_source", "weight_source", "frame_rate_source", "gop_length_source", "bit_rate_key_frame_source", "bit_rate_non-key_frame_source", "storage_cost_source", "encoding_protocol_source",
    "height_replica", "weight_replica", "frame_rate_replica", "gop_length_replica", "encoding_protocol_replica", "encoding_params",
)
PREDICTOR_REQUIRED = ("VID",) + PREDICTOR_FEATURE_REQUIRED + (
    "serving_speedup_value", "storage_overhead_radio", "storage_cost_replica", "serving_cost_replica",
)
VIDEO_ID_PATTERN = re.compile(r"^V_E_[A-Z2-7]{8,10}$")
PREDICTION_ID_PATTERN = re.compile(r"^V_T_[A-Z2-7]{8,10}$")
REPLICA_ID_PATTERN = re.compile(r"^C_E_[A-Z2-7]{8,10}_[A-Z2-7]{8,10}$")


@dataclass(frozen=True)
class DatasetBundle:
    originals: pd.DataFrame
    copies: pd.DataFrame


@dataclass(frozen=True)
class PredictorDataBundle:
    train: pd.DataFrame
    test: pd.DataFrame


def _data_root(path: str | Path) -> Path:
    """Accept project root, its ``data/`` directory, or either child directory."""
    path = Path(path).resolve()
    if path.is_file():
        path = path.parent
    if path.name in {"experiment", "prediciton"}:
        return path.parent
    if (path / "experiment").is_dir() or (path / "prediciton").is_dir():
        return path
    if (path / "data").is_dir():
        return path / "data"
    return path


def read_table(path: str | Path) -> pd.DataFrame:
    """Read a UTF-8 JSON array using the published semantic field names."""
    path = Path(path)
    if path.suffix.lower() != ".json":
        raise ValueError(f"WARM accepts JSON statistics only, not {path.name!r}")
    with path.open("r", encoding="utf-8-sig") as file:
        try:
            records = json.load(file)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(records, list):
        raise ValueError(f"{path}: expected a JSON array of objects")
    if not all(isinstance(row, dict) for row in records):
        raise ValueError(f"{path}: each array item must be a JSON object")
    df = pd.DataFrame.from_records(records)
    # Retain only documented JSON fields without internal renaming.
    return df[[name for name in df.columns if name in PUBLIC_FIELDS]].copy()



def _validate(df: pd.DataFrame, required: Iterable[str], name: str) -> None:
    missing = [key for key in required if key not in df.columns]
    if missing:
        raise ValueError(f"{name} missing JSON fields: {missing}")


def _id_column(df: pd.DataFrame, field: str, pattern, label: str) -> None:
    if not df[field].map(lambda x: isinstance(x, str) and bool(pattern.fullmatch(x))).all():
        raise ValueError(f"{label}: {field} must be an expected string identifier")


def _numeric(df: pd.DataFrame, names: Iterable[str], label: str) -> None:
    for field in names:
        if field not in df:
            continue
        df[field] = pd.to_numeric(df[field], errors="coerce")
        if df[field].isna().any() or not np.isfinite(df[field].to_numpy(float)).all():
            raise ValueError(f"{label}: {field} contains missing/non-numeric values")


def normalize_codec(value) -> str:
    return str(value).strip().lower().replace("h.264", "h264").replace("h.265", "hevc")


def read_source_statistics(data_dir: str | Path, expected_count: int | None = 1500) -> pd.DataFrame:
    path = Path(data_dir)
    if path.is_dir():
        path = _data_root(path) / "experiment" / "source_videos.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing source-video statistics JSON: {path}")
    df = read_table(path)
    _validate(df, ORIGINAL_REQUIRED, "source_videos.json")
    _id_column(df, "VID", VIDEO_ID_PATTERN, str(path))
    if df["VID"].duplicated().any():
        raise ValueError("source_videos.json contains duplicate VID values")
    _numeric(df, ("height_source", "weight_source", "frame_rate_source", "gop_length_source", "bit_rate_key_frame_source", "bit_rate_non-key_frame_source", "storage_cost_source", "serving_cost_source", "frame_nums_source"), path.name)
    for name in ("height_source", "weight_source", "frame_rate_source", "storage_cost_source", "serving_cost_source"):
        if (df[name] <= 0).any():
            raise ValueError(f"source_videos.json: {name} must be positive")
    df["encoding_protocol_source"] = df["encoding_protocol_source"].map(normalize_codec)
    if (df["encoding_protocol_source"] == "").any():
        raise ValueError("source_videos.json has empty codec")
    df["source_storage_mb"] = df["storage_cost_source"] * KB_TO_MB
    df["source_decode_s"] = df["serving_cost_source"] * US_TO_S
    if expected_count is not None and len(df) != expected_count:
        raise ValueError(f"Expected {expected_count} distinct videos; found {len(df)}")
    return df.reset_index(drop=True)


def _replica_directory(data_dir: str | Path) -> Path:
    return _data_root(data_dir) / "experiment" / "replicas"


def read_replica_statistics(
    data_dir: str | Path,
    originals: pd.DataFrame | None = None,
    *,
    require_complete: bool = True,
) -> pd.DataFrame:
    """Load per-video replica JSON arrays, retaining string CID/VID associations.

    The public planner uses only the *configuration* columns from these rows;
    serving_speedup_value and storage_overhead_radio measurements are not planning-time predictions.
    """
    if originals is None:
        originals = read_source_statistics(data_dir, expected_count=None)
    folder = _replica_directory(data_dir)
    if not folder.is_dir():
        raise FileNotFoundError(f"Missing replica JSON directory: {folder}")
    expected_files = {f"replica_{video_id}.json": video_id for video_id in originals["VID"]}
    available = {p.name: p for p in folder.glob("replica_V_E_*.json")}
    if require_complete:
        missing = sorted(set(expected_files) - set(available))
        extra = sorted(set(available) - set(expected_files))
        if missing or extra:
            raise ValueError(f"Replica JSON file set mismatch: missing={missing[:4]} ({len(missing)}), extra={extra[:4]} ({len(extra)})")
    all_records = []
    for filename, video_id in sorted(expected_files.items()):
        if filename not in available:
            continue
        part = read_table(available[filename])
        if part.empty:
            continue
        _validate(part, COPIES_REQUIRED, filename)
        _id_column(part, "VID", VIDEO_ID_PATTERN, filename)
        _id_column(part, "CID", REPLICA_ID_PATTERN, filename)
        if not part["VID"].eq(video_id).all():
            raise ValueError(f"{filename}: a replica refers to a different VID")
        # Link must use the exact suffix from its parent video ID.
        if not part["CID"].str.startswith("C_E_" + video_id[4:] + "_").all():
            raise ValueError(f"{filename}: CID does not match the owning VID")
        all_records.extend(part.to_dict("records"))
    if not all_records:
        return pd.DataFrame(columns=list(COPIES_REQUIRED))
    copies = pd.DataFrame.from_records(all_records)
    if copies["CID"].duplicated().any():
        raise ValueError("Multiple replica records have the same CID")
    return copies.reset_index(drop=True)


def normalize_tables(originals: pd.DataFrame, copies: pd.DataFrame) -> DatasetBundle:
    """Attach measured costs for testing/catalog analysis (NOT WARM plan)."""
    originals = originals.copy()
    copies = copies.copy()
    _validate(originals, ORIGINAL_REQUIRED, "source videos")
    _validate(copies, COPIES_REQUIRED, "replicas")
    _id_column(originals, "VID", VIDEO_ID_PATTERN, "source videos")
    _id_column(copies, "VID", VIDEO_ID_PATTERN, "replicas")
    _id_column(copies, "CID", REPLICA_ID_PATTERN, "replicas")
    if originals["VID"].duplicated().any() or copies["CID"].duplicated().any():
        raise ValueError("Duplicate VID or CID")

    _numeric(originals, ("height_source", "weight_source", "frame_rate_source", "gop_length_source", "bit_rate_key_frame_source", "bit_rate_non-key_frame_source", "storage_cost_source", "serving_cost_source", "frame_nums_source"), "source videos")
    _numeric(copies, ("height_replica", "weight_replica", "frame_rate_replica", "gop_length_replica", "serving_speedup_value", "storage_overhead_radio", "frame_nums_source"), "replicas")
    # Optional copy-side source descriptors can be missing and are populated
    # from the canonical source statistics below.
    for field in ("storage_cost_source", "serving_cost_source", "bit_rate_key_frame_source", "bit_rate_non-key_frame_source", "height_source", "weight_source", "frame_rate_source", "gop_length_source"):
        if field in copies:
            copies[field] = pd.to_numeric(copies[field], errors="coerce")
    # Per-replica source attributes may be missing; recover them from the
    # authoritative source table instead of the unrelated predictor files.
    source_cols = ("height_source", "weight_source", "frame_rate_source", "gop_length_source", "bit_rate_key_frame_source", "bit_rate_non-key_frame_source", "storage_cost_source", "serving_cost_source", "frame_nums_source", "encoding_protocol_source")
    srcmap = originals.set_index("VID")
    for field in source_cols:
        if field not in copies:
            copies[field] = copies["VID"].map(srcmap[field])
        elif field != "encoding_protocol_source":
            copies[field] = copies[field].fillna(copies["VID"].map(srcmap[field]))
    copies["encoding_protocol_source"] = copies["encoding_protocol_source"].map(normalize_codec)
    copies["encoding_protocol_replica"] = copies["encoding_protocol_replica"].map(normalize_codec)
    originals["encoding_protocol_source"] = originals["encoding_protocol_source"].map(normalize_codec)
    if copies[["VID", "CID", "serving_speedup_value", "storage_overhead_radio", "serving_cost_source", "storage_cost_source", "bit_rate_key_frame_source", "bit_rate_non-key_frame_source"]].isna().any().any():
        raise ValueError("Replica rows contain missing source links or costs")
    if not copies["VID"].isin(originals["VID"]).all():
        raise ValueError("Replica VID not present in source_videos.json")
    if (copies["serving_speedup_value"] <= 0).any() or (copies["storage_overhead_radio"] <= 0).any():
        raise ValueError("Replica serving_speedup_value/storage_overhead_radio must be positive")
    copies["true_storage_mb"] = copies["storage_cost_source"] * copies["storage_overhead_radio"] * KB_TO_MB
    copies["true_decode_s"] = copies["serving_cost_source"] * US_TO_S / copies["serving_speedup_value"]
    originals["source_storage_mb"] = originals["storage_cost_source"] * KB_TO_MB
    originals["source_decode_s"] = originals["serving_cost_source"] * US_TO_S
    return DatasetBundle(originals.reset_index(drop=True), copies.reset_index(drop=True))


def load_dataset(data_dir: str | Path) -> DatasetBundle:
    """Load real measured JSON statistics, separate from prediction training."""
    originals = read_source_statistics(data_dir, expected_count=None)
    copies = read_replica_statistics(data_dir, originals)
    if copies.empty:
        raise ValueError("No replica statistics found in per-video JSON files")
    return normalize_tables(originals, copies)


def _normalize_predictor_table(df: pd.DataFrame, name: str) -> pd.DataFrame:
    out = df.copy()
    _validate(out, PREDICTOR_REQUIRED, name)
    _id_column(out, "VID", PREDICTION_ID_PATTERN, name)
    _numeric(out, [c for c in PREDICTOR_REQUIRED if c not in ("VID", "encoding_protocol_source", "encoding_protocol_replica")], name)
    for field in ("encoding_protocol_source", "encoding_protocol_replica"):
        out[field] = out[field].map(normalize_codec)
    if (out[["serving_speedup_value", "storage_overhead_radio", "storage_cost_replica", "serving_cost_replica", "storage_cost_source"]] <= 0).any().any():
        raise ValueError(f"{name} contains nonpositive measured size/time/targets")
    return out.reset_index(drop=True)


def load_predictor_train_test(
    data_dir: str | Path,
    train_file: str = "train.json",
    test_file: str = "test.json",
) -> PredictorDataBundle:
    """Read *only* prediciton/train.json and test.json; no replica-catalog leakage."""
    folder = _data_root(data_dir) / "prediciton"
    train_path, test_path = folder / train_file, folder / test_file
    if not train_path.is_file():
        raise FileNotFoundError(f"Missing prediction training JSON: {train_path}")
    if not test_path.is_file():
        raise FileNotFoundError(f"Missing prediction test JSON: {test_path}")
    train = _normalize_predictor_table(read_table(train_path), train_path.name)
    test = _normalize_predictor_table(read_table(test_path), test_path.name)
    overlap = set(train["VID"]) & set(test["VID"])
    if overlap:
        raise ValueError(f"train.json/test.json VID namespaces overlap: {list(overlap)[:3]}")
    return PredictorDataBundle(train=train, test=test)


def split_predictor_train_calibration(
    train_df: pd.DataFrame,
    calibration_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Deterministic VIDEO-level split using pseudonymous train VID strings."""
    if not (0.0 < calibration_ratio < 0.5):
        raise ValueError("calibration_ratio must be in (0, 0.5)")
    df = train_df.reset_index(drop=True)
    if "VID" not in df or df["VID"].nunique() < 2:
        raise ValueError("train.json needs at least two distinct V_T video IDs for calibration")
    rng = np.random.default_rng(seed)
    vids = df["VID"].dropna().unique().copy()
    rng.shuffle(vids)
    n_cal = max(1, min(len(vids) - 1, int(round(len(vids) * calibration_ratio))))
    cal_vids = set(vids[:n_cal])
    cal = df[df["VID"].isin(cal_vids)].copy()
    fit = df[~df["VID"].isin(cal_vids)].copy()
    if fit.empty or cal.empty:
        raise ValueError("train.json is too small for fit/calibration split")
    return fit.reset_index(drop=True), cal.reset_index(drop=True)


def split_video_ids(
    originals: pd.DataFrame,
    train_ratio: float = 0.60,
    val_ratio: float = 0.20,
    test_ratio: float = 0.20,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-9:
        raise ValueError("split ratios must sum to 1")
    vids = originals["VID"].dropna().unique().copy()
    np.random.default_rng(seed).shuffle(vids)
    n = len(vids)
    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))
    train, val = vids[:n_train], vids[n_train:n_train + n_val]
    test = vids[n_train + n_val:]
    if len(test) == 0 and n:
        test = vids[-1:]
    return np.asarray(train), np.asarray(val), np.asarray(test)


def subset_by_vid(df: pd.DataFrame, vids: Iterable) -> pd.DataFrame:
    return df[df["VID"].isin(set(vids))].reset_index(drop=True)
