"""Statistics-only WARM algorithm study, NOT paper-results reproduction.

The public input is source-video statistics plus predictor train/validation
records. There is no source video byte stream, per-candidate measured catalog,
physical encoding, query-serving runtime, or actual-size materialization here.
Outputs are *predicted plans* only and must not be presented as generated replicas
or empirically measured serving results.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from .config import CandidateConfig, ExperimentConfig, OptimizerConfig
from .data import (
    KB_TO_MB, US_TO_S, load_predictor_train_test,
    read_source_statistics as _read_source_json,
    read_replica_statistics, split_predictor_train_calibration,
)
from .identifiers import replica_id
from .coupling import (
    OptimizationInstance, attach_singleton_gain,
    build_compatibility, prune_dominated_candidates,
)
from .prediction import make_predictor
from .optimization import (
    LazyBudgetedGreedy, EagerBudgetedGreedy, planner_for,
)
from .baselines import VStoreCoalescingPlanner, PriorZhaoVersionPlanner
from .video_prior_baselines import PriorVSAPlanner, PriorVSSCachePlanner
from .workload import build_workload
from .naming import DISPLAY_NAME, method_id, method_name


def read_source_statistics(path: str | Path, expected_count: int | None = 1500) -> pd.DataFrame:
    """Load data/experiment/source_videos.json (public entry-point alias)."""
    return _read_source_json(path, expected_count=expected_count)


def enumerate_candidate_specs(originals: pd.DataFrame, cfg: CandidateConfig) -> pd.DataFrame:
    """Enumerate logical *specifications*, not pre-generated physical replicas.

    As source-level statistics alone cannot reconstruct the paper's measured
    438,984-candidate catalog, this routine enumerates configurations from the
    published knobs; its count may differ from the original experiment.
    """
    rows = []
    for src in originals.itertuples(index=False):
        w1, h1, f1 = float(src.weight_source), float(src.height_source), float(src.frame_rate_source)
        widths = sorted(set(float(x) for x in cfg.widths) | {w1})
        frame_rates = sorted(set(float(x) for x in cfg.frame_rates) | {f1})
        for w in widths:
            if cfg.logical_only_not_richer_than_source and w > w1 + 1e-8:
                continue
            h = int(h1) if abs(w - w1) < 1e-8 else max(4, int(w / w1 * h1 / 4) * 4)
            if cfg.logical_only_not_richer_than_source and h > h1 + 1e-8:
                continue
            for fps in frame_rates:
                if cfg.logical_only_not_richer_than_source and fps > f1 + 1e-8:
                    continue
                for quality in cfg.qualities:
                    for codec in cfg.codecs:
                        for encoding_params in cfg.params:
                            rows.append((src.VID, int(h), int(w), float(fps), float(quality), str(codec), int(encoding_params)))
    if not rows:
        raise ValueError("No candidate specifications generated from source statistics")
    specs = pd.DataFrame(rows, columns=["VID", "height_replica", "weight_replica", "frame_rate_replica", "gop_length_replica", "encoding_protocol_replica", "encoding_params"])
    source_cols = [c for c in ("VID", "height_source", "weight_source", "frame_rate_source", "gop_length_source", "storage_cost_source", "serving_cost_source", "encoding_protocol_source", "bit_rate_key_frame_source", "bit_rate_non-key_frame_source", "frame_nums_source") if c in originals.columns]
    specs = specs.merge(originals[source_cols], on="VID", how="left", validate="many_to_one")
    def compact(value):
        number = float(value)
        return int(number) if number.is_integer() else number

    specs.insert(0, "CID", [
        replica_id(str(row.VID), compact(row.height_replica), compact(row.weight_replica),
                   compact(row.frame_rate_replica), compact(row.gop_length_replica), str(row.encoding_protocol_replica),
                   compact(row.encoding_params))
        for row in specs.itertuples(index=False)
    ])
    specs.insert(0, "cand_idx", np.arange(len(specs), dtype=np.int64))
    return specs


def candidate_specs_from_replica_json(
    data_dir: str | Path, originals: pd.DataFrame, cfg: CandidateConfig,
) -> pd.DataFrame:
    """Use published replica JSON *configurations*, never measured serving_speedup_value/storage_overhead_radio.

    Keeping the catalog configurations makes this useful for studying the
    original candidate universe. Physical costs are always obtained from the
    predictor, not from the replica catalog's measurements.
    """
    replicas = read_replica_statistics(data_dir, originals, require_complete=True)
    keys = ("VID", "CID", "height_replica", "weight_replica", "frame_rate_replica", "gop_length_replica", "encoding_protocol_replica", "encoding_params")
    if replicas.empty:
        raise ValueError("No replica configurations found in replica JSON files")
    specs = replicas[list(keys)].copy()
    if specs.duplicated("CID").any():
        raise ValueError("Repeated CID in replica configuration inputs")
    for key in ("height_replica", "weight_replica", "frame_rate_replica", "gop_length_replica", "encoding_params"):
        specs[key] = pd.to_numeric(specs[key], errors="raise")
        if not np.isfinite(specs[key].to_numpy(float)).all():
            raise ValueError(f"Replica JSON has invalid {key} values")
    if (specs[["height_replica", "weight_replica", "frame_rate_replica", "gop_length_replica"]] <= 0).any().any():
        raise ValueError("Replica configuration dimensions/GOP must be positive")
    specs["encoding_protocol_replica"] = specs["encoding_protocol_replica"].astype(str).str.lower()
    if specs["encoding_protocol_replica"].isin(("", "none", "nan")).any():
        raise ValueError("Replica JSON has empty encoding protocol")
    source_cols = [c for c in (
        "VID", "height_source", "weight_source", "frame_rate_source", "gop_length_source", "storage_cost_source", "serving_cost_source",
        "encoding_protocol_source", "bit_rate_key_frame_source", "bit_rate_non-key_frame_source", "frame_nums_source",
    ) if c in originals.columns]
    specs = specs.merge(originals[source_cols], on="VID", how="left", validate="many_to_one")
    if specs["height_source"].isna().any():
        raise ValueError("Replica configuration references an unknown source VID")
    if cfg.logical_only_not_richer_than_source:
        specs = specs[
            (specs["weight_replica"] <= specs["weight_source"] + 1e-9)
            & (specs["height_replica"] <= specs["height_source"] + 1e-9)
            & (specs["frame_rate_replica"] <= specs["frame_rate_source"] + 1e-9)
        ].copy()
    specs = specs.reset_index(drop=True)
    if specs.empty:
        raise ValueError("No compatible candidate configurations after filtering")
    specs.insert(0, "cand_idx", np.arange(len(specs), dtype=np.int64))
    assert not {"serving_speedup_value", "storage_overhead_radio", "storage_cost_replica", "serving_cost_replica", "true_storage_mb", "true_decode_s"} & set(specs.columns)
    return specs


def add_estimated_costs(predictions: pd.DataFrame, opt: OptimizerConfig) -> pd.DataFrame:
    """Create planning costs from FNN outputs only; no true-cost columns."""
    out = predictions.copy()
    if not {"storage_overhead_radio_pred", "serving_speedup_value_pred", "storage_cost_source", "serving_cost_source"}.issubset(out.columns):
        raise ValueError("Missing model estimates or source statistics")
    s = out["storage_cost_source"].to_numpy(float) * KB_TO_MB
    d = out["serving_cost_source"].to_numpy(float) * US_TO_S
    storage_ratio = np.maximum(out["storage_overhead_radio_pred"].to_numpy(float), 1e-12)
    serving_speedup = np.maximum(out["serving_speedup_value_pred"].to_numpy(float), 1e-12)
    storage_ratio_std = np.maximum(out.get("storage_overhead_radio_std", pd.Series(0, index=out.index)).to_numpy(float), 0)
    serving_speedup_std = np.maximum(out.get("serving_speedup_value_std", pd.Series(0, index=out.index)).to_numpy(float), 0)
    out["pred_storage_mb"] = s * storage_ratio
    out["pred_decode_s"] = d / serving_speedup
    out["robust_storage_mb"] = np.maximum(s * (storage_ratio + opt.beta_storage * storage_ratio_std), 1e-12)
    out["robust_decode_s"] = d / np.maximum(serving_speedup - opt.beta_decode * serving_speedup_std, 1e-12)
    return out


def estimated_instance(
    specs_with_predictions: pd.DataFrame,
    requests: pd.DataFrame,
    budget_mb: float,
    cfg: ExperimentConfig,
    *,
    prune: bool | None = None,
    robust: bool = False,
) -> OptimizationInstance:
    opt = cfg.optimizer
    if robust:
        opt = replace(opt, beta_storage=cfg.robust_beta_for_noise, beta_decode=cfg.robust_beta_for_noise)
    cand = add_estimated_costs(specs_with_predictions, opt)
    comp = build_compatibility(cand, requests, cfg.workload)
    cand = attach_singleton_gain(cand, requests, comp, storage_col="robust_storage_mb", decode_col="robust_decode_s")
    do_prune = opt.enable_dominance_pruning if prune is None else bool(prune)
    if do_prune:
        cand, comp, _ = prune_dominated_candidates(cand, comp, "robust_storage_mb", "robust_decode_s")
    return OptimizationInstance(cand, requests.sort_values("request_idx").reset_index(drop=True), comp, float(budget_mb), "robust_storage_mb", "robust_decode_s", False)


def _planner_for_study(name: str, cfg: ExperimentConfig):
    if name in {"warm", "lazy_unpruned", "robust_cost"}:
        return LazyBudgetedGreedy(best_singleton=cfg.optimizer.enable_best_singleton, min_gain=cfg.optimizer.min_gain)
    if name == "eager_dynamic":
        return EagerBudgetedGreedy(best_singleton=cfg.optimizer.enable_best_singleton, min_gain=cfg.optimizer.min_gain)
    if name == "vstore_coalesce":
        return VStoreCoalescingPlanner()
    if name == "prior_vsa_dp":
        return PriorVSAPlanner(max_versions=cfg.prior_vsa.max_versions, capacity_bins=cfg.prior_vsa.capacity_bins)
    if name == "prior_vss_cache":
        return PriorVSSCachePlanner(history_queries=cfg.prior_vss.history_queries, per_video_budget_multiplier=cfg.prior_vss.per_video_budget_multiplier, redundancy_weight=cfg.prior_vss.redundancy_weight, trace_seed=cfg.prior_vss.trace_seed)
    if name == "prior_zhao_version":
        return PriorZhaoVersionPlanner()
    return planner_for(name)


def check_data(data_dir: str | Path, cfg: ExperimentConfig) -> dict:
    root = Path(data_dir)
    originals = read_source_statistics(root, cfg.library_video_count if cfg.require_exact_library_size else None)
    train_test = load_predictor_train_test(root, cfg.predictor_train_file, cfg.predictor_test_file)
    fit, calib = split_predictor_train_calibration(train_test.train, cfg.predictor.calibration_ratio, cfg.seed)
    overlap = len(set(train_test.train["VID"]) & set(originals["VID"]))
    replicas = read_replica_statistics(root, originals, require_complete=True)
    return {
        "scope": "schema and provenance check; not performance verification",
        "source_videos": int(len(originals)),
        "predictor_training_rows": int(len(train_test.train)),
        "predictor_fit_rows": int(len(fit)),
        "predictor_calibration_rows": int(len(calib)),
        "predictor_validation_rows": int(len(train_test.test)),
        "train_video_overlap_with_library": overlap,
        "replica_configuration_records": int(len(replicas)),
        "replica_files": int(len(originals)),
        "source_storage_mb": float(originals.source_storage_mb.sum()),
    }


def planning_output_directory(
    output_dir: str | Path, candidate_source: str, budget_ratio: float, seed: int,
) -> Path:
    """Separate experiments by candidate source, budget, and random seed."""
    from decimal import Decimal

    ratio = Decimal(str(budget_ratio)) * Decimal(100)
    pct = format(ratio, "f").rstrip("0").rstrip(".") if "." in format(ratio, "f") else format(ratio, "f")
    percentage = pct.replace(".", "p")
    return Path(output_dir) / f"{candidate_source}_budget_{percentage}pct_seed_{int(seed)}"


def plan_from_statistics(
    data_dir: str | Path,
    output_dir: str | Path,
    cfg: ExperimentConfig,
    selected_methods: list[str] | tuple[str, ...] = ("WARM",),
    budget_ratio: float = 0.10,
    historical_log: str | Path | None = None,
    candidate_source: str = "replica-json",
) -> pd.DataFrame:
    """Generate predicted selection plans. Does not encode, generate, or measure videos."""
    if not (0 < float(budget_ratio) <= 1):
        raise ValueError("budget_ratio must lie in (0, 1]")
    cfg.validate()
    root = Path(data_dir)
    originals = read_source_statistics(root, cfg.library_video_count if cfg.require_exact_library_size else None)
    pdat = load_predictor_train_test(root, cfg.predictor_train_file, cfg.predictor_test_file)
    fit, calibration = split_predictor_train_calibration(pdat.train, cfg.predictor.calibration_ratio, cfg.seed)
    predictor = make_predictor("fnn", cfg.predictor, cfg.seed)
    predictor.fit(fit, calibration)
    if candidate_source == "replica-json":
        candidate_specs = candidate_specs_from_replica_json(root, originals, cfg.candidate)
    elif candidate_source == "enumerated":
        candidate_specs = enumerate_candidate_specs(originals, cfg.candidate)
    else:
        raise ValueError("candidate_source must be 'replica-json' or 'enumerated'")
    predicted = predictor.predict(candidate_specs)
    requests = build_workload(originals, cfg.workload, cfg.seed, historical_log=historical_log)
    budget_mb = float(originals.source_storage_mb.sum() * budget_ratio)
    rows = []
    parsed = list(dict.fromkeys(method_id(m) for m in selected_methods))
    if not parsed:
        raise ValueError("Specify at least one method")
    run_dir = planning_output_directory(output_dir, candidate_source, budget_ratio, cfg.seed)
    selected_dir = run_dir / "selected_replicas"
    selected_dir.mkdir(parents=True, exist_ok=True)
    # A repeated run with a different method subset must not leave stale method files.
    for old_name in DISPLAY_NAME.values():
        old_file = selected_dir / f"{old_name}.json"
        if old_file.exists():
            old_file.unlink()

    for name in parsed:
        instance = estimated_instance(
            predicted, requests, budget_mb, cfg,
            prune=False if name == "lazy_unpruned" else None,
            robust=name == "robust_cost",
        )
        planner = _planner_for_study(name, cfg)
        result = planner.optimize(instance, name)
        display_name = method_name(name)
        planned = result.plan.copy()

        # The source-only reference never selects replicas, so no empty plan file.
        if name != "source_only":
            fields = [
                "selected_order", "VID", "CID", "height_replica",
                "weight_replica", "frame_rate_replica", "gop_length_replica",
                "encoding_protocol_replica", "encoding_params",
                "pred_storage_mb", "pred_decode_s",
            ]
            chosen = planned[[field for field in fields if field in planned]].copy()
            chosen = chosen.rename(columns={
                "pred_storage_mb": "estimated_storage_mb",
                "pred_decode_s": "estimated_serving_cost_s",
            })
            if "selected_order" in chosen.columns:
                chosen = chosen.sort_values("selected_order")
            chosen.insert(0, "status", "selected_not_generated")
            (selected_dir / f"{display_name}.json").write_text(
                chosen.to_json(orient="records", indent=2, force_ascii=False, double_precision=15) + "\n",
                encoding="utf-8",
            )

        rows.append({
            "method": display_name,
            "status": "estimated_planning_only",
            "source_videos": int(len(originals)),
            "candidate_specifications": int(len(candidate_specs)),
            "budget_ratio": float(budget_ratio),
            "budget_mb": float(budget_mb),
            "selected_specifications": int(len(planned)),
            "estimated_storage_mb": float(result.storage_used),
            "estimated_weighted_saving_s": float(result.predicted_objective),
            "planner_runtime_s": float(result.runtime_s),
            "marginal_evaluations": int(result.marginal_evaluations),
        })
    summaries = pd.DataFrame(rows)
    (run_dir / "planning_summary.json").write_text(
        summaries.to_json(orient="records", indent=2, force_ascii=False, double_precision=15) + "\n",
        encoding="utf-8",
    )
    summaries.attrs["output_directory"] = str(run_dir)
    return summaries
