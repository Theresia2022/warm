from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import yaml


@dataclass
class CandidateConfig:
    widths: tuple[int, ...] = (360, 540, 720, 1280, 1920, 3840)
    frame_rates: tuple[float, ...] = (15.0, 20.0, 30.0, 45.0, 60.0)
    qualities: tuple[float, ...] = (0.25, 0.33, 0.5, 1.0)
    codecs: tuple[str, ...] = ("h264", "hevc", "wmv")
    params: tuple[int, ...] = (1, 3, 5)
    logical_only_not_richer_than_source: bool = True


@dataclass
class WorkloadConfig:
    total_historical_queries: int = 1_000_000
    video_distribution: str = "zipf"
    video_alpha: float = 1.0
    request_distribution: str = "zipf"
    request_alpha: float = 0.5
    max_requests_per_video: int = 16
    request_widths: tuple[int, ...] = (360, 480, 540, 720, 960, 1280, 1600, 1920, 2560, 3840)
    request_frame_rates: tuple[float, ...] = (10.0, 15.0, 20.0, 24.0, 30.0, 45.0, 60.0)
    logical_coverage_only: bool = True
    strict_codec: bool = False
    quality_constraint: bool = False


@dataclass
class PredictorConfig:
    kind: str = "fnn"
    hidden_dims: tuple[int, ...] = (128, 64)
    head_dim: int = 32
    dropout: float = 0.10
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 2048
    epochs: int = 120
    patience: int = 12
    huber_delta: float = 1.0
    storage_loss_weight: float = 1.0
    decode_loss_weight: float = 1.0
    seed: int = 42
    calibration_ratio: float = 0.15


@dataclass
class OptimizerConfig:
    beta_storage: float = 0.0
    beta_decode: float = 0.0
    enable_dominance_pruning: bool = True
    enable_lazy: bool = True
    enable_best_singleton: bool = True
    min_gain: float = 1e-12

    # V5: prediction-based planning + hard true-storage commit guard.
    # Ranking/optimization still uses predicted/robust costs. After a replica
    # is generated in staging space, its measured true size determines whether
    # it can be committed to the persistent replica store.
    enforce_true_budget_at_materialization: bool = True


@dataclass
class PriorMVSConfig:
    # Configurable, deterministic VLDB-2000-inspired candidate screening.
    # This is a heuristic pool, not dominance pruning.
    pool_cap: int = 12000
    per_video_density: int = 2
    per_video_gain: int = 1
    pair_top_density: int = 24
    pair_top_gain: int = 24


@dataclass
class PriorVSAConfig:
    # Song et al. 2009: feasible version sets -> multiple-choice knapsack DP.
    # The original uses a small fixed set of quality levels. The replica catalog
    # has many more versions, so representatives are capped explicitly.
    max_versions: int = 4
    capacity_bins: int = 2048


@dataclass
class PriorVSSConfig:
    # Haynes et al. 2021: online admission and modified LRU cache.
    # Original per-video budget defaults to 10x initial video size.
    # A separate global budget enforces comparability to WARM.
    history_queries: int = 20_000
    per_video_budget_multiplier: float = 10.0
    redundancy_weight: float = 1.0
    trace_seed: int = 1217


@dataclass
class ServingConfig:
    workers: int = 8
    query_count: int = 50_000
    arrival_process: str = "poisson"
    offered_load: float = 0.80
    seed: int = 4242
    sla_seconds: float = 0.200
    routing_policy: str = "min_cost"


@dataclass
class ExperimentConfig:
    seed: int = 42

    # Predictor data are read only from data/prediciton/*.json.
    predictor_train_file: str = "train.json"
    predictor_test_file: str = "test.json"

    # The public real-data mode expects metadata for all 1500 source videos.
    library_video_count: int = 1500
    require_exact_library_size: bool = True

    budget_ratios: tuple[float, ...] = (0.02, 0.05, 0.10, 0.15, 0.20)
    methods: tuple[str, ...] = (
        "vstore_coalesce",
        "prior_vsa_dp",
        "prior_vss_cache",
        "prior_zhao_version",
        "source_only",
        "warm",
        "static_density",
        "popularity_first",
        "smallest_storage",
        "benefit_only",
    )

    # Original paper protocol parameters retained as configuration references.
    workload_seeds: tuple[int, ...] = (41, 42, 43, 44, 45)
    bootstrap_samples: int = 2000

    # Secondary mechanism/system experiments, all still using 1500 videos.
    secondary_budget_ratio: float = 0.10
    load_factors: tuple[float, ...] = (0.50, 0.80, 1.00, 1.20, 1.50, 2.00)
    load_sweep_query_count: int = 100_000
    overlap_requests_per_video: int = 5
    overlap_modes: tuple[str, ...] = ("low", "natural", "high")
    global_budget_ratios: tuple[float, ...] = (0.05, 0.10, 0.20)
    request_scale_values: tuple[int, ...] = (8, 16, 32)
    noise_levels: tuple[float, ...] = (0.0, 0.05, 0.10, 0.20, 0.30)
    noise_seeds: tuple[int, ...] = (71, 72, 73)
    robust_beta_for_noise: float = 1.0

    # Decision-time acceleration ablation.
    # These experiments always use the full 1500-video library, but use small
    # storage budgets because Eager Greedy intentionally recomputes every
    # remaining candidate's exact marginal gain at every selection round.
    decision_ablation_budget_ratios: tuple[float, ...] = (0.005, 0.01, 0.02)
    decision_ablation_seeds: tuple[int, ...] = (41, 42, 43)

    candidate: CandidateConfig = field(default_factory=CandidateConfig)
    workload: WorkloadConfig = field(default_factory=WorkloadConfig)
    predictor: PredictorConfig = field(default_factory=PredictorConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    prior_mvs: PriorMVSConfig = field(default_factory=PriorMVSConfig)
    prior_vsa: PriorVSAConfig = field(default_factory=PriorVSAConfig)
    prior_vss: PriorVSSConfig = field(default_factory=PriorVSSConfig)
    serving: ServingConfig = field(default_factory=ServingConfig)

    def validate(self) -> None:
        if self.library_video_count <= 0:
            raise ValueError("library_video_count must be positive")
        if self.require_exact_library_size and self.library_video_count != 1500:
            raise ValueError(
                "Formal WARM experiments in this artifact require library_video_count=1500 "
                "when require_exact_library_size=true"
            )
        if any(x <= 0 for x in self.budget_ratios):
            raise ValueError("budget ratios must be positive")
        if self.serving.workers < 1:
            raise ValueError("serving.workers must be >= 1")
        if self.serving.routing_policy not in {"min_cost", "closest_feasible"}:
            raise ValueError("serving.routing_policy must be 'min_cost' or 'closest_feasible'")
        if not (0.0 < self.predictor.calibration_ratio < 0.5):
            raise ValueError("predictor.calibration_ratio must be in (0, 0.5)")
        if self.require_exact_library_size and len(self.workload_seeds) < 3:
            raise ValueError("expected at least three workload seeds in the study config")
        if self.bootstrap_samples < 100:
            raise ValueError("bootstrap_samples should be >= 100")
        if self.prior_mvs.pool_cap < 2:
            raise ValueError("prior_mvs.pool_cap must be >= 2")
        if min(
            self.prior_mvs.per_video_density, self.prior_mvs.per_video_gain,
            self.prior_mvs.pair_top_density, self.prior_mvs.pair_top_gain,
        ) < 0:
            raise ValueError("prior_mvs candidate-selection counts must be >= 0")
        if self.prior_vsa.max_versions < 1 or self.prior_vsa.max_versions > 10:
            raise ValueError("prior_vsa.max_versions must be between 1 and 10")
        if self.prior_vsa.capacity_bins < 1:
            raise ValueError("prior_vsa.capacity_bins must be >= 1")
        if self.prior_vss.history_queries < 1:
            raise ValueError("prior_vss.history_queries must be >= 1")
        if self.prior_vss.per_video_budget_multiplier <= 0:
            raise ValueError("prior_vss.per_video_budget_multiplier must be positive")
        if self.prior_vss.redundancy_weight < 0:
            raise ValueError("prior_vss.redundancy_weight must be nonnegative")
        if any(x <= 0 for x in self.decision_ablation_budget_ratios):
            raise ValueError("decision_ablation_budget_ratios must be positive")
        if self.require_exact_library_size and len(self.decision_ablation_seeds) < 3:
            raise ValueError("expected at least three decision-ablation seeds in the study config")


def _tupleize(cls, obj: dict[str, Any]) -> dict[str, Any]:
    annotations = getattr(cls, "__annotations__", {})
    out = dict(obj)
    for key, value in list(out.items()):
        if key in annotations and isinstance(value, list):
            out[key] = tuple(value)
    return out


def load_config(path: str | Path | None) -> ExperimentConfig:
    if path is None:
        cfg = ExperimentConfig()
        cfg.validate()
        return cfg
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    defaults = ExperimentConfig()
    cfg = ExperimentConfig(
        seed=data.get("seed", defaults.seed),
        predictor_train_file=data.get("predictor_train_file", defaults.predictor_train_file),
        predictor_test_file=data.get("predictor_test_file", defaults.predictor_test_file),
        library_video_count=int(data.get("library_video_count", defaults.library_video_count)),
        require_exact_library_size=bool(data.get("require_exact_library_size", defaults.require_exact_library_size)),
        budget_ratios=tuple(data.get("budget_ratios", defaults.budget_ratios)),
        methods=tuple(data.get("methods", defaults.methods)),
        workload_seeds=tuple(data.get("workload_seeds", defaults.workload_seeds)),
        bootstrap_samples=int(data.get("bootstrap_samples", defaults.bootstrap_samples)),
        secondary_budget_ratio=float(data.get("secondary_budget_ratio", defaults.secondary_budget_ratio)),
        load_factors=tuple(data.get("load_factors", defaults.load_factors)),
        load_sweep_query_count=int(data.get("load_sweep_query_count", defaults.load_sweep_query_count)),
        overlap_requests_per_video=int(data.get("overlap_requests_per_video", defaults.overlap_requests_per_video)),
        overlap_modes=tuple(data.get("overlap_modes", defaults.overlap_modes)),
        global_budget_ratios=tuple(data.get("global_budget_ratios", defaults.global_budget_ratios)),
        request_scale_values=tuple(data.get("request_scale_values", defaults.request_scale_values)),
        noise_levels=tuple(data.get("noise_levels", defaults.noise_levels)),
        noise_seeds=tuple(data.get("noise_seeds", defaults.noise_seeds)),
        robust_beta_for_noise=float(data.get("robust_beta_for_noise", defaults.robust_beta_for_noise)),
        decision_ablation_budget_ratios=tuple(data.get("decision_ablation_budget_ratios", defaults.decision_ablation_budget_ratios)),
        decision_ablation_seeds=tuple(data.get("decision_ablation_seeds", defaults.decision_ablation_seeds)),
        candidate=CandidateConfig(**_tupleize(CandidateConfig, data.get("candidate", {}))),
        workload=WorkloadConfig(**_tupleize(WorkloadConfig, data.get("workload", {}))),
        predictor=PredictorConfig(**_tupleize(PredictorConfig, data.get("predictor", {}))),
        optimizer=OptimizerConfig(**data.get("optimizer", {})),
        prior_mvs=PriorMVSConfig(**data.get("prior_mvs", {})),
        prior_vsa=PriorVSAConfig(**data.get("prior_vsa", {})),
        prior_vss=PriorVSSConfig(**data.get("prior_vss", {})),
        serving=ServingConfig(**data.get("serving", {})),
    )
    cfg.validate()
    return cfg


def dump_config(cfg: ExperimentConfig, path: str | Path) -> None:
    Path(path).write_text(
        yaml.safe_dump(asdict(cfg), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
