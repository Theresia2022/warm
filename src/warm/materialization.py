from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .coupling import OptimizationInstance

EPS = 1e-9


@dataclass(frozen=True)
class MaterializationResult:
    plan: pd.DataFrame
    attempted_count: int
    accepted_count: int
    skipped_budget_count: int
    skipped_invalid_storage_count: int
    pre_guard_true_storage_mb: float
    materialized_true_storage_mb: float
    remaining_true_budget_mb: float
    order_source: str
    budget_scope: str
    fallback_used: bool = False


def _ordered_plan(plan: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    if plan is None or len(plan) == 0:
        return (pd.DataFrame() if plan is None else plan.iloc[0:0].copy()), "empty"
    out = plan.copy()
    if "selected_order" in out.columns:
        out = out.sort_values(["selected_order", "cand_idx"], kind="stable")
        return out.reset_index(drop=True), "selected_order"
    out = out.reset_index(drop=True)
    out["selected_order"] = np.arange(len(out), dtype=np.int64)
    return out, "planner_row_order"


def _local_true_budget_allocations(
    instance: OptimizationInstance,
    method: str,
) -> dict[Any, float] | None:
    if method not in {"local_equal_lazy", "local_popularity_lazy"}:
        return None

    vids = list(instance.requests["VID"].drop_duplicates())
    if not vids:
        return {}

    if method == "local_equal_lazy":
        each = float(instance.budget_mb) / len(vids)
        return {video_id: each for video_id in vids}

    pop = instance.requests.groupby("VID")["weight"].sum().to_dict()
    total = sum(float(pop.get(v, 0.0)) for v in vids)
    if total <= EPS:
        each = float(instance.budget_mb) / len(vids)
        return {video_id: each for video_id in vids}
    return {
        video_id: float(instance.budget_mb) * float(pop.get(video_id, 0.0)) / total
        for video_id in vids
    }


def materialize_plan_under_true_budget(
    instance: OptimizationInstance,
    planned_plan: pd.DataFrame,
    method: str,
) -> MaterializationResult:
    """Replay a prediction-based plan against measured TRUE storage.

    Planner order is preserved. True storage is used only at the commit
    boundary: if a generated replica does not fit the remaining persistent
    budget it is skipped, and materialization proceeds to the next planned
    replica. True storage never re-ranks candidates.
    """
    ordered, order_source = _ordered_plan(planned_plan)

    if len(ordered) == 0:
        empty = instance.candidates.iloc[0:0].copy()
        return MaterializationResult(
            empty, 0, 0, 0, 0, 0.0, 0.0, float(instance.budget_mb),
            order_source, "local" if method.startswith("local_") else "global",
        )

    if "true_storage_mb" not in ordered.columns:
        raise ValueError("materialization guard requires true_storage_mb")

    true_series = pd.to_numeric(ordered["true_storage_mb"], errors="coerce")
    pre_true = float(true_series.where(np.isfinite(true_series), 0.0).fillna(0.0).sum())

    if not instance.enforce_true_budget_at_materialization:
        out = ordered.copy()
        out["materialized_order"] = np.arange(len(out), dtype=np.int64)
        used = float(out["true_storage_mb"].sum())
        return MaterializationResult(
            out.reset_index(drop=True), len(out), len(out), 0, 0,
            pre_true, used, float(instance.budget_mb) - used,
            order_source, "disabled",
        )

    local_remaining = _local_true_budget_allocations(instance, method)
    global_remaining = float(instance.budget_mb)
    accepted_rows: list[int] = []
    skipped_budget = 0
    skipped_invalid = 0
    attempted = 0

    for pos, row in ordered.iterrows():
        attempted += 1
        s = float(row["true_storage_mb"])
        if not np.isfinite(s) or s <= 0.0:
            skipped_invalid += 1
            continue

        if local_remaining is not None:
            video_id = row["VID"]
            remaining = float(local_remaining.get(video_id, 0.0))
            if s > remaining + EPS:
                skipped_budget += 1
                continue
            local_remaining[video_id] = max(remaining - s, 0.0)
            global_remaining = max(global_remaining - s, 0.0)
        else:
            if s > global_remaining + EPS:
                skipped_budget += 1
                continue
            global_remaining = max(global_remaining - s, 0.0)

        accepted_rows.append(pos)

    if accepted_rows:
        out = ordered.loc[accepted_rows].copy().reset_index(drop=True)
        out["materialized_order"] = np.arange(len(out), dtype=np.int64)
    else:
        out = ordered.iloc[0:0].copy()
        out["materialized_order"] = pd.Series(dtype=np.int64)

    used = float(out["true_storage_mb"].sum()) if len(out) else 0.0
    if used > float(instance.budget_mb) + EPS:
        raise RuntimeError(
            f"true-budget invariant violated: used={used:.9f} > "
            f"budget={instance.budget_mb:.9f} MB"
        )

    return MaterializationResult(
        plan=out,
        attempted_count=attempted,
        accepted_count=len(out),
        skipped_budget_count=skipped_budget,
        skipped_invalid_storage_count=skipped_invalid,
        pre_guard_true_storage_mb=pre_true,
        materialized_true_storage_mb=used,
        remaining_true_budget_mb=max(float(instance.budget_mb) - used, 0.0),
        order_source=order_source,
        budget_scope="local" if local_remaining is not None else "global",
    )
