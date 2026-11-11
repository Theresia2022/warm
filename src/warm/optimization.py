from __future__ import annotations

from dataclasses import dataclass
import heapq
import itertools
import time

import numpy as np
import pandas as pd

from .coupling import OptimizationInstance

EPS = 1e-12


@dataclass
class PlanResult:
    method: str
    plan: pd.DataFrame
    predicted_objective: float
    storage_used: float
    runtime_s: float
    marginal_evaluations: int = 0
    heap_pops: int = 0
    metadata: dict | None = None


def _empty_plan(instance: OptimizationInstance) -> pd.DataFrame:
    return instance.candidates.iloc[0:0].copy()


def evaluate_objective(
    instance: OptimizationInstance,
    plan: pd.DataFrame,
    decode_col: str | None = None,
) -> float:
    decode_col = decode_col or instance.decode_col
    req = instance.requests
    current = req["source_decode_cost"].to_numpy(float).copy()
    weights = req["weight"].to_numpy(float)
    if plan is None or len(plan) == 0:
        return 0.0
    tau_map = instance.candidates.set_index("cand_idx")[decode_col].to_dict()
    for cidx in plan["cand_idx"].astype(int):
        ridx = instance.compatibility.coverage.get(int(cidx), np.empty(0, dtype=np.int64))
        if len(ridx):
            current[ridx] = np.minimum(current[ridx], float(tau_map[cidx]))
    src = req["source_decode_cost"].to_numpy(float)
    return float(np.dot(weights, np.maximum(src - current, 0.0)))


class LazyBudgetedGreedy:
    def __init__(self, best_singleton: bool = True, min_gain: float = 1e-12):
        self.best_singleton = best_singleton
        self.min_gain = min_gain

    def optimize(self, instance: OptimizationInstance, method: str = "warm") -> PlanResult:
        start = time.perf_counter()
        cand = instance.candidates.copy()
        req = instance.requests
        weights = req["weight"].to_numpy(float)
        current = req["source_decode_cost"].to_numpy(float).copy()
        storage = cand.set_index("cand_idx")[instance.storage_col].astype(float).to_dict()
        decode = cand.set_index("cand_idx")[instance.decode_col].astype(float).to_dict()
        singleton = cand.set_index("cand_idx")["singleton_gain"].astype(float).to_dict()
        heap: list[tuple[float, float, int]] = []
        evals = 0
        pops = 0
        for cidx in cand["cand_idx"].astype(int):
            s = storage[cidx]
            g = singleton.get(cidx, 0.0)
            if not np.isfinite(s) or s <= 0 or s > instance.budget_mb + EPS or g <= self.min_gain:
                continue
            heapq.heappush(heap, (-(g / s), -g, cidx))

        selected: list[int] = []
        selected_set: set[int] = set()
        remaining = float(instance.budget_mb)
        objective = 0.0

        def marginal(cidx: int) -> float:
            nonlocal evals
            evals += 1
            ridx = instance.compatibility.coverage.get(cidx, np.empty(0, dtype=np.int64))
            if len(ridx) == 0:
                return 0.0
            diff = np.maximum(current[ridx] - decode[cidx], 0.0)
            return float(np.dot(weights[ridx], diff))

        while heap:
            _negd, _negg, cidx = heapq.heappop(heap)
            pops += 1
            if cidx in selected_set:
                continue
            s = storage[cidx]
            if s > remaining + EPS:
                continue
            g = marginal(cidx)
            if g <= self.min_gain:
                continue
            d = g / max(s, EPS)
            next_upper = -heap[0][0] if heap else -np.inf
            if d + EPS >= next_upper:
                selected.append(cidx)
                selected_set.add(cidx)
                remaining -= s
                objective += g
                ridx = instance.compatibility.coverage.get(cidx, np.empty(0, dtype=np.int64))
                if len(ridx):
                    current[ridx] = np.minimum(current[ridx], decode[cidx])
            else:
                heapq.heappush(heap, (-d, -g, cidx))

        greedy_selected = list(selected)
        plan = cand[cand["cand_idx"].isin(selected)].copy()
        if len(plan):
            order = {c: i for i, c in enumerate(selected)}
            plan["selected_order"] = plan["cand_idx"].map(order)
            plan = plan.sort_values("selected_order")

        best_singleton_chosen = False
        fallback_greedy_ids: list[int] = []
        if self.best_singleton and len(cand):
            feasible = cand[cand[instance.storage_col] <= instance.budget_mb + EPS]
            if len(feasible):
                best = feasible.loc[feasible["singleton_gain"].idxmax()]
                best_gain = float(best["singleton_gain"])
                if best_gain > objective + EPS:
                    fallback_greedy_ids = greedy_selected
                    plan = feasible.loc[[best.name]].copy()
                    plan["selected_order"] = 0
                    objective = best_gain
                    selected = [int(best["cand_idx"])]
                    best_singleton_chosen = True

        used = float(plan[instance.storage_col].sum()) if len(plan) else 0.0
        metadata = {
            "best_singleton_chosen": bool(best_singleton_chosen),
            "_fallback_greedy_ids": fallback_greedy_ids,
        }
        return PlanResult(
            method,
            plan.reset_index(drop=True),
            float(objective),
            used,
            time.perf_counter() - start,
            evals,
            pops,
            metadata,
        )


class EagerBudgetedGreedy:
    def __init__(self, best_singleton: bool = True, min_gain: float = 1e-12):
        self.best_singleton = best_singleton
        self.min_gain = min_gain

    def optimize(self, instance: OptimizationInstance, method: str = "eager_dynamic") -> PlanResult:
        start = time.perf_counter()
        cand = instance.candidates
        req = instance.requests
        weights = req["weight"].to_numpy(float)
        current = req["source_decode_cost"].to_numpy(float).copy()
        storage = cand.set_index("cand_idx")[instance.storage_col].astype(float).to_dict()
        decode = cand.set_index("cand_idx")[instance.decode_col].astype(float).to_dict()
        remaining = instance.budget_mb
        selected: list[int] = []
        available = set(int(x) for x in cand["cand_idx"])
        evals = 0
        objective = 0.0
        while available:
            best = None
            best_density = -np.inf
            best_gain = 0.0
            # Deterministic reference order.  Eager still recomputes the exact
            # marginal gain for EVERY feasible remaining candidate in EVERY
            # round; this tie-break only makes Lazy-vs-Eager equivalence easier
            # to audit.
            best_key = (-np.inf, -np.inf, -np.inf)
            for cidx in available:
                s = storage[cidx]
                if s > remaining + EPS or s <= 0:
                    continue
                ridx = instance.compatibility.coverage.get(cidx, np.empty(0, dtype=np.int64))
                evals += 1
                gain = float(np.dot(weights[ridx], np.maximum(current[ridx] - decode[cidx], 0.0))) if len(ridx) else 0.0
                density = gain / max(s, EPS)
                # Same conceptual preference as the Lazy heap:
                # higher density, then higher gain, then smaller cand_idx.
                key = (density, gain, -int(cidx))
                if key > best_key:
                    best, best_density, best_gain = cidx, density, gain
                    best_key = key
            if best is None or best_gain <= self.min_gain:
                break
            selected.append(best)
            available.remove(best)
            remaining -= storage[best]
            objective += best_gain
            ridx = instance.compatibility.coverage.get(best, np.empty(0, dtype=np.int64))
            if len(ridx):
                current[ridx] = np.minimum(current[ridx], decode[best])
        greedy_selected = list(selected)
        plan = cand[cand["cand_idx"].isin(selected)].copy()
        if len(plan):
            order = {c: i for i, c in enumerate(selected)}
            plan["selected_order"] = plan["cand_idx"].map(order)
            plan = plan.sort_values("selected_order")

        best_singleton_chosen = False
        fallback_greedy_ids: list[int] = []
        if self.best_singleton and len(cand):
            feasible = cand[cand[instance.storage_col] <= instance.budget_mb + EPS]
            if len(feasible):
                best_row = feasible.loc[feasible["singleton_gain"].idxmax()]
                if float(best_row["singleton_gain"]) > objective + EPS:
                    fallback_greedy_ids = greedy_selected
                    plan = feasible.loc[[best_row.name]].copy()
                    plan["selected_order"] = 0
                    objective = float(best_row["singleton_gain"])
                    best_singleton_chosen = True

        used = float(plan[instance.storage_col].sum()) if len(plan) else 0.0
        metadata = {
            "best_singleton_chosen": bool(best_singleton_chosen),
            "_fallback_greedy_ids": fallback_greedy_ids,
        }
        return PlanResult(
            method,
            plan.reset_index(drop=True),
            objective,
            used,
            time.perf_counter() - start,
            evals,
            metadata=metadata,
        )


def _greedy_by_score(instance: OptimizationInstance, score: pd.Series, method: str) -> PlanResult:
    start = time.perf_counter()
    cand = instance.candidates.copy()
    cand["_score"] = score.to_numpy(float)
    cand = cand.sort_values(["_score", "singleton_gain"], ascending=[False, False])
    used = 0.0
    chosen = []
    for row in cand.itertuples(index=False):
        s = float(getattr(row, instance.storage_col))
        if s > 0 and used + s <= instance.budget_mb + EPS:
            chosen.append(int(row.cand_idx))
            used += s
    plan = instance.candidates[instance.candidates["cand_idx"].isin(chosen)].copy()
    if len(plan):
        order = {c: i for i, c in enumerate(chosen)}
        plan["selected_order"] = plan["cand_idx"].map(order)
        plan = plan.sort_values("selected_order")
    obj = evaluate_objective(instance, plan)
    return PlanResult(method, plan.reset_index(drop=True), obj, used, time.perf_counter() - start)


class StaticDensityGreedy:
    def optimize(self, instance: OptimizationInstance, method: str = "static_density") -> PlanResult:
        return _greedy_by_score(instance, instance.candidates["singleton_density"], method)


class BenefitOnlyGreedy:
    def optimize(self, instance: OptimizationInstance, method: str = "benefit_only") -> PlanResult:
        return _greedy_by_score(instance, instance.candidates["singleton_gain"], method)


class SmallestStorageFirst:
    def optimize(self, instance: OptimizationInstance, method: str = "smallest_storage") -> PlanResult:
        score = 1.0 / np.maximum(instance.candidates[instance.storage_col].to_numpy(float), EPS)
        return _greedy_by_score(instance, pd.Series(score, index=instance.candidates.index), method)


class PopularityFirst:
    def optimize(self, instance: OptimizationInstance, method: str = "popularity_first") -> PlanResult:
        video_pop = instance.requests.groupby("VID")["weight"].sum().to_dict()
        score = instance.candidates["VID"].map(video_pop).fillna(0.0)
        # Within the same video, favor useful candidates before arbitrary IDs.
        score = score + 1e-6 * instance.candidates["singleton_density"]
        return _greedy_by_score(instance, score, method)


class SourceOnly:
    def optimize(self, instance: OptimizationInstance, method: str = "source_only") -> PlanResult:
        return PlanResult(method, _empty_plan(instance), 0.0, 0.0, 0.0)


class AllReplicas:
    def optimize(self, instance: OptimizationInstance, method: str = "all_replicas") -> PlanResult:
        start = time.perf_counter()
        plan = instance.candidates.copy()
        used = float(plan[instance.storage_col].sum())
        return PlanResult(method, plan, evaluate_objective(instance, plan), used, time.perf_counter() - start)


class ExactBruteforce:
    """Exact reference for tiny instances only."""
    def __init__(self, max_candidates: int = 24):
        self.max_candidates = max_candidates

    def optimize(self, instance: OptimizationInstance, method: str = "exact_small") -> PlanResult:
        start = time.perf_counter()
        cand = instance.candidates.reset_index(drop=True)
        n = len(cand)
        if n > self.max_candidates:
            raise ValueError(f"exact brute force supports <= {self.max_candidates} candidates, got {n}")
        best_obj, best_idx, best_storage = -1.0, (), 0.0
        stor = cand[instance.storage_col].to_numpy(float)
        for mask in range(1 << n):
            idx = [i for i in range(n) if mask & (1 << i)]
            s = float(stor[idx].sum()) if idx else 0.0
            if s > instance.budget_mb + EPS:
                continue
            plan = cand.iloc[idx] if idx else cand.iloc[0:0]
            obj = evaluate_objective(instance, plan)
            if obj > best_obj + EPS:
                best_obj, best_idx, best_storage = obj, tuple(idx), s
        plan = cand.iloc[list(best_idx)].copy() if best_idx else cand.iloc[0:0].copy()
        return PlanResult(method, plan.reset_index(drop=True), best_obj, best_storage, time.perf_counter() - start)



class LocalBudgetLazy:
    """Per-video Lazy greedy under a fixed local budget partition.

    This is a C2 mechanism baseline: candidates from different videos do *not*
    compete for storage after the initial budget allocation. ``equal`` assigns
    the same budget to every video, while ``popularity`` allocates budget in
    proportion to each video's workload mass. Any unused local budget is
    stranded rather than redistributed, which is precisely the limitation this
    baseline is intended to expose.
    """
    def __init__(self, allocation: str = "equal", best_singleton: bool = True, min_gain: float = 1e-12):
        if allocation not in {"equal", "popularity"}:
            raise ValueError(allocation)
        self.allocation = allocation
        self.best_singleton = best_singleton
        self.min_gain = min_gain

    def optimize(self, instance: OptimizationInstance, method: str | None = None) -> PlanResult:
        start = time.perf_counter()
        cand = instance.candidates
        req = instance.requests
        weights = req["weight"].to_numpy(float)
        source = req["source_decode_cost"].to_numpy(float)
        storage = cand.set_index("cand_idx")[instance.storage_col].astype(float).to_dict()
        decode = cand.set_index("cand_idx")[instance.decode_col].astype(float).to_dict()
        singleton = cand.set_index("cand_idx")["singleton_gain"].astype(float).to_dict()
        vids = list(req["VID"].drop_duplicates())
        if not vids:
            return PlanResult(method or f"local_{self.allocation}_lazy", _empty_plan(instance), 0.0, 0.0, time.perf_counter()-start)

        if self.allocation == "equal":
            allocations = {video_id: float(instance.budget_mb) / len(vids) for video_id in vids}
        else:
            pop = req.groupby("VID")["weight"].sum().to_dict()
            total = sum(float(pop.get(v, 0.0)) for v in vids)
            if total <= EPS:
                allocations = {video_id: float(instance.budget_mb) / len(vids) for video_id in vids}
            else:
                allocations = {video_id: float(instance.budget_mb) * float(pop.get(video_id, 0.0)) / total for video_id in vids}

        selected: list[int] = []
        evals = 0
        pops = 0

        for video_id, group in cand.groupby("VID", sort=False):
            budget = float(allocations.get(video_id, 0.0))
            if budget <= EPS or len(group) == 0:
                continue
            # Requests are global-indexed; this is okay because coverage stores
            # global request positions and all candidates in this group only
            # cover requests for the same video.
            current = source.copy()
            heap: list[tuple[float, float, int]] = []
            for cidx in group["cand_idx"].astype(int):
                s = storage[cidx]
                g = singleton.get(cidx, 0.0)
                if s > 0 and s <= budget + EPS and g > self.min_gain:
                    heapq.heappush(heap, (-(g / s), -g, cidx))

            local_selected: list[int] = []
            local_set: set[int] = set()
            remaining = budget
            local_obj = 0.0

            while heap:
                _negd, _negg, cidx = heapq.heappop(heap)
                pops += 1
                if cidx in local_set:
                    continue
                s = storage[cidx]
                if s > remaining + EPS:
                    continue
                ridx = instance.compatibility.coverage.get(cidx, np.empty(0, dtype=np.int64))
                evals += 1
                gain = float(np.dot(weights[ridx], np.maximum(current[ridx] - decode[cidx], 0.0))) if len(ridx) else 0.0
                if gain <= self.min_gain:
                    continue
                density = gain / max(s, EPS)
                next_upper = -heap[0][0] if heap else -np.inf
                if density + EPS >= next_upper:
                    local_selected.append(cidx)
                    local_set.add(cidx)
                    remaining -= s
                    local_obj += gain
                    if len(ridx):
                        current[ridx] = np.minimum(current[ridx], decode[cidx])
                else:
                    heapq.heappush(heap, (-density, -gain, cidx))

            if self.best_singleton:
                feasible = group[group[instance.storage_col] <= budget + EPS]
                if len(feasible):
                    best_row = feasible.loc[feasible["singleton_gain"].idxmax()]
                    if float(best_row["singleton_gain"]) > local_obj + EPS:
                        local_selected = [int(best_row["cand_idx"])]

            selected.extend(local_selected)

        plan = cand[cand["cand_idx"].isin(selected)].copy()
        if len(plan):
            order = {c: i for i, c in enumerate(selected)}
            plan["selected_order"] = plan["cand_idx"].map(order)
            plan = plan.sort_values("selected_order")
        obj = evaluate_objective(instance, plan)
        used = float(plan[instance.storage_col].sum()) if len(plan) else 0.0
        metadata = {
            "local_budget_allocation": self.allocation,
            "local_budget_stranded_mb": max(float(instance.budget_mb) - used, 0.0),
        }
        return PlanResult(
            method or f"local_{self.allocation}_lazy",
            plan.reset_index(drop=True),
            obj,
            used,
            time.perf_counter() - start,
            evals,
            pops,
            metadata,
        )

def planner_for(method: str):
    method = method.lower()
    if method == "warm":
        return LazyBudgetedGreedy(best_singleton=True)
    if method == "eager_dynamic":
        return EagerBudgetedGreedy(best_singleton=True)
    if method == "static_density":
        return StaticDensityGreedy()
    if method == "benefit_only":
        return BenefitOnlyGreedy()
    if method == "smallest_storage":
        return SmallestStorageFirst()
    if method == "popularity_first":
        return PopularityFirst()
    if method == "source_only":
        return SourceOnly()
    if method == "local_equal_lazy":
        return LocalBudgetLazy("equal")
    if method == "local_popularity_lazy":
        return LocalBudgetLazy("popularity")
    if method == "all_replicas":
        return AllReplicas()
    raise KeyError(method)
