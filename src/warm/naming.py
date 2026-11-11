"""Public method names. Keep internal identifiers for traceability of adapters."""

from __future__ import annotations

# All labels are the names used in the WARM manuscript.
DISPLAY_NAME: dict[str, str] = {
    "warm": "WARM",
    "source_only": "Source-only",
    "prior_vss_cache": "VSS",
    "vstore_coalesce": "VStore",
    "prior_vsa_dp": "VSA",
    "prior_zhao_version": "VACTS",
    "popularity_first": "wm-pop",
    "smallest_storage": "wm-smin",
    "benefit_only": "wm-ben",
    "static_density": "wm-dens",
    "local_equal_lazy": "wm-eqbd",
    "local_popularity_lazy": "wm-pwbd",
    "eager_dynamic": "wm-eagr",
    "lazy_unpruned": "wm-unpr",
    "robust_cost": "wm-rbst",
}

PAPER_MAIN_METHODS = (
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


def method_id(public_or_internal: str) -> str:
    """Accept published display names or implementation method IDs."""
    needle = public_or_internal.strip().lower()
    for internal, display in DISPLAY_NAME.items():
        if needle in {internal.lower(), display.lower()}:
            return internal
    raise ValueError(
        f"Unknown method {public_or_internal!r}; choose from: "
        + ", ".join(DISPLAY_NAME.values())
    )


def method_name(internal: str) -> str:
    return DISPLAY_NAME[internal]
