"""WARM public command-line interface for algorithm research."""
from __future__ import annotations

import argparse
import pandas as pd

from .config import load_config
from .naming import DISPLAY_NAME
from .study import check_data, plan_from_statistics


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="warm", description="WARM workload-aware replica materialization planner")
    sub = ap.add_subparsers(dest="command", required=True)
    sub.add_parser("methods", help="list the permitted paper baseline and ablation labels")

    ck = sub.add_parser("check-data", help="validate source statistics and FNN train/validation tables")
    ck.add_argument("--data-dir", required=True)
    ck.add_argument("--config", default="config/paper_full.yaml")

    pl = sub.add_parser("plan", help="produce a predicted replica SELECTION plan (not generated video)")
    pl.add_argument("--data-dir", required=True)
    pl.add_argument("--output-dir", default="outputs/planning")
    pl.add_argument("--config", default="config/paper_full.yaml")
    pl.add_argument("--methods", default="WARM", help="comma-separated public names, e.g. WARM,VSS,wm-dens")
    pl.add_argument("--budget-ratio", type=float, default=0.10)
    pl.add_argument("--historical-log", default=None)
    pl.add_argument("--candidate-source", choices=("replica-json", "enumerated"),
                    default="replica-json",
                    help="use per-video replica JSON configurations (default), or enumerate from YAML")

    args = ap.parse_args(argv)
    if args.command == "methods":
        for display in DISPLAY_NAME.values():
            print(display)
        return 0
    cfg = load_config(args.config)
    if args.command == "check-data":
        print(pd.Series(check_data(args.data_dir, cfg)).to_string())
        return 0
    if args.command == "plan":
        methods = [x.strip() for x in args.methods.split(",") if x.strip()]
        res = plan_from_statistics(args.data_dir, args.output_dir, cfg, methods,
                                   args.budget_ratio, args.historical_log,
                                   candidate_source=args.candidate_source)
        print(res.to_string(index=False))
        print(f"Results: {res.attrs['output_directory']}")
        return 0
    raise AssertionError(args.command)
