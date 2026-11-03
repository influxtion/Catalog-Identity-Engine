import argparse
import json
from pathlib import Path

from catalog_identity.evaluation.config import RunConfig, load_config
from catalog_identity.evaluation.runner import run_evaluation
from catalog_identity.evaluation.synthetic import SYNTHETIC_VERSION, write_synthetic
from catalog_identity.ingestion.audit import profile
from catalog_identity.ingestion.loaders import load_listings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Catalog Identity Engine baseline tools")
    commands = parser.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("demo", help="Run a clearly labeled synthetic experiment")
    demo.add_argument("--output", type=Path, required=True)
    demo.add_argument("--seed", type=int, default=42)
    evaluate = commands.add_parser("evaluate", help="Evaluate official local WDC pair files")
    evaluate.add_argument("--config", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    profiling = commands.add_parser(
        "profile", help="Validate and profile canonical CSV/JSONL listings"
    )
    profiling.add_argument("input", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "profile":
            print(json.dumps(profile(load_listings(args.input)), indent=2))
            return 0
        if args.command == "demo":
            if args.output.exists():
                raise ValueError("output already exists; choose a fresh run directory")
            paths = write_synthetic(args.output / "fixture", args.seed)
            config = RunConfig(
                dataset_name="fictional-products",
                dataset_version=SYNTHETIC_VERSION,
                synthetic=True,
                **{k: v.resolve() for k, v in paths.items()},
                test_entity_disjoint=True,
                seed=args.seed,
            )
            output = args.output / "evaluation"
        else:
            config, output = load_config(args.config), args.output
        report = run_evaluation(config, output)
        print(report["warning"])
        print(json.dumps(report["splits"]["test"]["pair_classification"], indent=2))
        print(f"Report: {(output / 'report.json').resolve()}")
    except (ValueError, OSError) as exc:
        parser.exit(2, f"error: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
