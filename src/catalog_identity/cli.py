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
    development = commands.add_parser("develop", help="Compare matchers using validation only")
    development.add_argument("--config", type=Path, required=True)
    development.add_argument("--output", type=Path, required=True)
    development.add_argument("--negative-policy", choices=["all", "random_only"], default="all")
    development.add_argument("--feature-version", choices=["2", "3", "4", "5", "6"], default="2")
    development.add_argument("--num-leaves", type=int, default=15)
    development.add_argument("--n-estimators", type=int, default=200)
    development.add_argument(
        "--learners", nargs="+", choices=["logistic", "lightgbm"], default=["logistic", "lightgbm"]
    )
    screening = commands.add_parser(
        "screen-matchers", help="Fixed train-only entity-disjoint screening"
    )
    screening.add_argument("--config", type=Path, required=True)
    screening.add_argument("--output", type=Path, required=True)
    input_audit = commands.add_parser(
        "audit-training", help="Audit training ambiguity and prepare local blinded review"
    )
    input_audit.add_argument("--config", type=Path, required=True)
    input_audit.add_argument("--output", type=Path, required=True)
    input_audit.add_argument("--tokenizer-directory", type=Path)
    input_audit.add_argument("--control-report", type=Path)
    semantic_screening = commands.add_parser(
        "screen-semantic", help="Fixed offline semantic matcher screening"
    )
    semantic_screening.add_argument("--config", type=Path, required=True)
    semantic_screening.add_argument("--output", type=Path, required=True)
    semantic_screening.add_argument("--cache", type=Path, default=Path("runs/semantic-cache"))
    joint_pilot = commands.add_parser(
        "pilot-joint", help="Timing-only offline supervised matcher gate"
    )
    joint_pilot.add_argument("--config", type=Path, required=True)
    joint_pilot.add_argument("--output", type=Path, required=True)
    joint_pilot.add_argument("--validation-report", type=Path, required=True)
    joint_pilot.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    joint_screen = commands.add_parser(
        "screen-joint", help="Gated train-only supervised matcher screening"
    )
    joint_screen.add_argument("--config", type=Path, required=True)
    joint_screen.add_argument("--output", type=Path, required=True)
    joint_screen.add_argument("--timing", type=Path, required=True)
    joint_screen.add_argument("--control", type=Path, required=True)
    joint_screen.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    joint_develop = commands.add_parser(
        "develop-joint", help="Validate a qualifying joint matcher once"
    )
    joint_develop.add_argument("--config", type=Path, required=True)
    joint_develop.add_argument("--output", type=Path, required=True)
    joint_develop.add_argument("--screening", type=Path, required=True)
    joint_develop.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    for action in ("pilot", "screen", "develop"):
        command = commands.add_parser(f"{action}-full", help="Registered conventional full matcher")
        command.add_argument("--config", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
        if action == "pilot":
            command.add_argument("--validation-report", type=Path, required=True)
        elif action == "screen":
            command.add_argument("--timing", type=Path, required=True)
            command.add_argument("--control", type=Path, required=True)
        else:
            command.add_argument("--screening", type=Path, required=True)
    diagnostics = commands.add_parser(
        "validation-errors", help="Inspect saved matcher validation errors"
    )
    diagnostics.add_argument("--config", type=Path, required=True)
    diagnostics.add_argument("--model", type=Path, required=True)
    diagnostics.add_argument("--output", type=Path, required=True)
    ablation = commands.add_parser("retrieval-ablation", help="Compare retrieval on validation")
    ablation.add_argument("--config", type=Path, required=True)
    ablation.add_argument("--output", type=Path, required=True)
    ablation.add_argument("--allow-model-download", action="store_true")
    frozen = commands.add_parser("evaluate-frozen", help="Evaluate a saved model without refitting")
    frozen.add_argument("--config", type=Path, required=True)
    frozen.add_argument("--model", type=Path, required=True)
    frozen.add_argument("--output", type=Path, required=True)
    frozen.add_argument("--hybrid", action="store_true")
    server = commands.add_parser("serve", help="Start the local inspection API")
    server.add_argument("--port", type=int, default=8000)
    worker = commands.add_parser("worker", help="Process durable catalog jobs")
    worker.add_argument("--once", action="store_true")
    worker.add_argument("--config-root", type=Path, default=Path("configs"))
    worker.add_argument("--output-root", type=Path, default=Path("runs/jobs"))
    commands.add_parser("seed-showcase", help="Load original fictional showcase listings")
    clusters = commands.add_parser(
        "cluster-showcase", help="Evaluate synthetic hierarchical clusters"
    )
    clusters.add_argument("--output", type=Path, required=True)
    incremental = commands.add_parser(
        "incremental-showcase", help="Compare synthetic update behavior"
    )
    incremental.add_argument("--output", type=Path, required=True)
    scale = commands.add_parser(
        "benchmark-scale", help="Measure lexical retrieval on synthetic data"
    )
    scale.add_argument("--output", type=Path, required=True)
    scale.add_argument("--sizes", nargs="+", type=int, default=[1000, 10000])
    scale.add_argument("--queries", type=int, default=100)
    profiling = commands.add_parser(
        "profile", help="Validate and profile canonical CSV/JSONL listings"
    )
    profiling.add_argument("input", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "audit-training":
            from catalog_identity.evaluation.training_audit import run_audit

            report = run_audit(
                load_config(args.config), args.output, args.tokenizer_directory, args.control_report
            )
            print(
                json.dumps(
                    {k: report[k] for k in ("pairs", "pair_collisions", "review_sampling")},
                    indent=2,
                )
            )
            return 0
        if args.command in ("pilot-full", "screen-full", "develop-full"):
            from catalog_identity.evaluation import full_encoder

            config = load_config(args.config)
            if args.command == "pilot-full":
                report = full_encoder.pilot(
                    config, args.output, args.validation_report, args.device
                )
                result = {
                    k: report[k]
                    for k in (
                        "gate_passed",
                        "projected_screening_seconds",
                        "projected_full_fit_validation_seconds",
                    )
                }
            elif args.command == "screen-full":
                report = full_encoder.screen(
                    config, args.output, args.timing, args.control, args.device
                )
                result = {"selected_candidate": report["selected_candidate"]}
            else:
                report = full_encoder.develop(config, args.output, args.screening, args.device)
                result = report["full_validation_classification"]
            print(json.dumps(result, indent=2))
            return 0
        if args.command == "develop-joint":
            from catalog_identity.evaluation.cross_encoder import develop_joint

            report = develop_joint(
                load_config(args.config), args.output, args.screening, args.device
            )
            print(json.dumps(report["full_validation_classification"], indent=2))
            return 0
        if args.command == "pilot-joint":
            from catalog_identity.evaluation.cross_encoder import pilot

            report = pilot(
                load_config(args.config), args.output, args.validation_report, device=args.device
            )
            print(
                json.dumps(
                    {
                        key: report[key]
                        for key in (
                            "gate_passed",
                            "projected_screening_seconds",
                            "projected_full_fit_validation_seconds",
                        )
                    },
                    indent=2,
                )
            )
            return 0
        if args.command == "screen-joint":
            from catalog_identity.evaluation.cross_encoder import screen_joint

            report = screen_joint(
                load_config(args.config), args.output, args.timing, args.control, args.device
            )
            print(json.dumps({"selected_candidate": report["selected_candidate"]}, indent=2))
            return 0
        if args.command == "screen-semantic":
            from catalog_identity.evaluation.semantic_screening import screen_semantic

            report = screen_semantic(load_config(args.config), args.output, args.cache)
            print(json.dumps({"selected_candidate": report["selected_candidate"]}, indent=2))
            return 0
        if args.command == "screen-matchers":
            from catalog_identity.evaluation.screening import screen

            report = screen(load_config(args.config), args.output)
            print(json.dumps({"selected_candidate": report["selected_candidate"]}, indent=2))
            return 0
        if args.command == "validation-errors":
            from catalog_identity.evaluation.validation_errors import validation_errors

            report = validation_errors(load_config(args.config), args.model, args.output)
            print(json.dumps(report["classification"], indent=2))
            return 0
        if args.command == "incremental-showcase":
            from catalog_identity.evaluation.incremental import incremental_showcase

            print(json.dumps(incremental_showcase(args.output)["steps"], indent=2))
            return 0
        if args.command == "cluster-showcase":
            from catalog_identity.evaluation.cluster_showcase import cluster_showcase

            report = cluster_showcase(args.output)
            print(json.dumps(report["levels"], indent=2))
            return 0
        if args.command == "benchmark-scale":
            from catalog_identity.evaluation.scaling import scaling

            report = scaling(args.output, args.sizes, args.queries)
            print(json.dumps(report["runs"], indent=2))
            return 0
        if args.command == "serve":
            import uvicorn

            uvicorn.run(
                "catalog_identity.api.app:create_app",
                factory=True,
                host="127.0.0.1",
                port=args.port,
            )
            return 0
        if args.command in ("worker", "seed-showcase"):
            import os
            import time

            from catalog_identity.storage.catalog import Catalog
            from catalog_identity.storage.database import Database

            database = Database(os.getenv("DATABASE_URL", "sqlite:///catalog.db"))
            model_directory = os.getenv("CATALOG_MODEL_DIR")
            catalog = Catalog(database, Path(model_directory) if model_directory else None)
            try:
                if args.command == "seed-showcase":
                    from catalog_identity.showcase import seed

                    print(json.dumps(seed(catalog), indent=2))
                else:
                    from catalog_identity.storage.worker import run_one

                    while True:
                        job = run_one(catalog, args.config_root, args.output_root)
                        if args.once:
                            break
                        if job is None:
                            time.sleep(1)
            finally:
                database.close()
            return 0
        if args.command == "evaluate-frozen":
            from catalog_identity.evaluation.frozen import evaluate_frozen

            report = evaluate_frozen(load_config(args.config), args.model, args.output, args.hybrid)
            print(json.dumps(report["classification"], indent=2))
            return 0
        if args.command == "retrieval-ablation":
            from catalog_identity.evaluation.retrieval_ablation import retrieval_ablation

            report = retrieval_ablation(
                load_config(args.config), args.output, args.allow_model_download
            )
            print(
                json.dumps(
                    {
                        name: value["labeled_pair_recall_at_k"]
                        for name, value in report["channels"].items()
                    },
                    indent=2,
                )
            )
            return 0
        if args.command == "develop":
            from catalog_identity.evaluation.development import develop

            report = develop(
                load_config(args.config),
                args.output,
                negative_policy=args.negative_policy,
                feature_version=args.feature_version,
                num_leaves=args.num_leaves,
                n_estimators=args.n_estimators,
                learners=tuple(args.learners),
            )
            print(
                json.dumps(
                    {name: value["classification"] for name, value in report["models"].items()},
                    indent=2,
                )
            )
            return 0
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
