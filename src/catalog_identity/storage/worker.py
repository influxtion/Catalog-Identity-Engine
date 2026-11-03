from pathlib import Path


def run_one(catalog, config_root: Path, output_root: Path):
    job = catalog.claim_job()
    if job is None:
        return None
    try:
        if job["kind"] == "rebuild":
            result = catalog.rebuild()
        else:
            from catalog_identity.evaluation.config import load_config
            from catalog_identity.evaluation.runner import run_evaluation

            name = job["payload"].get("config_name", "")
            path = (config_root / name).resolve()
            if not name or Path(name).name != name or path.parent != config_root.resolve():
                raise ValueError("evaluation config must be a filename in the configured directory")
            config = load_config(path)
            output = output_root / job["id"]
            mode = job["payload"].get("mode", "baseline")
            if mode == "validation":
                result = develop_job(config, output, job["payload"].get("feature_version", "3"))
            elif mode == "baseline":
                report = run_evaluation(config, output)
                result = {
                    "phase": "baseline_test_evaluation",
                    "synthetic": report["synthetic"],
                    "dataset": report["dataset_name"],
                    "threshold_selection": report["threshold_selection"],
                    "test": report["splits"]["test"]["pair_classification"],
                    "retrieval": report["splits"]["test"]["retrieval"],
                    "report_path": str((output / "report.json").resolve()),
                }
            else:
                raise ValueError("unknown evaluation mode")
        catalog.finish_job(job["id"], result=result)
    except Exception as exc:
        catalog.finish_job(job["id"], error=f"{type(exc).__name__}: {exc}"[:2000])
    return job["id"]


def develop_job(config, output, feature_version):
    from catalog_identity.evaluation.development import develop
    from catalog_identity.evaluation.validation_errors import validation_errors

    report = develop(config, output, feature_version=feature_version)
    models = {}
    for learner in report["models"]:
        diagnostic = validation_errors(config, output / learner, output / f"diagnostics-{learner}")
        models[learner] = {
            "classification": diagnostic["classification"],
            "policy_classification": report["models"][learner]["classification"],
            "average_precision": diagnostic["probability_quality"]["average_precision"],
            "match_threshold": diagnostic["match_threshold"],
        }
    return {
        "phase": "validation_only",
        "test_opened": False,
        "synthetic": config.synthetic,
        "dataset": config.dataset_name,
        "protocol": config.protocol.model_dump() if config.protocol else None,
        "feature_version": feature_version,
        "models": models,
        "inputs": report["inputs"],
        "warning": "Development-data diagnostics; not an independent performance estimate.",
        "report_path": str((output / "validation.json").resolve()),
    }
