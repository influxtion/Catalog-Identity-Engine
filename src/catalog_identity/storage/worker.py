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
            report = run_evaluation(load_config(path), output_root / job["id"])
            result = {
                "synthetic": report["synthetic"],
                "dataset": report["dataset_name"],
                "threshold_selection": report["threshold_selection"],
                "test": report["splits"]["test"]["pair_classification"],
                "retrieval": report["splits"]["test"]["retrieval"],
                "report_path": str((output_root / job["id"] / "report.json").resolve()),
            }
        catalog.finish_job(job["id"], result=result)
    except Exception as exc:
        catalog.finish_job(job["id"], error=f"{type(exc).__name__}: {exc}"[:2000])
    return job["id"]
