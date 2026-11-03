from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from catalog_identity.evaluation.runner import file_hash, git_metadata


def implementation_manifest():
    package = Path(__file__).resolve().parents[1]
    root = package.parents[1]
    dependencies = {}
    for name in ("numpy", "scikit-learn", "lightgbm", "sentence-transformers", "faiss-cpu"):
        try:
            dependencies[name] = version(name)
        except PackageNotFoundError:
            pass
    return {
        "git": git_metadata(root),
        "dependencies": dependencies,
        "source_sha256": {
            p.relative_to(package).as_posix(): file_hash(p) for p in sorted(package.rglob("*.py"))
        },
        "lock_sha256": file_hash(root / "uv.lock") if (root / "uv.lock").exists() else None,
    }
