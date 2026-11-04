"""Explicit, fingerprinted CPU/CUDA execution for the unchanged joint architecture."""

import hashlib
import json
import os
import platform
import subprocess
import sys
from importlib import metadata

EXECUTION_VERSION = 1


def fingerprint(manifest):
    body = {key: value for key, value in manifest.items() if key != "fingerprint"}
    return hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def validate_execution(manifest):
    expected = {
        "schema_version",
        "device",
        "python",
        "platform",
        "dependencies",
        "torch_build",
        "cuda_runtime",
        "cudnn_version",
        "gpu",
        "precision",
        "deterministic_algorithms",
        "warn_only",
        "cudnn_benchmark",
        "cudnn_deterministic",
        "attention",
        "autocast",
        "optimizer",
        "cublas_workspace_config",
        "torch_threads",
        "torch_interop_threads",
        "fingerprint",
    }
    if not isinstance(manifest, dict) or set(manifest) != expected:
        raise ValueError("invalid execution manifest fields")
    if (
        manifest["schema_version"] != 1
        or manifest["device"] not in ("cpu", "cuda:0")
        or manifest["precision"] != {"global": "ieee", "cuda_matmul": "ieee", "cudnn": "ieee"}
        or manifest["deterministic_algorithms"] is not True
        or manifest["warn_only"] is not False
        or manifest["cudnn_benchmark"] is not False
        or manifest["cudnn_deterministic"] is not True
        or manifest["attention"] != "math"
        or manifest["autocast"] is not False
        or manifest["optimizer"] != {"foreach": False, "fused": False}
        or manifest["cublas_workspace_config"] != ":4096:8"
        or manifest["torch_threads"] != 14
    ):
        raise ValueError("execution manifest differs from registered execution flags")
    if manifest["device"] == "cuda:0":
        if not isinstance(manifest["gpu"], dict) or set(manifest["gpu"]) != {
            "name",
            "capability",
            "total_memory",
            "driver_version",
        }:
            raise ValueError("CUDA execution requires complete GPU metadata")
    elif manifest["gpu"] is not None:
        raise ValueError("CPU execution cannot claim a selected GPU")
    if manifest["fingerprint"] != fingerprint(manifest):
        raise ValueError("execution manifest fingerprint differs")
    return manifest


def configure_execution(device="cpu"):
    if device not in ("cpu", "cuda", "cuda:0"):
        raise ValueError("device must explicitly be cpu or cuda:0")
    # Must precede any CUDA context initialization. No persistent environment change.
    previous_workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    loaded_torch = sys.modules.get("torch")
    if (
        loaded_torch is not None
        and loaded_torch.cuda.is_initialized()
        and previous_workspace != ":4096:8"
    ):
        raise ValueError("CUDA was initialized before registered workspace configuration")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    import torch

    device = "cuda:0" if device.startswith("cuda") else "cpu"
    if device == "cuda:0" and not torch.cuda.is_available():
        raise ValueError("CUDA explicitly requested but unavailable; no CPU fallback")
    torch.set_num_threads(14)
    torch.manual_seed(42)
    if device == "cuda:0":
        torch.cuda.manual_seed_all(42)
    torch.use_deterministic_algorithms(True, warn_only=False)
    # Torch 2.15's new precision API only; do not mix with legacy allow_tf32 setters.
    torch.backends.fp32_precision = "ieee"
    torch.backends.cuda.matmul.fp32_precision = "ieee"
    torch.backends.cudnn.fp32_precision = "ieee"
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    gpu = None
    if device == "cuda:0":
        properties = torch.cuda.get_device_properties(0)
        driver = (
            subprocess.run(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                check=True,
                capture_output=True,
                text=True,
                timeout=10,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
            .stdout.strip()
            .splitlines()[0]
        )
        gpu = {
            "name": properties.name,
            "capability": list(torch.cuda.get_device_capability(0)),
            "total_memory": properties.total_memory,
            "driver_version": driver,
        }
    packages = {
        distribution.metadata["Name"].lower().replace("_", "-"): distribution.version
        for distribution in metadata.distributions()
        if distribution.metadata.get("Name")
    }
    result = {
        "schema_version": 1,
        "device": device,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "dependencies": dict(sorted(packages.items())),
        "torch_build": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "gpu": gpu,
        "precision": {
            "global": torch.backends.fp32_precision,
            "cuda_matmul": torch.backends.cuda.matmul.fp32_precision,
            "cudnn": torch.backends.cudnn.fp32_precision,
        },
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "attention": "math",
        "autocast": False,
        "optimizer": {"foreach": False, "fused": False},
        "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
    }
    result["fingerprint"] = fingerprint(result)
    return validate_execution(result)


def require_same_execution(expected, actual):
    validate_execution(expected)
    validate_execution(actual)
    if expected["fingerprint"] != actual["fingerprint"]:
        raise ValueError("execution environment/device differs from timing or frozen selection")
