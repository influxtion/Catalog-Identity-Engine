"""Measured lexical retrieval costs on explicitly synthetic populations."""

import threading
from pathlib import Path
from time import perf_counter

import numpy as np

from catalog_identity.evaluation.provenance import implementation_manifest
from catalog_identity.evaluation.runner import write_json
from catalog_identity.normalization import normalize
from catalog_identity.retrieval import CandidateIndex
from catalog_identity.schema import Listing


def scaling(output: Path, sizes=(1000, 10000), queries=100, seed=42):
    import psutil

    if output.exists() or any(size < 2 for size in sizes) or queries < 1:
        raise ValueError("use a fresh output directory, positive query count and sizes >= 2")
    report = {
        "synthetic": True,
        "seed": seed,
        "implementation": implementation_manifest(),
        "scope": "Lexical retrieval only; no accuracy or end-to-end service scaling claim.",
        "memory_scope": (
            "Sampled process RSS at 5 ms intervals, including interpreter and imported libraries. "
            "Not an exact peak."
        ),
        "runs": [],
    }
    process = psutil.Process()
    for size in sizes:
        stop = threading.Event()
        samples = [process.memory_info().rss]

        def sample(stop_event=stop, rss_samples=samples):
            while not stop_event.wait(0.005):
                rss_samples.append(process.memory_info().rss)

        monitor = threading.Thread(target=sample, daemon=True)
        monitor.start()
        try:
            records = [
                normalize(
                    Listing(
                        source=f"synthetic-{i % 3}",
                        source_id=str(i),
                        title=(
                            f"Fictional brand {i % 31} model Q{i // 2} "
                            f"storage {64 if i % 2 else 128}GB"
                        ),
                        brand=f"Brand{i % 31}",
                        model=f"Q{i // 2}",
                    )
                )
                for i in range(size)
            ]
            started = perf_counter()
            index = CandidateIndex().fit(records).build(records)
            build_seconds = perf_counter() - started
            rng = np.random.default_rng(seed)
            chosen = rng.choice(size, min(queries, size), replace=False)
            durations = []
            returned = 0
            for i in chosen:
                started = perf_counter()
                returned += len(index.query(records[int(i)], 50))
                durations.append(perf_counter() - started)
            samples.append(process.memory_info().rss)
            report["runs"].append(
                {
                    "listings": size,
                    "queries": len(chosen),
                    "returned_candidates": returned,
                    "build_seconds_including_vocabulary": build_seconds,
                    "build_listings_per_second": size / build_seconds,
                    "query_p50_ms": float(np.percentile(durations, 50) * 1000),
                    "query_p95_ms": float(np.percentile(durations, 95) * 1000),
                    "queries_per_second": len(chosen) / sum(durations),
                    "sampled_peak_rss_bytes": max(samples),
                    "rss_before_bytes": samples[0],
                    "sparse_matrix_bytes": index.matrix.data.nbytes
                    + index.matrix.indices.nbytes
                    + index.matrix.indptr.nbytes,
                }
            )
        finally:
            stop.set()
            monitor.join()
    output.mkdir(parents=True)
    write_json(output / "scaling.json", report)
    return report
