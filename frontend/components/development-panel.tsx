"use client";

import { useState } from "react";
import {
  api,
  DevelopmentBenchmark,
  DevelopmentModel,
  EvaluationReport,
  shortId,
} from "@/lib/api";

function ModelTable({ models }: { models: DevelopmentModel[] }) {
  return (
    <div className="table-scroll">
      <table className="metrics-table">
        <thead>
          <tr>
            <th>Matcher</th>
            <th>Correct / false matches</th>
            <th>Precision</th>
            <th>Recall</th>
            <th>Average precision</th>
          </tr>
        </thead>
        <tbody>
          {models.map((model) => (
            <tr key={model.name}>
              <td>
                <b>{model.name}</b>
              </td>
              <td>
                {model.true_positives} / {model.false_positives}
              </td>
              <td>
                {model.precision === null
                  ? "Undefined"
                  : `${(model.precision * 100).toFixed(1)}%`}
              </td>
              <td>{((model.recall || 0) * 100).toFixed(1)}%</td>
              <td>{model.average_precision.toFixed(3)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function DevelopmentPanel({
  benchmark,
  reports,
  onQueued,
}: {
  benchmark: DevelopmentBenchmark | null;
  reports: EvaluationReport[];
  onQueued: () => Promise<void>;
}) {
  const [size, setSize] = useState("large");
  const [version, setVersion] = useState("3");
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");
  return (
    <div className="retrieval-chart">
      <h3>Precision development</h3>
      {benchmark && (
        <>
          <p>
            {benchmark.protocol} · {benchmark.pairs.toLocaleString("en-US")}{" "}
            validation pairs
          </p>
          <ModelTable models={benchmark.models} />
          <p className="evaluation-note">{benchmark.warning}</p>
        </>
      )}
      <div className="development-controls">
        <label>
          Development split
          <select
            aria-label="Development split"
            value={size}
            onChange={(e) => setSize(e.target.value)}
          >
            <option value="large">Large official split</option>
            <option value="small">Small official split</option>
          </select>
        </label>
        <label>
          Pair features
          <select
            aria-label="Pair features"
            value={version}
            onChange={(e) => setVersion(e.target.value)}
          >
            <option value="3">Specification features</option>
            <option value="2">Previous text features</option>
          </select>
        </label>
        <button
          className="secondary"
          disabled={busy}
          onClick={async () => {
            setBusy(true);
            setError("");
            setMessage("");
            try {
              await api("/evaluations", {
                config_name:
                  size === "large"
                    ? "wdc-products-large.json"
                    : "wdc-products.example.json",
                mode: "validation",
                feature_version: version,
              });
              setMessage(
                "Validation comparison queued. Test data stays unopened.",
              );
              await onQueued();
            } catch (error) {
              setError((error as Error).message);
            } finally {
              setBusy(false);
            }
          }}
        >
          {busy ? "Queueing…" : "Queue validation comparison"}
        </button>
      </div>
      <p className="helper">
        Requires the chosen local train/validation files and an active worker.
        Completed comparisons appear below.
      </p>
      {message && (
        <p role="status" className="helper">
          {message}
        </p>
      )}
      {error && (
        <p role="alert" className="conflict-text">
          {error}
        </p>
      )}
      {reports
        .filter((report) => report.data.phase === "validation_only")
        .map((report) => (
          <div key={report.id}>
            <h4>
              Run {shortId(report.id)} · {report.data.dataset}
              {report.data.synthetic ? " · Synthetic" : ""}
            </h4>
            <p className="helper">
              Full-validation diagnostics include development data.
            </p>
            <ModelTable
              models={Object.entries(report.data.models || {}).map(
                ([name, model]) => ({
                  name,
                  ...model.classification,
                  average_precision: model.average_precision,
                }),
              )}
            />
          </div>
        ))}
    </div>
  );
}
