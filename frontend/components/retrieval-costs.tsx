import { RetrievalCostReport } from "@/lib/api";

export function RetrievalCosts({
  report,
}: {
  report: RetrievalCostReport | null;
}) {
  if (!report) return null;
  return (
    <div className="retrieval-chart">
      <h3>Retrieval cost measurements</h3>
      <p>
        Synthetic listings · lexical index · 100 queries per size. Local machine
        measurements, not live service latency.
      </p>
      <div className="table-scroll">
        <table className="metrics-table">
          <thead>
            <tr>
              <th>Listings</th>
              <th>Index build</th>
              <th>Median / p95</th>
              <th>Queries / sec</th>
              <th>Sampled RSS</th>
            </tr>
          </thead>
          <tbody>
            {report.runs.map((run) => (
              <tr key={run.listings}>
                <td>{run.listings.toLocaleString("en-US")}</td>
                <td>{run.build_seconds_including_vocabulary.toFixed(2)} s</td>
                <td>
                  {run.query_p50_ms.toFixed(1)} / {run.query_p95_ms.toFixed(1)}{" "}
                  ms
                </td>
                <td>{run.queries_per_second.toFixed(1)}</td>
                <td>{(run.sampled_peak_rss_bytes / 1048576).toFixed(0)} MiB</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="helper">
        Build includes vocabulary fitting. Memory includes the Python process;
        sampling can miss brief peaks. Measurements exclude service requests and
        embedding inference.
      </p>
    </div>
  );
}
