"use client";

import { useCallback, useEffect, useState } from "react";
import { Icon } from "@/components/icons";
import { ProductIcon, Tags } from "@/components/product";
import { PairDialog } from "@/components/pair-dialog";
import { ImportDialog } from "@/components/import-dialog";
import { RetrievalCosts } from "@/components/retrieval-costs";
import { DevelopmentPanel } from "@/components/development-panel";
import {
  api,
  Benchmark,
  Candidate,
  Identity,
  Job,
  Listing,
  Overview,
  Pair,
  RetrievalCostReport,
  DevelopmentBenchmark,
  EvaluationReport,
  shortId,
  priceLabel,
} from "@/lib/api";

const navigation = [
  { key: "explorer", label: "Identity explorer", icon: "grid" },
  { key: "review", label: "Review queue", icon: "review" },
  { key: "graph", label: "Identity graph", icon: "graph" },
  { key: "evaluation", label: "Evaluations", icon: "chart" },
];

export default function Dashboard() {
  const [section, setSection] = useState("explorer");
  const [listings, setListings] = useState<Listing[]>([]);
  const [families, setFamilies] = useState<Identity[]>([]);
  const [variants, setVariants] = useState<Identity[]>([]);
  const [queue, setQueue] = useState<Pair[]>([]);
  const [overview, setOverview] = useState<Overview | null>(null);
  const [jobs, setJobs] = useState<Job[]>([]);
  const [benchmarks, setBenchmarks] = useState<Benchmark[]>([]);
  const [costs, setCosts] = useState<RetrievalCostReport | null>(null);
  const [development, setDevelopment] = useState<DevelopmentBenchmark | null>(
    null,
  );
  const [reports, setReports] = useState<EvaluationReport[]>([]);
  const [query, setQuery] = useState("");
  const [selected, setSelected] = useState<string | null>(null);
  const [candidates, setCandidates] = useState<Candidate[]>([]);
  const [pair, setPair] = useState<Pair | null>(null);
  const [importing, setImporting] = useState(false);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState(false);
  const [notice, setNotice] = useState("");
  const [graphLevel, setGraphLevel] = useState("family");
  const load = useCallback(async () => {
    try {
      // Read job status first so completed-job views cannot predate its commit.
      const j = await api<Job[]>("/jobs");
      const [rows, f, v, q, o, b, c, d, e] = await Promise.all([
        api<Listing[]>("/listings?limit=200"),
        api<Identity[]>("/identities?level=family"),
        api<Identity[]>("/identities?level=variant"),
        api<Pair[]>("/reviews"),
        api<Overview>("/overview"),
        api<Benchmark[]>("/benchmarks"),
        api<RetrievalCostReport | null>("/performance"),
        api<DevelopmentBenchmark | null>("/development-benchmarks"),
        api<EvaluationReport[]>("/evaluations"),
      ]);
      setListings(rows);
      setFamilies(f);
      setVariants(v);
      setQueue(q);
      setOverview(o);
      setJobs(j);
      setBenchmarks(b);
      setCosts(c);
      setDevelopment(d);
      setReports(e);
      setSelected((current) =>
        current && rows.some((row) => row.id === current)
          ? current
          : rows.find((row) => row.data.source_id === "h1-black")?.id ||
            rows[0]?.id ||
            null,
      );
    } catch (error) {
      setError((error as Error).message);
    } finally {
      setLoading(false);
    }
  }, []);
  useEffect(() => {
    load();
  }, [load]);
  useEffect(() => {
    if (!jobs.some((job) => ["pending", "running"].includes(job.status)))
      return;
    const timer = setInterval(load, 3000);
    return () => clearInterval(timer);
  }, [jobs, load]);
  useEffect(() => {
    let canceled = false;
    setCandidates([]);
    if (selected)
      api<Candidate[]>(`/listings/${selected}/candidates?k=4`)
        .then((value) => {
          if (!canceled) setCandidates(value);
        })
        .catch((error) => {
          if (!canceled) setError(error.message);
        });
    return () => {
      canceled = true;
    };
  }, [selected, overview?.catalog_revision]);
  const selectedListing = listings.find((row) => row.id === selected);
  const familyOf = (id: string) =>
    families.find((group) => group.members.includes(id));
  const variantOf = (id: string) =>
    variants.find((group) => group.members.includes(id));
  const visible = listings.filter((row) =>
    `${row.data.title} ${row.data.brand} ${row.data.model} ${row.data.source} ${row.data.source_id} ${familyOf(row.id)?.id || ""} ${variantOf(row.id)?.id || ""}`
      .toLowerCase()
      .includes(query.toLowerCase()),
  );
  const inspect = async (left: string, right: string) => {
    setBusy(true);
    try {
      setPair(await api<Pair>("/pairs/inspect", { left, right }));
    } catch (error) {
      setError((error as Error).message);
    } finally {
      setBusy(false);
    }
  };
  const rebuild = async () => {
    try {
      await api("/identities/rebuild", {});
      setNotice("Rebuild queued. Keep the worker running to process it.");
      await load();
    } catch (error) {
      setError((error as Error).message);
    }
  };
  const synthetic =
    listings.length > 0 &&
    listings.every((row) => row.data.raw?.synthetic === true);
  const currentNav = navigation.find((item) => item.key === section)!;
  return (
    <div className="workspace">
      <aside className="sidebar">
        <a className="brand" href="/" aria-label="Catalog Identity Engine home">
          <span className="brand-mark">
            <Icon name="graph" size={27} />
          </span>
          <span>
            catalog<span className="brand-sub">IDENTITY ENGINE</span>
          </span>
        </a>
        <div className="workspace-label">
          <span className="workspace-avatar">C</span>
          <span>
            Catalog workspace<small>Local development</small>
          </span>
          <span className="workspace-dot" />
        </div>
        <div className="nav-label">WORKSPACE</div>
        <nav aria-label="Main navigation">
          {navigation.map((item) => (
            <button
              key={item.key}
              className={section === item.key ? "active" : ""}
              onClick={() => setSection(item.key)}
            >
              <Icon name={item.icon} />
              <span>{item.label}</span>
              {item.key === "review" && queue.length > 0 && (
                <em>{queue.length}</em>
              )}
            </button>
          ))}
        </nav>
        <div className="sidebar-bottom">
          <div className="sidebar-note">
            <span className="status-dot" />
            <span>{overview ? "Catalog connected" : "Connecting…"}</span>
          </div>
          <p>
            Every match has evidence.
            <br />
            Every identity has a history.
          </p>
          <div className="sidebar-version">
            Revision {overview?.catalog_revision ?? "—"}
            <span>v0.2</span>
          </div>
        </div>
      </aside>
      <main>
        <div className="topbar">
          <span>
            Workspace <span className="slash">/</span> {currentNav.label}
          </span>
          <span className="environment">
            <i />
            {synthetic ? "Synthetic showcase" : "Local catalog"}
          </span>
        </div>
        <div className="page-heading">
          <div>
            <div className="eyebrow">PRODUCT RESOLUTION</div>
            <h1>{currentNav.label}</h1>
            <p>
              {section === "explorer"
                ? "Different listings. One clear view of the product."
                : section === "review"
                  ? "Resolve uncertainty with the evidence in front of you."
                  : section === "graph"
                    ? "Follow the links. Inspect the boundaries."
                    : "Measured results, with their limits in view."}
            </p>
          </div>
          <div className="heading-actions">
            <button className="secondary" onClick={rebuild}>
              <Icon name="refresh" size={16} />
              Rebuild identities
            </button>
            <button className="primary" onClick={() => setImporting(true)}>
              <Icon name="plus" size={17} />
              Add listings
            </button>
          </div>
        </div>
        {error && (
          <div className="banner error" role="alert">
            <span>{error}</span>
            <button
              onClick={() => {
                setError("");
                load();
              }}
            >
              Retry
            </button>
            <button onClick={() => setError("")} aria-label="Dismiss error">
              ×
            </button>
          </div>
        )}
        {notice && (
          <div className="banner" role="status">
            {notice}
            <button onClick={() => setNotice("")} aria-label="Dismiss notice">
              ×
            </button>
          </div>
        )}
        {overview?.identities_stale && (
          <div className="banner warning">
            Identity memberships are awaiting a rebuild. Displayed memberships
            may be outdated.
          </div>
        )}
        <div className="stats">
          <div>
            <span>Loaded listings</span>
            <strong>
              {loading ? "—" : listings.length}
              <small>
                across {new Set(listings.map((x) => x.data.source)).size}{" "}
                sources
              </small>
            </strong>
          </div>
          <div>
            <span>Product families</span>
            <strong>
              {families.length}
              <small>underlying models</small>
            </strong>
          </div>
          <div>
            <span>Exact variants</span>
            <strong>
              {variants.length}
              <small>distinct configurations</small>
            </strong>
          </div>
          <div>
            <span>Needs review</span>
            <strong className="amber">
              {queue.length}
              <small>scored pairs to inspect</small>
            </strong>
          </div>
        </div>
        {loading ? (
          <div className="empty">Connecting to your catalog…</div>
        ) : (
          <>
            {section === "explorer" && (
              <div className="explorer-grid">
                <section className="listing-panel">
                  <div className="panel-toolbar">
                    <label className="search">
                      <Icon name="search" size={18} />
                      <input
                        aria-label="Search listings"
                        placeholder="Search listings, models, sources…"
                        value={query}
                        onChange={(event) => setQuery(event.target.value)}
                      />
                      <kbd>⌕</kbd>
                    </label>
                    <span className="count">{visible.length} listings</span>
                  </div>
                  <div className="table-scroll">
                    <table className="listing-table">
                      <thead>
                        <tr>
                          <th>Product listing</th>
                          <th>Source</th>
                          <th>Family</th>
                          <th>Price</th>
                        </tr>
                      </thead>
                      <tbody>
                        {visible.map((row) => (
                          <tr
                            key={row.id}
                            className={selected === row.id ? "selected" : ""}
                            onClick={() => setSelected(row.id)}
                          >
                            <td>
                              <button
                                className="product-cell"
                                aria-label={`Select ${row.data.title}`}
                                onClick={() => setSelected(row.id)}
                              >
                                <ProductIcon listing={row} />
                                <span>
                                  <b>{row.data.title}</b>
                                  <small>
                                    {row.data.brand || "Unknown brand"} <i>·</i>{" "}
                                    {row.data.model || "Unknown model"}
                                  </small>
                                  <Tags listing={row} />
                                </span>
                              </button>
                            </td>
                            <td>
                              <span
                                className={`source-badge ${row.data.source}`}
                              >
                                {row.data.source}
                              </span>
                            </td>
                            <td>
                              <code>
                                {familyOf(row.id)
                                  ? shortId(familyOf(row.id)!.id)
                                  : "Pending"}
                              </code>
                            </td>
                            <td className="price">
                              {priceLabel(row)}
                              <small>{row.data.currency || ""}</small>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                  {visible.length === 0 && (
                    <div className="empty">
                      {listings.length
                        ? "No listings match your search."
                        : "Your catalog is empty. Add listings to begin."}
                    </div>
                  )}
                  <footer className="table-footer">
                    <span>
                      {synthetic
                        ? "Original fictional listings · no benchmark data"
                        : "Showing up to 200 loaded listings"}
                    </span>
                    <span>
                      Source IDs preserved <span className="tiny-check">✓</span>
                    </span>
                  </footer>
                </section>
                <aside className="detail-panel">
                  {selectedListing ? (
                    <>
                      <div className="detail-eyebrow">
                        LISTING DETAIL{" "}
                        <span>REV {selectedListing.revision}</span>
                      </div>
                      <div className="product-stage">
                        <ProductIcon listing={selectedListing} large />
                        <span>
                          {selectedListing.data.category || "Uncategorized"}
                        </span>
                      </div>
                      <div className="detail-content">
                        <span className="source-name">
                          {selectedListing.data.source}
                        </span>
                        <h2>{selectedListing.data.title}</h2>
                        <Tags listing={selectedListing} />
                        {(["family", "variant"] as const).map((level) => {
                          const group =
                            level === "family"
                              ? familyOf(selectedListing.id)
                              : variantOf(selectedListing.id);
                          return (
                            <button
                              key={level}
                              className="identity-summary"
                              disabled={!group}
                              aria-label={`Show ${level} members`}
                              onClick={() => group && setQuery(group.id)}
                            >
                              <Icon name="link" size={18} />
                              <div>
                                <span>
                                  {level === "family"
                                    ? "Product family"
                                    : "Variant group"}
                                </span>
                                <code>
                                  {group?.id.slice(0, 18) || "Awaiting rebuild"}
                                </code>
                              </div>
                              <span className="member-count">
                                {group?.members.length ?? 1} listings
                              </span>
                            </button>
                          );
                        })}
                        <div className="section-heading">
                          <h3>Candidate matches</h3>
                          <span>Top {candidates.length}</span>
                        </div>
                        <div className="candidate-list">
                          {candidates.map((candidate) => {
                            const row = listings.find(
                              (x) => x.id === candidate.listing_id,
                            );
                            return (
                              <button
                                key={candidate.listing_id}
                                disabled={busy}
                                onClick={() =>
                                  inspect(
                                    selectedListing.id,
                                    candidate.listing_id,
                                  )
                                }
                                aria-label={`Compare with ${row?.data.title || shortId(candidate.listing_id)}`}
                              >
                                <span className="rank">
                                  {candidate.rank.toString().padStart(2, "0")}
                                </span>
                                <span>
                                  <b>
                                    {row?.data.title ||
                                      shortId(candidate.listing_id)}
                                  </b>
                                  <small>
                                    {candidate.sources.join(" + ")} · similarity{" "}
                                    {(candidate.text_score * 100).toFixed(0)}%
                                  </small>
                                </span>
                                <Icon name="arrow" size={16} />
                              </button>
                            );
                          })}
                        </div>
                        <p className="helper">
                          Retrieval similarity finds candidates. It is not a
                          match probability.
                        </p>
                      </div>
                    </>
                  ) : (
                    <div className="empty">
                      Select a listing to inspect its evidence.
                    </div>
                  )}
                </aside>
              </div>
            )}
            {section === "review" && (
              <section className="content-panel">
                <div className="section-heading">
                  <h2>Pairs awaiting a decision</h2>
                  <span>{queue.length} open</span>
                </div>
                {queue.length ? (
                  queue.map((item) => (
                    <button
                      className="review-row"
                      key={item.id}
                      onClick={() => inspect(item.left, item.right)}
                    >
                      <span className="review-symbol">
                        <Icon name="review" />
                      </span>
                      <span>
                        <b>
                          {listings.find((x) => x.id === item.left)?.data
                            .title || shortId(item.left)}
                        </b>
                        <small>
                          with{" "}
                          {listings.find((x) => x.id === item.right)?.data
                            .title || shortId(item.right)}
                        </small>
                      </span>
                      <span className="pill warning">
                        {item.hierarchy.missing.length
                          ? "Missing attributes"
                          : "Uncertain score"}
                      </span>
                      <Icon name="arrow" />
                    </button>
                  ))
                ) : (
                  <div className="empty">
                    No scored pairs are awaiting review. Inspect a candidate
                    pair to add evidence.
                  </div>
                )}
                <p className="helper">
                  Reviews apply to the listing revisions you inspect. Changes
                  invalidate earlier decisions.
                </p>
              </section>
            )}
            {section === "graph" && (
              <section className="content-panel">
                <div className="section-heading">
                  <div>
                    <h2>Connections with constraints</h2>
                    <p>
                      Solid links were accepted. Dashed links were rejected.
                    </p>
                  </div>
                  <div className="segmented">
                    <button
                      className={graphLevel === "family" ? "active" : ""}
                      onClick={() => setGraphLevel("family")}
                    >
                      Families
                    </button>
                    <button
                      className={graphLevel === "variant" ? "active" : ""}
                      onClick={() => setGraphLevel("variant")}
                    >
                      Variants
                    </button>
                  </div>
                </div>
                <div className="graph-stage">
                  <svg
                    viewBox={`0 0 920 ${Math.max(300, Math.ceil(listings.length / 6) * 140)}`}
                    role="group"
                    aria-label="Identity relationship graph"
                  >
                    {[
                      ...(overview?.graph.accepted || []),
                      ...(overview?.graph.rejected || []),
                    ]
                      .filter((edge) => edge.level === graphLevel)
                      .map((edge, i) => {
                        const a = listings.findIndex((x) => x.id === edge.left),
                          b = listings.findIndex((x) => x.id === edge.right);
                        if (a < 0 || b < 0) return null;
                        return (
                          <line
                            key={i}
                            x1={70 + (a % 6) * 155}
                            y1={65 + Math.floor(a / 6) * 140}
                            x2={70 + (b % 6) * 155}
                            y2={65 + Math.floor(b / 6) * 140}
                            stroke={edge.reason ? "#b86843" : "#91b59b"}
                            strokeWidth={edge.reason ? 2 : 3}
                            strokeDasharray={edge.reason ? "6 5" : undefined}
                          >
                            <title>
                              {edge.reason || "Accepted structured evidence"}
                            </title>
                          </line>
                        );
                      })}
                    {listings.map((listing, i) => (
                      <g
                        key={listing.id}
                        role="button"
                        tabIndex={0}
                        aria-label={`Open ${listing.data.title} from ${listing.data.source}`}
                        onKeyDown={(event) => {
                          if (event.key === "Enter" || event.key === " ") {
                            event.preventDefault();
                            setSelected(listing.id);
                            setSection("explorer");
                          }
                        }}
                        transform={`translate(${70 + (i % 6) * 155},${65 + Math.floor(i / 6) * 140})`}
                        onClick={() => {
                          setSelected(listing.id);
                          setSection("explorer");
                        }}
                        style={{ cursor: "pointer" }}
                      >
                        <circle
                          r="26"
                          fill="#f7faf5"
                          stroke="#3d6550"
                          strokeWidth="2"
                        />
                        <text
                          textAnchor="middle"
                          dy="5"
                          fontSize="12"
                          fontWeight="600"
                          fill="#264632"
                        >
                          {listing.data.model || "?"}
                        </text>
                        <text
                          textAnchor="middle"
                          y="47"
                          fontSize="10"
                          fill="#58665e"
                        >
                          {listing.data.source} ·{" "}
                          {listing.data.source_id.slice(0, 12)}
                        </text>
                      </g>
                    ))}
                  </svg>
                </div>
                <div className="graph-legend">
                  <span>
                    <i className="accepted-line" />
                    Accepted evidence
                  </span>
                  <span>
                    <i className="rejected-line" />
                    Rejected merge
                  </span>
                  <span>
                    {overview?.graph.rejected?.filter(
                      (x) => x.level === graphLevel,
                    ).length || 0}{" "}
                    rejected links
                  </span>
                </div>
                <div className="section-heading history-heading">
                  <h3>Identity history</h3>
                  <span>{overview?.history.length || 0} recorded events</span>
                </div>
                <div className="history-list">
                  {overview?.history
                    .slice(-6)
                    .reverse()
                    .map((event, i) => (
                      <div key={i}>
                        <span className="history-dot" />
                        <b>{event.event.replaceAll("_", " ")}</b>
                        <span>{event.level}</span>
                        <code>
                          {shortId(
                            event.identity_id || event.previous_id || "",
                          )}
                        </code>
                        <small>revision {event.revision}</small>
                      </div>
                    ))}
                </div>
              </section>
            )}
            {section === "evaluation" && (
              <section className="content-panel">
                <div className="section-heading">
                  <div>
                    <h2>Benchmark progress</h2>
                    <p>
                      Official WDC entity labels. Family and variant quality
                      require separate evaluation.
                    </p>
                  </div>
                  <span className="pill warning">Precision target not met</span>
                </div>
                <div className="evaluation-note">
                  The improved matcher finds more matches, but 80% test
                  precision falls short of the 99% aspiration. Its scores are
                  inspection evidence, not authorization to merge families or
                  variants.
                </div>
                <table className="metrics-table">
                  <thead>
                    <tr>
                      <th>Model</th>
                      <th>Test precision</th>
                      <th>Test recall</th>
                      <th>Average precision</th>
                    </tr>
                  </thead>
                  <tbody>
                    {benchmarks.map((item) => (
                      <tr key={item.name}>
                        <td>
                          <b>{item.name}</b>
                          <small>{item.phase}</small>
                        </td>
                        <td>
                          {item.precision === null
                            ? "Undefined"
                            : `${(item.precision * 100).toFixed(1)}%`}
                        </td>
                        <td>{((item.recall || 0) * 100).toFixed(1)}%</td>
                        <td>{item.average_precision.toFixed(3)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
                <div className="retrieval-chart">
                  <h3>Candidate retrieval recall</h3>
                  <p>
                    Directed gold matches found in the 1,000-offer test
                    population.
                  </p>
                  {benchmarks.map((item) => (
                    <div className="chart-group" key={item.name}>
                      <h4>
                        {item.name === "Original logistic"
                          ? "Lexical retrieval"
                          : "Hybrid retrieval"}
                      </h4>
                      {Object.entries(item.retrieval).map(([k, value]) => (
                        <div className="bar-row" key={k}>
                          <span>Top {k}</span>
                          <div>
                            <i style={{ width: `${value * 100}%` }} />
                          </div>
                          <b>{(value * 100).toFixed(1)}%</b>
                        </div>
                      ))}
                    </div>
                  ))}
                </div>
                <RetrievalCosts report={costs} />
                <DevelopmentPanel
                  benchmark={development}
                  reports={reports}
                  onQueued={load}
                />
                <div className="section-heading history-heading">
                  <h3>Evaluation jobs</h3>
                  <button
                    className="secondary"
                    onClick={async () => {
                      try {
                        await api("/evaluations", {
                          config_name: "wdc-products.example.json",
                          mode: "baseline",
                        });
                        setNotice(
                          "Evaluation queued. This runs the original baseline against locally supplied WDC files.",
                        );
                        await load();
                      } catch (error) {
                        setError((error as Error).message);
                      }
                    }}
                  >
                    Queue baseline evaluation
                  </button>
                </div>
                <p className="helper">
                  Runs require local benchmark files and an active worker.
                  Repeated test runs are not a model-selection workflow.
                </p>
                {jobs
                  .filter((job) => job.kind === "evaluate")
                  .map((job) => (
                    <div className="job-row" key={job.id}>
                      <code>{shortId(job.id)}</code>
                      <span className="pill">{job.status}</span>
                      {job.error && (
                        <span className="conflict-text">{job.error}</span>
                      )}
                    </div>
                  ))}
              </section>
            )}
          </>
        )}
        <footer className="page-footer">
          <span>
            Catalog Identity Engine <span>·</span> Independent, open-source
            product resolution
          </span>
          <span>
            {
              jobs.filter((job) => ["pending", "running"].includes(job.status))
                .length
            }{" "}
            active jobs
          </span>
        </footer>
      </main>
      {pair && (
        <PairDialog
          pair={pair}
          listings={listings}
          onClose={() => {
            setPair(null);
            load();
          }}
          onSaved={load}
        />
      )}
      {importing && (
        <ImportDialog
          onClose={() => setImporting(false)}
          onImported={() => {
            setNotice("Listings saved. Identity rebuild is queued.");
            load();
          }}
        />
      )}
    </div>
  );
}
