"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { Icon } from "@/components/icons";
import "./annotation.css";

type Relation = "same" | "different" | "uncertain" | null;
type Evidence = {
  side: "left" | "right";
  field: string;
  quote: string;
  role: string;
  interpretation: string;
};
type Source = { url: string; checked_on: string; note: string };
type Judgment = {
  input_sufficiency: string | null;
  benchmark_identity: Relation;
  family_relation: Relation;
  variant_relation: Relation;
  evidence: Evidence[];
  sources: Source[];
  reason: string;
};
type Listing = {
  title: string;
  brand?: string;
  description?: string;
  category?: string;
  model?: string;
  gtin?: string;
  attributes: Record<string, string>;
  price?: string;
  currency?: string;
  product_url?: string;
};
type Case = {
  review_id: string;
  left: Listing;
  right: Listing;
  case_hash: string;
  guideline_version: string;
  revision: number;
  status: string;
  judgment: Judgment;
  resolution?: string;
  independent_reviews?: { reviewer: string; judgment: Judgment }[];
};
type QueueItem = {
  id: string;
  left_title: string;
  right_title: string;
  status: string;
  resolution: string | null;
};
type Session = {
  name: string;
  role: string;
  batch_id: string;
  batch: { name: string; case_count: number; guideline_version: string };
};
const fields = ["title", "brand", "description", "category", "model", "gtin"];
const roles = {
  own_model: "Own product model / MPN",
  series: "Product series",
  compatibility: "Compatible device",
  measurement: "Measurement",
  variant_attribute: "Variant / configuration",
  product_type: "Product type",
  manufacturer: "Manufacturer",
  other: "Other observation",
};
const label = (value: string) => value.replaceAll("_", " ");

async function request<T>(
  key: string,
  path: string,
  body?: unknown,
): Promise<T> {
  const response = await fetch(`/api/catalog/annotations${path}`, {
    method: body ? "POST" : "GET",
    cache: "no-store",
    headers: {
      Authorization: `Bearer ${key}`,
      "Content-Type": "application/json",
    },
    body: body ? JSON.stringify(body) : undefined,
  });
  const result = await response.json();
  if (!response.ok)
    throw new Error(
      typeof result.detail === "string"
        ? result.detail
        : Array.isArray(result.detail)
          ? result.detail.map((e: { msg: string }) => e.msg).join("; ")
          : "Unable to save. Try again.",
    );
  return result;
}

function Observation({ side, listing }: { side: string; listing: Listing }) {
  return (
    <article className="ann-observation" aria-label={`${side} listing`}>
      <div className="ann-eyebrow">{side} listing</div>
      <h2>{listing.title || "Untitled listing"}</h2>
      <dl>
        {[
          "brand",
          "category",
          "model",
          "gtin",
          ...(listing.price ? ["price", "currency"] : []),
          ...(listing.product_url ? ["product_url"] : []),
        ].map((field) => (
          <div key={field}>
            <dt>{label(field)}</dt>
            <dd>
              {(listing[field as keyof Listing] as string) || (
                <span className="ann-missing">Not supplied</span>
              )}
            </dd>
          </div>
        ))}
      </dl>
      <details open>
        <summary>Description</summary>
        <p className="ann-description">
          {listing.description || "No description supplied."}
        </p>
      </details>
      {Object.keys(listing.attributes || {}).length > 0 && (
        <details>
          <summary>Attributes</summary>
          <dl>
            {Object.entries(listing.attributes).map(([name, value]) => (
              <div key={name}>
                <dt>{name}</dt>
                <dd>{value || "Not supplied"}</dd>
              </div>
            ))}
          </dl>
        </details>
      )}
    </article>
  );
}

export default function Annotate() {
  const [key, setKey] = useState("");
  const [entry, setEntry] = useState("");
  const [session, setSession] = useState<Session | null>(null);
  const [queue, setQueue] = useState<QueueItem[]>([]);
  const [current, setCurrent] = useState<Case | null>(null);
  const [filter, setFilter] = useState("pending");
  const [busy, setBusy] = useState(false);
  const [dirty, setDirty] = useState(false);
  const [confirming, setConfirming] = useState(false);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");

  async function connect(accessKey: string) {
    setBusy(true);
    setError("");
    try {
      const active = await request<Session>(accessKey, "/session");
      const items = await request<QueueItem[]>(accessKey, "/cases");
      const next = items.find((item) => item.status !== "submitted");
      const selected = next
        ? await request<Case>(accessKey, `/cases/${next.id}`)
        : null;
      setSession(active);
      setQueue(items);
      setKey(accessKey);
      setEntry("");
      sessionStorage.setItem("annotation-access-key", accessKey);
      setCurrent(selected);
      setDirty(false);
      setNotice("");
      setConfirming(false);
    } catch (e) {
      setError((e as Error).message);
      sessionStorage.removeItem("annotation-access-key");
    } finally {
      setBusy(false);
    }
  }
  useEffect(() => {
    const saved = sessionStorage.getItem("annotation-access-key");
    if (saved) void connect(saved);
  }, []);
  useEffect(() => {
    const warn = (event: BeforeUnloadEvent) => {
      if (dirty) {
        event.preventDefault();
        event.returnValue = "";
      }
    };
    window.addEventListener("beforeunload", warn);
    return () => window.removeEventListener("beforeunload", warn);
  }, [dirty]);

  async function openCase(id: string) {
    if (dirty && !window.confirm("Discard unsaved changes to this review?"))
      return;
    setBusy(true);
    setError("");
    setNotice("");
    setConfirming(false);
    try {
      setCurrent(await request<Case>(key, `/cases/${id}`));
      setDirty(false);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }
  function change(patch: Partial<Judgment>) {
    if (!current) return;
    setCurrent({ ...current, judgment: { ...current.judgment, ...patch } });
    setDirty(true);
    setNotice("");
    setConfirming(false);
    setFilter("pending");
  }
  async function save(submit: boolean) {
    if (!current) return;
    setBusy(true);
    setError("");
    try {
      const result = await request<{ revision: number; status: string }>(
        key,
        `/cases/${current.review_id}`,
        {
          revision: current.revision,
          case_hash: current.case_hash,
          guideline_version: current.guideline_version,
          judgment: current.judgment,
          submit,
        },
      );
      setCurrent({ ...current, ...result });
      setDirty(false);
      setConfirming(false);
      setNotice(
        submit
          ? "Submitted and locked. Thank you for recording the evidence."
          : "Draft saved. You can return to it later.",
      );
      setQueue(await request<QueueItem[]>(key, "/cases"));
    } catch (e) {
      setError((e as Error).message);
      setConfirming(false);
    } finally {
      setBusy(false);
    }
  }
  function logout() {
    if (dirty && !window.confirm("Discard unsaved changes and sign out?"))
      return;
    sessionStorage.removeItem("annotation-access-key");
    setKey("");
    setSession(null);
    setQueue([]);
    setCurrent(null);
    setDirty(false);
    setNotice("");
    setError("");
    setConfirming(false);
  }
  const done = queue.filter((item) => item.status === "submitted").length;
  const visible = queue.filter(
    (item) =>
      filter === "all" ||
      (filter === "submitted"
        ? item.status === "submitted"
        : item.status !== "submitted"),
  );
  const judgment = current?.judgment;
  const locked = current?.status === "submitted";
  const complete =
    judgment &&
    judgment.input_sufficiency &&
    judgment.benchmark_identity &&
    judgment.family_relation &&
    judgment.variant_relation &&
    judgment.reason.trim().length >= 10;

  return (
    <div className="annotation-app">
      <header className="ann-header">
        <Link
          href="/"
          className="ann-home"
          onClick={(event) => {
            if (
              dirty &&
              !window.confirm("Leave with unsaved changes to this review?")
            )
              event.preventDefault();
          }}
        >
          <Icon name="graph" />
          <span>
            Catalog <b>/ Evidence review</b>
          </span>
        </Link>
        {session && (
          <div className="ann-account">
            <span>
              {session.name} <small>{session.role}</small>
            </span>
            <button disabled={busy} onClick={logout}>
              Sign out
            </button>
          </div>
        )}
      </header>
      {!session ? (
        <main className="ann-login">
          <div className="ann-eyebrow">Independent review</div>
          <h1>
            Turn uncertain matches
            <br />
            into useful evidence.
          </h1>
          <p>
            Use your assigned access key to open your review cases. Your work is
            private until you submit it.
          </p>
          <form
            onSubmit={(e) => {
              e.preventDefault();
              void connect(entry.trim());
            }}
          >
            <label>
              Reviewer access key
              <input
                type="password"
                autoComplete="off"
                value={entry}
                onChange={(e) => setEntry(e.target.value)}
                required
              />
            </label>
            <button className="ann-primary" disabled={busy || !entry.trim()}>
              {busy ? "Opening workspace…" : "Open review workspace"}
            </button>
          </form>
          {error && (
            <p role="alert" className="ann-error">
              {error}
            </p>
          )}
          <p className="ann-help">
            Ask the batch coordinator for your individual key. Use a different
            person for each independent review.
          </p>
        </main>
      ) : (
        <div className="ann-layout">
          <aside className="ann-queue">
            <div className="ann-eyebrow">
              {session.role === "adjudicator"
                ? "Adjudication"
                : "Your assignment"}
            </div>
            <h2>{session.batch.name}</h2>
            <p>
              {done} of {queue.length} submitted
            </p>
            <progress
              value={done}
              max={queue.length || 1}
              aria-label="Submission progress"
            />
            <label className="ann-filter">
              Show cases
              <select
                aria-label="Show cases"
                value={filter}
                onChange={(e) => setFilter(e.target.value)}
              >
                <option value="pending">To review</option>
                <option value="submitted">Submitted</option>
                <option value="all">All cases</option>
              </select>
            </label>
            <div className="ann-case-list">
              {visible.map((item, index) => (
                <button
                  key={item.id}
                  disabled={busy}
                  aria-current={
                    current?.review_id === item.id ? "true" : undefined
                  }
                  onClick={() => void openCase(item.id)}
                >
                  <span className="ann-case-number">{index + 1}</span>
                  <span>
                    <strong>{item.left_title || "Untitled listing"}</strong>
                    <small>
                      {item.resolution === "needs_adjudication"
                        ? "Disagreement"
                        : label(item.status)}
                    </small>
                  </span>
                </button>
              ))}
            </div>
            {!visible.length && (
              <p className="ann-help">
                {session.role === "adjudicator"
                  ? "Cases appear after both reviewers submit. Refresh to check for completed reviews."
                  : "No cases in this view."}
              </p>
            )}
            <button disabled={busy || dirty} onClick={() => void connect(key)}>
              Refresh queue
            </button>
          </aside>
          <main className="ann-workspace">
            <div className="ann-intro">
              <div>
                <div className="ann-eyebrow">Evidence before assumptions</div>
                <h1>
                  {session.role === "adjudicator"
                    ? "Resolve the evidence."
                    : "What do these listings support?"}
                </h1>
              </div>
              <span className="ann-badge">
                {session.role === "adjudicator"
                  ? "Two reviews locked"
                  : "Independent review"}
              </span>
            </div>
            <details className="ann-guide">
              <summary>Review guidance</summary>
              <p>
                Benchmark identity, product family and exact variant are
                separate decisions. A shared series or compatibility code does
                not establish identity. Record what a code refers to, and choose
                uncertain when the available evidence cannot settle the
                question.
              </p>
              <p>
                Copy evidence exactly from a listing field. Record source checks
                with a URL, date and finding. Do not infer identifiers from
                benchmark labels. Submitted reviews are locked; disagreement
                goes to a separate adjudicator.
              </p>
            </details>
            {error && (
              <div role="alert" className="ann-error">
                {error}
                <button
                  disabled={busy || !current}
                  onClick={() => current && void openCase(current.review_id)}
                >
                  Reload saved review
                </button>
              </div>
            )}
            {notice && (
              <div role="status" className="ann-notice">
                {notice}
              </div>
            )}
            {!current || !judgment ? (
              <section className="ann-empty">
                <Icon name="review" size={36} />
                <h2>No review open</h2>
                <p>
                  Choose a case from the queue. Submitted work remains available
                  in the Submitted view.
                </p>
              </section>
            ) : (
              <>
                <div className="ann-case-meta">
                  <span>Case {current.review_id.slice(0, 8)}</span>
                  <span>
                    {locked
                      ? "Submitted · locked"
                      : dirty
                        ? "Unsaved changes"
                        : current.status === "draft"
                          ? "Saved draft"
                          : "Not started"}
                  </span>
                </div>
                <div className="ann-observations">
                  <Observation side="Left" listing={current.left} />
                  <Observation side="Right" listing={current.right} />
                </div>
                {current.independent_reviews && (
                  <section className="ann-panel">
                    <h2>Independent reviews</h2>
                    <p>
                      Both reviews were locked before becoming visible here.
                    </p>
                    <div className="ann-prior-reviews">
                      {current.independent_reviews.map((review) => (
                        <article key={review.reviewer}>
                          <h3>{review.reviewer}</h3>
                          <dl>
                            {[
                              "input_sufficiency",
                              "benchmark_identity",
                              "family_relation",
                              "variant_relation",
                            ].map((field) => (
                              <div key={field}>
                                <dt>{label(field)}</dt>
                                <dd>
                                  {label(
                                    review.judgment[
                                      field as keyof Judgment
                                    ] as string,
                                  )}
                                </dd>
                              </div>
                            ))}
                          </dl>
                          <p>{review.judgment.reason}</p>
                          {review.judgment.evidence.map((ev, i) => (
                            <blockquote key={i}>
                              “{ev.quote}”{" "}
                              <small>
                                {ev.side} {ev.field} ·{" "}
                                {roles[ev.role as keyof typeof roles]}
                              </small>
                              <p>{ev.interpretation}</p>
                            </blockquote>
                          ))}
                          {review.judgment.sources.map((source, i) => (
                            <p key={i}>
                              {source.url} · {source.checked_on}
                              <br />
                              {source.note}
                            </p>
                          ))}
                        </article>
                      ))}
                    </div>
                  </section>
                )}
                <section className="ann-panel">
                  <h2>
                    {session.role === "adjudicator"
                      ? "Your resolution"
                      : "Your review"}
                  </h2>
                  <fieldset disabled={busy || locked} className="ann-form">
                    <div className="ann-decisions">
                      <label>
                        Input sufficiency
                        <select
                          aria-label="Input sufficiency"
                          value={judgment.input_sufficiency || ""}
                          onChange={(e) =>
                            change({
                              input_sufficiency: e.target.value || null,
                            })
                          }
                        >
                          <option value="">Choose…</option>
                          <option value="sufficient">Sufficient</option>
                          <option value="insufficient">Insufficient</option>
                          <option value="conflicting">Conflicting</option>
                          <option value="needs_source_check">
                            Needs source check
                          </option>
                        </select>
                      </label>
                      {(
                        [
                          ["benchmark_identity", "Benchmark identity"],
                          ["family_relation", "Product family"],
                          ["variant_relation", "Exact variant"],
                        ] as const
                      ).map(([field, title]) => (
                        <label key={field}>
                          {title}
                          <select
                            aria-label={title}
                            value={judgment[field] || ""}
                            onChange={(e) =>
                              change({ [field]: e.target.value || null })
                            }
                          >
                            <option value="">Choose…</option>
                            <option value="same">Same</option>
                            <option value="different">Different</option>
                            <option value="uncertain">Uncertain</option>
                          </select>
                        </label>
                      ))}
                    </div>
                    <div className="ann-section-heading">
                      <div>
                        <h3>Observed evidence</h3>
                        <p>
                          Quote the listing and explain what the observation
                          identifies.
                        </p>
                      </div>
                      <button
                        onClick={() =>
                          change({
                            evidence: [
                              ...judgment.evidence,
                              {
                                side: "left",
                                field: "title",
                                quote: "",
                                role: "own_model",
                                interpretation: "",
                              },
                            ],
                          })
                        }
                      >
                        + Add evidence
                      </button>
                    </div>
                    {judgment.evidence.map((ev, index) => {
                      const edit = (patch: Partial<Evidence>) =>
                        change({
                          evidence: judgment.evidence.map((v, i) =>
                            i === index ? { ...v, ...patch } : v,
                          ),
                        });
                      return (
                        <div className="ann-evidence" key={index}>
                          <div className="ann-evidence-options">
                            <label>
                              Listing
                              <select
                                aria-label="Listing"
                                value={ev.side}
                                onChange={(e) =>
                                  edit({
                                    side: e.target.value as "left" | "right",
                                  })
                                }
                              >
                                <option value="left">Left</option>
                                <option value="right">Right</option>
                              </select>
                            </label>
                            <label>
                              Field
                              <select
                                aria-label="Field"
                                value={ev.field}
                                onChange={(e) =>
                                  edit({ field: e.target.value })
                                }
                              >
                                {fields.map((field) => (
                                  <option key={field} value={field}>
                                    {label(field)}
                                  </option>
                                ))}
                              </select>
                            </label>
                            <label>
                              Evidence role
                              <select
                                aria-label="Evidence role"
                                value={ev.role}
                                onChange={(e) => edit({ role: e.target.value })}
                              >
                                {Object.entries(roles).map(([value, title]) => (
                                  <option value={value} key={value}>
                                    {title}
                                  </option>
                                ))}
                              </select>
                            </label>
                          </div>
                          <label>
                            Exact quote
                            <textarea
                              aria-label="Exact quote"
                              rows={2}
                              value={ev.quote}
                              onChange={(e) => edit({ quote: e.target.value })}
                              placeholder="Copy from the selected listing field"
                            />
                          </label>
                          <label>
                            What this supports
                            <input
                              aria-label="What this supports"
                              value={ev.interpretation}
                              onChange={(e) =>
                                edit({ interpretation: e.target.value })
                              }
                            />
                          </label>
                          <button
                            className="ann-remove"
                            onClick={() =>
                              change({
                                evidence: judgment.evidence.filter(
                                  (_, i) => i !== index,
                                ),
                              })
                            }
                          >
                            Remove evidence
                          </button>
                        </div>
                      );
                    })}
                    <div className="ann-section-heading">
                      <div>
                        <h3>Source checks</h3>
                        <p>
                          Only add sources you actually checked. Pages and
                          images are not fetched automatically.
                        </p>
                      </div>
                      <button
                        onClick={() =>
                          change({
                            sources: [
                              ...judgment.sources,
                              { url: "", checked_on: "", note: "" },
                            ],
                          })
                        }
                      >
                        + Add source
                      </button>
                    </div>
                    {judgment.sources.map((source, index) => {
                      const edit = (patch: Partial<Source>) =>
                        change({
                          sources: judgment.sources.map((v, i) =>
                            i === index ? { ...v, ...patch } : v,
                          ),
                        });
                      return (
                        <div key={index} className="ann-evidence">
                          <div className="ann-source-fields">
                            <label>
                              Source URL
                              <input
                                type="url"
                                value={source.url}
                                onChange={(e) => edit({ url: e.target.value })}
                              />
                            </label>
                            <label>
                              Date checked
                              <input
                                type="date"
                                value={source.checked_on}
                                onChange={(e) =>
                                  edit({ checked_on: e.target.value })
                                }
                              />
                            </label>
                          </div>
                          <label>
                            Finding and provenance
                            <textarea
                              aria-label="Finding and provenance"
                              value={source.note}
                              onChange={(e) => edit({ note: e.target.value })}
                              placeholder="What did the source establish, and how was it verified?"
                            />
                          </label>
                          <button
                            className="ann-remove"
                            onClick={() =>
                              change({
                                sources: judgment.sources.filter(
                                  (_, i) => i !== index,
                                ),
                              })
                            }
                          >
                            Remove source
                          </button>
                        </div>
                      );
                    })}
                    <label>
                      Reasoning and remaining uncertainty
                      <textarea
                        aria-label="Reasoning and remaining uncertainty"
                        rows={4}
                        value={judgment.reason}
                        onChange={(e) => change({ reason: e.target.value })}
                        placeholder="Explain your decision and identify any evidence that is still missing."
                      />
                    </label>
                    <p className="ann-help">
                      Complete evidence lines before saving. Definite identity
                      decisions need a quote or source check; uncertain
                      decisions are valid outcomes.
                    </p>
                  </fieldset>
                  {!locked && (
                    <div className="ann-actions">
                      <button
                        disabled={busy || !dirty}
                        onClick={() => void save(false)}
                      >
                        Save draft
                      </button>
                      {!confirming ? (
                        <button
                          className="ann-primary"
                          disabled={busy || !complete}
                          onClick={() => setConfirming(true)}
                        >
                          Submit review…
                        </button>
                      ) : (
                        <div
                          className="ann-confirm"
                          role="group"
                          aria-label="Confirm submission"
                        >
                          <p>
                            Lock this review? Submitted decisions cannot be
                            edited.
                          </p>
                          <button
                            onClick={() => setConfirming(false)}
                            disabled={busy}
                          >
                            Keep editing
                          </button>
                          <button
                            className="ann-primary"
                            onClick={() => void save(true)}
                            disabled={busy}
                          >
                            Submit and lock
                          </button>
                        </div>
                      )}
                    </div>
                  )}
                  {locked && (
                    <div className="ann-actions">
                      <span>
                        Submitted reviews are preserved in the audit history.
                      </span>
                      <button
                        className="ann-primary"
                        disabled={
                          busy ||
                          !queue.some((item) => item.status !== "submitted")
                        }
                        onClick={() => {
                          const next = queue.find(
                            (item) => item.status !== "submitted",
                          );
                          if (next) void openCase(next.id);
                        }}
                      >
                        Next unreviewed case
                      </button>
                    </div>
                  )}
                </section>
              </>
            )}
          </main>
        </div>
      )}
    </div>
  );
}
