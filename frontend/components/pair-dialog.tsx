"use client";
import { useEffect, useRef, useState } from "react";
import { Icon } from "@/components/icons";
import { ProductIcon } from "@/components/product";
import { api, Listing, Pair, relationLabel } from "@/lib/api";

export function PairDialog({
  pair,
  listings,
  onClose,
  onSaved,
}: {
  pair: Pair;
  listings: Listing[];
  onClose: () => void;
  onSaved: () => void;
}) {
  const ref = useRef<HTMLDialogElement>(null);
  const [level, setLevel] = useState<"family" | "variant">("variant");
  const [note, setNote] = useState("");
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState("");
  const [reviewError, setReviewError] = useState("");
  useEffect(() => {
    ref.current?.showModal();
  }, []);
  const selected = [pair.left, pair.right].map((id) =>
    listings.find((x) => x.id === id),
  );
  const canMatch =
    level === "family"
      ? pair.hierarchy.family !== "different"
      : pair.hierarchy.variant === "same";
  const save = async (decision: string) => {
    setSaving(true);
    setReviewError("");
    try {
      await api("/reviews", {
        left: pair.left,
        right: pair.right,
        left_revision: pair.left_revision,
        right_revision: pair.right_revision,
        level,
        decision,
        note,
      });
      setSaved("Review saved. Identity updates are queued for the worker.");
      onSaved();
    } catch (error) {
      setReviewError((error as Error).message);
    } finally {
      setSaving(false);
    }
  };
  return (
    <dialog
      ref={ref}
      className="pair-dialog"
      onCancel={onClose}
      aria-labelledby="pair-title"
    >
      <header className="dialog-header">
        <div>
          <div className="eyebrow">PAIR INSPECTION</div>
          <h2 id="pair-title">Look at the evidence.</h2>
        </div>
        <button
          className="icon-button"
          onClick={onClose}
          aria-label="Close comparison"
        >
          <Icon name="close" />
        </button>
      </header>
      <div
        className={`relation ${pair.hierarchy.relation === "uncertain" ? "warning" : ""}`}
      >
        <Icon name="link" />
        <strong>{relationLabel[pair.hierarchy.relation]}</strong>
        <span>Structured identity rules</span>
      </div>
      <div className="comparison">
        {selected.map((listing, i) => (
          <article key={i}>
            {listing ? (
              <>
                <ProductIcon listing={listing} large />
                <div className="source-name">
                  {listing.data.source}{" "}
                  <span>· revision {listing.revision}</span>
                </div>
                <h3>{listing.data.title}</h3>
                <dl>
                  {[
                    ["Brand", listing.data.brand],
                    ["Model", listing.data.model],
                    ["Category", listing.data.category],
                    ...Object.entries(listing.data.attributes),
                  ].map(([key, value]) => (
                    <div key={key}>
                      <dt>{key}</dt>
                      <dd>{value || "Unknown"}</dd>
                    </div>
                  ))}
                </dl>
              </>
            ) : (
              <p>Listing is outside the current loaded page.</p>
            )}
          </article>
        ))}
      </div>
      <div className="evidence-box">
        <h3>Why this decision</h3>
        <p>
          Family: <b>{pair.hierarchy.family}</b> · Exact variant:{" "}
          <b>{pair.hierarchy.variant}</b>
        </p>
        {pair.hierarchy.conflicts.length > 0 && (
          <p className="conflict-text">
            Conflicting: {pair.hierarchy.conflicts.join(", ")}
          </p>
        )}
        {pair.hierarchy.missing.length > 0 && (
          <p>
            Unknown: {pair.hierarchy.missing.join(", ")}. Missing values do not
            establish a match.
          </p>
        )}
        {pair.probability !== null && (
          <p>
            Benchmark entity score:{" "}
            <b>{(pair.probability * 100).toFixed(1)}%</b> ·{" "}
            {pair.model_decision}. This score does not establish family or
            variant identity.
          </p>
        )}
        <details>
          <summary>Inspect numerical evidence</summary>
          <div className="feature-grid">
            {Object.entries(pair.features).map(([key, value]) => (
              <div key={key}>
                <span>{key.replaceAll("_", " ")}</span>
                <code>{value.toFixed(3)}</code>
              </div>
            ))}
          </div>
        </details>
      </div>
      <section className="review-form">
        <div className="section-heading">
          <h3>Record your review</h3>
          <label>
            Identity level{" "}
            <select
              value={level}
              onChange={(event) =>
                setLevel(event.target.value as "family" | "variant")
              }
            >
              <option value="variant">Exact variant</option>
              <option value="family">Product family</option>
            </select>
          </label>
        </div>
        <label className="sr-only" htmlFor="review-note">
          Review note
        </label>
        <textarea
          id="review-note"
          placeholder="What did you verify? Add a reason for your decision."
          value={note}
          onChange={(event) => setNote(event.target.value)}
        />
        {!canMatch && (
          <p className="helper">
            Resolve the conflicting or missing defining attributes before
            confirming a match.
          </p>
        )}
        <div className="review-actions">
          <button
            className="secondary"
            disabled={saving || note.trim().length < 3}
            onClick={() => save("reject")}
          >
            Keep separate
          </button>
          <button
            className="secondary"
            disabled={saving || note.trim().length < 3}
            onClick={() => save("uncertain")}
          >
            Keep in review
          </button>
          <button
            className="primary"
            disabled={saving || note.trim().length < 3 || !canMatch}
            onClick={() => save("match")}
          >
            Confirm {level} match
          </button>
        </div>
        {reviewError && (
          <p role="alert" className="conflict-text">
            {reviewError}
          </p>
        )}
        {saved && (
          <p role="status" className="success-text">
            {saved}
          </p>
        )}
      </section>
    </dialog>
  );
}
