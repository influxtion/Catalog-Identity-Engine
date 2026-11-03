"use client";
import { useEffect, useRef, useState } from "react";
import { Icon } from "@/components/icons";
import { api } from "@/lib/api";

export function ImportDialog({
  onClose,
  onImported,
}: {
  onClose: () => void;
  onImported: () => void;
}) {
  const ref = useRef<HTMLDialogElement>(null);
  const [text, setText] = useState(
    '[\n  {\n    "source": "my-store",\n    "source_id": "example-1",\n    "title": "Aster H1 Headphones Black",\n    "brand": "Aster",\n    "model": "H1",\n    "category": "headphones",\n    "attributes": { "color": "black" }\n  }\n]',
  );
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  useEffect(() => {
    ref.current?.showModal();
  }, []);
  const submit = async () => {
    setBusy(true);
    setError("");
    try {
      const rows = JSON.parse(text);
      if (!Array.isArray(rows))
        throw new Error("Provide an array of listings.");
      await api("/listings", { listings: rows });
      onImported();
      onClose();
    } catch (error) {
      setError((error as Error).message);
    } finally {
      setBusy(false);
    }
  };
  return (
    <dialog
      ref={ref}
      className="import-dialog"
      onCancel={onClose}
      aria-labelledby="import-title"
    >
      <header className="dialog-header">
        <div>
          <div className="eyebrow">CATALOG INPUT</div>
          <h2 id="import-title">Add listings</h2>
        </div>
        <button
          className="icon-button"
          onClick={onClose}
          aria-label="Close import"
        >
          <Icon name="close" />
        </button>
      </header>
      <p>
        Paste canonical listing records. Source and source ID identify a
        listing; sending an updated record preserves its identity and
        invalidates old evidence.
      </p>
      <label htmlFor="listing-json">Listing JSON</label>
      <textarea
        id="listing-json"
        className="json-input"
        value={text}
        onChange={(event) => setText(event.target.value)}
        spellCheck={false}
      />
      {error && (
        <p role="alert" className="conflict-text">
          {error}
        </p>
      )}
      <button className="primary" disabled={busy} onClick={submit}>
        {busy ? "Importing…" : "Import listings"}
      </button>
    </dialog>
  );
}
