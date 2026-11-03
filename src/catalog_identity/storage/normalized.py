"""JSON-safe normalized evidence, with the original record stored separately."""

from dataclasses import asdict


def normalized_payload(record):
    return {
        "version": record.version,
        **{
            name: getattr(record, name)
            for name in ("title", "description", "brand", "category", "model", "gtin", "currency")
        },
        "attributes": dict(record.attributes),
        "price": str(record.price) if record.price is not None else None,
        "issues": list(record.issues),
        "changes": [asdict(change) for change in record.changes],
    }
