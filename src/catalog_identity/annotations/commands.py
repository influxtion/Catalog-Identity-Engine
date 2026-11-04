import json
import os

from catalog_identity.annotations.store import AnnotationStore
from catalog_identity.storage.database import Database


def execute(args):
    # All files contain private local material. Refuse overwrite and ignore the entire folder.
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / ".gitignore").write_text("*\n", encoding="utf-8")
    database = Database(os.getenv("DATABASE_URL", "sqlite:///catalog.db"))
    try:
        store = AnnotationStore(database)
        if args.command == "annotation-import":
            result = store.initialize(
                args.packet, args.name, args.reviewers, args.adjudicator, args.audit_report
            )
            filename = "access-keys.json"
        elif args.command == "annotation-rotate":
            result = store.rotate(args.batch, args.reviewer)
            filename = "access-key.json"
        else:
            result = store.export(args.batch)
            if args.review_key:
                from catalog_identity.annotations.report import analyze

                report = analyze(result, args.review_key)
                (args.output / "analysis.json").write_text(
                    json.dumps(report, indent=2), encoding="utf-8"
                )
            filename = "annotations.json"
            (args.output / "summary.json").write_text(
                json.dumps(
                    {
                        "batch_id": result["batch_id"],
                        "summary": result["summary"],
                        "provenance": result["batch"]["provenance"],
                        "warning": result["warning"],
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        path = args.output / filename
        path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        # Never echo credentials or product observations to terminal logs.
        print(json.dumps({"batch_id": result["batch_id"], "file": str(path.resolve())}))
    finally:
        database.close()
    return 0
