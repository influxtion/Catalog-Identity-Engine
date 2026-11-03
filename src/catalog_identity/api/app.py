import json
import os
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field

from catalog_identity.schema import Listing
from catalog_identity.storage.catalog import Catalog, Conflict, NotFound
from catalog_identity.storage.database import Database


class Request(BaseModel):
    model_config = ConfigDict(extra="forbid")


class IngestRequest(Request):
    listings: list[Listing] = Field(min_length=1, max_length=1000)


class PairRequest(Request):
    left: str
    right: str


class ReviewRequest(PairRequest):
    level: Literal["family", "variant"]
    decision: Literal["match", "reject", "uncertain"]
    note: str = Field(min_length=3, max_length=2000)
    left_revision: int = Field(ge=1)
    right_revision: int = Field(ge=1)


class EvaluationRequest(Request):
    config_name: str = Field(pattern=r"^[A-Za-z0-9_-]+(?:\.[A-Za-z0-9_-]+)*\.json$")
    idempotency_key: str | None = Field(default=None, max_length=200)


def create_app(database_url=None, model_directory=None):
    database = Database(database_url or os.getenv("DATABASE_URL", "sqlite:///catalog.db"))
    configured_model = model_directory or os.getenv("CATALOG_MODEL_DIR")
    catalog = Catalog(database, Path(configured_model) if configured_model else None)
    app = FastAPI(title="Catalog Identity Engine", version="0.2.0")
    app.state.catalog = catalog
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )

    def call(function, *args, **kwargs):
        try:
            return function(*args, **kwargs)
        except NotFound as exc:
            raise HTTPException(404, str(exc)) from exc
        except Conflict as exc:
            raise HTTPException(409, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.get("/health")
    def health():
        return {"status": "ok", "schema_version": 1}

    @app.get("/overview")
    def overview():
        return call(catalog.overview)

    @app.post("/listings", status_code=201)
    def ingest(request: IngestRequest):
        return call(catalog.ingest, request.listings, enqueue_rebuild=True)

    @app.get("/listings")
    def search(q: str = "", limit: int = Query(50, ge=1, le=200), offset: int = Query(0, ge=0)):
        return call(catalog.search, q, limit, offset)

    @app.get("/listings/{key}")
    def listing(key: str):
        return call(catalog.listing, key)

    @app.get("/listings/{key}/candidates")
    def candidates(key: str, k: int = Query(20, ge=1, le=100)):
        return call(catalog.retrieve, key, k)

    @app.post("/pairs/inspect")
    def inspect(request: PairRequest):
        return call(catalog.inspect_pair, request.left, request.right)

    @app.get("/identities")
    def identities(level: Literal["family", "variant"] = "family"):
        return call(catalog.identities, level)

    @app.get("/identities/{key}")
    def identity(key: str):
        return call(catalog.identity, key)

    @app.post("/identities/rebuild", status_code=202)
    def rebuild():
        return call(catalog.enqueue, "rebuild")

    @app.get("/reviews")
    def review_queue():
        return call(catalog.review_queue)

    @app.post("/reviews", status_code=201)
    def review(request: ReviewRequest):
        return call(catalog.submit_review, **request.model_dump(), enqueue_rebuild=True)

    @app.get("/jobs")
    def jobs():
        return call(catalog.jobs)

    @app.post("/evaluations", status_code=202)
    def evaluate(request: EvaluationRequest):
        return call(
            catalog.enqueue,
            "evaluate",
            {"config_name": request.config_name},
            request.idempotency_key,
        )

    @app.get("/evaluations")
    def evaluations():
        return call(catalog.evaluation_reports)

    @app.get("/benchmarks")
    def benchmarks():
        path = Path(__file__).resolve().parents[1] / "benchmarks.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else []

    @app.get("/performance")
    def performance():
        path = Path(__file__).resolve().parents[1] / "performance.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

    return app
