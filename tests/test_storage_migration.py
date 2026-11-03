import pytest

pytest.importorskip("sqlalchemy")
from sqlalchemy import create_engine, insert, select

from catalog_identity.schema import Listing
from catalog_identity.storage import database as tables
from catalog_identity.storage.catalog import Catalog, digest
from catalog_identity.storage.database import Database


def test_v1_upgrade_preserves_source_records_and_backfills_normalization(tmp_path):
    url = "sqlite:///" + str(tmp_path / "old.db")
    engine = create_engine(url)
    tables.metadata.create_all(engine, tables=[tables.state, tables.listings])
    record = Listing(
        source="fixture",
        source_id="a",
        title="  ASTER  Lamp ",
        attributes={"color": "Grey", "weight": "1 kg"},
    )
    payload = record.model_dump(mode="json")
    with engine.begin() as connection:
        connection.execute(
            insert(tables.state).values(
                key="catalog",
                data={
                    "schema_version": 1,
                    "revision": 1,
                    "identities_stale": True,
                    "family_registry": None,
                    "variant_registry": None,
                    "graph": {},
                },
            )
        )
        connection.execute(
            insert(tables.listings).values(
                id=record.key,
                source=record.source,
                source_id=record.source_id,
                title_search="aster lamp",
                revision=1,
                digest=digest(payload),
                data=payload,
            )
        )
    engine.dispose()
    database = Database(url)
    with database.transaction() as connection:
        assert connection.execute(select(tables.state.c.data)).scalar_one()["schema_version"] == 2
        raw = connection.execute(select(tables.listings.c.data)).scalar_one()
        assert raw == payload
        normalized = connection.execute(select(tables.normalized_listings)).mappings().one()
        assert normalized["listing_revision"] == 1
        assert normalized["data"]["title"] == "aster lamp"
        assert normalized["data"]["attributes"] == {"color": "gray", "weight": "1000 g"}
        assert normalized["data"]["changes"]
    catalog = Catalog(database)
    assert catalog.ingest([record])["changed"] == 0
    updated = record.model_copy(update={"attributes": {"color": "black"}})
    catalog.ingest([updated])
    assert catalog.listing(record.key)["normalized"]["attributes"] == {"color": "black"}
    with database.transaction() as connection:
        assert (
            connection.execute(select(tables.normalized_listings.c.listing_revision)).scalar_one()
            == 2
        )
    database.close()


def test_future_schema_is_rejected_before_adding_tables(tmp_path):
    url = "sqlite:///" + str(tmp_path / "future.db")
    engine = create_engine(url)
    tables.metadata.create_all(engine, tables=[tables.state])
    with engine.begin() as connection:
        connection.execute(insert(tables.state).values(key="catalog", data={"schema_version": 99}))
    engine.dispose()
    with pytest.raises(ValueError, match="unsupported database schema"):
        Database(url)
    from sqlalchemy import inspect

    assert inspect(engine).get_table_names() == ["catalog_state"]
    engine.dispose()
