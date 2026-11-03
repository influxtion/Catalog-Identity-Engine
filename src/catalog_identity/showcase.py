"""Fictional structured examples with explicit synthetic family/variant labels."""

from catalog_identity.schema import Listing


def showcase():
    rows = [
        (
            "north",
            "h1-black",
            "Aster H1 Wireless Headphones · Black",
            "Aster",
            "H1",
            "headphones",
            {"color": "black"},
            "129.00",
        ),
        (
            "market",
            "h1-b",
            "Aster H-1 Noise Cancelling Headphones, Black",
            "Aster",
            "H-1",
            "headphones",
            {"color": "Black"},
            "124.00",
        ),
        (
            "north",
            "h1-silver",
            "Aster H1 Wireless Headphones · Silver",
            "Aster",
            "H1",
            "headphones",
            {"color": "silver"},
            "129.00",
        ),
        (
            "studio",
            "h1-s",
            "Aster H1 Silver Over-ear Headphones",
            "Aster",
            "H1",
            "headphones",
            {"color": "silver"},
            "135.00",
        ),
        (
            "market",
            "h1-unknown",
            "Aster H1 Wireless Headphones",
            "Aster",
            "H1",
            "headphones",
            {},
            "119.00",
        ),
        (
            "north",
            "h2-black",
            "Aster H2 Wireless Headphones · Black",
            "Aster",
            "H2",
            "headphones",
            {"color": "black"},
            "179.00",
        ),
        (
            "north",
            "p4-128",
            "Lumen P4 Phone · 128 GB · Graphite",
            "Lumen",
            "P4",
            "phone",
            {"capacity": "128GB", "color": "black"},
            "499.00",
        ),
        (
            "market",
            "p4-128",
            "Lumen P-4 128GB Smartphone Black",
            "Lumen",
            "P-4",
            "phone",
            {"capacity": "128 GB", "color": "black"},
            "489.00",
        ),
        (
            "studio",
            "p4-256",
            "Lumen P4 Phone · 256 GB · Graphite",
            "Lumen",
            "P4",
            "phone",
            {"capacity": "256GB", "color": "black"},
            "579.00",
        ),
        (
            "north",
            "s8-m",
            "Morrow S8 Cotton Overshirt · Medium",
            "Morrow",
            "S8",
            "apparel",
            {"color": "blue", "size": "M"},
            "79.00",
        ),
        (
            "market",
            "s8-l",
            "Morrow S8 Cotton Overshirt · Large",
            "Morrow",
            "S8",
            "apparel",
            {"color": "blue", "size": "L"},
            "79.00",
        ),
        (
            "studio",
            "mystery",
            "Wireless headphones, open box",
            None,
            None,
            "headphones",
            {},
            "65.00",
        ),
    ]
    return [
        Listing(
            source=source,
            source_id=id,
            title=title,
            brand=brand,
            model=model,
            category=category,
            attributes=attributes,
            price=price,
            currency="USD",
            raw={"synthetic": True, "fixture_version": "1"},
        )
        for source, id, title, brand, model, category, attributes, price in rows
    ]


def seed(catalog):
    records = showcase()
    result = catalog.ingest(records)
    catalog.rebuild()
    for a, b in ((0, 1), (0, 2), (0, 4), (0, 5), (6, 8), (0, 11)):
        catalog.inspect_pair(records[a].key, records[b].key)
    return result
