# Catalog Identity Engine

Resolve product listings across stores, with reproducible experiments and inspectable evidence.
Independent project inspired by public catalog-matching research; not affiliated with Shopify.

The first milestone is a typed ingestion pipeline and an evaluated lexical/logistic baseline.
Product-family and exact-variant identity are distinct concepts. WDC's binary entity labels must
not be treated as hierarchical family/variant annotations.

## Development

Python 3.11 or newer. Using [uv](https://docs.astral.sh/uv/):

```sh
uv sync --extra dev
uv run pytest
uv run ruff check .
```

Alternatively, create a virtual environment and run `python -m pip install -e ".[dev]"`.
The committed lockfile is the reproducible dependency specification for uv installations.

Read [PROJECT_SPEC.md](PROJECT_SPEC.md) for the full design and
[docs/datasets.md](docs/datasets.md) for benchmark semantics and data restrictions.

## License

Project code and original synthetic fixtures are MIT licensed. Third-party datasets and
product images are not included and are not covered by this license.
