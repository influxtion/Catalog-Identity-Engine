"""Input records contain observations, never benchmark labels or entity IDs."""

from enum import StrEnum
from typing import Any
from uuid import UUID, uuid5

from pydantic import BaseModel, ConfigDict, Field, field_validator

LISTING_NAMESPACE = UUID("ac05c061-a175-4f7c-babf-4f6cb46def19")


class IdentityLevel(StrEnum):
    BENCHMARK_ENTITY = "benchmark_entity"
    FAMILY = "family"
    VARIANT = "variant"


class Relation(StrEnum):
    SAME_VARIANT = "same_exact_variant"
    SAME_FAMILY = "same_family_different_variant"
    DIFFERENT = "different_product"
    UNCERTAIN = "uncertain"


class Listing(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    source: str
    source_id: str
    title: str = ""
    description: str | None = None
    brand: str | None = None
    category: str | None = None
    model: str | None = None
    gtin: str | None = None
    attributes: dict[str, str | None] = Field(default_factory=dict)
    price: str | None = None
    currency: str | None = None
    product_url: str | None = None
    image_url: str | None = None
    observed_at: str | None = None
    raw: dict[str, Any] = Field(default_factory=dict)

    @field_validator("source", "source_id")
    @classmethod
    def require_identity(cls, value: str) -> str:
        if not value.strip() or value != value.strip():
            raise ValueError("source and source_id must be nonempty without surrounding whitespace")
        return value

    @property
    def key(self) -> str:
        # Length prefix avoids ambiguity when source IDs contain delimiters.
        return str(uuid5(LISTING_NAMESPACE, f"{len(self.source)}:{self.source}{self.source_id}"))
