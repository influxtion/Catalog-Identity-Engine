from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, model_validator

GUIDELINE_VERSION = "identity-evidence-v1"
Relation = Literal["same", "different", "uncertain"]
Sufficiency = Literal["sufficient", "insufficient", "conflicting", "needs_source_check"]
EvidenceField = Literal["title", "brand", "description", "category", "model", "gtin"]
Role = Literal[
    "own_model",
    "series",
    "compatibility",
    "measurement",
    "variant_attribute",
    "product_type",
    "manufacturer",
    "other",
]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class VisibleListing(BaseModel):
    # Preserve raw whitespace: evidence quotes must refer to the actual observation.
    model_config = ConfigDict(extra="forbid", strict=True)
    title: str
    brand: str | None = None
    description: str | None = None
    category: str | None = None
    model: str | None = None
    gtin: str | None = None
    attributes: dict[str, str | None] = Field(default_factory=dict)
    price: str | None = None
    currency: str | None = None
    product_url: str | None = None


class Evidence(StrictModel):
    side: Literal["left", "right"]
    field: EvidenceField
    quote: str = Field(min_length=1, max_length=2000)
    start: int | None = Field(default=None, ge=0)
    role: Role
    interpretation: str = Field(min_length=3, max_length=2000)


class Source(StrictModel):
    url: HttpUrl
    checked_on: date
    note: str = Field(min_length=10, max_length=3000)

    @model_validator(mode="after")
    def no_future_date(self):
        if self.checked_on > date.today():
            raise ValueError("source observation date cannot be in the future")
        return self


class Draft(StrictModel):
    input_sufficiency: Sufficiency | None = None
    benchmark_identity: Relation | None = None
    family_relation: Relation | None = None
    variant_relation: Relation | None = None
    evidence: list[Evidence] = Field(default_factory=list, max_length=30)
    sources: list[Source] = Field(default_factory=list, max_length=10)
    reason: str = Field(default="", max_length=5000)


class Judgment(Draft):
    input_sufficiency: Sufficiency
    benchmark_identity: Relation
    family_relation: Relation
    variant_relation: Relation
    reason: str = Field(min_length=10, max_length=5000)

    @model_validator(mode="after")
    def coherent_evidence(self):
        if self.variant_relation == "same" and self.family_relation != "same":
            raise ValueError("same variant requires same family")
        if self.family_relation == "different" and self.variant_relation != "different":
            raise ValueError("different families require different variants")
        if any(
            v != "uncertain"
            for v in (self.benchmark_identity, self.family_relation, self.variant_relation)
        ) and not (self.evidence or self.sources):
            raise ValueError(
                "a definite identity decision needs quoted or verified source evidence"
            )
        return self


class SaveRequest(StrictModel):
    revision: int = Field(ge=0)
    case_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    guideline_version: Literal["identity-evidence-v1"]
    submit: bool = False
    judgment: Draft


class PacketCase(BaseModel):
    model_config = ConfigDict(extra="forbid")
    review_id: str = Field(pattern=r"^[a-f0-9]{64}$")
    left: VisibleListing
    right: VisibleListing
    judgment: dict

    @model_validator(mode="after")
    def unannotated(self):
        expected = {
            "input_sufficiency",
            "benchmark_identity",
            "family_relation",
            "variant_relation",
            "evidence_fields",
            "reason",
        }
        if set(self.judgment) != expected or any(
            v != ([] if k == "evidence_fields" else None) for k, v in self.judgment.items()
        ):
            raise ValueError("import only unannotated blind packets, never labels or review keys")
        return self


DECISION_FIELDS = ("input_sufficiency", "benchmark_identity", "family_relation", "variant_relation")


def decision(judgment):
    return {key: judgment[key] for key in DECISION_FIELDS}


def disagreement(a, b):
    if decision(a) != decision(b):
        return True
    roles = {}
    for annotation in (a, b):
        for evidence in annotation["evidence"]:
            key = (evidence["side"], evidence["field"], evidence["quote"], evidence.get("start"))
            roles.setdefault(key, set()).add(evidence["role"])
    return any(len(values) > 1 for values in roles.values())
