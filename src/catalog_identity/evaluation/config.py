from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class WdcProtocol(BaseModel):
    model_config = ConfigDict(extra="forbid")
    corner_cases: Literal[20, 50, 80]
    development_size: Literal["small", "medium", "large"]
    test_unseen: Literal[0, 50, 100]


class RunConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    dataset_name: str = Field(min_length=1)
    dataset_version: str = Field(min_length=1)
    synthetic: bool = False
    train: Path
    validation: Path
    test: Path
    protocol: WdcProtocol | None = None
    test_entity_disjoint: bool = False
    seed: int = Field(default=42, ge=0, le=2**32 - 1)
    max_features: int = Field(default=50_000, ge=1)
    regularization: float = Field(default=1.0, gt=0, allow_inf_nan=False)
    target_precision: float = Field(default=0.99, gt=0, le=1)
    minimum_validation_matches: int = Field(default=5, ge=1)
    reject_threshold: float = Field(default=0.1, ge=0, lt=1)
    retrieval_k: list[int] = Field(default_factory=lambda: [10, 20, 50], min_length=1)
    pipeline_k: int = Field(default=20, ge=1)

    @model_validator(mode="after")
    def validate_protocol(self):
        if not self.synthetic and self.protocol is None:
            raise ValueError("real WDC evaluation requires the benchmark protocol")
        if self.protocol and self.protocol.test_unseen == 100 and not self.test_entity_disjoint:
            raise ValueError("100% unseen test requires the entity-disjoint audit")
        if self.protocol and self.protocol.test_unseen != 100 and self.test_entity_disjoint:
            raise ValueError("entity-disjoint test requires the 100% unseen protocol")
        if any(type(k) is not int or k < 1 for k in self.retrieval_k):
            raise ValueError("retrieval_k must contain positive integers")
        self.retrieval_k = sorted(set(self.retrieval_k))
        return self


def load_config(path: Path) -> RunConfig:
    config = RunConfig.model_validate_json(path.read_text(encoding="utf-8"))
    return config.model_copy(
        update={
            name: (path.parent / getattr(config, name)).resolve()
            for name in ("train", "validation", "test")
        }
    )
