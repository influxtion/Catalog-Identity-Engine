"""Optional supervised joint-text matcher. Heavy runtimes are loaded only on use."""

import hashlib
import json
import math
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler

from catalog_identity.matching.distinctions import DISTINCTION_NAMES, DistinctionFeatures
from catalog_identity.matching.execution import configure_execution, validate_execution
from catalog_identity.matching.semantic_features import MODEL, REVISION, OfflineEncoder, digest
from catalog_identity.normalization import normalize

ARCHITECTURE = {
    "version": "joint-minilm-v1",
    "pair_tokens": 256,
    "title_tokens": 96,
    "brand_tokens": 8,
    "frozen_layers": 4,
    "backbone_layers": 6,
    "head_hidden": 128,
    "dropout": 0.1,
    "numeric_feature_version": "3",
    "inference": "mean_orientation_logits_then_sigmoid",
}
TRAINING = {
    "seed": 42,
    "epochs": 3,
    "backbone_lr": 2e-5,
    "head_lr": 1e-3,
    "weight_decay": 0.01,
    "warmup_fraction": 0.1,
    "gradient_clip": 1.0,
    "microbatch": 8,
    "accumulation": 4,
    "torch_threads": 14,
    "dtype": "float32",
}


class PairTokenizer:
    """Balanced listing budgets with field labels included in the token limit."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        if tokenizer.num_special_tokens_to_add(pair=True) != 3 or any(
            getattr(tokenizer, key, None) is None
            for key in ("cls_token_id", "sep_token_id", "pad_token_id")
        ):
            raise ValueError("registered BERT pair special-token format is required")
        self.labels = {
            field: tokenizer.encode(f"{field}: ", add_special_tokens=False)
            for field in ("title", "brand", "description")
        }

    def parts(self, listing):
        record = normalize(listing)
        return {
            field: self.tokenizer.encode(getattr(record, field), add_special_tokens=False)
            for field in self.labels
        }

    def _needed(self, parts):
        return sum(
            len(self.labels[k]) + min(len(v), {"title": 96, "brand": 8}.get(k, len(v)))
            for k, v in parts.items()
            if v or k == "title"
        )

    def _side(self, parts, budget):
        fields = [k for k, value in parts.items() if value or k == "title"]
        remaining = budget - sum(len(self.labels[k]) for k in fields)
        if remaining < 0:
            raise ValueError("listing budget cannot preserve field labels")
        result, truncated = [], {}
        for field in fields:
            raw = parts[field]
            take = min(len(raw), {"title": 96, "brand": 8}.get(field, remaining), remaining)
            result.extend(self.labels[field] + raw[:take])
            remaining -= take
            truncated[field] = take < len(raw)
        return result, truncated

    def encode_parts(self, left, right):
        available = 256 - self.tokenizer.num_special_tokens_to_add(pair=True)
        needs = [self._needed(left), self._needed(right)]
        budgets = [min(available // 2, needs[0]), min(available - available // 2, needs[1])]
        spare = available - sum(budgets)
        for side in (0, 1):
            addition = min(spare, needs[side] - budgets[side])
            budgets[side] += addition
            spare -= addition
        a, at = self._side(left, budgets[0])
        b, bt = self._side(right, budgets[1])
        # Transformers 5 removed prepare_for_model. Assemble the pinned BERT pair
        # format directly, preserving already-budgeted WordPiece IDs exactly.
        while len(a) + len(b) > available:  # Final longest-first guard.
            (a if len(a) > len(b) else b).pop()
        ids = [
            self.tokenizer.cls_token_id,
            *a,
            self.tokenizer.sep_token_id,
            *b,
            self.tokenizer.sep_token_id,
        ]
        length = len(ids)
        encoded = {
            "input_ids": ids + [self.tokenizer.pad_token_id] * (256 - length),
            "attention_mask": [1] * length + [0] * (256 - length),
            "token_type_ids": [0] * (len(a) + 2) + [1] * (len(b) + 1) + [0] * (256 - length),
        }
        return {
            k: np.asarray(encoded[k], dtype=np.int64)
            for k in ("input_ids", "attention_mask", "token_type_ids")
        }, {
            "tokens": int(sum(encoded["attention_mask"])),
            "left_truncated": at,
            "right_truncated": bt,
        }

    def pair(self, left, right):
        return self.encode_parts(self.parts(left), self.parts(right))


def build_network(backbone, numeric_dimension):
    import torch
    from torch import nn

    if len(backbone.encoder.layer) != 6 or backbone.config.type_vocab_size != 2:
        raise ValueError("joint matcher requires six backbone layers and two token types")
    for parameter in backbone.parameters():
        parameter.requires_grad = False
    for layer in backbone.encoder.layer[4:]:
        for parameter in layer.parameters():
            parameter.requires_grad = True

    class JointNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = backbone
            self.head = nn.Sequential(
                nn.Linear(backbone.config.hidden_size + numeric_dimension, 128),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(128, 1),
            )

        def train(self, mode=True):
            super().train(mode)
            self.backbone.embeddings.eval()
            for layer in self.backbone.encoder.layer[:4]:
                layer.eval()
            return self

        def forward(self, tokens, numeric):
            from torch.nn.attention import SDPBackend, sdpa_kernel

            with (
                torch.autocast(device_type=tokens["input_ids"].device.type, enabled=False),
                sdpa_kernel([SDPBackend.MATH]),
            ):
                cls = self.backbone(**tokens).last_hidden_state[:, 0]
                return self.head(torch.cat((cls, numeric), dim=1)).squeeze(1)

    return JointNetwork().float().cpu()


class JointMatcher:
    def __init__(self, backbone=None, tokenizer=None, encoder_metadata=None, device="cpu"):
        self.execution = configure_execution(device)
        self.device = self.execution["device"]
        if backbone is None:
            source = OfflineEncoder()
            backbone = source.encoder.model[0].auto_model
            tokenizer = source.encoder.model.tokenizer
            encoder_metadata = source.metadata
        self.encoder_metadata = encoder_metadata
        self.tokenizer = PairTokenizer(tokenizer)
        self.network = build_network(backbone, len(DISTINCTION_NAMES)).to(self.device)
        self.features = DistinctionFeatures()
        self.scaler = StandardScaler()
        self.calibration = None
        self.training_record = None
        self.provenance = {}
        self.training_execution = None

    def fit_features(self, train):
        if train.name != "train":
            raise ValueError("joint feature fitting requires training split")
        values = self.features.fit(train).split(train)
        return self.scaler.fit_transform(values).astype(np.float32)

    def prepare(self, split, numeric=None):
        if numeric is None:
            numeric = self.scaler.transform(self.features.split(split)).astype(np.float32)
        parts = {key: self.tokenizer.parts(listing) for key, listing in split.listings.items()}
        orientations, statistics = [], []
        for reverse in (False, True):
            rows, stats = [], []
            for pair in split.pairs:
                a, b = (pair.right, pair.left) if reverse else (pair.left, pair.right)
                row, stat = self.tokenizer.encode_parts(parts[a], parts[b])
                rows.append(row)
                stats.append(stat)
            orientations.append(
                {
                    key: np.stack([row[key] for row in rows])
                    for key in ("input_ids", "attention_mask", "token_type_ids")
                }
            )
            statistics.append(
                {
                    "pairs": len(rows),
                    "mean_tokens": float(np.mean([s["tokens"] for s in stats])),
                    "maximum_tokens": max(s["tokens"] for s in stats),
                    "truncated_sides": sum(
                        any(s[side].values())
                        for s in stats
                        for side in ("left_truncated", "right_truncated")
                    ),
                    "truncated_fields": {
                        field: sum(
                            s[side].get(field, False)
                            for s in stats
                            for side in ("left_truncated", "right_truncated")
                        )
                        for field in ("title", "brand", "description")
                    },
                }
            )
        return {
            "orientations": orientations,
            "numeric": numeric,
            "statistics": statistics,
            "labels": np.asarray([p.label for p in split.pairs], dtype=np.float32),
            "initial_orientation": np.asarray(
                [
                    int(hashlib.sha256(f"42:{p.pair_id}".encode()).hexdigest(), 16) % 2
                    for p in split.pairs
                ],
                dtype=np.int64,
            ),
        }

    def batch(self, prepared, indices, orientation):
        import torch

        if np.isscalar(orientation):
            data = {
                key: value[indices] for key, value in prepared["orientations"][orientation].items()
            }
        else:
            data = {
                key: np.where(
                    orientation[:, None] == 0,
                    prepared["orientations"][0][key][indices],
                    prepared["orientations"][1][key][indices],
                )
                for key in prepared["orientations"][0]
            }
        return (
            {key: torch.from_numpy(value).to(self.device) for key, value in data.items()},
            torch.from_numpy(prepared["numeric"][indices]).to(self.device),
        )

    def optimizer(self):
        import torch

        return torch.optim.AdamW(
            [
                {
                    "params": [p for p in self.network.backbone.parameters() if p.requires_grad],
                    "lr": 2e-5,
                },
                {"params": self.network.head.parameters(), "lr": 1e-3},
            ],
            weight_decay=0.01,
            foreach=False,
            fused=False,
        )

    def train_prepared(self, prepared, microbatch=8, progress=None):
        import torch

        if microbatch not in (4, 8):
            raise ValueError("registered microbatch must be 8, or memory fallback 4")
        accumulation = 32 // microbatch
        batches = math.ceil(len(prepared["labels"]) / microbatch)
        steps = math.ceil(batches / accumulation) * 3
        warmup = max(1, math.ceil(steps * 0.1))
        optimizer = self.optimizer()

        def multiplier(step):
            return (
                step / warmup
                if step < warmup
                else max(0.0, (steps - step) / max(1, steps - warmup))
            )

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)
        self.network.train()
        optimizer.zero_grad(set_to_none=True)
        for epoch in range(3):
            order = np.random.default_rng(42 + epoch).permutation(len(prepared["labels"]))
            for batch_index, start in enumerate(range(0, len(order), microbatch)):
                indices = order[start : start + microbatch]
                orientation = (prepared["initial_orientation"][indices] + epoch) % 2
                tokens, numeric = self.batch(prepared, indices, orientation)
                logits = self.network(tokens, numeric)
                labels = torch.from_numpy(prepared["labels"][indices]).to(self.device)
                # Normalize each final partial accumulation group by its actual example count.
                group_start = (start // 32) * 32
                group_size = min(32, len(order) - group_start)
                loss = (
                    torch.nn.functional.binary_cross_entropy_with_logits(
                        logits, labels, reduction="sum"
                    )
                    / group_size
                )
                loss.backward()
                if not torch.isfinite(loss):
                    raise ValueError("nonfinite joint training loss; execution stopped")
                if (batch_index + 1) % accumulation == 0 or batch_index + 1 == batches:
                    torch.nn.utils.clip_grad_norm_(
                        self.network.parameters(), 1.0, error_if_nonfinite=True
                    )
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                if progress is not None:
                    progress(epoch, batch_index + 1, batches)
        self.training_record = {
            **TRAINING,
            "microbatch": microbatch,
            "accumulation": accumulation,
            "optimizer_steps": steps,
            "warmup_steps": warmup,
            "checkpoint": "final_epoch_only",
            "training_pairs": len(prepared["labels"]),
        }
        self.training_execution = self.execution
        return self

    def fit(self, train, microbatch=8, progress=None):
        numeric = self.fit_features(train)
        prepared = self.prepare(train, numeric)
        self.train_prepared(prepared, microbatch, progress)
        return prepared["statistics"]

    def logits_prepared(self, prepared, microbatch=8):
        import torch

        self.network.eval()
        values = []
        with torch.no_grad():
            for start in range(0, len(prepared["labels"]), microbatch):
                indices = np.arange(start, min(start + microbatch, len(prepared["labels"])))
                logits = [
                    self.network(*self.batch(prepared, indices, side)).detach().cpu().numpy()
                    for side in (0, 1)
                ]
                values.extend(((logits[0] + logits[1]) / 2).tolist())
        values = np.asarray(values)
        if not np.isfinite(values).all():
            raise ValueError("nonfinite joint inference logits; execution stopped")
        return values

    def predict(self, split, calibrated=True):
        logits = self.logits_prepared(self.prepare(split))
        if calibrated and self.calibration is not None:
            logits = logits * self.calibration["slope"] + self.calibration["intercept"]
        return 1 / (1 + np.exp(-np.clip(logits, -700, 700)))

    def save(self, directory: Path):
        from safetensors.torch import save_file

        directory.mkdir(parents=True, exist_ok=False)
        self.features.save(directory / "features.json")
        self.tokenizer.tokenizer.save_pretrained(directory / "tokenizer")
        self.network.backbone.config.save_pretrained(directory / "backbone")
        metadata = {
            "architecture": ARCHITECTURE,
            "training": self.training_record,
            "encoder": self.encoder_metadata,
            "feature_names": list(DISTINCTION_NAMES),
            "scaler": {"mean": self.scaler.mean_.tolist(), "scale": self.scaler.scale_.tolist()},
            "calibration": self.calibration,
            "provenance": self.provenance,
            "backbone_config_sha256": digest(directory / "backbone/config.json"),
            "tokenizer_files": {
                p.relative_to(directory / "tokenizer").as_posix(): digest(p)
                for p in sorted((directory / "tokenizer").rglob("*"))
                if p.is_file()
            },
        }
        if self.training_execution is not None:
            metadata["execution"] = validate_execution(self.training_execution)
        (directory / "matcher.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        save_file(
            {
                name: value.detach().cpu().contiguous()
                for name, value in self.network.state_dict().items()
            },
            str(directory / "weights.safetensors"),
        )

    @classmethod
    def load(
        cls, directory: Path, backbone=None, tokenizer=None, encoder_metadata=None, device="cpu"
    ):
        from safetensors.torch import load_file

        metadata = json.loads((directory / "matcher.json").read_text(encoding="utf-8"))
        base_keys = {
            "architecture",
            "training",
            "encoder",
            "feature_names",
            "scaler",
            "calibration",
            "tokenizer_files",
            "backbone_config_sha256",
            "provenance",
        }
        if set(metadata) not in (base_keys, base_keys | {"execution"}):
            raise ValueError("unexpected joint matcher metadata keys")
        if "execution" in metadata:
            validate_execution(metadata["execution"])
        if metadata.get("architecture") != ARCHITECTURE or metadata.get("feature_names") != list(
            DISTINCTION_NAMES
        ):
            raise ValueError("joint matcher architecture/feature metadata differs")
        training = metadata["training"]
        expected_keys = set(TRAINING) | {
            "optimizer_steps",
            "warmup_steps",
            "checkpoint",
            "training_pairs",
        }
        if (
            not isinstance(training, dict)
            or set(training) != expected_keys
            or any(
                training[k] != v
                for k, v in TRAINING.items()
                if k not in ("microbatch", "accumulation")
            )
            or training["microbatch"] not in (4, 8)
            or training["accumulation"] != 32 // training["microbatch"]
            or training["checkpoint"] != "final_epoch_only"
        ):
            raise ValueError("joint matcher training metadata differs")
        actual_tokenizer_files = {
            p.relative_to(directory / "tokenizer").as_posix(): digest(p)
            for p in sorted((directory / "tokenizer").rglob("*"))
            if p.is_file()
        }
        if not actual_tokenizer_files or actual_tokenizer_files != metadata["tokenizer_files"]:
            raise ValueError("joint matcher tokenizer files differ")
        if digest(directory / "backbone/config.json") != metadata["backbone_config_sha256"]:
            raise ValueError("joint matcher backbone configuration differs")
        calibration = metadata["calibration"]
        if calibration is not None and (
            set(calibration) != {"slope", "intercept"}
            or not all(isinstance(v, (int, float)) and np.isfinite(v) for v in calibration.values())
            or calibration["slope"] <= 0
        ):
            raise ValueError("invalid joint matcher calibration")
        mean, scale = (
            np.asarray(metadata["scaler"][key], dtype=float) for key in ("mean", "scale")
        )
        if (
            mean.shape != (len(DISTINCTION_NAMES),)
            or scale.shape != mean.shape
            or not np.isfinite(mean).all()
            or not np.isfinite(scale).all()
            or (scale <= 0).any()
        ):
            raise ValueError("invalid joint matcher scaler")
        if backbone is None:
            from transformers import AutoTokenizer, BertConfig, BertModel

            configuration = BertConfig.from_pretrained(
                str(directory / "backbone"), local_files_only=True
            )
            required = {
                "model_type": "bert",
                "hidden_size": 384,
                "num_hidden_layers": 6,
                "num_attention_heads": 12,
                "intermediate_size": 1536,
                "type_vocab_size": 2,
                "vocab_size": 30522,
                "max_position_embeddings": 512,
                "hidden_act": "gelu",
                "hidden_dropout_prob": 0.1,
                "attention_probs_dropout_prob": 0.1,
            }
            if any(getattr(configuration, k) != v for k, v in required.items()):
                raise ValueError("saved backbone differs from registered MiniLM architecture")
            if (
                metadata["encoder"].get("model") != MODEL
                or metadata["encoder"].get("revision") != REVISION
            ):
                raise ValueError("saved backbone provenance differs from registered source")
            backbone = BertModel(configuration)
            tokenizer = AutoTokenizer.from_pretrained(
                str(directory / "tokenizer"),
                local_files_only=True,
                token=False,
                trust_remote_code=False,
            )
            encoder_metadata = metadata["encoder"]
        result = cls(backbone, tokenizer, encoder_metadata, device=device)
        if metadata["encoder"] != result.encoder_metadata:
            raise ValueError("joint matcher encoder requirements differ from local assets")
        state = load_file(str(directory / "weights.safetensors"), device="cpu")
        expected = result.network.state_dict()
        if set(state) != set(expected) or any(
            state[k].shape != expected[k].shape or state[k].dtype != expected[k].dtype
            for k in expected
        ):
            raise ValueError("joint matcher tensor keys, shapes or dtypes differ")
        import torch

        if any(not torch.isfinite(value).all() for value in state.values()):
            raise ValueError("joint matcher tensors must be finite")
        result.network.load_state_dict(state, strict=True)
        result.features = DistinctionFeatures.load(directory / "features.json")
        result.scaler.mean_, result.scaler.scale_ = mean, scale
        result.scaler.var_ = scale**2
        result.scaler.n_features_in_ = len(mean)
        result.calibration, result.training_record = metadata["calibration"], metadata["training"]
        result.provenance = metadata["provenance"]
        result.training_execution = metadata.get("execution")
        result.network.eval()
        return result
