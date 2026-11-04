"""Registered conventional all-layer CLS-only matcher; optional runtimes stay lazy."""

import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

from catalog_identity.matching.cross_encoder import JointMatcher, PairTokenizer
from catalog_identity.matching.execution import configure_execution, validate_execution
from catalog_identity.matching.semantic_features import MODEL, REVISION, OfflineEncoder, digest

ARCHITECTURE = {
    "version": "full-minilm-cls-v1",
    "pair_tokens": 256,
    "title_tokens": 96,
    "brand_tokens": 8,
    "backbone_layers": 6,
    "trainable_layers": 6,
    "trainable_embeddings": True,
    "pooler": None,
    "head": "CLS_dropout_linear",
    "dropout": 0.1,
    "classifier_init_std": 0.02,
    "classifier_init_bias": 0.0,
    "numeric_features": False,
    "inference": "mean_orientation_logits_then_sigmoid",
}
TRAINING = {
    "seed": 42,
    "epochs": 3,
    "learning_rate": 2e-5,
    "weight_decay": 0.01,
    "bias_layernorm_weight_decay": 0.0,
    "warmup_fraction": 0.1,
    "gradient_clip": 1.0,
    "microbatch": 8,
    "accumulation": 4,
    "torch_threads": 14,
    "dtype": "float32",
}


def build_full_network(backbone):
    import torch
    from torch import nn

    if len(backbone.encoder.layer) != 6 or backbone.config.type_vocab_size != 2:
        raise ValueError("full matcher requires six BERT layers and two token types")
    backbone.pooler = None  # CLS-only loss has no path through the unused BERT pooler.
    for parameter in backbone.parameters():
        parameter.requires_grad = True

    class FullNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = backbone
            self.dropout = nn.Dropout(0.1)
            self.classifier = nn.Linear(backbone.config.hidden_size, 1)
            nn.init.normal_(self.classifier.weight, mean=0.0, std=0.02)
            nn.init.zeros_(self.classifier.bias)

        def forward(self, tokens, numeric=None):
            from torch.nn.attention import SDPBackend, sdpa_kernel

            if numeric is not None and numeric.shape[1] != 0:
                raise ValueError("CLS-only matcher cannot accept numeric features")
            with (
                torch.autocast(device_type=tokens["input_ids"].device.type, enabled=False),
                sdpa_kernel([SDPBackend.MATH]),
            ):
                cls = self.backbone(**tokens).last_hidden_state[:, 0]
                return self.classifier(self.dropout(cls)).squeeze(1)

    return FullNetwork().float().cpu()


class FullMatcher(JointMatcher):
    """Shares token preparation and symmetric inference, never fits numeric features."""

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
        self.network = build_full_network(backbone).to(self.device)
        self.calibration = None
        self.training_record = None
        self.training_execution = None
        self.provenance = {}

    def fit_features(self, train):
        if train.name != "train":
            raise ValueError("full matcher fitting requires training split")
        # Zero-width transport preserves the shared runner interface without any features.
        return np.empty((len(train.pairs), 0), dtype=np.float32)

    def prepare(self, split, numeric=None):
        if numeric is None:
            numeric = np.empty((len(split.pairs), 0), dtype=np.float32)
        if numeric.shape != (len(split.pairs), 0):
            raise ValueError("CLS-only matcher cannot prepare numeric features")
        return super().prepare(split, numeric)

    def optimizer(self):
        import torch

        groups = {True: [], False: []}
        for name, parameter in self.network.named_parameters():
            if not parameter.requires_grad:
                raise ValueError("all full-matcher parameters must be trainable")
            groups[name.endswith(".bias") or "LayerNorm." in name].append(parameter)
        return torch.optim.AdamW(
            [
                {"params": groups[False], "weight_decay": 0.01},
                {"params": groups[True], "weight_decay": 0.0},
            ],
            lr=2e-5,
            foreach=False,
            fused=False,
        )

    def audit_parameters(self):
        return {
            "embeddings": self.network.backbone.embeddings.word_embeddings.weight,
            "early_layer": self.network.backbone.encoder.layer[0].attention.self.query.weight,
            "late_layer": self.network.backbone.encoder.layer[5].attention.self.query.weight,
            "classifier": self.network.classifier.weight,
        }

    def train_prepared(self, prepared, microbatch=8, progress=None):
        import torch

        if microbatch != 8:
            raise ValueError("full matcher requires registered microbatch 8; no fallback")
        count = len(prepared["labels"])
        if count < 1 or not np.isin(prepared["labels"], [0, 1]).all():
            raise ValueError("nonempty binary training labels required")
        batches = math.ceil(count / 8)
        steps = math.ceil(count / 32) * 3
        warmup = max(1, math.ceil(steps * 0.1))
        optimizer = self.optimizer()
        before = {k: p.detach().cpu().clone() for k, p in self.audit_parameters().items()}

        def multiplier(step):
            return (
                step / warmup
                if step < warmup
                else max(0.0, (steps - step) / max(1, steps - warmup))
            )

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)
        self.network.train()
        optimizer.zero_grad(set_to_none=True)
        audits, actual_steps = [], 0
        for epoch in range(3):
            order = np.random.default_rng(42 + epoch).permutation(count)
            loss_sum, norms, rates, group_sizes = 0.0, [], [], []
            orientations = [0, 0]
            for batch_index, start in enumerate(range(0, count, 8)):
                indices = order[start : start + 8]
                orientation = (prepared["initial_orientation"][indices] + epoch) % 2
                for side in (0, 1):
                    orientations[side] += int((orientation == side).sum())
                logits = self.network(*self.batch(prepared, indices, orientation))
                labels = torch.from_numpy(prepared["labels"][indices]).to(self.device)
                total_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    logits, labels, reduction="sum"
                )
                if not torch.isfinite(total_loss):
                    raise ValueError("nonfinite full-matcher training loss")
                loss_sum += float(total_loss.detach())
                group_size = min(32, count - (start // 32) * 32)
                (total_loss / group_size).backward()
                if (batch_index + 1) % 4 == 0 or batch_index + 1 == batches:
                    if any(p.grad is None for p in self.network.parameters()):
                        raise ValueError("intended trainable parameter has no gradient")
                    norm = torch.nn.utils.clip_grad_norm_(
                        self.network.parameters(), 1.0, error_if_nonfinite=True
                    )
                    norms.append(float(norm))
                    rates.append(optimizer.param_groups[0]["lr"])
                    group_sizes.append(group_size)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    actual_steps += 1
                if progress is not None:
                    progress(epoch, batch_index + 1, batches)
            audits.append(
                {
                    "epoch": epoch + 1,
                    "example_weighted_mean_loss": loss_sum / count,
                    "optimizer_steps": len(norms),
                    "learning_rate_min": min(rates),
                    "learning_rate_max": max(rates),
                    "preclip_gradient_norm_min": min(norms),
                    "preclip_gradient_norm_mean": float(np.mean(norms)),
                    "preclip_gradient_norm_max": max(norms),
                    "all_gradients_present_and_finite": True,
                    "orientation_counts": orientations,
                    "effective_batch_size_counts": dict(Counter(group_sizes)),
                    "final_accumulation_examples": group_sizes[-1],
                    "final_accumulation_microbatches": math.ceil(group_sizes[-1] / 8),
                }
            )
        changes = {}
        for name, parameter in self.audit_parameters().items():
            delta = (parameter.detach().cpu() - before[name]).abs()
            changes[name] = {
                "changed_elements": int((delta > 0).sum()),
                "maximum_absolute_change": float(delta.max()),
            }
            if not torch.isfinite(delta).all() or changes[name]["changed_elements"] == 0:
                raise ValueError(f"optimization audit failed for {name}")
        if actual_steps != steps:
            raise ValueError("optimizer step count differs from fixed schedule")
        self.training_record = {
            **TRAINING,
            "optimizer_steps": actual_steps,
            "warmup_steps": warmup,
            "checkpoint": "final_epoch_only",
            "training_pairs": count,
            "class_counts": {
                "positive": int(prepared["labels"].sum()),
                "negative": int(count - prepared["labels"].sum()),
            },
            "epoch_audits": audits,
            "parameter_changes": changes,
            "all_backbone_parameters_trainable": all(
                p.requires_grad for p in self.network.backbone.parameters()
            ),
            "trainable_parameter_count": sum(p.numel() for p in self.network.parameters()),
            "pooler_removed": self.network.backbone.pooler is None,
        }
        self.training_execution = self.execution
        return self

    def save(self, directory: Path):
        from safetensors.torch import save_file

        directory.mkdir(parents=True, exist_ok=False)
        self.tokenizer.tokenizer.save_pretrained(directory / "tokenizer")
        self.network.backbone.config.save_pretrained(directory / "backbone")
        metadata = {
            "architecture": ARCHITECTURE,
            "training": self.training_record,
            "encoder": self.encoder_metadata,
            "execution": validate_execution(self.training_execution),
            "calibration": self.calibration,
            "provenance": self.provenance,
            "backbone_config_sha256": digest(directory / "backbone/config.json"),
            "tokenizer_files": {
                p.relative_to(directory / "tokenizer").as_posix(): digest(p)
                for p in sorted((directory / "tokenizer").rglob("*"))
                if p.is_file()
            },
        }
        (directory / "matcher.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        save_file(
            {k: v.detach().cpu().contiguous() for k, v in self.network.state_dict().items()},
            str(directory / "weights.safetensors"),
        )

    @classmethod
    def load(
        cls, directory: Path, backbone=None, tokenizer=None, encoder_metadata=None, device="cpu"
    ):
        from safetensors.torch import load_file

        metadata = json.loads((directory / "matcher.json").read_text(encoding="utf-8"))
        if (
            set(metadata)
            != {
                "architecture",
                "training",
                "encoder",
                "execution",
                "calibration",
                "provenance",
                "backbone_config_sha256",
                "tokenizer_files",
            }
            or metadata["architecture"] != ARCHITECTURE
        ):
            raise ValueError("full matcher architecture/metadata differs")
        validate_execution(metadata["execution"])
        validate_training_record(metadata["training"])
        actual = {
            p.relative_to(directory / "tokenizer").as_posix(): digest(p)
            for p in sorted((directory / "tokenizer").rglob("*"))
            if p.is_file()
        }
        if (
            not actual
            or actual != metadata["tokenizer_files"]
            or digest(directory / "backbone/config.json") != metadata["backbone_config_sha256"]
        ):
            raise ValueError("full matcher tokenizer/config files differ")
        calibration = metadata["calibration"]
        if calibration is not None and (
            set(calibration) != {"slope", "intercept"}
            or not all(isinstance(v, (int, float)) and np.isfinite(v) for v in calibration.values())
            or calibration["slope"] <= 0
        ):
            raise ValueError("invalid full matcher calibration")
        if backbone is None:
            from transformers import AutoTokenizer, BertConfig, BertModel

            config = BertConfig.from_pretrained(str(directory / "backbone"), local_files_only=True)
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
            if (
                any(getattr(config, k) != v for k, v in required.items())
                or metadata["encoder"].get("model") != MODEL
                or metadata["encoder"].get("revision") != REVISION
            ):
                raise ValueError("saved full backbone differs from registered MiniLM")
            backbone = BertModel(config, add_pooling_layer=False)
            tokenizer = AutoTokenizer.from_pretrained(
                str(directory / "tokenizer"),
                local_files_only=True,
                token=False,
                trust_remote_code=False,
            )
            encoder_metadata = metadata["encoder"]
        result = cls(backbone, tokenizer, encoder_metadata, device=device)
        if metadata["encoder"] != result.encoder_metadata:
            raise ValueError("full matcher encoder provenance differs")
        state = load_file(str(directory / "weights.safetensors"), device="cpu")
        expected = result.network.state_dict()
        if set(state) != set(expected) or any(
            state[k].shape != expected[k].shape or state[k].dtype != expected[k].dtype
            for k in expected
        ):
            raise ValueError("full matcher tensor keys, shapes or dtypes differ")
        import torch

        if any(not torch.isfinite(v).all() for v in state.values()):
            raise ValueError("full matcher tensors must be finite")
        if metadata["training"]["trainable_parameter_count"] != sum(
            p.numel() for p in result.network.parameters()
        ) or any(
            metadata["training"]["parameter_changes"][k]["changed_elements"] > p.numel()
            for k, p in result.audit_parameters().items()
        ):
            raise ValueError("full matcher recorded parameter counts differ")
        result.network.load_state_dict(state, strict=True)
        result.calibration, result.training_record = calibration, metadata["training"]
        result.training_execution, result.provenance = metadata["execution"], metadata["provenance"]
        result.network.eval()
        return result


def validate_training_record(record):
    expected = set(TRAINING) | {
        "optimizer_steps",
        "warmup_steps",
        "checkpoint",
        "training_pairs",
        "class_counts",
        "epoch_audits",
        "parameter_changes",
        "all_backbone_parameters_trainable",
        "trainable_parameter_count",
        "pooler_removed",
    }
    if (
        not isinstance(record, dict)
        or set(record) != expected
        or any(record[k] != v for k, v in TRAINING.items())
        or record["checkpoint"] != "final_epoch_only"
    ):
        raise ValueError("full matcher training metadata differs")
    count = record["training_pairs"]
    steps = math.ceil(count / 32) * 3 if type(count) is int and count > 0 else 0
    classes = record["class_counts"]
    if (
        not steps
        or type(record["optimizer_steps"]) is not int
        or record["optimizer_steps"] != steps
        or record["warmup_steps"] != max(1, math.ceil(steps * 0.1))
        or not isinstance(record["epoch_audits"], list)
        or len(record["epoch_audits"]) != 3
        or record["all_backbone_parameters_trainable"] is not True
        or record["pooler_removed"] is not True
        or type(record["trainable_parameter_count"]) is not int
        or record["trainable_parameter_count"] <= 0
        or not isinstance(classes, dict)
        or set(classes) != {"positive", "negative"}
        or any(type(v) is not int or v < 0 for v in classes.values())
        or sum(classes.values()) != count
    ):
        raise ValueError("invalid full matcher optimization audit")
    expected_groups = {str(k): v for k, v in ((32, count // 32), (count % 32, 1)) if k and v}
    final_examples = count % 32 or 32
    previous_orientations = None
    epoch_keys = {
        "epoch",
        "example_weighted_mean_loss",
        "optimizer_steps",
        "learning_rate_min",
        "learning_rate_max",
        "preclip_gradient_norm_min",
        "preclip_gradient_norm_mean",
        "preclip_gradient_norm_max",
        "all_gradients_present_and_finite",
        "orientation_counts",
        "effective_batch_size_counts",
        "final_accumulation_examples",
        "final_accumulation_microbatches",
    }

    def finite_nonnegative(value):
        return type(value) in (int, float) and np.isfinite(value) and value >= 0

    for epoch, audit in enumerate(record["epoch_audits"], 1):
        if not isinstance(audit, dict) or set(audit) != epoch_keys:
            raise ValueError("invalid full matcher epoch audit keys")
        orientation = audit["orientation_counts"]
        groups = audit["effective_batch_size_counts"]
        scalars = [
            audit[k]
            for k in (
                "example_weighted_mean_loss",
                "learning_rate_min",
                "learning_rate_max",
                "preclip_gradient_norm_min",
                "preclip_gradient_norm_mean",
                "preclip_gradient_norm_max",
            )
        ]
        if (
            type(audit["epoch"]) is not int
            or audit["epoch"] != epoch
            or type(audit["optimizer_steps"]) is not int
            or audit["optimizer_steps"] != steps // 3
            or not isinstance(orientation, list)
            or len(orientation) != 2
            or any(type(v) is not int or v < 0 for v in orientation)
            or sum(orientation) != count
            or (previous_orientations is not None and orientation != previous_orientations[::-1])
            or not isinstance(groups, dict)
            or any(
                type(k) not in (str, int) or type(v) is not int or v <= 0 for k, v in groups.items()
            )
            or {str(k): v for k, v in groups.items()} != expected_groups
            or type(audit["final_accumulation_examples"]) is not int
            or audit["final_accumulation_examples"] != final_examples
            or type(audit["final_accumulation_microbatches"]) is not int
            or audit["final_accumulation_microbatches"] != math.ceil(final_examples / 8)
            or audit["all_gradients_present_and_finite"] is not True
            or not all(finite_nonnegative(v) for v in scalars)
        ):
            raise ValueError("invalid full matcher epoch audit")
        if not (
            audit["learning_rate_min"] <= audit["learning_rate_max"] <= 2e-5
            and audit["preclip_gradient_norm_min"]
            <= audit["preclip_gradient_norm_mean"]
            <= audit["preclip_gradient_norm_max"]
        ):
            raise ValueError("invalid full matcher gradient/learning-rate range")
        previous_orientations = orientation
    changes = record["parameter_changes"]
    if not isinstance(changes, dict) or set(changes) != {
        "embeddings",
        "early_layer",
        "late_layer",
        "classifier",
    }:
        raise ValueError("invalid full matcher parameter-change audit")
    for value in changes.values():
        if (
            not isinstance(value, dict)
            or set(value) != {"changed_elements", "maximum_absolute_change"}
            or type(value["changed_elements"]) is not int
            or value["changed_elements"] <= 0
            or not finite_nonnegative(value["maximum_absolute_change"])
            or value["maximum_absolute_change"] <= 0
        ):
            raise ValueError("invalid full matcher parameter-change audit")
