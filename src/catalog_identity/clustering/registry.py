from dataclasses import dataclass, field
from uuid import uuid4


@dataclass
class IdentityRegistry:
    """IDs persist across recomputation; merges/splits remain auditable.

    Each old ID nominates its largest-overlap child. Merged nominations retain the
    oldest ID. Other split children receive new opaque IDs. Retired IDs stay in
    history and are never reused for unrelated memberships.
    """

    active: dict[str, tuple[str, ...]] = field(default_factory=dict)
    birth: dict[str, int] = field(default_factory=dict)
    history: list[dict] = field(default_factory=list)
    revision: int = 0

    def reconcile(self, clusters):
        groups = sorted(tuple(sorted(group)) for group in clusters)
        flat = [key for group in groups for key in group]
        if any(not group for group in groups) or len(flat) != len(set(flat)):
            raise ValueError("clusters must form a disjoint nonempty partition")
        if groups == sorted(self.active.values()):
            return self.active
        old = self.active
        nominations = {}
        for identity, members in old.items():
            overlaps = [(len(set(members) & set(group)), group) for group in groups]
            overlaps = [item for item in overlaps if item[0]]
            if overlaps:
                winner = sorted(overlaps, key=lambda item: (-item[0], item[1]))[0][1]
                nominations.setdefault(winner, []).append(identity)
        result = {}
        for group in groups:
            nominees = nominations.get(group, [])
            identity = (
                min(nominees, key=lambda key: (self.birth[key], key)) if nominees else str(uuid4())
            )
            if identity not in self.birth:
                self.birth[identity] = len(self.birth)
            result[identity] = group
        self.revision += 1
        for identity, members in old.items():
            descendants = sorted(key for key, group in result.items() if set(members) & set(group))
            if len(descendants) != 1 or descendants[0] != identity or result[identity] != members:
                self.history.append(
                    {
                        "revision": self.revision,
                        "event": "split" if len(descendants) > 1 else "update_or_merge",
                        "previous_id": identity,
                        "next_ids": descendants,
                        "previous_members": list(members),
                    }
                )
        for identity, members in result.items():
            if identity not in old:
                self.history.append(
                    {
                        "revision": self.revision,
                        "event": "created",
                        "identity_id": identity,
                        "members": list(members),
                    }
                )
        self.active = result
        return result

    def to_dict(self):
        return {
            "active": self.active,
            "birth": self.birth,
            "history": self.history,
            "revision": self.revision,
        }

    @classmethod
    def from_dict(cls, value):
        return cls(
            {k: tuple(v) for k, v in value["active"].items()},
            value["birth"],
            value["history"],
            value["revision"],
        )
