from __future__ import annotations

import hashlib


def stable_group_bucket(group_id: str, n_buckets: int = 10) -> int:
    digest = hashlib.md5(group_id.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % n_buckets


def assign_group_split(group_id: str) -> str:
    bucket = stable_group_bucket(group_id, n_buckets=10)
    if bucket <= 6:
        return "train"
    if bucket == 7:
        return "val"
    return "test"
