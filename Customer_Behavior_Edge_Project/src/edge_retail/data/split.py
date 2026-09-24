"""Chia train/val/test theo đơn vị sequence / subject / ngày (không bao giờ theo frame)."""

from __future__ import annotations

import random
from collections.abc import Iterable, Mapping

SPLITS = ("train", "val", "test")


def assign_by_list(items: Iterable[str], lists: Mapping[str, Iterable[str]]) -> dict[str, str]:
    """Gán split theo danh sách tường minh trong config.

    ``lists`` dạng ``{"train": [...], "val": [...], "test": [...]}``. Item không có trong
    danh sách nào sẽ bị bỏ qua (không đưa vào kết quả).
    """
    lookup: dict[str, str] = {}
    for split, names in lists.items():
        if split not in SPLITS:
            raise ValueError(f"Split không hợp lệ: {split}")
        for name in names or []:
            if name in lookup and lookup[name] != split:
                raise ValueError(f"'{name}' xuất hiện ở cả {lookup[name]} và {split}")
            lookup[name] = split
    return {item: lookup[item] for item in items if item in lookup}


def assign_by_ratio(
    items: Iterable[str], ratio: Mapping[str, float], seed: int = 42
) -> dict[str, str]:
    """Chia ngẫu nhiên (tái tạo được) theo tỉ lệ, ví dụ ``{"train": .7, "val": .15, "test": .15}``."""
    names = sorted(set(items))
    total = sum(ratio.get(s, 0.0) for s in SPLITS)
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Tổng tỉ lệ phải bằng 1, hiện là {total}")
    rng = random.Random(seed)
    rng.shuffle(names)
    n = len(names)
    n_train = round(n * ratio.get("train", 0.0))
    n_val = round(n * ratio.get("val", 0.0))
    # đảm bảo mỗi split có ít nhất 1 phần tử khi tỉ lệ > 0 và đủ item
    if n >= 3:
        n_train = max(n_train, 1) if ratio.get("train", 0) > 0 else 0
        n_val = max(n_val, 1) if ratio.get("val", 0) > 0 else 0
    n_train = min(n_train, n)
    n_val = min(n_val, n - n_train)
    out: dict[str, str] = {}
    for i, name in enumerate(names):
        if i < n_train:
            out[name] = "train"
        elif i < n_train + n_val:
            out[name] = "val"
        else:
            out[name] = "test"
    return out


def check_no_overlap(assignment: Mapping[str, str]) -> None:
    """Xác nhận mỗi đơn vị chỉ thuộc đúng một split (bảo vệ khỏi data leakage)."""
    seen: dict[str, str] = {}
    for item, split in assignment.items():
        if item in seen and seen[item] != split:
            raise ValueError(f"Rò rỉ dữ liệu: '{item}' thuộc cả {seen[item]} và {split}")
        seen[item] = split
