#!/usr/bin/env python3
"""Sinh dữ liệu mẫu giả lập đúng định dạng MOT để smoke-test pipeline khi chưa tải dataset thật.

Tạo N sequence, mỗi sequence có vài "khách hàng" đi theo quỹ đạo ngẫu nhiên trong một cửa hàng
giả lập (ảnh nền xám + hình chữ nhật màu). Đầu ra: data/sample/MOT17-like/train/<seq>/{img1,gt,seqinfo.ini}

Ví dụ:
    python scripts/make_sample_data.py --n-seq 4 --frames 60
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from edge_retail.data.mot import MOT_COLUMNS, SeqInfo, write_gt, write_seqinfo  # noqa: E402


def make_sequence(seq_dir: Path, name: str, frames: int, n_people: int, W: int, H: int, rng: random.Random) -> None:
    img_dir = seq_dir / "img1"
    img_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    people = []
    for pid in range(1, n_people + 1):
        w, h = rng.randint(30, 50), rng.randint(80, 120)
        x, y = rng.uniform(0, W - w), rng.uniform(0, H - h)
        vx, vy = rng.uniform(-4, 4), rng.uniform(-2, 2)
        start = rng.randint(1, max(1, frames // 3))
        color = (rng.randint(50, 255), rng.randint(50, 255), rng.randint(50, 255))
        people.append([pid, x, y, w, h, vx, vy, start, color])

    for f in range(1, frames + 1):
        img = np.full((H, W, 3), 110, dtype=np.uint8)
        # "kệ hàng" tĩnh
        cv2.rectangle(img, (0, 0), (W, 40), (70, 70, 70), -1)
        cv2.rectangle(img, (W - 60, 40), (W, H), (70, 70, 70), -1)
        for p in people:
            pid, x, y, w, h, vx, vy, start, color = p
            if f < start:
                continue
            x += vx
            y += vy
            if x < 0 or x + w > W:
                vx = -vx
                x = min(max(x, 0), W - w)
            if y < 0 or y + h > H:
                vy = -vy
                y = min(max(y, 0), H - h)
            p[1], p[2], p[5], p[6] = x, y, vx, vy
            cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), color, -1)
            vis = rng.choice([1.0, 1.0, 1.0, 0.8, 0.6, 0.4])
            rows.append([f, pid, round(x, 2), round(y, 2), w, h, 1, 1, vis])
        cv2.imwrite(str(img_dir / f"{f:06d}.jpg"), img)

    write_gt(pd.DataFrame(rows, columns=MOT_COLUMNS), seq_dir / "gt" / "gt.txt")
    write_seqinfo(seq_dir, SeqInfo(name=name, im_dir="img1", frame_rate=10, seq_length=frames, im_width=W, im_height=H))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(PROJECT / "data" / "sample" / "MOT17-like" / "train"))
    ap.add_argument("--n-seq", type=int, default=4)
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--people", type=int, default=5)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=384)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    rng = random.Random(args.seed)
    out = Path(args.out)
    for i in range(1, args.n_seq + 1):
        name = f"SAMPLE-{i:02d}"
        make_sequence(out / name, name, args.frames, args.people, args.width, args.height, rng)
        print(f"tạo {out / name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
