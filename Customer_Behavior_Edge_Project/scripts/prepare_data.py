#!/usr/bin/env python3
"""Chuẩn hoá và chia dữ liệu theo configs/split.yaml (Bước 2).

Hỗ trợ hiện tại (G0):
  --task tracking    : copy/symlink sequence MOT vào processed/tracking/{train,val,test}
  --task detection   : xuất ảnh + nhãn YOLO từ sequence tracking (và CrowdHuman nếu có)
  --task trajectory  : sinh CSV quỹ đạo từ ground truth MOT (và ATC nếu có)
  --task stats       : in thống kê từng split

Ví dụ:
    # dữ liệu thật
    python scripts/prepare_data.py --task tracking detection trajectory stats
    # dữ liệu mẫu (sau khi chạy make_sample_data.py)
    python scripts/prepare_data.py --source data/sample/MOT17-like --dataset SAMPLE \
        --ratio 0.5 0.25 0.25 --task tracking detection trajectory stats --out data/sample/processed
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

import pandas as pd
import yaml

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from edge_retail.data.mot import list_sequences, read_gt, read_seqinfo, sequence_stats, validate_gt  # noqa: E402
from edge_retail.data.split import SPLITS, assign_by_list, assign_by_ratio, check_no_overlap  # noqa: E402
from edge_retail.data.yolo import (  # noqa: E402
    crowdhuman_odgt_to_yolo,
    gt_to_trajectory_csv,
    mot_to_yolo_labels,
    summarize_yolo_split,
    write_yolo_dataset_yaml,
)


def load_yaml(p: Path) -> dict:
    with open(p, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


# ----------------------------------------------------------------------------- helpers


def link_or_copy_dir(src: Path, dst: Path, copy: bool) -> None:
    if dst.exists() or dst.is_symlink():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if copy:
        shutil.copytree(src, dst)
    else:
        os.symlink(src.resolve(), dst, target_is_directory=True)


def collect_mot_sequences(source: Path, variant: str | None) -> dict[str, Path]:
    """Tìm mọi sequence trong ``source/train`` (và ``source/test`` nếu có nhãn)."""
    found: dict[str, Path] = {}
    for split_dir in (source / "train", source / "test", source):
        for seq in list_sequences(split_dir):
            if not (seq / "gt" / "gt.txt").exists():
                continue
            name = seq.name
            if variant and "-" in name and name.count("-") >= 2:
                # MOT17-02-FRCNN -> chỉ giữ biến thể chọn
                if not name.endswith(variant):
                    continue
                name = name.rsplit("-", 1)[0]
            found[name] = seq
    return found


def decide_assignment(names: list[str], dataset: str, split_cfg: dict, ratio: list[float] | None, seed: int) -> dict[str, str]:
    if ratio:
        return assign_by_ratio(names, dict(zip(SPLITS, ratio)), seed=seed)
    ds_cfg = split_cfg.get("tracking", {}).get(dataset)
    if ds_cfg is None:
        raise SystemExit(f"Không có cấu hình split cho '{dataset}' trong split.yaml; dùng --ratio.")
    if "ratio" in ds_cfg:
        return assign_by_ratio(names, ds_cfg["ratio"], seed=seed)
    return assign_by_list(names, {s: ds_cfg.get(s, []) for s in SPLITS})


# ----------------------------------------------------------------------------- tasks


def task_tracking(seqs: dict[str, Path], assignment: dict[str, str], out: Path, copy: bool) -> None:
    print("\n[tracking] chia sequence:")
    for name, split in sorted(assignment.items(), key=lambda kv: (kv[1], kv[0])):
        src = seqs[name]
        info = read_seqinfo(src)
        errors = validate_gt(read_gt(src / "gt" / "gt.txt"), info.im_width, info.im_height)
        if errors:
            print(f"  ! {name}: {'; '.join(errors)}")
        link_or_copy_dir(src, out / "tracking" / split / name, copy)
        print(f"  {split:<5} {name}")


def task_detection(seqs: dict[str, Path], assignment: dict[str, str], out: Path, det_cfg: dict, copy: bool, crowdhuman: Path | None) -> None:
    print("\n[detection] xuất YOLO:")
    root = out / "detection"
    for name, split in assignment.items():
        n = mot_to_yolo_labels(
            seqs[name],
            root / "images" / split,
            root / "labels" / split,
            frame_stride=int(det_cfg.get("frame_stride", 5)),
            min_visibility=float(det_cfg.get("min_visibility", 0.25)),
            min_box_size=int(det_cfg.get("min_box_size", 8)),
            copy_images=copy,
        )
        print(f"  {split:<5} {name}: {n} ảnh")
    if crowdhuman and crowdhuman.exists():
        for split, odgt in (("train", "annotation_train.odgt"), ("val", "annotation_val.odgt")):
            p = crowdhuman / odgt
            if p.exists():
                n = crowdhuman_odgt_to_yolo(p, crowdhuman / "Images", root / "labels" / split)
                print(f"  {split:<5} CrowdHuman: {n} nhãn (ảnh giữ tại {crowdhuman / 'Images'}, cần symlink vào images/{split})")
    write_yolo_dataset_yaml(root, root / "detection.yaml", det_cfg.get("class_names", ["person"]))
    print(f"  ghi {root / 'detection.yaml'}")


def task_trajectory(seqs: dict[str, Path], assignment: dict[str, str], out: Path, atc_dir: Path | None) -> None:
    print("\n[trajectory] sinh CSV quỹ đạo từ GT:")
    root = out / "trajectory"
    for name, split in assignment.items():
        df = gt_to_trajectory_csv(seqs[name], root / split / f"{name}.csv")
        print(f"  {split:<5} {name}: {df['track_id'].nunique()} quỹ đạo, {len(df)} điểm")
    if atc_dir and atc_dir.exists():
        csvs = sorted(atc_dir.glob("*.csv"))
        if csvs:
            assign = assign_by_ratio([c.stem for c in csvs], {"train": 0.7, "val": 0.15, "test": 0.15})
            for c in csvs:
                dst = root / assign[c.stem] / f"ATC_{c.name}"
                dst.parent.mkdir(parents=True, exist_ok=True)
                if not dst.exists():
                    shutil.copy2(c, dst)
            print(f"  ATC: {len(csvs)} ngày → {dict(pd.Series(assign).value_counts())}")


def task_stats(out: Path) -> None:
    print("\n[stats]")
    rows = []
    for split in SPLITS:
        for seq in list_sequences(out / "tracking" / split):
            s = sequence_stats(seq)
            s["split"] = split
            rows.append(s)
    if rows:
        df = pd.DataFrame(rows)[["split", "sequence", "frames", "tracks", "boxes", "density", "occluded_ratio", "fps", "size"]]
        print(df.to_string(index=False))
        agg = df.groupby("split")[["frames", "tracks", "boxes"]].sum()
        agg["density"] = (agg["boxes"] / agg["frames"]).round(2)
        print("\nTổng theo split:\n" + agg.to_string())
        d = agg["density"]
        if {"train", "val"} <= set(d.index) and d["train"] > 0:
            skew = abs(d["val"] - d["train"]) / d["train"]
            flag = "OK" if skew <= 0.3 else "CẢNH BÁO > 30 %"
            print(f"\nLệch mật độ val/train: {skew:.1%} ({flag})")
        out.joinpath("stats_tracking.csv").write_text(df.to_csv(index=False))
    for split in SPLITS:
        lbl = out / "detection" / "labels" / split
        if lbl.exists():
            print(f"detection/{split}: {summarize_yolo_split(lbl)}")


# ----------------------------------------------------------------------------- main


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", nargs="+", default=["tracking", "detection", "trajectory", "stats"],
                    choices=["tracking", "detection", "trajectory", "stats"])
    ap.add_argument("--source", default=str(PROJECT / "data" / "raw" / "MOT17"), help="thư mục dataset MOT gốc")
    ap.add_argument("--dataset", default="MOT17", help="tên trong split.yaml (MOT17, MOT20, CAVIAR, ...)")
    ap.add_argument("--out", default=str(PROJECT / "data" / "processed"))
    ap.add_argument("--split-config", default=str(PROJECT / "configs" / "split.yaml"))
    ap.add_argument("--ratio", nargs=3, type=float, metavar=("TRAIN", "VAL", "TEST"), help="ghi đè: chia ngẫu nhiên theo tỉ lệ")
    ap.add_argument("--copy", action="store_true", help="copy thay vì symlink (dùng trên Windows/ổ khác)")
    ap.add_argument("--crowdhuman", default=str(PROJECT / "data" / "raw" / "CrowdHuman"))
    ap.add_argument("--atc", default=str(PROJECT / "data" / "raw" / "ATC"))
    args = ap.parse_args(argv)

    split_cfg = load_yaml(Path(args.split_config))
    seed = int(split_cfg.get("seed", 42))
    source, out = Path(args.source), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    needs_seqs = {"tracking", "detection", "trajectory"} & set(args.task)
    assignment: dict[str, str] = {}
    seqs: dict[str, Path] = {}
    if needs_seqs:
        variant = split_cfg.get("tracking", {}).get(args.dataset, {}).get("detector_variant")
        seqs = collect_mot_sequences(source, variant)
        if not seqs:
            raise SystemExit(f"Không tìm thấy sequence MOT nào trong {source}")
        assignment = decide_assignment(sorted(seqs), args.dataset, split_cfg, args.ratio, seed)
        check_no_overlap(assignment)
        skipped = sorted(set(seqs) - set(assignment))
        if skipped:
            print(f"Bỏ qua (không có trong split.yaml): {skipped}")

    if "tracking" in args.task:
        task_tracking(seqs, assignment, out, args.copy)
    if "detection" in args.task:
        task_detection(seqs, assignment, out, split_cfg.get("detection", {}), args.copy, Path(args.crowdhuman))
    if "trajectory" in args.task:
        task_trajectory(seqs, assignment, out, Path(args.atc))
    if "stats" in args.task:
        task_stats(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
