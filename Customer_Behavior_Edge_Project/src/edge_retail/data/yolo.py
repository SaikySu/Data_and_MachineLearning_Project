"""Chuyển nhãn MOT sang YOLO format để fine-tune detector 1 lớp ``person``."""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd
import yaml

from .mot import read_gt, read_seqinfo


def _clip(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def mot_to_yolo_labels(
    seq_dir: Path,
    images_out: Path,
    labels_out: Path,
    frame_stride: int = 5,
    min_visibility: float = 0.25,
    min_box_size: int = 8,
    copy_images: bool = True,
) -> int:
    """Xuất ảnh + nhãn YOLO từ một sequence MOT. Trả về số ảnh đã xuất.

    Mỗi dòng nhãn: ``0 cx cy w h`` (chuẩn hoá 0–1). Box được cắt vào trong ảnh; box quá nhỏ
    hoặc bị che quá ``min_visibility`` bị loại.
    """
    seq_dir = Path(seq_dir)
    info = read_seqinfo(seq_dir)
    gt = read_gt(seq_dir / "gt" / "gt.txt")
    gt = gt[gt["vis"] >= min_visibility]
    gt = gt[(gt["w"] >= min_box_size) & (gt["h"] >= min_box_size)]

    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    W, H = float(info.im_width), float(info.im_height)
    n_written = 0
    for frame in range(1, info.seq_length + 1, frame_stride):
        src = seq_dir / info.im_dir / f"{frame:06d}{info.im_ext}"
        if not src.exists():
            continue
        stem = f"{info.name}_{frame:06d}"
        rows = gt[gt["frame"] == frame]
        lines = []
        for r in rows.itertuples(index=False):
            x1, y1 = _clip(r.x, 0, W), _clip(r.y, 0, H)
            x2, y2 = _clip(r.x + r.w, 0, W), _clip(r.y + r.h, 0, H)
            bw, bh = x2 - x1, y2 - y1
            if bw < min_box_size or bh < min_box_size:
                continue
            cx, cy = (x1 + x2) / 2 / W, (y1 + y2) / 2 / H
            lines.append(f"0 {cx:.6f} {cy:.6f} {bw / W:.6f} {bh / H:.6f}")
        (labels_out / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        dst = images_out / f"{stem}{info.im_ext}"
        if copy_images:
            shutil.copy2(src, dst)
        else:
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            dst.symlink_to(src.resolve())
        n_written += 1
    return n_written


def write_yolo_dataset_yaml(root: Path, out_file: Path, names: list[str] | None = None) -> None:
    """Sinh file yaml cho Ultralytics: ``path, train, val, test, names``."""
    names = names or ["person"]
    cfg = {
        "path": str(Path(root).resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {i: n for i, n in enumerate(names)},
    }
    Path(out_file).write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))


def summarize_yolo_split(labels_dir: Path) -> dict:
    """Đếm số ảnh, số box, số ảnh không có box trong một split YOLO."""
    labels_dir = Path(labels_dir)
    files = sorted(labels_dir.glob("*.txt"))
    n_boxes = 0
    n_empty = 0
    for f in files:
        lines = [ln for ln in f.read_text().splitlines() if ln.strip()]
        n_boxes += len(lines)
        n_empty += int(len(lines) == 0)
    return {"images": len(files), "boxes": n_boxes, "empty_images": n_empty}


def crowdhuman_odgt_to_yolo(
    odgt_path: Path, images_dir: Path, labels_out: Path, min_box_size: int = 8
) -> int:
    """Chuyển annotation CrowdHuman (.odgt, JSON lines) sang YOLO dùng box full-body.

    Trả về số file nhãn đã ghi. Cần Pillow để đọc kích thước ảnh.
    """
    import json

    from PIL import Image

    labels_out = Path(labels_out)
    labels_out.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(odgt_path, encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            img = Path(images_dir) / f"{rec['ID']}.jpg"
            if not img.exists():
                continue
            with Image.open(img) as im:
                W, H = im.size
            lines = []
            for gb in rec.get("gtboxes", []):
                if gb.get("tag") != "person":
                    continue
                if gb.get("extra", {}).get("ignore", 0) == 1:
                    continue
                x, y, w, h = gb["fbox"]
                x1, y1 = _clip(x, 0, W), _clip(y, 0, H)
                x2, y2 = _clip(x + w, 0, W), _clip(y + h, 0, H)
                bw, bh = x2 - x1, y2 - y1
                if bw < min_box_size or bh < min_box_size:
                    continue
                lines.append(
                    f"0 {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} {bw / W:.6f} {bh / H:.6f}"
                )
            (labels_out / f"{rec['ID']}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
            n += 1
    return n


def gt_to_trajectory_csv(seq_dir: Path, out_csv: Path) -> pd.DataFrame:
    """Sinh quỹ đạo ``t, track_id, x, y`` (pixel, tâm đáy box) từ ground truth MOT.

    Dùng cho module analytics (heatmap, dwell, flow) khi chưa có homography.
    """
    seq_dir = Path(seq_dir)
    info = read_seqinfo(seq_dir)
    gt = read_gt(seq_dir / "gt" / "gt.txt")
    traj = pd.DataFrame(
        {
            "t": (gt["frame"] - 1) / info.frame_rate,
            "track_id": gt["id"],
            "x": gt["x"] + gt["w"] / 2,
            "y": gt["y"] + gt["h"],
            "sequence": info.name,
        }
    ).sort_values(["track_id", "t"])
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    traj.to_csv(out_csv, index=False)
    return traj
