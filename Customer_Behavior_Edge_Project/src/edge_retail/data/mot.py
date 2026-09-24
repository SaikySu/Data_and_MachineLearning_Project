"""Đọc / ghi định dạng MOTChallenge.

gt.txt: ``frame, id, x, y, w, h, conf, class, visibility`` (x, y là góc trên-trái, pixel).
seqinfo.ini: thông tin sequence (imDir, frameRate, seqLength, imWidth, imHeight).
"""

from __future__ import annotations

import configparser
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

MOT_COLUMNS = ["frame", "id", "x", "y", "w", "h", "conf", "cls", "vis"]

PEDESTRIAN_CLASS = 1  # theo quy ước MOT17/MOT20


@dataclass(frozen=True)
class SeqInfo:
    name: str
    im_dir: str
    frame_rate: float
    seq_length: int
    im_width: int
    im_height: int
    im_ext: str = ".jpg"


def read_seqinfo(seq_dir: Path) -> SeqInfo:
    """Đọc ``seqinfo.ini`` của một sequence MOT."""
    seq_dir = Path(seq_dir)
    parser = configparser.ConfigParser()
    ini = seq_dir / "seqinfo.ini"
    if not ini.exists():
        raise FileNotFoundError(f"Không tìm thấy {ini}")
    parser.read(ini)
    s = parser["Sequence"]
    return SeqInfo(
        name=s.get("name", seq_dir.name),
        im_dir=s.get("imDir", "img1"),
        frame_rate=float(s.get("frameRate", 30)),
        seq_length=int(s["seqLength"]),
        im_width=int(s["imWidth"]),
        im_height=int(s["imHeight"]),
        im_ext=s.get("imExt", ".jpg"),
    )


def write_seqinfo(seq_dir: Path, info: SeqInfo) -> None:
    seq_dir = Path(seq_dir)
    seq_dir.mkdir(parents=True, exist_ok=True)
    text = (
        "[Sequence]\n"
        f"name={info.name}\n"
        f"imDir={info.im_dir}\n"
        f"frameRate={info.frame_rate:g}\n"
        f"seqLength={info.seq_length}\n"
        f"imWidth={info.im_width}\n"
        f"imHeight={info.im_height}\n"
        f"imExt={info.im_ext}\n"
    )
    (seq_dir / "seqinfo.ini").write_text(text, encoding="utf-8")


def read_gt(gt_path: Path, pedestrian_only: bool = True) -> pd.DataFrame:
    """Đọc ``gt/gt.txt``. Với ``pedestrian_only`` chỉ giữ class 1 và conf != 0."""
    gt_path = Path(gt_path)
    df = pd.read_csv(gt_path, header=None, names=MOT_COLUMNS)
    if pedestrian_only:
        df = df[(df["cls"] == PEDESTRIAN_CLASS) & (df["conf"] != 0)]
    return df.reset_index(drop=True)


def write_gt(df: pd.DataFrame, gt_path: Path) -> None:
    gt_path = Path(gt_path)
    gt_path.parent.mkdir(parents=True, exist_ok=True)
    df[MOT_COLUMNS].to_csv(gt_path, header=False, index=False)


def list_sequences(split_dir: Path) -> list[Path]:
    """Liệt kê các thư mục sequence hợp lệ (có seqinfo.ini) trong một thư mục MOT."""
    split_dir = Path(split_dir)
    if not split_dir.exists():
        return []
    return sorted(p for p in split_dir.iterdir() if (p / "seqinfo.ini").exists())


def sequence_stats(seq_dir: Path) -> dict:
    """Thống kê nhanh cho một sequence: frame, track, box, mật độ, tỉ lệ che khuất."""
    info = read_seqinfo(seq_dir)
    gt = read_gt(Path(seq_dir) / "gt" / "gt.txt")
    n_frames = info.seq_length
    n_boxes = len(gt)
    return {
        "sequence": info.name,
        "frames": n_frames,
        "tracks": int(gt["id"].nunique()),
        "boxes": n_boxes,
        "density": round(n_boxes / n_frames, 2) if n_frames else 0.0,
        "occluded_ratio": round(float((gt["vis"] < 0.5).mean()), 3) if n_boxes else 0.0,
        "fps": info.frame_rate,
        "size": f"{info.im_width}x{info.im_height}",
    }


def validate_gt(gt: pd.DataFrame, im_width: int, im_height: int) -> list[str]:
    """Trả về danh sách lỗi (rỗng nếu hợp lệ): box âm, box ngoài ảnh, w/h <= 0."""
    errors: list[str] = []
    if (gt[["w", "h"]] <= 0).any().any():
        errors.append("có box với w hoặc h <= 0")
    if ((gt["x"] + gt["w"]) < 0).any() or ((gt["y"] + gt["h"]) < 0).any():
        errors.append("có box nằm hoàn toàn ngoài ảnh (âm)")
    if (gt["x"] > im_width).any() or (gt["y"] > im_height).any():
        errors.append("có box nằm hoàn toàn ngoài ảnh (vượt kích thước)")
    if gt.duplicated(["frame", "id"]).any():
        errors.append("có (frame, id) trùng lặp")
    return errors
