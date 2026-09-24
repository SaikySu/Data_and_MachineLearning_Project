"""Smoke test cho module edge_retail.data với dữ liệu MOT giả lập."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from edge_retail.data.mot import read_gt, read_seqinfo, sequence_stats, validate_gt  # noqa: E402
from edge_retail.data.split import assign_by_list, assign_by_ratio, check_no_overlap  # noqa: E402
from edge_retail.data.yolo import gt_to_trajectory_csv, mot_to_yolo_labels, summarize_yolo_split  # noqa: E402


@pytest.fixture(scope="module")
def sample_root(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("sample") / "train"
    subprocess.run(
        [sys.executable, str(PROJECT / "scripts" / "make_sample_data.py"), "--out", str(out),
         "--n-seq", "3", "--frames", "20", "--people", "4"],
        check=True,
    )
    return out


def test_sample_sequences_valid(sample_root: Path):
    seqs = sorted(p for p in sample_root.iterdir() if p.is_dir())
    assert len(seqs) == 3
    for seq in seqs:
        info = read_seqinfo(seq)
        gt = read_gt(seq / "gt" / "gt.txt")
        assert validate_gt(gt, info.im_width, info.im_height) == []
        st = sequence_stats(seq)
        assert st["frames"] == 20 and st["tracks"] == 4 and st["boxes"] > 0


def test_split_by_list_rejects_overlap():
    with pytest.raises(ValueError):
        assign_by_list(["a", "b"], {"train": ["a"], "val": ["a"]})
    out = assign_by_list(["a", "b", "c"], {"train": ["a"], "val": ["b"], "test": []})
    assert out == {"a": "train", "b": "val"}


def test_split_by_ratio_is_reproducible_and_complete():
    items = [f"seq{i}" for i in range(10)]
    a = assign_by_ratio(items, {"train": 0.7, "val": 0.15, "test": 0.15}, seed=1)
    b = assign_by_ratio(items, {"train": 0.7, "val": 0.15, "test": 0.15}, seed=1)
    assert a == b and set(a) == set(items)
    assert {"train", "val", "test"} == set(a.values())
    check_no_overlap(a)


def test_mot_to_yolo_and_trajectory(sample_root: Path, tmp_path: Path):
    seq = sorted(sample_root.iterdir())[0]
    n = mot_to_yolo_labels(seq, tmp_path / "img", tmp_path / "lbl", frame_stride=5, min_visibility=0.25)
    assert n == 4  # frames 1, 6, 11, 16
    summary = summarize_yolo_split(tmp_path / "lbl")
    assert summary["images"] == 4 and summary["boxes"] > 0
    for f in (tmp_path / "lbl").glob("*.txt"):
        for line in f.read_text().splitlines():
            cls, cx, cy, w, h = line.split()
            assert cls == "0"
            assert all(0.0 <= float(v) <= 1.0 for v in (cx, cy, w, h))
    traj = gt_to_trajectory_csv(seq, tmp_path / "traj.csv")
    assert set(traj.columns) >= {"t", "track_id", "x", "y"}
    assert traj["track_id"].nunique() == 4


def test_prepare_data_end_to_end(sample_root: Path, tmp_path: Path):
    out = tmp_path / "processed"
    res = subprocess.run(
        [sys.executable, str(PROJECT / "scripts" / "prepare_data.py"),
         "--source", str(sample_root.parent), "--dataset", "SAMPLE", "--ratio", "0.34", "0.33", "0.33",
         "--out", str(out), "--task", "tracking", "detection", "trajectory", "stats", "--copy"],
        capture_output=True, text=True,
    )
    assert res.returncode == 0, res.stderr
    for split in ("train", "val", "test"):
        assert any((out / "tracking" / split).iterdir())
        assert any((out / "detection" / "labels" / split).glob("*.txt"))
        assert any((out / "trajectory" / split).glob("*.csv"))
    assert (out / "detection" / "detection.yaml").exists()
    assert (out / "stats_tracking.csv").exists()
