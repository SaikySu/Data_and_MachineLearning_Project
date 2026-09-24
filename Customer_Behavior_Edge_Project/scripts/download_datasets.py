#!/usr/bin/env python3
"""Tải (hoặc in hướng dẫn tải) các dataset trong configs/datasets.yaml.

Ví dụ:
    python scripts/download_datasets.py --list
    python scripts/download_datasets.py MOT17 CAVIAR
    python scripts/download_datasets.py --all

Chỉ MOT17/MOT20 có link tải trực tiếp (zip). Các dataset còn lại yêu cầu chấp nhận điều khoản
hoặc không có link cố định nên script in hướng dẫn và tạo sẵn thư mục đích.
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

import yaml

PROJECT = Path(__file__).resolve().parents[1]
CONFIG = PROJECT / "configs" / "datasets.yaml"

DIRECT_DOWNLOAD = {"MOT17", "MOT20"}


def load_config() -> dict:
    with open(CONFIG, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def download_file(url: str, dest: Path, chunk: int = 1 << 20) -> Path:
    import requests
    from tqdm import tqdm

    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"  đã có {dest}, bỏ qua tải")
        return dest
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(tmp, "wb") as fh, tqdm(total=total, unit="B", unit_scale=True, desc=dest.name) as bar:
            for part in r.iter_content(chunk_size=chunk):
                fh.write(part)
                bar.update(len(part))
    tmp.rename(dest)
    return dest


def extract_zip(zip_path: Path, dest_dir: Path) -> None:
    print(f"  giải nén {zip_path.name} → {dest_dir}")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest_dir)


def handle(name: str, spec: dict, root: Path) -> None:
    dest = root / spec["dest"]
    dest.mkdir(parents=True, exist_ok=True)
    print(f"\n=== {name} ===")
    print(f"  giấy phép : {spec.get('license', 'n/a')}")
    print(f"  thư mục   : {dest}")
    if name in DIRECT_DOWNLOAD:
        zip_path = dest.parent / f"{name}.zip"
        try:
            download_file(spec["url"], zip_path)
            extract_zip(zip_path, dest.parent)
            print("  hoàn tất.")
            return
        except Exception as exc:  # noqa: BLE001
            print(f"  không tải tự động được ({exc}).")
    print(f"  Tải thủ công từ: {spec['url']}")
    if spec.get("notes"):
        print(f"  Ghi chú: {spec['notes'].strip()}")
    print(f"  Sau đó đặt dữ liệu vào: {dest}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("names", nargs="*", help="tên dataset (xem --list)")
    ap.add_argument("--all", action="store_true", help="xử lý mọi dataset có enabled: true")
    ap.add_argument("--list", action="store_true", help="liệt kê dataset trong config")
    args = ap.parse_args(argv)

    cfg = load_config()
    root = PROJECT / cfg.get("root", "data")
    datasets: dict = cfg["datasets"]

    if args.list or (not args.names and not args.all):
        print(f"{'Tên':<14}{'Bật':<6}{'Task':<24}Giấy phép")
        for n, s in datasets.items():
            print(f"{n:<14}{str(s.get('enabled', False)):<6}{','.join(s.get('task', [])):<24}{s.get('license', '')}")
        return 0

    names = [n for n, s in datasets.items() if s.get("enabled")] if args.all else args.names
    unknown = [n for n in names if n not in datasets]
    if unknown:
        print(f"Không có dataset: {unknown}", file=sys.stderr)
        return 1
    for n in names:
        handle(n, datasets[n], root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
