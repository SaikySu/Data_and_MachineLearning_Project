# EdgeRetail Insight – Theo dõi chuyển động & phân tích hành vi khách hàng trên Edge

Dự án xây dựng hệ thống chạy trên thiết bị Edge (Jetson Orin Nano / Raspberry Pi 5 + Hailo / RK3588) để
phát hiện, theo dõi khách hàng từ camera CCTV và sinh các chỉ số hành vi ẩn danh: đếm vào/ra, heatmap,
dwell time theo khu, luồng di chuyển, tương tác với kệ hàng.

## Tiến độ theo các bước

| Bước | Nội dung | Trạng thái | Tài liệu |
|---|---|---|---|
| 1 | Khảo sát dataset phù hợp | ✅ | [docs/01_khao_sat_dataset.md](docs/01_khao_sat_dataset.md) |
| 2 | Chia dữ liệu thành các mục | ✅ (thiết kế + script) | [docs/02_to_chuc_du_lieu.md](docs/02_to_chuc_du_lieu.md) |
| 3 | Đề xuất dự án | ✅ | [docs/03_de_xuat_du_an.md](docs/03_de_xuat_du_an.md) |
| 4 | Xây dựng toàn bộ dự án | ⏳ chờ yêu cầu tiếp theo | mục 9 trong tài liệu bước 3 |

## Cấu trúc

```
Customer_Behavior_Edge_Project/
├── docs/                     # 3 tài liệu của bước 1–3
├── configs/
│   ├── datasets.yaml         # nguồn dữ liệu, giấy phép, thư mục đích
│   └── split.yaml            # quy tắc chia train/val/test (theo sequence/subject/ngày)
├── scripts/
│   ├── download_datasets.py  # tải MOT17/MOT20, in hướng dẫn cho dataset cần tải thủ công
│   ├── make_sample_data.py   # sinh dữ liệu MOT giả lập để smoke-test
│   └── prepare_data.py       # chuẩn hoá: tracking (MOT) / detection (YOLO) / trajectory (CSV) / stats
├── src/edge_retail/data/     # đọc-ghi MOT, chuyển YOLO, chia split, thống kê, kiểm tra nhãn
├── tests/                    # pytest smoke test
└── data/                     # raw/, processed/, sample/ – không commit
```

## Chạy nhanh

```bash
cd Customer_Behavior_Edge_Project
pip install -r requirements.txt

# 1) Kiểm thử pipeline bằng dữ liệu mẫu (không cần tải gì)
python scripts/make_sample_data.py --n-seq 4 --frames 60
python scripts/prepare_data.py --source data/sample/MOT17-like --dataset SAMPLE \
    --ratio 0.5 0.25 0.25 --out data/sample/processed --copy
pytest tests -q

# 2) Dữ liệu thật
python scripts/download_datasets.py --list
python scripts/download_datasets.py MOT17          # tải + giải nén vào data/raw/MOT17
python scripts/download_datasets.py --all          # in hướng dẫn cho các dataset còn lại
python scripts/prepare_data.py                     # MOT17 theo split.yaml -> data/processed/
python scripts/prepare_data.py --source data/raw/CAVIAR --dataset CAVIAR --task tracking trajectory
```

Đầu ra `data/processed/`:

* `tracking/{train,val,test}/<seq>/` – MOT format, dùng trực tiếp với TrackEval / ByteTrack.
* `detection/{images,labels}/{train,val,test}` + `detection.yaml` – huấn luyện Ultralytics YOLO 1 lớp `person`.
* `trajectory/{train,val,test}/*.csv` – `t, track_id, x, y` cho module analytics.
* `stats_tracking.csv` – thống kê frame/track/box/mật độ/tỉ lệ che khuất theo split.

## Ghi chú

* Mọi dataset công khai chỉ dùng cho nghiên cứu; xem mục 4 tài liệu bước 1.
* Chia dữ liệu luôn theo sequence / subject / ngày, không theo frame, để tránh rò rỉ.
* Các bước G1–G6 (detector, tracker, analytics, action, edge app, tối ưu) sẽ được xây dựng ở bước 4.
