# Thư mục dữ liệu

Thư mục này **không commit dữ liệu** lên git (xem `.gitignore`). Cấu trúc chuẩn sau khi chạy
`scripts/download_datasets.py` và `scripts/prepare_data.py`:

```
data/
├── raw/                      # dữ liệu gốc tải về, giữ nguyên định dạng của nhà cung cấp
│   ├── MOT17/
│   ├── MOT20/
│   ├── CrowdHuman/
│   ├── MERL_Shopping/
│   ├── CAVIAR/
│   └── ATC/
├── processed/                # dữ liệu đã chuẩn hoá cho từng nhóm bài toán
│   ├── detection/            # YOLO format: images/{train,val,test}, labels/{train,val,test}
│   ├── tracking/             # MOT format theo split: {train,val,test}/<seq>/{img1,gt,seqinfo.ini}
│   ├── action/               # clip + nhãn hành động (MERL)
│   ├── trajectory/           # CSV quỹ đạo (ATC, CAVIAR gt) cho phân tích hành vi
│   └── anomaly/              # chuỗi keypoint + nhãn bất thường (RetailS)
└── sample/                   # dữ liệu mẫu giả lập để smoke-test pipeline (tạo bởi make_sample_data.py)
```
