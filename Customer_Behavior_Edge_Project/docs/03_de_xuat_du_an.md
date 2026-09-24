# Bước 3 – Đề xuất dự án: **EdgeRetail Insight** – Theo dõi chuyển động và phân tích hành vi khách hàng trên thiết bị Edge

## 1. Mục tiêu

Xây dựng hệ thống chạy **hoàn toàn trên thiết bị Edge** đặt tại cửa hàng, nhận luồng video từ camera CCTV, và trả về các chỉ số hành vi khách hàng **theo thời gian thực, ẩn danh, không gửi video lên cloud**:

1. Đếm khách vào/ra và số người hiện diện theo thời gian.
2. Bản đồ nhiệt (heatmap) vị trí khách trong cửa hàng.
3. Thời gian dừng (dwell time) tại từng khu/kệ hàng.
4. Luồng di chuyển giữa các khu (ma trận chuyển vùng, đường đi phổ biến).
5. Nhận diện tương tác với kệ hàng (với tay, cầm xem sản phẩm) và cảnh báo bất thường (tuỳ chọn).
6. Dashboard tổng hợp theo giờ/ngày chạy ngay trên Edge, đồng bộ số liệu (không phải video) về server khi có mạng.

## 2. Kiến trúc hệ thống

```
Camera RTSP/USB
      │
      ▼
┌───────────────────────────── Edge device ─────────────────────────────┐
│ 1. Capture & resize (OpenCV/GStreamer, 640×384, 10–15 fps)            │
│ 2. Detector: YOLO11n / YOLO26n fine-tune "person" (INT8, ONNX/TensorRT/│
│    NCNN/Hailo)                                                         │
│ 3. Tracker: ByteTrack (mặc định) hoặc BoT-SORT khi cần ReID nhẹ        │
│ 4. Geometry: homography pixel → mặt sàn, gán zone                     │
│ 5. Analytics engine (SQLite/DuckDB):                                   │
│      counting line │ heatmap grid │ dwell/zone │ flow matrix          │
│ 6. Action head (tuỳ chọn): pose (YOLO-pose-n) + TCN nhỏ trên keypoint │
│ 7. API (FastAPI) + dashboard (Streamlit/HTML) + MQTT publisher         │
└────────────────────────────────────────────────────────────────────────┘
      │  chỉ số ẩn danh (JSON), không video
      ▼
Server/Cloud tuỳ chọn: tổng hợp nhiều cửa hàng, BI
```

## 3. Lựa chọn công nghệ

| Thành phần | Lựa chọn chính | Phương án thay thế | Lý do |
|---|---|---|---|
| Phần cứng Edge | **NVIDIA Jetson Orin Nano 8GB** | Raspberry Pi 5 + AI HAT (Hailo-8L); RK3588 | Orin Nano chạy YOLO-n INT8 > 60 FPS, đủ 2–4 camera; Pi 5 + Hailo phù hợp bản demo 1 camera giá rẻ |
| Detector | Ultralytics **YOLO11n/YOLO26n**, fine-tune CrowdHuman + MOT17 + ảnh cửa hàng | YOLOv8n, RT-DETR-tiny | Nhỏ, có sẵn export INT8/TensorRT/NCNN/TFLite, NMS-free (YOLO26) giảm hậu xử lý trên Edge |
| Tracker | **ByteTrack** | BoT-SORT (+ ReID OSNet-0.25) | Không cần mạng ReID, chạy CPU, giữ track khi che khuất nhờ dùng cả box tin cậy thấp |
| Chiếu mặt sàn | Homography 4 điểm hiệu chuẩn tay | Calibration đầy đủ (EPFL) | Đủ cho zone/heatmap ở cửa hàng nhỏ |
| Hành động | YOLO11n-pose + TCN/GRU nhỏ trên chuỗi keypoint (train trên MERL) | Video-classifier X3D-XS | Giữ riêng tư (chỉ keypoint), rẻ tính toán |
| Lưu trữ | SQLite (event) + Parquet (aggregate) | DuckDB | Không cần server DB |
| Giao diện | FastAPI + Streamlit | Grafana qua MQTT | Nhanh, thuần Python |
| Đánh giá | TrackEval (HOTA/MOTA/IDF1), Ultralytics val (mAP), script tự viết cho chỉ số nghiệp vụ | | Chuẩn benchmark |

## 4. Kế hoạch dữ liệu (tóm tắt từ Bước 1–2)

| Tầng | Dataset train | Dataset đánh giá |
|---|---|---|
| Detection | CrowdHuman + MOT17 (train) + Roboflow retail | CrowdHuman val, MOT17-09/11 |
| Tracking | MOT17 train (fine-tune detector) | MOT17-09/11 (val), CAVIAR shopping, MOT20 (stress) |
| Analytics | Không cần train; kiểm định bằng quỹ đạo GT | ATC (zone/dwell/flow), CAVIAR (vào/ra cửa hàng) |
| Action | MERL Shopping train | MERL test (theo subject) |
| Anomaly (tuỳ chọn) | RetailS | RetailS test thật + dàn dựng |

## 5. Chỉ số thành công (KPI)

| Nhóm | Chỉ số | Mục tiêu |
|---|---|---|
| Detector | mAP50 (person) trên CrowdHuman val, mô hình INT8 | ≥ 0.80 |
| Tracker | HOTA / IDF1 trên MOT17 val (09, 11) | ≥ 55 / ≥ 65 |
| Đếm vào/ra | MAE so với GT trên CAVIAR & video tự quay | ≤ 5 % |
| Dwell time | Sai số trung bình theo zone (ATC) | ≤ 10 % |
| Action | F1 5 lớp trên MERL test | ≥ 0.70 |
| Hiệu năng Edge | FPS end-to-end 1 camera 640×384 trên Orin Nano / Pi5+Hailo | ≥ 25 / ≥ 12 |
| Tài nguyên | RAM < 2 GB, khởi động < 30 s, hoạt động offline | đạt |
| Riêng tư | Không ghi video, không lưu khuôn mặt, chỉ số ẩn danh | đạt |

## 6. Lộ trình triển khai (Bước 4 sẽ thực hiện theo yêu cầu tiếp theo)

| Giai đoạn | Nội dung | Đầu ra |
|---|---|---|
| G0. Chuẩn bị dữ liệu (đã có script) | Tải, chuẩn hoá, chia split, thống kê | `data/processed/*`, báo cáo thống kê |
| G1. Detector | Fine-tune YOLO-n trên detection set; export ONNX/TensorRT INT8; benchmark FPS | `weights/person_yolo11n.{pt,onnx,engine}` |
| G2. Tracker | Tích hợp ByteTrack, đánh giá trên MOT17 val + CAVIAR bằng TrackEval | Báo cáo HOTA/IDF1 |
| G3. Analytics engine | Homography, zone, counting line, heatmap, dwell, flow; kiểm định trên ATC | Module `analytics/`, unit test |
| G4. Action head | Pose + TCN trên MERL; gộp vào pipeline | `weights/action_tcn.onnx` |
| G5. Edge app | Pipeline đa luồng, SQLite, FastAPI, dashboard, Docker cho Jetson/Pi | Image + hướng dẫn cài |
| G6. Đánh giá & tối ưu | Đo FPS/RAM, quantization, batch, đa camera; kiểm thử thực địa | Báo cáo tổng kết |

## 7. Rủi ro & biện pháp

| Rủi ro | Biện pháp |
|---|---|
| Dataset công khai chỉ cho nghiên cứu | Dùng để phát triển; khi triển khai thật thu thập dữ liệu riêng có đồng thuận |
| Góc camera cửa hàng khác dữ liệu train → detector giảm chính xác | Fine-tune thêm 200–500 ảnh tự gán nhãn từ camera thật; augmentation góc cao |
| Che khuất nặng làm đổi ID → sai dwell time | ByteTrack + ngưỡng track buffer dài; gộp track theo khoảng cách không gian-thời gian |
| Edge thiếu tài nguyên khi thêm action head | Chạy action head theo lịch (mỗi N frame) hoặc chỉ trong zone quan tâm |
| Quy định bảo vệ dữ liệu cá nhân | Không lưu ảnh, làm mờ trong dashboard, bật/tắt module theo cấu hình |

## 8. Cấu trúc mã nguồn dự kiến

```
Customer_Behavior_Edge_Project/
├── docs/                 # 01 khảo sát, 02 tổ chức dữ liệu, 03 đề xuất (tài liệu này)
├── configs/              # datasets.yaml, split.yaml, (sau) pipeline.yaml, zones.yaml
├── scripts/              # download_datasets.py, prepare_data.py, make_sample_data.py, (sau) train/export/benchmark
├── src/edge_retail/
│   ├── data/             # đọc/ghi MOT, CrowdHuman, chia split, thống kê
│   ├── detect/           # (G1) wrapper YOLO, ONNX/TensorRT runtime
│   ├── track/            # (G2) ByteTrack
│   ├── analytics/        # (G3) zone, counting, heatmap, dwell, flow
│   ├── action/           # (G4) pose + TCN
│   └── app/              # (G5) pipeline, API, dashboard
├── tests/
└── data/                 # không commit
```

## 9. Câu hỏi cần chốt trước Bước 4

1. Phần cứng Edge mục tiêu: Jetson Orin Nano, Raspberry Pi 5 (+Hailo), hay chỉ CPU x86?
2. Số camera và loại nguồn (RTSP hay file video để demo)?
3. Phạm vi ưu tiên: chỉ đếm + heatmap + dwell (G1–G3, G5) hay bao gồm nhận diện hành động (G4)?
4. Có dữ liệu camera thật của cửa hàng để fine-tune không?
5. Giao diện mong muốn: dashboard web trên Edge, hay chỉ API/MQTT đẩy về server?
